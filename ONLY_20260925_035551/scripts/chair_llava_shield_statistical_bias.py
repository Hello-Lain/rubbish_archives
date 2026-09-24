#!/usr/bin/env python3
"""Evaluate SHIELD's Statistical Bias module on the fixed CHAIR-500 split."""

# The script adds the repository's ``src`` directory before importing the
# project packages; those imports intentionally follow the path setup.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adapters.chair_metrics import ChairScorer, sha256_file
from methods.shield_statistical_bias import ShieldStatisticalBias, StatisticalBiasConfig
from pope_llava_only_compare import (
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    HUB_ROOT,
    ImageDataset,
    configure_cuda,
    fp8_call_counts,
    format_pope_prompt,
    generate_batch,
    git_sha,
    image_collate,
    image_token_count,
    load_runtime,
    require_hub_path,
    resolve_device,
    resolve_dtype,
    tokenize_legacy_image_prompt,
)


DEFAULT_QUESTIONS = HUB_ROOT / "chair/shield/questions.jsonl"
DEFAULT_TRAIN_INSTANCES = HUB_ROOT / "coco2014/annotations/instances_train2014.json"
DEFAULT_VAL_INSTANCES = HUB_ROOT / "coco2014/annotations/instances_val2014.json"
DEFAULT_TRAIN_CAPTIONS = HUB_ROOT / "coco2014/annotations/captions_train2014.json"
DEFAULT_VAL_CAPTIONS = HUB_ROOT / "coco2014/annotations/captions_val2014.json"
DEFAULT_CHAIR_CACHE = PROJECT_ROOT / "outputs/chair_shield_ground_truth_v2.json"
# SHIELD's official CHAIR runner consumes the pre-generated naive-caption
# file, not the final 512-token Vanilla evaluation captions.
DEFAULT_CAPTIONS = PROJECT_ROOT / "outputs/shield_llava15_chair_first_caption_20260922.jsonl"
DEFAULT_PROMPT = "Please describe this image in detail."
DEFAULT_CLIP_MODEL = (
    HUB_ROOT
    / "models--openai--clip-vit-large-patch14-336"
    / "snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/chair_llava_7b_shield_statistical_bias_sourcecap_b32_20260922"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--caption-file", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--train-instances-path", type=Path, default=DEFAULT_TRAIN_INSTANCES)
    parser.add_argument("--val-instances-path", type=Path, default=DEFAULT_VAL_INSTANCES)
    parser.add_argument("--train-captions-path", type=Path, default=DEFAULT_TRAIN_CAPTIONS)
    parser.add_argument("--val-captions-path", type=Path, default=DEFAULT_VAL_CAPTIONS)
    parser.add_argument("--chair-cache", type=Path, default=DEFAULT_CHAIR_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp8-scope", choices=("text_mlp", "text_all"), default="text_mlp")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--image-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required for the reproducible CHAIR protocol.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--decoding", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--gamma-gain", type=float, default=3.0)
    parser.add_argument("--gain-per", type=float, default=0.55)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_questions(path: Path, start: int, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    end = None if limit is None else start + limit
    selected = rows[start:end]
    if not selected:
        raise ValueError(f"No CHAIR rows selected from {path}")
    return selected


def image_id_from_name(image_name: str) -> int:
    return int(Path(image_name).stem.split("_")[-1])


def load_caption_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = row.get("image")
        text = row.get("text", row.get("caption"))
        if image is None or text is None:
            raise KeyError("caption file rows must contain image and text/caption")
        lookup[str(image)] = str(text)
    if not lookup:
        raise ValueError(f"Caption file is empty: {path}")
    return lookup


def process_caption(caption: str) -> str:
    """Match SHIELD's caption normalization used for LLaVA token injection."""

    processed = ".".join(caption.replace("\n\n", "").split(".")[:-1])
    return processed + ". "


@torch.inference_mode()
def build_raw_image_feature_cache(
    model: LlavaForConditionalGeneration,
    processor: Any,
    image_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    progress: bool,
) -> dict[str, torch.Tensor]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": image_collate,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(ImageDataset(image_paths), **loader_kwargs)
    vision_layer = model.config.vision_feature_layer
    select_strategy = model.config.vision_feature_select_strategy
    cache: dict[str, torch.Tensor] = {}
    iterator = tqdm(loader, desc="vision-features", disable=not progress)
    for images, paths in iterator:
        pixel_values = processor.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        if device.type == "cuda":
            pixel_values = pixel_values.pin_memory()
        pixel_values = pixel_values.to(
            device=device,
            dtype=dtype,
            non_blocking=device.type == "cuda",
        )
        outputs = model.vision_tower(
            pixel_values,
            output_hidden_states=True,
        )
        if not isinstance(vision_layer, int):
            raise ValueError("Statistical Bias runner expects one vision_feature_layer")
        features = outputs.hidden_states[vision_layer]
        if select_strategy == "default":
            features = features[:, 1:]
        for path, feature in zip(paths, features, strict=True):
            cache[path] = feature.detach()
    if len(cache) != len(image_paths):
        raise RuntimeError(
            f"Vision feature cache incomplete: {len(cache)}/{len(image_paths)}"
        )
    return cache


@torch.inference_mode()
def build_clip_feature_cache(
    clip_model: CLIPModel,
    clip_processor: Any,
    rows: list[dict[str, Any]],
    captions: dict[str, str],
    device: torch.device,
    batch_size: int,
    progress: bool,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    iterator = range(0, len(rows), batch_size)
    if progress:
        iterator = tqdm(iterator, desc="clip-text-features")
    for start in iterator:
        batch_rows = rows[start : start + batch_size]
        texts = [process_caption(captions[str(row["image"])]) for row in batch_rows]
        inputs = clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state[:, 1:]
        lengths = attention_mask.sum(dim=-1).tolist()
        for row, feature, length in zip(batch_rows, hidden, lengths, strict=True):
            token_count = max(int(length) - 1, 0)
            cache[str(row["image"])] = feature[:token_count].detach()
    if len(cache) != len(rows):
        raise RuntimeError(f"CLIP feature cache incomplete: {len(cache)}/{len(rows)}")
    return cache


@torch.inference_mode()
def build_statistical_prompt_batch(
    model: LlavaForConditionalGeneration,
    processor: Any,
    rows: list[dict[str, Any]],
    image_root: Path,
    raw_image_cache: dict[str, torch.Tensor],
    clip_feature_cache: dict[str, torch.Tensor],
    captions: dict[str, str],
    device: torch.device,
    module: ShieldStatisticalBias,
    pope_answer_instruction: bool = False,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    image_count = image_token_count(processor)
    prompts = [
        format_pope_prompt(
            str(row["text"]),
            one_word=pope_answer_instruction,
        )
        for row in rows
    ]
    input_id_rows = [
        tokenize_legacy_image_prompt(
            processor.tokenizer,
            prompt,
            int(model.config.image_token_index),
            image_count,
        )
        for prompt in prompts
    ]
    tokenized = processor.tokenizer.pad(
        [
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
            for input_ids in input_id_rows
        ],
        padding=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    text_embeds = model.get_input_embeddings()(input_ids)
    input_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    selected_counts: list[int] = []
    for index, row in enumerate(rows):
        image_name = str(row["image"])
        image_path = str((image_root / image_name).resolve())
        try:
            raw_features = raw_image_cache[image_path]
            clip_features = clip_feature_cache[image_name]
        except KeyError as exc:
            raise KeyError(f"Missing cached features for {image_name}") from exc
        enhanced, selected_indices = module.enhance(raw_features, clip_features)
        projected = model.multi_modal_projector(
            enhanced.to(dtype=text_embeds.dtype).unsqueeze(0)
        ).squeeze(0)

        image_positions = torch.where(
            input_ids[index] == int(model.config.image_token_index)
        )[0]
        if image_positions.numel() != image_count:
            raise RuntimeError(
                f"Expected {image_count} image tokens, got {image_positions.numel()}"
            )
        image_start = int(image_positions[0].item())
        image_end = int(image_positions[-1].item())

        caption_text = process_caption(captions[image_name])
        input_caption_ids = processor.tokenizer(
            caption_text,
            return_tensors="pt",
        )["input_ids"][0].to(device)
        input_caption_embeds = model.get_input_embeddings()(input_caption_ids)
        selected_caption_embeds = module.map_text_segments(
            selected_indices,
            int(clip_features.shape[0]),
            input_caption_embeds,
        ).to(dtype=text_embeds.dtype)

        input_rows.append(
            torch.cat(
                (
                    text_embeds[index, :image_start],
                    projected,
                    selected_caption_embeds,
                    text_embeds[index, image_end + 1 :],
                ),
                dim=0,
            )
        )
        mask_rows.append(
            torch.cat(
                (
                    attention_mask[index, :image_start],
                    torch.ones(
                        projected.shape[0] + selected_caption_embeds.shape[0],
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                    attention_mask[index, image_end + 1 :],
                ),
                dim=0,
            )
        )
        selected_counts.append(int(selected_caption_embeds.shape[0]))

    max_length = max(row.shape[0] for row in input_rows)
    padded_embeds: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for row_embeds, row_mask in zip(input_rows, mask_rows, strict=True):
        pad_length = max_length - row_embeds.shape[0]
        padded_embeds.append(
            torch.cat(
                (
                    torch.zeros(
                        (pad_length, row_embeds.shape[1]),
                        dtype=row_embeds.dtype,
                        device=device,
                    ),
                    row_embeds,
                ),
                dim=0,
            )
        )
        padded_masks.append(
            torch.cat(
                (
                    torch.zeros(
                        pad_length,
                        dtype=row_mask.dtype,
                        device=device,
                    ),
                    row_mask,
                ),
                dim=0,
            )
        )
    attention_mask_batch = torch.stack(padded_masks, dim=0)
    # The custom caption injection makes prompt lengths differ within a
    # batch.  Keep left padding for the last-token readout, but restore the
    # logical per-example positions that single-example SHIELD uses.
    position_ids = attention_mask_batch.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask_batch == 0, 0)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    dummy_input_ids = torch.full(
        (len(rows), max_length),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    return {
        "input_ids": dummy_input_ids,
        "attention_mask": attention_mask_batch,
        "position_ids": position_ids,
        "inputs_embeds": torch.stack(padded_embeds, dim=0),
    }, selected_counts


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("num_workers must be >= 0 and prefetch_factor must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    configure_cuda(device, args.tf32)
    model_path = require_hub_path(args.model, "model")
    questions_path = require_hub_path(args.questions_path, "questions")
    images_root = require_hub_path(args.images_root, "images")
    clip_path = require_hub_path(args.clip_model, "CLIP model")
    for path in (
        args.train_instances_path,
        args.val_instances_path,
        args.train_captions_path,
        args.val_captions_path,
    ):
        require_hub_path(path, "COCO annotation")
    for path in (model_path, clip_path):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Incomplete model snapshot: {path}")
    if not args.caption_file.is_file():
        raise FileNotFoundError(f"Caption file not found: {args.caption_file}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {images_root}")

    rows = load_questions(args.questions_path, args.start, args.limit)
    captions = load_caption_lookup(args.caption_file)
    missing_captions = [row["image"] for row in rows if row["image"] not in captions]
    if missing_captions:
        raise KeyError(f"Missing captions for {len(missing_captions)} images")
    image_paths = sorted(
        {(images_root / str(row["image"])).resolve() for row in rows}
    )
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images, first={missing_images[0]}")

    model, state, fp8_recipe = load_runtime(args, model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    clip_model = CLIPModel.from_pretrained(
        str(clip_path),
        local_files_only=True,
    ).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(
        str(clip_path),
        local_files_only=True,
        use_fast=False,
    )

    feature_started = time.perf_counter()
    raw_image_cache = build_raw_image_feature_cache(
        model=model,
        processor=processor,
        image_paths=image_paths,
        device=device,
        dtype=dtype,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        progress=args.progress,
    )
    clip_feature_cache = build_clip_feature_cache(
        clip_model=clip_model,
        clip_processor=clip_processor,
        rows=rows,
        captions=captions,
        device=device,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    feature_elapsed = time.perf_counter() - feature_started
    state.image_feature_count = len(raw_image_cache)

    module = ShieldStatisticalBias(
        StatisticalBiasConfig(
            threshold=args.threshold,
            gamma_gain=args.gamma_gain,
            gain_per=args.gain_per,
        )
    )
    records: list[dict[str, Any]] = []
    selected_counts: list[int] = []
    generation_started = time.perf_counter()
    iterator = range(0, len(rows), args.batch_size)
    if args.progress:
        iterator = tqdm(iterator, desc="shield-statistical-bias")
    for start in iterator:
        real_rows = rows[start : start + args.batch_size]
        inputs, batch_selected_counts = build_statistical_prompt_batch(
            model=model,
            processor=processor,
            rows=real_rows,
            image_root=images_root,
            raw_image_cache=raw_image_cache,
            clip_feature_cache=clip_feature_cache,
            captions=captions,
            device=device,
            module=module,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        answers, generations = generate_batch(
            model=model,
            processor=processor,
            inputs=inputs,
            image_positions=[[] for _ in real_rows],
            method="vanilla",
            args=args,
            state=state,
            fp8_recipe=fp8_recipe,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for row, answer, generation, selected_count in zip(
            real_rows,
            answers,
            generations,
            batch_selected_counts,
            strict=True,
        ):
            records.append(
                {
                    "question_id": int(row["question_id"]),
                    "image_id": image_id_from_name(str(row["image"])),
                    "image": str(row["image"]),
                    "prompt": str(row["text"]),
                    "caption": answer,
                    "generated_tokens": int(generation["generated_tokens"]),
                    "statistical_bias_selected_tokens": selected_count,
                }
            )
            selected_counts.append(selected_count)
        if args.progress:
            iterator.set_postfix(  # type: ignore[union-attr]
                rows=len(records),
                tokens=sum(int(record["generated_tokens"]) for record in records),
            )
    generation_elapsed = time.perf_counter() - generation_started

    if state.fp8_requested:
        state.fp8_calls, state.fp8_native_fallback_calls = fp8_call_counts(model)

    image_ids = [int(row["image_id"]) for row in records]
    scorer = ChairScorer(
        train_instances_path=require_hub_path(args.train_instances_path, "COCO annotation"),
        val_instances_path=require_hub_path(args.val_instances_path, "COCO annotation"),
        train_captions_path=require_hub_path(args.train_captions_path, "COCO annotation"),
        val_captions_path=require_hub_path(args.val_captions_path, "COCO annotation"),
        image_ids=image_ids,
        cache_path=args.chair_cache,
    )
    scored = scorer.score(records)
    for record, sentence in zip(records, scored["sentences"], strict=True):
        record["chair"] = sentence["metrics"]
        record["objects"] = sentence["generated_objects"]

    acceleration = state.as_dict()
    acceleration_ok = (
        acceleration["effective_attention"] == "flash_attention_2"
        and acceleration["fp8_effective"] is True
        and acceleration["fp8_native_fallback_calls"] == 0
        and acceleration["fallback_events"] == []
    )
    full_protocol = len(records) == 500
    status = "PASS" if full_protocol and acceleration_ok else "SMOKE"
    output_prefix = args.output.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_prefix.with_suffix(".jsonl")
    metrics_path = output_prefix.with_suffix(".metrics.json")
    predictions_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "status": status,
        "method": "shield_statistical_bias",
        "model": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_revision": model_path.name,
        "questions_path": str(questions_path),
        "questions_sha256": sha256_file(questions_path),
        "caption_file": str(args.caption_file.resolve()),
        "caption_file_sha256": sha256_file(args.caption_file),
        "clip_model": str(clip_path),
        "clip_model_config_sha256": sha256_file(clip_path / "config.json"),
        "images_root": str(images_root),
        "count": len(records),
        "protocol": {
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "image_batch_size": args.image_batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "cache": "dynamic",
            "image_feature_cache": True,
            "prompt": DEFAULT_PROMPT,
        },
        "chair_protocol": {
            "name": "SHIELD-compatible",
            "source": "hukcc/SHIELD",
            "questions": "500 fixed COCO val2014 images",
            "prompt": DEFAULT_PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "ground_truth": "COCO train2014+val2014 instances and captions",
            "synonyms": "SHIELD chair_eval.py",
            "counting": (
                "CHAIRs=hallucinated_captions/captions; "
                "CHAIRi=hallucinated_object_mentions/object_mentions"
            ),
            "recall": "macro mean of per-image grounded COCO object coverage",
        },
        "acceleration": acceleration,
        "statistical_bias": {
            "threshold": args.threshold,
            "gamma_gain": args.gamma_gain,
            "gain_per": args.gain_per,
            "bias_weight": 0.0,
            "use_cd": False,
            "selected_caption_tokens": sum(selected_counts),
        },
        "chair": {
            **scored["overall_metrics"],
            "captions_with_hallucination": sum(
                int(record["chair"]["CHAIRs"]) for record in records
            ),
        },
        "runtime": {
            "feature_cache_seconds": feature_elapsed,
            "generation_seconds": generation_elapsed,
            "total_seconds": feature_elapsed + generation_elapsed,
            "rows_per_second": (
                len(records) / generation_elapsed if generation_elapsed else 0.0
            ),
            "generated_tokens": sum(int(record["generated_tokens"]) for record in records),
            "selected_caption_tokens": sum(selected_counts),
        },
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "predictions_path": str(predictions_path),
        "scorer": {
            "schema_version": scored["schema_version"],
            "cache_path": str(args.chair_cache.resolve()),
            "cache_sha256": sha256_file(args.chair_cache),
        },
        "git_sha": git_sha(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
