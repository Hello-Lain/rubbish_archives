#!/usr/bin/env python3
"""Evaluate the LLaVA Vanilla baseline on the fixed SHIELD-compatible CHAIR split."""

# The script reuses the shared model runtime and CHAIR adapter.  Those imports
# intentionally follow the path setup below.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adapters.chair_metrics import ChairScorer, sha256_file
from pope_llava_only_compare import (
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    HUB_ROOT,
    build_image_feature_cache,
    build_prompt_batch,
    configure_cuda,
    fp8_call_counts,
    generate_batch,
    git_sha,
    load_runtime,
    require_hub_path,
    resolve_device,
    resolve_dtype,
)


DEFAULT_QUESTIONS = HUB_ROOT / "chair/shield/questions.jsonl"
DEFAULT_TRAIN_INSTANCES = HUB_ROOT / "coco2014/annotations/instances_train2014.json"
DEFAULT_VAL_INSTANCES = HUB_ROOT / "coco2014/annotations/instances_val2014.json"
DEFAULT_TRAIN_CAPTIONS = HUB_ROOT / "coco2014/annotations/captions_train2014.json"
DEFAULT_VAL_CAPTIONS = HUB_ROOT / "coco2014/annotations/captions_val2014.json"
DEFAULT_CHAIR_CACHE = PROJECT_ROOT / "outputs/chair_shield_ground_truth_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/chair_llava_7b_shield_vanilla_fixed_b32_20260922"
)
DEFAULT_PROMPT = "Please describe this image in detail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(
        "--train-instances-path",
        type=Path,
        default=DEFAULT_TRAIN_INSTANCES,
    )
    parser.add_argument("--val-instances-path", type=Path, default=DEFAULT_VAL_INSTANCES)
    parser.add_argument(
        "--train-captions-path",
        type=Path,
        default=DEFAULT_TRAIN_CAPTIONS,
    )
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
    parser.add_argument(
        "--fp8-scope",
        choices=("text_mlp", "text_all"),
        default="text_mlp",
    )
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
    annotation_paths = (
        args.train_instances_path,
        args.val_instances_path,
        args.train_captions_path,
        args.val_captions_path,
    )
    for path in annotation_paths:
        require_hub_path(path, "COCO annotation")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete model snapshot: {model_path}")
    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)

    rows = load_questions(args.questions_path, args.start, args.limit)
    image_paths = sorted(
        {(images_root / str(row["image"])).resolve() for row in rows}
    )
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} images, first={missing_images[0]}"
        )

    model, state, fp8_recipe = load_runtime(args, model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    feature_started = time.perf_counter()
    if not args.image_feature_cache:
        raise ValueError(
            "This evaluator requires --image-feature-cache for the reproducible "
            "CHAIR protocol; disablement is intentionally unsupported."
        )
    image_cache = build_image_feature_cache(
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
    feature_elapsed = time.perf_counter() - feature_started
    state.image_feature_count = len(image_cache)

    records: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    iterator = range(0, len(rows), args.batch_size)
    if args.progress:
        iterator = tqdm(iterator, desc="vanilla")
    for start in iterator:
        batch_rows = rows[start : start + args.batch_size]
        inputs, image_positions = build_prompt_batch(
            model=model,
            processor=processor,
            rows=batch_rows,
            image_root=images_root,
            image_cache=image_cache,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        answers, generations = generate_batch(
            model=model,
            processor=processor,
            inputs=inputs,
            image_positions=image_positions,
            method="vanilla",
            args=args,
            state=state,
            fp8_recipe=fp8_recipe,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for row, answer, generation in zip(
            batch_rows,
            answers,
            generations,
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
                }
            )
        if args.progress:
            iterator.set_postfix(  # type: ignore[union-attr]
                rows=len(records),
                tokens=sum(int(record["generated_tokens"]) for record in records),
            )
    generation_elapsed = time.perf_counter() - generation_started

    if state.fp8_requested:
        state.fp8_calls, state.fp8_native_fallback_calls = fp8_call_counts(model)

    scorer = ChairScorer(
        train_instances_path=require_hub_path(
            args.train_instances_path,
            "COCO annotation",
        ),
        val_instances_path=require_hub_path(args.val_instances_path, "COCO annotation"),
        train_captions_path=require_hub_path(
            args.train_captions_path,
            "COCO annotation",
        ),
        val_captions_path=require_hub_path(
            args.val_captions_path,
            "COCO annotation",
        ),
        image_ids=[int(record["image_id"]) for record in records],
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
        "method": "vanilla",
        "model": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_revision": model_path.name,
        "questions_path": str(questions_path),
        "questions_sha256": sha256_file(questions_path),
        "train_instances_path": str(args.train_instances_path.resolve()),
        "train_instances_sha256": sha256_file(args.train_instances_path),
        "val_instances_path": str(args.val_instances_path.resolve()),
        "val_instances_sha256": sha256_file(args.val_instances_path),
        "train_captions_path": str(args.train_captions_path.resolve()),
        "train_captions_sha256": sha256_file(args.train_captions_path),
        "val_captions_path": str(args.val_captions_path.resolve()),
        "val_captions_sha256": sha256_file(args.val_captions_path),
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
        "only": None,
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
            "only_changed_steps": 0,
            "only_positive_steps": 0,
            "only_negative_steps": 0,
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
