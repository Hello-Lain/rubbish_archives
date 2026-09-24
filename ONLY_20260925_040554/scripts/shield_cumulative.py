#!/usr/bin/env python3
"""Run cumulative SHIELD ablations on CHAIR or POPE."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from adapters.chair_metrics import ChairScorer, sha256_file  # noqa: E402
from methods.shield_adaptive_plausibility import (  # noqa: E402
    AdaptivePlausibilityConfig,
)
from methods.shield_inherent_bias import InherentBiasConfig  # noqa: E402
from methods.shield_pipeline import (  # noqa: E402
    ShieldStageSpec,
    build_shield_pipeline,
    custom_stage_spec,
    normalize_component_names,
    stage_spec,
)
from methods.shield_statistical_bias import StatisticalBiasConfig  # noqa: E402
from methods.shield_vulnerability_defense import (  # noqa: E402
    VulnerabilityAttack,
    VulnerabilityDefenseConfig,
)
from pope_llava_only_compare import (  # noqa: E402
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    HUB_ROOT,
    answer_label,
    compute_metrics,
    configure_cuda,
    fp8_call_counts,
    git_sha,
    load_rows,
    load_runtime,
    require_hub_path,
    resolve_device,
    resolve_dtype,
)
from shield_cumulative_runtime import (  # noqa: E402
    build_attacked_pixel_cache,
    build_clean_shield_caches,
    build_clip_feature_cache,
    build_llava_pixel_cache,
    build_noise_bias_feature,
    build_projected_feature_cache,
    build_vision_feature_cache,
    load_caption_lookup,
)
from shield_vulnerability_runtime import (  # noqa: E402
    DEFAULT_CLIP_MODEL,
    build_vulnerability_prompt_batch,
    generate_contrastive_batch,
    unique_image_rows,
)


DEFAULT_QUESTIONS = HUB_ROOT / "chair/shield/questions.jsonl"
DEFAULT_POPE = HUB_ROOT / "pope/coco/coco_pope_adversarial.json"
DEFAULT_TRAIN_INSTANCES = HUB_ROOT / "coco2014/annotations/instances_train2014.json"
DEFAULT_VAL_INSTANCES = HUB_ROOT / "coco2014/annotations/instances_val2014.json"
DEFAULT_TRAIN_CAPTIONS = HUB_ROOT / "coco2014/annotations/captions_train2014.json"
DEFAULT_VAL_CAPTIONS = HUB_ROOT / "coco2014/annotations/captions_val2014.json"
DEFAULT_CHAIR_CACHE = PROJECT_ROOT / "outputs/chair_shield_ground_truth_v2.json"
DEFAULT_CHAIR_CAPTIONS = (
    PROJECT_ROOT / "outputs/shield_llava15_chair_first_caption_20260922.jsonl"
)
DEFAULT_POPE_CAPTIONS = (
    PROJECT_ROOT / "outputs/shield_llava15_coco_pope_first_caption_20260922.jsonl"
)
DEFAULT_PROMPT = "Please describe this image in detail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chair", "pope"), required=True)
    parser.add_argument(
        "--stage",
        choices=("adaptive", "vulnerability", "statistical", "inherent", "full"),
        help=(
            "Named cumulative ablation stage. Use --components for an explicit "
            "leave-one-out/component subset."
        ),
    )
    parser.add_argument(
        "--components",
        help=(
            "Comma-separated SHIELD component subset, e.g. "
            "adaptive_plausibility,statistical_bias."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--pope-path", type=Path)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--caption-file", type=Path)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--train-instances-path", type=Path, default=DEFAULT_TRAIN_INSTANCES)
    parser.add_argument("--val-instances-path", type=Path, default=DEFAULT_VAL_INSTANCES)
    parser.add_argument("--train-captions-path", type=Path, default=DEFAULT_TRAIN_CAPTIONS)
    parser.add_argument("--val-captions-path", type=Path, default=DEFAULT_VAL_CAPTIONS)
    parser.add_argument("--chair-cache", type=Path, default=DEFAULT_CHAIR_CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp8-scope", choices=("text_mlp", "text_all"), default="text_mlp")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument(
        "--image-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--pope-answer-instruction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Append SHIELD/LLaVA's official one-word POPE answer instruction. "
            "Defaults to true for POPE and false for CHAIR."
        ),
    )
    parser.add_argument("--decoding", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cd-alpha", type=float, default=2.0)
    parser.add_argument("--cd-beta", type=float, default=0.35)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--gamma-gain", type=float, default=3.0)
    parser.add_argument("--gain-per", type=float)
    parser.add_argument("--bias-weight", type=float)
    parser.add_argument("--bias-sample-num", type=int, default=32)
    parser.add_argument("--bias-seed", type=int)
    parser.add_argument("--epsilon", type=float, default=0.14)
    parser.add_argument("--attack-steps", type=int, default=30)
    parser.add_argument("--attack-c", type=float, default=12.0)
    parser.add_argument("--attack-lr", type=float)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_chair_questions(
    path: Path,
    start: int,
    limit: int | None,
) -> list[dict[str, Any]]:
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


def resolve_stage(args: argparse.Namespace) -> ShieldStageSpec:
    if args.stage is not None and args.components is not None:
        raise ValueError("Use either --stage or --components, not both")
    if args.components is not None:
        raw_components = tuple(
            component.strip()
            for component in args.components.split(",")
            if component.strip()
        )
        if not raw_components:
            raise ValueError("--components must contain at least one component")
        # Normalize once here so output names and manifests use canonical order.
        return custom_stage_spec(normalize_component_names(raw_components))
    if args.stage is None:
        raise ValueError("One of --stage or --components is required")
    return stage_spec(args.stage)


def resolve_defaults(args: argparse.Namespace) -> None:
    args.pope_answer_instruction = resolve_pope_answer_instruction(
        args.dataset,
        args.pope_answer_instruction,
    )
    if args.dataset == "chair":
        args.questions_path = args.questions_path or DEFAULT_QUESTIONS
        args.caption_file = args.caption_file or DEFAULT_CHAIR_CAPTIONS
        args.batch_size = args.batch_size or 32
        args.image_batch_size = args.image_batch_size or 16
        args.num_workers = 4 if args.num_workers is None else args.num_workers
        args.prefetch_factor = 4 if args.prefetch_factor is None else args.prefetch_factor
        args.max_new_tokens = args.max_new_tokens or 512
        args.threshold = 0.002 if args.threshold is None else args.threshold
        args.gain_per = 0.55 if args.gain_per is None else args.gain_per
        args.bias_weight = 0.01 if args.bias_weight is None else args.bias_weight
        args.attack_lr = 0.02 if args.attack_lr is None else args.attack_lr
    else:
        args.pope_path = args.pope_path or DEFAULT_POPE
        args.caption_file = args.caption_file or DEFAULT_POPE_CAPTIONS
        args.batch_size = args.batch_size or 8
        args.image_batch_size = args.image_batch_size or 16
        args.num_workers = 4 if args.num_workers is None else args.num_workers
        args.prefetch_factor = 4 if args.prefetch_factor is None else args.prefetch_factor
        args.max_new_tokens = args.max_new_tokens or 8
        args.threshold = 0.011 if args.threshold is None else args.threshold
        args.gain_per = 0.5 if args.gain_per is None else args.gain_per
        args.bias_weight = 0.1 if args.bias_weight is None else args.bias_weight
        args.attack_lr = 0.14 if args.attack_lr is None else args.attack_lr

    args.bias_seed = args.seed if args.bias_seed is None else args.bias_seed

    if args.output is None:
        args.output = PROJECT_ROOT / (
            f"outputs/{args.dataset}_llava_7b_shield_cumulative_"
            f"{args.stage}_20260922"
        )


def resolve_pope_answer_instruction(
    dataset: str,
    requested: bool | None,
) -> bool:
    """Use the one-word suffix only for POPE's official prompt contract."""

    if requested is None:
        return dataset == "pope"
    if dataset == "chair" and requested:
        raise ValueError(
            "CHAIR does not use the POPE one-word answer instruction; "
            "pass --no-pope-answer-instruction or leave it unspecified"
        )
    return requested


def validate_args(args: argparse.Namespace, stage: ShieldStageSpec) -> None:
    if args.batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("invalid DataLoader parameters")
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid decoding arguments")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("top_k must be positive when provided")
    if args.cd_beta <= 0 or args.cd_beta > 1:
        raise ValueError("cd_beta must be in (0, 1]")
    if args.bias_sample_num < 1:
        raise ValueError("bias_sample_num must be positive")
    if stage.use_inherent_bias and args.bias_weight < 0:
        raise ValueError("bias_weight must be non-negative")
    if args.bias_seed < 0:
        raise ValueError("bias_seed must be non-negative")


def require_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    model_path = require_hub_path(args.model, "model")
    images_root = require_hub_path(args.images_root, "image root")
    if args.dataset == "chair":
        data_path = require_hub_path(args.questions_path, "questions")
        annotation_paths = [
            args.train_instances_path,
            args.val_instances_path,
            args.train_captions_path,
            args.val_captions_path,
        ]
    else:
        data_path = require_hub_path(args.pope_path, "POPE file")
        annotation_paths = []
    for path in annotation_paths:
        require_hub_path(path, "COCO annotation")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete model snapshot: {model_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    return model_path, images_root, annotation_paths


def acceleration_ok(acceleration: dict[str, Any]) -> bool:
    return (
        acceleration["effective_attention"] == "flash_attention_2"
        and acceleration["fp8_effective"] is True
        and acceleration["fp8_native_fallback_calls"] == 0
        and acceleration["fallback_events"] == []
    )


def main() -> None:
    args = parse_args()
    stage = resolve_stage(args)
    args.stage = stage.name
    resolve_defaults(args)
    validate_args(args, stage)
    pipeline = build_shield_pipeline(
        stage.components,
        adaptive_config=AdaptivePlausibilityConfig(beta=args.cd_beta),
        vulnerability_config=VulnerabilityDefenseConfig(
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            epsilon=args.epsilon,
            attack_steps=args.attack_steps,
            attack_c=args.attack_c,
            attack_lr=args.attack_lr,
        ),
        statistical_config=StatisticalBiasConfig(
            threshold=args.threshold,
            gamma_gain=args.gamma_gain,
            gain_per=args.gain_per,
        ),
        inherent_config=InherentBiasConfig(
            bias_weight=args.bias_weight,
            sample_num=args.bias_sample_num,
            seed=args.bias_seed,
        ),
        caption_token_injection=stage.caption_token_injection,
    )
    set_seed(args.seed)
    os.environ.setdefault("HF_HOME", str(HUB_ROOT.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HUB_ROOT))

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    configure_cuda(device, args.tf32)
    model_path, images_root, annotation_paths = require_dataset_paths(args)

    if args.dataset == "chair":
        rows = load_chair_questions(args.questions_path, args.start, args.limit)
        feature_rows = rows
    else:
        rows = load_rows(args.pope_path, args.start, args.limit)
        feature_rows = unique_image_rows(rows)
    image_paths = sorted(
        {(images_root / str(row["image"])).resolve() for row in feature_rows}
    )
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} images, first={missing_images[0]}"
        )

    captions: dict[str, str] = {}
    needs_caption = pipeline.requires_caption
    if needs_caption:
        if not args.caption_file.is_file():
            raise FileNotFoundError(args.caption_file)
        captions = load_caption_lookup(args.caption_file)
        missing_captions = [
            str(row["image"])
            for row in feature_rows
            if str(row["image"]) not in captions
        ]
        if missing_captions:
            raise KeyError(f"Missing captions for {len(missing_captions)} images")

    model, state, fp8_recipe = load_runtime(args, model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "left"

    feature_started = time.perf_counter()
    clean_pixels = build_llava_pixel_cache(
        processor,
        image_paths,
        args.image_batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.progress,
    )
    raw_clean = build_vision_feature_cache(
        model,
        image_paths,
        clean_pixels,
        device,
        dtype,
        args.image_batch_size,
        args.progress,
        "clean-vision-features",
    )

    clip_model: CLIPModel | None = None
    clip_processor: CLIPProcessor | None = None
    clip_features: dict[str, torch.Tensor] = {}
    if pipeline.requires_clip:
        clip_path = require_hub_path(args.clip_model, "CLIP model")
        clip_model = CLIPModel.from_pretrained(
            str(clip_path),
            local_files_only=True,
        ).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(
            str(clip_path),
            local_files_only=True,
            use_fast=False,
        )
    if pipeline.statistical is not None:
        assert clip_model is not None and clip_processor is not None
        clip_features = build_clip_feature_cache(
            clip_model,
            clip_processor,
            feature_rows,
            captions,
            device,
            args.batch_size,
            args.progress,
        )

    attacked_pixels: dict[str, torch.Tensor] = {}
    raw_adversarial: dict[str, torch.Tensor] = {}
    attack_config = pipeline.vulnerability.config if pipeline.vulnerability else None
    if pipeline.requires_adversarial_branch:
        assert clip_model is not None and clip_processor is not None
        assert attack_config is not None
        attacked_pixels = build_attacked_pixel_cache(
            image_paths,
            captions,
            clip_model,
            clip_processor,
            VulnerabilityAttack(attack_config),
            clean_pixels,
            device,
            args.progress,
        )
        raw_adversarial = build_vision_feature_cache(
            model,
            image_paths,
            attacked_pixels,
            device,
            dtype,
            args.image_batch_size,
            args.progress,
            "adversarial-vision-features",
        )

    bias_features: torch.Tensor | None = None
    if pipeline.inherent is not None:
        first_pixels = clean_pixels[str(image_paths[0])]
        bias_features = build_noise_bias_feature(
            model,
            first_pixels,
            pipeline.inherent.config.sample_num,
            device,
            dtype,
            seed=pipeline.inherent.config.seed,
        )

    if pipeline.requires_feature_transform:
        clean_cache, caption_embedding_cache, selected_counts = (
            build_clean_shield_caches(
                model,
                processor,
                image_paths,
                raw_clean,
                clip_features,
                captions,
                bias_features,
                pipeline,
                device=device,
                dtype=dtype,
                progress=args.progress,
            )
        )
    else:
        clean_cache = build_projected_feature_cache(
            model,
            image_paths,
            raw_clean,
            device,
            dtype,
            args.image_batch_size,
            args.progress,
            "clean-projected-features",
        )
        caption_embedding_cache = {}
        selected_counts = {}

    if pipeline.requires_adversarial_branch:
        adversarial_cache = build_projected_feature_cache(
            model,
            image_paths,
            raw_adversarial,
            device,
            dtype,
            args.image_batch_size,
            args.progress,
            "adversarial-projected-features",
        )
    else:
        adversarial_cache = clean_cache

    if clip_model is not None:
        del clip_model, clip_processor
    del clean_pixels, attacked_pixels, raw_clean, raw_adversarial
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    feature_elapsed = time.perf_counter() - feature_started
    state.image_feature_count = len(clean_cache)

    predictions: list[int] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    iterator = range(0, len(rows), args.batch_size)
    if args.progress:
        iterator = tqdm(iterator, desc=f"shield-{args.stage}")
    for start in iterator:
        batch_rows = rows[start : start + args.batch_size]
        clean_inputs = build_vulnerability_prompt_batch(
            model,
            processor,
            batch_rows,
            images_root,
            clean_cache,
            device,
            caption_embedding_cache=caption_embedding_cache,
            pope_answer_instruction=args.pope_answer_instruction,
        )
        adversarial_inputs = build_vulnerability_prompt_batch(
            model,
            processor,
            batch_rows,
            images_root,
            adversarial_cache,
            device,
            pope_answer_instruction=args.pope_answer_instruction,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        answers, generations = generate_contrastive_batch(
            model,
            processor,
            clean_inputs,
            adversarial_inputs,
            args,
            state,
            fp8_recipe,
            pipeline=pipeline,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for row, answer, generation in zip(
            batch_rows,
            answers,
            generations,
            strict=True,
        ):
            if args.dataset == "chair":
                records.append(
                    {
                        "question_id": int(row["question_id"]),
                        "image_id": image_id_from_name(str(row["image"])),
                        "image": str(row["image"]),
                        "prompt": str(row["text"]),
                        "caption": answer,
                        "generated_tokens": int(generation["generated_tokens"]),
                        "selected_caption_tokens": selected_counts.get(
                            str(row["image"]),
                            0,
                        ),
                    }
                )
            else:
                label = 1 if str(row["label"]).lower() == "yes" else 0
                prediction = answer_label(answer)
                predictions.append(prediction)
                labels.append(label)
                records.append(
                    {
                        "question_id": row.get("question_id"),
                        "image": str(row["image"]),
                        "question": str(row["text"]),
                        "label": label,
                        "prediction": prediction,
                        "answer": answer,
                        "correct": prediction == label,
                        "generated_tokens": int(generation["generated_tokens"]),
                        "selected_caption_tokens": selected_counts.get(
                            str(row["image"]),
                            0,
                        ),
                    }
                )
        if args.progress and args.dataset == "pope":
            iterator.set_postfix(  # type: ignore[union-attr]
                rows=len(records),
                acc=f"{compute_metrics(predictions, labels)['accuracy']:.4f}",
            )
    generation_elapsed = time.perf_counter() - generation_started
    if state.fp8_requested:
        state.fp8_calls, state.fp8_native_fallback_calls = fp8_call_counts(model)

    summary: dict[str, Any] = {
        "status": "SMOKE",
        "method": "shield_cumulative",
        "stage": args.stage,
        "dataset": args.dataset,
        "model": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_revision": model_path.name,
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
            "pope_answer_instruction": args.pope_answer_instruction,
        },
        "shield_components": pipeline.metadata(),
        "shield_ablation": {
            "use_adaptive_plausibility": stage.use_adaptive_plausibility,
            "use_vulnerability_defense": stage.use_vulnerability_defense,
            "use_statistical_bias": stage.use_statistical_bias,
            "use_inherent_bias": stage.use_inherent_bias,
            "caption_token_injection": stage.caption_token_injection,
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "threshold": args.threshold,
            "gamma_gain": args.gamma_gain,
            "gain_per": args.gain_per,
            "bias_weight": (
                args.bias_weight if stage.use_inherent_bias else 0.0
            ),
            "bias_sample_num": args.bias_sample_num,
            "bias_seed": args.bias_seed if stage.use_inherent_bias else None,
            "bias_input_dtype": (
                "float16"
                if stage.use_inherent_bias and device.type == "cuda"
                else str(dtype).replace("torch.", "")
            ),
            "epsilon": args.epsilon,
            "attack_steps": args.attack_steps,
            "attack_c": args.attack_c,
            "attack_lr": args.attack_lr,
        },
        "acceleration": state.as_dict(),
        "runtime": {
            "feature_cache_seconds": feature_elapsed,
            "generation_seconds": generation_elapsed,
            "total_seconds": feature_elapsed + generation_elapsed,
            "rows_per_second": (
                len(records) / generation_elapsed if generation_elapsed else 0.0
            ),
            "generated_tokens": sum(
                int(record["generated_tokens"]) for record in records
            ),
            "selected_caption_tokens": sum(selected_counts.values()),
        },
        "predictions_path": None,
        "git_sha": git_sha(),
    }

    if args.dataset == "chair":
        scorer = ChairScorer(
            train_instances_path=require_hub_path(
                args.train_instances_path,
                "COCO annotation",
            ),
            val_instances_path=require_hub_path(
                args.val_instances_path,
                "COCO annotation",
            ),
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
        summary["chair"] = {
            **scored["overall_metrics"],
            "captions_with_hallucination": sum(
                int(record["chair"]["CHAIRs"]) for record in records
            ),
        }
        summary["questions_path"] = str(args.questions_path)
        summary["questions_sha256"] = sha256_file(args.questions_path)
    else:
        summary["pope_path"] = str(args.pope_path)
        summary["pope_sha256"] = sha256_file(args.pope_path)
        summary["unique_images"] = len(image_paths)
        summary["metrics"] = compute_metrics(predictions, labels)

    output_prefix = args.output.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_prefix.with_suffix(".jsonl")
    metrics_path = output_prefix.with_suffix(".metrics.json")
    predictions_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary["predictions_path"] = str(predictions_path)
    expected_count = 500 if args.dataset == "chair" else 3000
    if len(records) == expected_count and acceleration_ok(summary["acceleration"]):
        summary["status"] = "PASS"
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
