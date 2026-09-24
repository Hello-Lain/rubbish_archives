#!/usr/bin/env python3
"""Evaluate SHIELD Statistical Bias on the fixed POPE adversarial split."""

# The script reuses the shared model runtime and the CHAIR-side Statistical
# Bias feature preparation.  Those imports intentionally follow the path setup.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adapters.chair_metrics import sha256_file
from chair_llava_shield_statistical_bias import (
    DEFAULT_CLIP_MODEL,
    build_clip_feature_cache,
    build_raw_image_feature_cache,
    build_statistical_prompt_batch,
    load_caption_lookup,
)
from methods.shield_statistical_bias import ShieldStatisticalBias, StatisticalBiasConfig
from pope_llava_only_compare import (
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    HUB_ROOT,
    answer_label,
    compute_metrics,
    configure_cuda,
    format_pope_prompt,
    fp8_call_counts,
    generate_batch,
    git_sha,
    load_runtime,
    load_rows,
    require_hub_path,
    resolve_device,
    resolve_dtype,
)


DEFAULT_POPE = HUB_ROOT / "pope/coco/coco_pope_adversarial.json"
DEFAULT_CAPTIONS = PROJECT_ROOT / (
    "outputs/shield_llava15_coco_pope_first_caption_20260922.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/pope_llava_7b_adversarial_shield_statistical_bias_sourcecap_20260922"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pope-path", type=Path, default=DEFAULT_POPE)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--caption-file", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--image-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required for the reproducible POPE protocol.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--pope-answer-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append SHIELD/LLaVA's official one-word POPE answer instruction.",
    )
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
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("top_k must be positive when provided")


def unique_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        image = str(row["image"])
        if image not in seen:
            seen.add(image)
            unique.append(row)
    return unique


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    os.environ.setdefault("HF_HOME", str(HUB_ROOT.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HUGGINGFACE_DATASETS_CACHE", str(HUB_ROOT))

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    configure_cuda(device, args.tf32)
    model_path = require_hub_path(args.model, "model")
    pope_path = require_hub_path(args.pope_path, "POPE file")
    images_root = require_hub_path(args.images_root, "image root")
    clip_path = require_hub_path(args.clip_model, "CLIP model")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model snapshot is incomplete: {model_path}")
    if not pope_path.is_file():
        raise FileNotFoundError(pope_path)
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    if not args.caption_file.is_file():
        raise FileNotFoundError(f"Caption file not found: {args.caption_file}")
    if not (clip_path / "config.json").is_file():
        raise FileNotFoundError(f"CLIP snapshot is incomplete: {clip_path}")

    rows = load_rows(pope_path, args.start, args.limit)
    unique_rows = unique_image_rows(rows)
    captions = load_caption_lookup(args.caption_file)
    missing_captions = [
        str(row["image"])
        for row in unique_rows
        if str(row["image"]) not in captions
    ]
    if missing_captions:
        raise KeyError(
            f"Missing captions for {len(missing_captions)} unique POPE images"
        )
    image_paths = sorted(
        {(images_root / str(row["image"])).resolve() for row in unique_rows}
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
        rows=unique_rows,
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
    predictions: list[int] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    selected_counts: list[int] = []
    generation_started = time.perf_counter()
    iterator = range(0, len(rows), args.batch_size)
    if args.progress:
        iterator = tqdm(iterator, desc="shield-statistical-bias")
    for start in iterator:
        batch_rows = rows[start : start + args.batch_size]
        inputs, batch_selected_counts = build_statistical_prompt_batch(
            model=model,
            processor=processor,
            rows=batch_rows,
            image_root=images_root,
            raw_image_cache=raw_image_cache,
            clip_feature_cache=clip_feature_cache,
            captions=captions,
            device=device,
            module=module,
            pope_answer_instruction=args.pope_answer_instruction,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        answers, generations = generate_batch(
            model=model,
            processor=processor,
            inputs=inputs,
            image_positions=[[] for _ in batch_rows],
            method="vanilla",
            args=args,
            state=state,
            fp8_recipe=fp8_recipe,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for row, answer, generation, selected_count in zip(
            batch_rows,
            answers,
            generations,
            batch_selected_counts,
            strict=True,
        ):
            prediction = answer_label(answer)
            label = 1 if str(row["label"]).lower() == "yes" else 0
            predictions.append(prediction)
            labels.append(label)
            selected_counts.append(selected_count)
            records.append(
                {
                    "question_id": int(row["question_id"]),
                    "image": str(row["image"]),
                    "question": str(row["text"]),
                    "label": label,
                    "prediction": prediction,
                    "answer": answer,
                    "correct": prediction == label,
                    "statistical_bias_selected_tokens": selected_count,
                    **generation,
                }
            )
        if args.progress:
            iterator.set_postfix(  # type: ignore[union-attr]
                rows=len(records),
                acc=f"{compute_metrics(predictions, labels)['accuracy']:.4f}",
            )
    generation_elapsed = time.perf_counter() - generation_started

    if state.fp8_requested:
        state.fp8_calls, state.fp8_native_fallback_calls = fp8_call_counts(model)
    metrics = compute_metrics(predictions, labels)
    acceleration = state.as_dict()
    acceleration_ok = (
        acceleration["effective_attention"] == "flash_attention_2"
        and acceleration["fp8_effective"] is True
        and acceleration["fp8_native_fallback_calls"] == 0
        and acceleration["fallback_events"] == []
    )
    full_protocol = len(records) == 3000
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
        "pope_path": str(pope_path),
        "pope_sha256": sha256_file(pope_path),
        "caption_file": str(args.caption_file.resolve()),
        "caption_file_sha256": sha256_file(args.caption_file),
        "clip_model": str(clip_path),
        "clip_model_config_sha256": sha256_file(clip_path / "config.json"),
        "images_root": str(images_root),
        "count": len(records),
        "unique_images": len(unique_rows),
        "start": args.start,
        "limit": args.limit,
        "protocol": {
            "dtype": state.dtype,
            "attention": state.effective_attention,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": (
                "A chat between a curious human and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite "
                "answers to the human's questions."
            ),
            "prompt_template": (
                "{system_prompt} USER: <image>\\n{question}"
                " Please answer this question with one word. ASSISTANT:"
                if args.pope_answer_instruction
                else "{system_prompt} USER: <image>\\n{question} ASSISTANT:"
            ),
            "pope_answer_instruction": args.pope_answer_instruction,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "image_batch_size": args.image_batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "cache": "dynamic",
            "image_feature_cache": True,
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
        "metrics": metrics,
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
            "selected_caption_tokens": sum(selected_counts),
        },
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "predictions_path": str(predictions_path),
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
