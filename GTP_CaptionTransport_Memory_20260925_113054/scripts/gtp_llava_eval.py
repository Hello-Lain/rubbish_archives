#!/usr/bin/env python3
"""Evaluate GTP on the fixed LLaVA POPE adversarial or CHAIR-500 protocol."""

# ruff: noqa: E402, I001

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from adapters.chair_metrics import ChairScorer, sha256_file
from adapters.gtp_llava import LlavaGTPBatchBackend, extract_coco_claims
from data.pope import PopeDataset
from methods.gtp import GTPMethod
from pope_llava_only_compare import (
    DEFAULT_SYSTEM_PROMPT,
    build_image_feature_cache,
    build_prompt_batch,
    compute_metrics,
    configure_cuda,
    fp8_call_counts,
    fp8_context,
    generate_batch,
    git_sha,
    load_runtime,
    require_hub_path,
    resolve_device,
    resolve_dtype,
)
from hydra.utils import instantiate

LOG = logging.getLogger("gtp_eval")
HUB_ROOT = Path("/data/lcq/.cache/huggingface/hub").resolve()
EXPECTED_POPE_SHA256 = "185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac"
EXPECTED_CHAIR_QUESTIONS_SHA256 = "d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e"
EXPECTED_LLAVA_CONFIG_SHA256 = "0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f"
CHAIR_PROMPT = "Please describe this image in detail."
CHAIR_CLAIM_PROMPT = (
    "List the clearly visible objects in this image as short object names separated by commas. "
    "Include only objects you can directly see; omit uncertain objects."
)


def _load_chair_rows(path: Path, start: int, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    selected = rows[start:] if limit is None else rows[start : start + limit]
    if not selected:
        raise ValueError(f"No CHAIR questions selected from {path}")
    return selected


def _build_chair_dev_records(
    *,
    annotations_path: Path,
    images_root: Path,
    excluded_image_ids: set[int],
    start: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if start < 0 or count < 1:
        raise ValueError("CHAIR dev start must be non-negative and count must be positive")

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    eligible = [
        image for image in annotations["images"] if int(image["id"]) not in excluded_image_ids
    ]
    eligible.sort(
        key=lambda image: hashlib.sha256(
            f"gtp-chair-dev-v1:{seed}:{int(image['id'])}".encode("ascii")
        ).digest()
    )
    selected = eligible[start : start + count]
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} CHAIR dev images at offset {start}, but only found {len(selected)}"
        )

    records: list[dict[str, Any]] = []
    for image in selected:
        image_id = int(image["id"])
        filename = str(image["file_name"])
        image_path = (images_root / filename).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append(
            {
                "id": f"dev-{image_id}",
                "question_id": image_id,
                "image_id": image_id,
                "image": filename,
                "image_path": str(image_path),
                "text": CHAIR_PROMPT,
            }
        )
    return records


def _build_claim_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    prompt: str,
) -> tuple[list[dict[str, Any]], str]:
    if mode == "caption_prefix":
        return rows, "greedy_clean_caption_prefix"
    if mode == "visible_object_list":
        return (
            [{**row, "text": prompt} for row in rows],
            "greedy_visible_object_list",
        )
    raise ValueError("chair_claim_mode must be 'caption_prefix' or 'visible_object_list'")


def _load_records(cfg: DictConfig, dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name == "pope":
        dataset = PopeDataset(
            path=str(cfg.eval.pope_path),
            images_root=str(cfg.eval.images_root),
            start=int(cfg.start),
            limit=None if cfg.limit is None else int(cfg.limit),
            validate_images=True,
        )
        records = [dataset[index] for index in range(len(dataset))]
        for record in records:
            record["text"] = str(record["prompt"])
        return records

    if dataset_name in {"chair", "chair_dev"}:
        images_root = Path(str(cfg.eval.images_root)).expanduser().resolve()
        if dataset_name == "chair":
            rows = _load_chair_rows(
                Path(str(cfg.eval.chair_questions_path)),
                int(cfg.start),
                None if cfg.limit is None else int(cfg.limit),
            )
        else:
            canonical_rows = _load_chair_rows(
                Path(str(cfg.eval.chair_questions_path)),
                0,
                None,
            )
            excluded_image_ids = {
                int(Path(str(row["image"])).stem.split("_")[-1]) for row in canonical_rows
            }
            dev_count = int(cfg.chair_dev_count if cfg.limit is None else cfg.limit)
            rows = _build_chair_dev_records(
                annotations_path=Path(str(cfg.eval.chair_val_instances_path)),
                images_root=images_root,
                excluded_image_ids=excluded_image_ids,
                start=int(cfg.start),
                count=dev_count,
                seed=int(cfg.chair_dev_seed),
            )

        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            image = str(row["image"])
            image_path = (images_root / image).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append(
                {
                    **row,
                    "id": str(row.get("question_id", int(cfg.start) + index)),
                    "image_id": int(Path(image).stem.split("_")[-1]),
                    "image_path": str(image_path),
                    "text": str(row.get("text", CHAIR_PROMPT)),
                }
            )
        return records

    raise ValueError(f"Unsupported dataset={dataset_name!r}; expected 'pope' or 'chair'")


def _as_runtime_args(cfg: DictConfig, *, max_new_tokens: int) -> Namespace:
    return Namespace(
        attention=str(cfg.model.attn_implementation),
        fp8=bool(cfg.model.fp8),
        fp8_scope=str(cfg.model.fp8_scope),
        tf32=bool(cfg.model.tf32),
        fallback=bool(cfg.model.fallback),
        image_feature_cache=True,
        max_new_tokens=max_new_tokens,
        decoding=str(cfg.decoding),
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        top_k=None if cfg.top_k is None else int(cfg.top_k),
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _prepare_output_prefix(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_canonical_protocol(
    cfg: DictConfig,
    *,
    dataset_name: str,
    batch_size: int,
    max_new_tokens: int,
) -> bool:
    expected_batch_size = 8 if dataset_name == "pope" else 32
    expected_max_new_tokens = 8 if dataset_name == "pope" else 512
    shared_match = (
        str(cfg.decoding) == "sample"
        and float(cfg.temperature) == 1.0
        and float(cfg.top_p) == 1.0
        and cfg.top_k is None
        and int(cfg.seed) == 42
        and batch_size == expected_batch_size
        and int(cfg.image_batch_size) == 16
        and int(cfg.num_workers) == 4
        and int(cfg.prefetch_factor) == 4
        and str(cfg.model.torch_dtype) == "bfloat16"
        and bool(cfg.model.tf32)
        and bool(cfg.model.fp8)
        and str(cfg.model.fp8_scope) == "text_mlp"
        and str(cfg.model.attn_implementation) == "auto"
    )
    claim_config_match = (
        str(cfg.chair_claim_mode) == "caption_prefix" and int(cfg.claim_max_new_tokens) == 32
    ) or (
        str(cfg.chair_claim_mode) == "visible_object_list"
        and int(cfg.claim_max_new_tokens) == 48
        and str(cfg.chair_claim_prompt) == CHAIR_CLAIM_PROMPT
    )
    dataset_match = max_new_tokens == expected_max_new_tokens and (
        dataset_name != "chair"
        or (
            str(cfg.eval.variant) == "gtp"
            and str(cfg.method._target_) == "methods.gtp.GTPMethod"
            and claim_config_match
        )
    )
    return shared_match and dataset_match


def _run(cfg: DictConfig) -> dict[str, Any]:
    dataset_name = str(cfg.dataset).lower()
    if dataset_name not in {"pope", "chair", "chair_dev"}:
        raise ValueError("dataset must be 'pope', 'chair', or 'chair_dev'")
    experiment_variant = str(cfg.eval.variant).lower()
    if experiment_variant not in {"gtp", "vanilla"}:
        raise ValueError("eval.variant must be either 'gtp' or 'vanilla'")
    if experiment_variant == "vanilla" and dataset_name != "chair_dev":
        raise ValueError("the Vanilla variant is enabled only for chair_dev comparisons")
    if dataset_name == "pope" and experiment_variant != "gtp":
        raise ValueError("POPE evaluation requires eval.variant=gtp")
    if int(cfg.start) < 0:
        raise ValueError("start must be non-negative")
    if int(cfg.chair_dev_seed) < 0:
        raise ValueError("chair_dev_seed must be non-negative")

    pope_path = require_hub_path(Path(str(cfg.eval.pope_path)), "POPE dataset")
    chair_questions_path = require_hub_path(
        Path(str(cfg.eval.chair_questions_path)),
        "CHAIR questions",
    )
    images_root = require_hub_path(Path(str(cfg.eval.images_root)), "COCO images")
    model_path = require_hub_path(Path(str(cfg.model.path)), "LLaVA model")
    for key in (
        "chair_train_instances_path",
        "chair_val_instances_path",
        "chair_train_captions_path",
        "chair_val_captions_path",
    ):
        require_hub_path(Path(str(cfg.eval[key])), f"COCO annotation {key}")

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete LLaVA snapshot: {model_path}")
    model_config_sha256 = sha256_file(model_path / "config.json")
    if model_config_sha256 != EXPECTED_LLAVA_CONFIG_SHA256:
        raise ValueError(
            f"LLaVA config hash differs from the fixed project snapshot: {model_config_sha256}"
        )
    pope_sha256 = sha256_file(pope_path)
    chair_questions_sha256 = sha256_file(chair_questions_path)
    if pope_sha256 != EXPECTED_POPE_SHA256:
        raise ValueError(f"Unexpected POPE SHA-256: {pope_sha256}")
    if chair_questions_sha256 != EXPECTED_CHAIR_QUESTIONS_SHA256:
        raise ValueError(f"Unexpected CHAIR questions SHA-256: {chair_questions_sha256}")

    records = _load_records(cfg, dataset_name)
    chair_cache_path: Path | None = None
    chair_cache_sha256: str | None = None
    if dataset_name in {"chair", "chair_dev"}:
        chair_cache_path = Path(str(cfg.eval.chair_cache)).expanduser()
        if not chair_cache_path.is_absolute():
            chair_cache_path = PROJECT_ROOT / chair_cache_path
        if not chair_cache_path.is_file():
            raise FileNotFoundError(
                f"CHAIR ground-truth cache must exist before inference: {chair_cache_path}"
            )
        chair_cache_payload = json.loads(chair_cache_path.read_text(encoding="utf-8"))
        if chair_cache_payload.get("schema_version") != "shield_chair_ground_truth_v2":
            raise ValueError(f"Unexpected CHAIR cache schema: {chair_cache_path}")
        if dataset_name == "chair":
            cached_image_ids = {
                int(image_id) for image_id in chair_cache_payload.get("ground_truth", {})
            }
            requested_image_ids = {int(record["image_id"]) for record in records}
            missing_cached_ids = requested_image_ids - cached_image_ids
            if missing_cached_ids:
                raise ValueError(
                    "CHAIR ground-truth cache is missing requested image IDs: "
                    f"{sorted(missing_cached_ids)[:10]}"
                )
        chair_cache_sha256 = sha256_file(chair_cache_path)

    images = sorted({(images_root / str(record["image"])).resolve() for record in records})
    missing_images = [path for path in images if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} image files; first={missing_images[0]}"
        )

    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.env.device))
    dtype = resolve_dtype(str(cfg.model.torch_dtype), device)
    configure_cuda(device, bool(cfg.model.tf32))
    args = _as_runtime_args(
        cfg,
        max_new_tokens=(
            int(cfg.pope_max_new_tokens)
            if dataset_name == "pope"
            else int(cfg.chair_max_new_tokens)
        ),
    )
    model, runtime_state, fp8_recipe = load_runtime(args, model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    feature_started = time.perf_counter()
    image_cache = build_image_feature_cache(
        model=model,
        processor=processor,
        image_paths=images,
        device=device,
        dtype=dtype,
        batch_size=int(cfg.image_batch_size),
        num_workers=int(cfg.num_workers),
        prefetch_factor=int(cfg.prefetch_factor),
        progress=bool(cfg.progress),
    )
    feature_elapsed = time.perf_counter() - feature_started
    runtime_state.image_feature_count = len(image_cache)

    method: GTPMethod | None = None
    if experiment_variant == "gtp":
        method = instantiate(cfg.method)
        if dataset_name == "pope" and method.grounding_source != "coco_claims":
            raise ValueError("POPE requires grounding_source='coco_claims'")
        backend = LlavaGTPBatchBackend(
            model_path=str(model_path),
            model=model,
            processor=processor,
            image_cache=image_cache,
            images_root=images_root,
            device=device,
            max_new_tokens=args.max_new_tokens,
            decoding=args.decoding,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            forward_context=lambda: fp8_context(runtime_state, fp8_recipe),
        )
        backend.setup(device)
        method.setup(backend, device)

    batch_size = int(cfg.pope_batch_size if dataset_name == "pope" else cfg.chair_batch_size)
    output_records: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    for batch_start in range(0, len(records), batch_size):
        batch_rows = records[batch_start : batch_start + batch_size]
        prepared, image_positions = build_prompt_batch(
            model=model,
            processor=processor,
            rows=batch_rows,
            image_root=images_root,
            image_cache=image_cache,
            device=device,
            pope_answer_instruction=dataset_name == "pope",
        )

        if dataset_name in {"chair", "chair_dev"} and method is not None:
            claim_rows, claim_source = _build_claim_rows(
                batch_rows,
                mode=str(cfg.chair_claim_mode),
                prompt=str(cfg.chair_claim_prompt),
            )
            claim_inputs = prepared
            claim_image_positions = image_positions
            if claim_source == "greedy_visible_object_list":
                claim_inputs, claim_image_positions = build_prompt_batch(
                    model=model,
                    processor=processor,
                    rows=claim_rows,
                    image_root=images_root,
                    image_cache=image_cache,
                    device=device,
                    pope_answer_instruction=False,
                )
            claim_args = _as_runtime_args(
                cfg,
                max_new_tokens=int(cfg.claim_max_new_tokens),
            )
            claim_args.decoding = "greedy"
            drafts, _ = generate_batch(
                model=model,
                processor=processor,
                inputs=claim_inputs,
                image_positions=claim_image_positions,
                method="vanilla",
                args=claim_args,
                state=runtime_state,
                fp8_recipe=fp8_recipe,
            )
            for row, draft in zip(batch_rows, drafts, strict=True):
                row["gtp_claim_draft"] = draft
                row["gtp_claims"] = extract_coco_claims(draft)
                row["gtp_claim_source"] = claim_source
        elif dataset_name == "pope":
            for row in batch_rows:
                row["gtp_claim_draft"] = None
                row["gtp_claims"] = extract_coco_claims(str(row["text"]))
                row["gtp_claim_source"] = "question_object"

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if method is not None:
            generated = method.generate(
                {
                    "records": batch_rows,
                    "inputs": prepared,
                    "image_positions": image_positions,
                }
            )
        else:
            answers, diagnostics = generate_batch(
                model=model,
                processor=processor,
                inputs=prepared,
                image_positions=image_positions,
                method="vanilla",
                args=args,
                state=runtime_state,
                fp8_recipe=fp8_recipe,
            )
            generated = [
                {
                    "id": row.get("id"),
                    "answer": answer,
                    "prediction": answer,
                    "generated_tokens": int(diagnostics[index]["generated_tokens"]),
                }
                for index, (row, answer) in enumerate(zip(batch_rows, answers, strict=True))
            ]
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for row, result in zip(batch_rows, generated, strict=True):
            if dataset_name == "pope":
                prediction = int(
                    __import__("pope_llava_only_compare").answer_label(result["answer"])
                )
                label = 1 if str(row["label"]).lower() == "yes" else 0
                output_records.append(
                    {
                        **result,
                        "question_id": row["question_id"],
                        "label": label,
                        "prediction_label": prediction,
                        "correct": prediction == label,
                        "gtp_claim_source": row["gtp_claim_source"],
                    }
                )
            else:
                output_records.append(
                    {
                        **result,
                        "question_id": int(row["question_id"]),
                        "image_id": int(row["image_id"]),
                        "caption": result["answer"],
                        "evaluation_variant": experiment_variant,
                        **(
                            {"gtp_claim_source": row["gtp_claim_source"]}
                            if method is not None
                            else {}
                        ),
                    }
                )

    generation_elapsed = time.perf_counter() - generation_started
    if runtime_state.fp8_requested:
        (
            runtime_state.fp8_calls,
            runtime_state.fp8_native_fallback_calls,
        ) = fp8_call_counts(model)

    result_metrics: dict[str, Any] | None = None
    if dataset_name == "pope":
        prediction_labels = [int(row["prediction_label"]) for row in output_records]
        labels = [int(row["label"]) for row in output_records]
        result_metrics = compute_metrics(prediction_labels, labels)
    else:
        if chair_cache_path is None:
            raise RuntimeError("CHAIR ground-truth cache path was not initialized")
        scorer = ChairScorer(
            train_instances_path=Path(str(cfg.eval.chair_train_instances_path)),
            val_instances_path=Path(str(cfg.eval.chair_val_instances_path)),
            train_captions_path=Path(str(cfg.eval.chair_train_captions_path)),
            val_captions_path=Path(str(cfg.eval.chair_val_captions_path)),
            image_ids=[int(row["image_id"]) for row in output_records],
            cache_path=chair_cache_path,
        )
        scored = scorer.score(output_records)
        for row, sentence in zip(
            output_records,
            scored["sentences"],
            strict=True,
        ):
            row["chair"] = sentence["metrics"]
            row["objects"] = sentence["generated_objects"]
            row["hallucinated_objects"] = sentence["hallucinated_objects"]
        result_metrics = scored["overall_metrics"]

    acceleration = runtime_state.as_dict()
    acceleration_ok = (
        acceleration["effective_attention"] == "flash_attention_2"
        and acceleration["fp8_effective"] is True
        and acceleration["fp8_native_fallback_calls"] == 0
        and acceleration["fallback_events"] == []
    )
    expected_count = 3000 if dataset_name == "pope" else 500
    full_protocol = (
        dataset_name in {"pope", "chair"}
        and experiment_variant == "gtp"
        and len(output_records) == expected_count
        and int(cfg.start) == 0
        and _is_canonical_protocol(
            cfg,
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    )
    status = (
        "DEV"
        if dataset_name == "chair_dev"
        else "PASS"
        if full_protocol and acceleration_ok
        else "SMOKE"
    )
    output_prefix = _prepare_output_prefix(str(cfg.output_prefix))
    predictions_path = output_prefix.with_suffix(".jsonl")
    metrics_path = output_prefix.with_suffix(".metrics.json")
    _write_jsonl(predictions_path, output_records)

    summary = {
        "status": status,
        "dataset": dataset_name,
        "method": (
            "GTP-CaptionTransport+Memory"
            if method is not None
            and method.grounding_source == "caption_tokens"
            and method.inject_caption_memory
            else "GTP-CaptionTransport"
            if method is not None and method.grounding_source == "caption_tokens"
            else "GTP"
            if method is not None
            else "Vanilla"
        ),
        "model": str(model_path),
        "model_revision": model_path.name,
        "model_config_sha256": model_config_sha256,
        "data_path": str(
            pope_path
            if dataset_name == "pope"
            else Path(str(cfg.eval.chair_val_instances_path))
            if dataset_name == "chair_dev"
            else chair_questions_path
        ),
        "data_sha256": (
            pope_sha256
            if dataset_name == "pope"
            else sha256_file(Path(str(cfg.eval.chair_val_instances_path)))
            if dataset_name == "chair_dev"
            else chair_questions_sha256
        ),
        "images_root": str(images_root),
        "count": len(output_records),
        "start": int(cfg.start),
        "limit": None if cfg.limit is None else int(cfg.limit),
        "protocol": {
            "decoding": str(cfg.decoding),
            "temperature": float(cfg.temperature),
            "top_p": float(cfg.top_p),
            "top_k": None if cfg.top_k is None else int(cfg.top_k),
            "max_new_tokens": args.max_new_tokens,
            "seed": int(cfg.seed),
            "batch_size": batch_size,
            "image_batch_size": int(cfg.image_batch_size),
            "num_workers": int(cfg.num_workers),
            "prefetch_factor": int(cfg.prefetch_factor),
            "cache": "dynamic",
            "image_feature_cache": True,
            "prompt": (
                "official LLaVA POPE one-word prompt" if dataset_name == "pope" else CHAIR_PROMPT
            ),
            "claim_generation": (
                {
                    "source": "question_object",
                    "additional_generation": False,
                }
                if dataset_name == "pope"
                else {
                    "source": "none",
                    "additional_generation": False,
                }
                if method is None
                else {
                    "source": str(cfg.chair_claim_mode),
                    "prompt": (
                        str(cfg.chair_claim_prompt)
                        if str(cfg.chair_claim_mode) == "visible_object_list"
                        else CHAIR_PROMPT
                    ),
                    "max_new_tokens": int(cfg.claim_max_new_tokens),
                    "additional_generation": True,
                }
            ),
            "patch_feature_space": "LLaVA multimodal-projector output",
            "claim_feature_space": (
                "none"
                if method is None
                else "individual LLaVA LM input-token embeddings"
                if method.grounding_source == "caption_tokens"
                else "mean LLaVA LM input-token embeddings"
            ),
            "claim_in_prompt": bool(method and method.inject_caption_memory),
            "caption_memory": {
                "enabled": bool(method and method.inject_caption_memory),
                "branch": "grounded_only" if method and method.inject_caption_memory else None,
                "slots": method.caption_memory_slots if method else 0,
                "insertion_point": (
                    "after_image_patches" if method and method.inject_caption_memory else None
                ),
            },
            "dual_cache": (
                "separate clean and grounded DynamicCache with independent sequence lengths"
                if method and method.inject_caption_memory
                else "separate clean and grounded DynamicCache"
            ),
        },
        "gtp": (
            {
                "a_u": method.a_u,
                "a_h": method.a_h,
                "tau": method.tau,
                "tau_e": method.tau_e,
                "kappa": method.kappa,
                "grounding_source": method.grounding_source,
                "inject_caption_memory": method.inject_caption_memory,
                "caption_memory_slots": method.caption_memory_slots,
                "invalid_grounding_count": sum(
                    not bool(row["gtp_grounding_valid"]) for row in output_records
                ),
                "clean_top1_excluded_steps": sum(
                    int(row["gtp_clean_top1_excluded_steps"]) for row in output_records
                ),
                "mean_projected_support": (
                    sum(float(row["gtp_mean_projected_support"]) for row in output_records)
                    / max(len(output_records), 1)
                ),
            }
            if method is not None
            else None
        ),
        "acceleration": acceleration,
        "acceleration_contract_pass": acceleration_ok,
        "metrics": result_metrics,
        "chair_protocol": (
            {
                "name": (
                    "SHIELD-compatible"
                    if dataset_name == "chair"
                    else "COCO val2014 development subset"
                ),
                "questions": (
                    "500 fixed COCO val2014 images"
                    if dataset_name == "chair"
                    else f"{len(output_records)} images disjoint from fixed CHAIR-500"
                ),
                "ground_truth": "COCO train2014+val2014 instances and captions",
                "chair_cache": str(chair_cache_path.resolve()),
                "chair_cache_sha256": chair_cache_sha256,
                **(
                    {
                        "selection_seed": int(cfg.chair_dev_seed),
                        "excluded_questions_sha256": chair_questions_sha256,
                        "image_ids_sha256": hashlib.sha256(
                            ",".join(str(int(row["image_id"])) for row in output_records).encode(
                                "ascii"
                            )
                        ).hexdigest(),
                    }
                    if dataset_name == "chair_dev"
                    else {}
                ),
            }
            if dataset_name in {"chair", "chair_dev"}
            else None
        ),
        "runtime": {
            "feature_cache_seconds": feature_elapsed,
            "generation_seconds": generation_elapsed,
            "total_seconds": feature_elapsed + generation_elapsed,
            "rows_per_second": (
                len(output_records) / generation_elapsed if generation_elapsed else 0.0
            ),
            "generated_tokens": sum(int(row["generated_tokens"]) for row in output_records),
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "predictions_path": str(predictions_path),
        "git_sha": git_sha(),
        "method_sha256": sha256_file(PROJECT_ROOT / "src/methods/gtp.py"),
        "adapter_sha256": sha256_file(PROJECT_ROOT / "src/adapters/gtp_llava.py"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "system_prompt": DEFAULT_SYSTEM_PROMPT if dataset_name == "pope" else None,
    }
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    LOG.info("GTP run summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="gtp_eval")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OmegaConf.resolve(cfg)
    _run(cfg)


if __name__ == "__main__":
    main()
