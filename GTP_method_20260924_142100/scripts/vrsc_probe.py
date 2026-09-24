#!/usr/bin/env python3
"""Capture paired visual responses or run small, protocol-matched VRSC experiments."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import shield_cumulative as canonical  # noqa: E402
from shield_cumulative_runtime import (  # noqa: E402
    build_clean_shield_caches,
    build_clip_feature_cache,
    build_projected_feature_cache,
    build_vision_feature_cache,
    process_input_caption,
)
from shield_vulnerability_runtime import (  # noqa: E402
    DEFAULT_CLIP_MODEL,
    _forward_branch,
    build_llava_pixel_cache,
    build_vulnerability_prompt_batch,
    generate_contrastive_batch,
    load_caption_lookup,
    unique_image_rows,
)
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, set_seed  # noqa: E402

from methods.evidence_consensus import (  # noqa: E402
    EvidenceConsensus,
    EvidenceConsensusConfig,
    transport_features,
)
from methods.gtp import (  # noqa: E402
    GTPConfig,
    GroundedTokenProjection,
    counterfactual_patch_scores,
)
from methods.shield_adaptive_plausibility import ShieldAdaptivePlausibility  # noqa: E402
from methods.shield_pipeline import build_shield_pipeline  # noqa: E402
from methods.shield_statistical_bias import (  # noqa: E402
    ShieldStatisticalBias,
    StatisticalBiasConfig,
)
from methods.vrsc import (  # noqa: E402
    VRSC,
    VRSCConfig,
    energy_preserving_probe,
    patch_caption_scores,
)

LOGGER = logging.getLogger(__name__)
METHODS = ("vanilla", "adaptive", "shrink", "vrsc", "vrsc_residual", "shuffled", "as")
EXTRA_METHODS = (
    "evidence_consensus",
    "evidence_shrink",
    "evidence_visual",
    "gtp",
    "gtp_patch",
)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "generate"), default="capture")
    parser.add_argument("--dataset", choices=("pope", "chair"), default="pope")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--model", type=Path, default=canonical.DEFAULT_MODEL)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--images-root", type=Path, default=canonical.DEFAULT_IMAGES)
    parser.add_argument("--pope-path", type=Path, default=canonical.DEFAULT_POPE)
    parser.add_argument("--questions-path", type=Path, default=canonical.DEFAULT_QUESTIONS)
    parser.add_argument("--caption-file", type=Path)
    parser.add_argument("--chair-cache", type=Path, default=canonical.DEFAULT_CHAIR_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp8-scope", default="text_mlp")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--evidence-weight", type=float, default=2.0)
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--response-clip", type=float, default=1.0)
    parser.add_argument("--consensus-weight", type=float, default=0.75)
    parser.add_argument("--transport-trust", type=float, default=1.0)
    parser.add_argument("--feature-gain", type=float, default=1.0)
    parser.add_argument("--gtp-mode", choices=("legacy", "selective"), default="legacy")
    parser.add_argument(
        "--gtp-caption-injection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gtp-max-evidence-weight", type=float, default=0.75)
    parser.add_argument("--gtp-minimum-evidence-fraction", type=float, default=0.0)
    parser.add_argument("--gtp-trust", type=float, default=1.0)
    parser.add_argument("--gtp-signal-scale", type=float, default=0.000001)
    parser.add_argument("--gtp-disagreement-scale", type=float, default=0.08)
    parser.add_argument("--gtp-sharpness", type=float, default=1.0)
    parser.add_argument("--gtp-specificity-temperature", type=float, default=0.5)
    parser.add_argument("--gtp-support-margin", type=float, default=0.0)
    parser.add_argument("--gtp-correction-clip", type=float, default=4.0)
    parser.add_argument("--gtp-candidate-top-k", type=int, default=128)
    parser.add_argument("--gtp-patch-gain", type=float, default=0.35)
    parser.add_argument("--gtp-patch-temperature", type=float, default=0.05)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.methods = args.methods.split(",")
    allowed_methods = METHODS + EXTRA_METHODS
    if not set(args.methods) <= set(allowed_methods) or len(args.methods) != len(
        set(args.methods)
    ):
        parser.error("methods must be unique names from " + ",".join(allowed_methods))
    if args.mode == "capture" and args.dataset != "pope":
        parser.error("first-token capture is a POPE-only diagnostic")
    args.batch_size = (
        args.batch_size if args.batch_size is not None else (32 if args.dataset == "chair" else 8)
    )
    args.max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (512 if args.dataset == "chair" else 8)
    )
    if (
        min(
            args.batch_size,
            args.image_batch_size,
            args.max_new_tokens,
            args.prefetch_factor,
            args.limit,
        )
        < 1
        or min(args.start, args.num_workers) < 0
    ):
        parser.error("invalid batch, worker, token, or subset parameters")
    args.caption_file = args.caption_file or (
        canonical.DEFAULT_CHAIR_CAPTIONS
        if args.dataset == "chair"
        else canonical.DEFAULT_POPE_CAPTIONS
    )
    args.pope_answer_instruction = args.dataset == "pope"
    args.decoding, args.temperature, args.top_p, args.top_k = "sample", 1.0, 1.0, None
    args.fallback, args.image_feature_cache = False, True
    VRSCConfig(args.eta, args.evidence_weight, args.trust, args.response_clip)
    GTPConfig(
        args.gtp_mode,
        args.gtp_max_evidence_weight,
        args.gtp_minimum_evidence_fraction,
        args.gtp_trust,
        args.gtp_signal_scale,
        args.gtp_disagreement_scale,
        args.gtp_sharpness,
        args.gtp_specificity_temperature,
        args.gtp_support_margin,
        args.gtp_correction_clip,
        args.gtp_candidate_top_k,
    )
    if args.gtp_patch_gain < 0 or not torch.isfinite(torch.tensor(args.gtp_patch_gain)):
        parser.error("gtp-patch-gain must be finite and nonnegative")
    if args.gtp_patch_temperature <= 0 or not torch.isfinite(
        torch.tensor(args.gtp_patch_temperature)
    ):
        parser.error("gtp-patch-temperature must be finite and positive")
    return args


def shuffled_scores(scores: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    order = torch.randperm(scores.numel(), generator=generator, device="cpu")
    return scores[order.to(scores.device)]


def answer_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    """Include only whole decoded yes/no tokens, not arbitrary first subwords."""
    result: dict[str, list[int]] = {"yes": [], "no": []}
    for token_id in range(len(tokenizer)):
        token = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        if token in result:
            result[token].append(token_id)
    if not all(result.values()):
        raise ValueError("tokenizer does not expose standalone Yes/No tokens")
    return result


def branch_names(method: str) -> tuple[str, str, str | None]:
    """Keep routing explicit: a missing semantic probe silently becomes Shrink."""
    branches = {
        "vanilla": ("base", "base", None),
        "adaptive": ("base", "base", None),
        "shrink": ("base", "base", None),
        "vrsc": ("base", "semantic", None),
        "vrsc_residual": ("base", "semantic", "shuffled"),
        "shuffled": ("base", "shuffled", None),
        "as": ("as", "as", None),
        "evidence_consensus": ("base", "evidence", None),
        "evidence_shrink": ("evidence", "evidence", None),
        "evidence_visual": ("base", "evidence_visual", None),
        "gtp": ("base", "gtp_positive", "gtp_control"),
        "gtp_patch": ("base", "gtp_patch", None),
    }
    return branches[method]


class DecodeRule:
    def __init__(
        self,
        method: str,
        config: VRSCConfig,
        evidence_config: EvidenceConsensusConfig | None = None,
        gtp_config: GTPConfig | None = None,
    ) -> None:
        self.method = method
        if method == "shrink":
            config = VRSCConfig(config.eta, 0.0, config.trust, config.response_clip)
        elif method == "vrsc_residual":
            config = VRSCConfig(
                config.eta,
                config.evidence_weight,
                config.trust,
                config.response_clip,
                response_mode="orthogonal_residual",
            )
        self.vrsc = VRSC(config)
        self.consensus = (
            EvidenceConsensus(evidence_config)
            if method in ("evidence_consensus", "evidence_shrink", "evidence_visual")
            else None
        )
        self.gtp = GroundedTokenProjection(gtp_config) if method == "gtp" else None
        self.config = (
            self.gtp.config
            if self.gtp is not None
            else self.consensus.config
            if self.consensus is not None
            else config
        )
        self.adaptive = ShieldAdaptivePlausibility()
        self.calls = 0
        self.semantic_delta_max = 0.0

    def combine_logits(
        self,
        base: torch.Tensor,
        probe: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        if self.method == "vrsc_residual" or self.consensus is not None or self.gtp is not None:
            self.semantic_delta_max = max(
                self.semantic_delta_max, (base.float() - probe.float()).abs().max().item()
            )
        if self.method in ("adaptive", "as"):
            return self.adaptive.apply(base, base)
        if self.method == "vanilla":
            return base
        if self.consensus is not None:
            return self.consensus.combine_logits(base, probe)
        if self.gtp is not None:
            return self.gtp.combine_logits(base, probe, reference)
        return self.vrsc.combine_logits(base, probe, reference)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def acceleration(model: Any, state: Any) -> dict[str, Any]:
    state.fp8_calls, state.fp8_native_fallback_calls = canonical.fp8_call_counts(model)
    result = state.as_dict()
    if not canonical.acceleration_ok(result) or state.fp8_calls == 0:
        raise RuntimeError(f"canonical acceleration rejected: {result}")
    return result


def chair_score(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    scorer = canonical.ChairScorer(
        train_instances_path=canonical.DEFAULT_TRAIN_INSTANCES,
        val_instances_path=canonical.DEFAULT_VAL_INSTANCES,
        train_captions_path=canonical.DEFAULT_TRAIN_CAPTIONS,
        val_captions_path=canonical.DEFAULT_VAL_CAPTIONS,
        image_ids=[row["image_id"] for row in records],
        cache_path=args.chair_cache,
    )
    scored = scorer.score(records)
    for row, detail in zip(records, scored["sentences"], strict=True):
        row["chair"] = detail["metrics"]
        row["objects"] = detail["generated_objects"]
    return scored["overall_metrics"]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    # Reserve a new run directory before any costly work; never overwrite evidence.
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hub = canonical.HUB_ROOT
    for key in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_DATASETS_DOWNLOADED_DATASETS_PATH",
        "DATASETS_ROOT",
    ):
        os.environ[key] = str(hub)
    os.environ["HF_HOME"] = str(hub.parent)
    os.environ["HF_HUB_OFFLINE"] = "1"
    device = canonical.resolve_device(args.device)
    dtype = canonical.resolve_dtype(args.dtype, device)
    canonical.configure_cuda(device, args.tf32)
    set_seed(args.seed)
    model_path = canonical.require_hub_path(args.model, "model")
    clip_path = canonical.require_hub_path(args.clip_model, "CLIP model")
    images_root = canonical.require_hub_path(args.images_root, "images")
    data_path = canonical.require_hub_path(
        args.pope_path if args.dataset == "pope" else args.questions_path, "dataset"
    )
    rows = canonical.load_rows(data_path, args.start, args.limit)
    full_count = args.start == 0 and len(rows) == (500 if args.dataset == "chair" else 3000)
    completion_status = "PASS" if full_count and args.mode == "generate" else "SMOKE"
    feature_rows = unique_image_rows(rows)
    paths = sorted({(images_root / row["image"]).resolve() for row in rows})
    for path in paths:
        canonical.require_hub_path(path, "image")
        if not path.is_file():
            raise FileNotFoundError(path)
    captions = load_caption_lookup(args.caption_file)
    config = VRSCConfig(args.eta, args.evidence_weight, args.trust, args.response_clip)
    evidence_config = EvidenceConsensusConfig(
        args.consensus_weight, args.trust, args.transport_trust, args.feature_gain
    )
    gtp_config = GTPConfig(
        args.gtp_mode,
        args.gtp_max_evidence_weight,
        args.gtp_minimum_evidence_fraction,
        args.gtp_trust,
        args.gtp_signal_scale,
        args.gtp_disagreement_scale,
        args.gtp_sharpness,
        args.gtp_specificity_temperature,
        args.gtp_support_margin,
        args.gtp_correction_clip,
        args.gtp_candidate_top_k,
    )
    rules = {
        method: DecodeRule(method, config, evidence_config, gtp_config)
        for method in args.methods
    }
    required_caches = {
        name for method in args.methods for name in branch_names(method) if name is not None
    }
    if "gtp" in args.methods:
        required_caches.update(("semantic", "shuffled"))
    if "gtp_patch" in args.methods:
        required_caches.update(("semantic", "shuffled", "gtp_patch"))
    if args.mode == "capture":
        required_caches.update(("base", "semantic", "shuffled"))
    protocol = {
        key: getattr(args, key)
        for key in (
            "seed",
            "batch_size",
            "image_batch_size",
            "num_workers",
            "prefetch_factor",
            "max_new_tokens",
            "decoding",
            "temperature",
            "top_p",
            "top_k",
            "pope_answer_instruction",
        )
    }
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "mode": args.mode,
        "dataset": args.dataset,
        "count": len(rows),
        "start": args.start,
        "limit": args.limit,
        "unique_images": len(paths),
        "method_config": asdict(config),
        "method_configs": {
            method: asdict(rule.config) for method, rule in rules.items()
        },
        "branch_routing": {method: branch_names(method) for method in args.methods},
        "protocol": {**protocol, "cache": "dynamic"},
        "model": str(model_path),
        "model_revision": model_path.name,
        "model_config_sha256": canonical.sha256_file(model_path / "config.json"),
        "clip_model": str(clip_path),
        "clip_config_sha256": canonical.sha256_file(clip_path / "config.json"),
        "data_path": str(data_path),
        "data_sha256": canonical.sha256_file(data_path),
        "caption_file": str(args.caption_file),
        "caption_sha256": canonical.sha256_file(args.caption_file),
        "image_hashes": {path.name: canonical.sha256_file(path) for path in paths},
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "argv": sys.argv,
        "git_sha": canonical.git_sha(),
        "source_hashes": {
            str(path.relative_to(ROOT)): canonical.sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "src/methods/vrsc.py",
                ROOT / "src/methods/evidence_consensus.py",
                ROOT / "src/methods/gtp.py",
                ROOT / "scripts/shield_vulnerability_runtime.py",
            )
        },
        "limitations": [
            "Exploratory method; full-count runs use the canonical dataset size.",
            "Naive captions are reused; their generation cost is not timed.",
            "Generation decode_slots include post-EOS padding in the reused runtime.",
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    feature_start = time.perf_counter()
    model, state, recipe = canonical.load_runtime(args, model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    manifest["protocol"]["generation_vocab_size"] = len(processor.tokenizer)
    write_json(args.output / "manifest.json", manifest)
    pixels = build_llava_pixel_cache(
        processor,
        paths,
        args.image_batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.progress,
    )
    raw = build_vision_feature_cache(
        model, paths, pixels, device, dtype, args.image_batch_size, args.progress, "vrsc-vision"
    )
    clip_model = CLIPModel.from_pretrained(str(clip_path), local_files_only=True).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(
        str(clip_path), local_files_only=True, use_fast=False
    )
    caption_features = build_clip_feature_cache(
        clip_model,
        clip_processor,
        feature_rows,
        captions,
        device,
        args.batch_size,
        args.progress,
    )
    del clip_model, clip_processor, pixels
    gc.collect()
    torch.cuda.empty_cache()
    raw_variants: dict[str, dict[str, torch.Tensor]] = {}
    if "base" in required_caches:
        raw_variants["base"] = raw
    for name in ("semantic", "shuffled", "gtp_patch"):
        if name in required_caches:
            raw_variants[name] = {}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    diagnostics = []
    for path in paths:
        x = raw[str(path)]
        scores = patch_caption_scores(x, caption_features[path.name])
        shuffled = shuffled_scores(scores, generator)
        diagnostic: dict[str, Any] = {
            "image": path.name,
            "score_min": scores.min().item(),
            "score_max": scores.max().item(),
        }
        patch_consensus = counterfactual_patch_scores(
            scores, shuffled, args.gtp_patch_temperature
        )
        for name, values in (
            ("semantic", scores),
            ("shuffled", shuffled),
            ("gtp_patch", patch_consensus),
        ):
            if name not in required_caches:
                continue
            gain = args.gtp_patch_gain if name == "gtp_patch" else args.eta
            altered = energy_preserving_probe(x, values, gain)
            raw_variants[name][str(path)] = altered
            diagnostic[name + "_relative_change"] = (
                (altered.float() - x.float()).norm() / x.float().norm()
            ).item()
            diagnostic[name + "_norm_ratio"] = (altered.float().norm() / x.float().norm()).item()
        diagnostics.append(diagnostic)
    caches = {
        name: build_projected_feature_cache(
            model,
            paths,
            features,
            device,
            dtype,
            args.image_batch_size,
            args.progress,
            "project-" + name,
        )
        for name, features in raw_variants.items()
    }
    if "gtp" in args.methods:
        caches["gtp_positive"] = caches["semantic"]
        caches["gtp_control"] = caches["shuffled"]
    caption_cache: dict[str, torch.Tensor] = {}
    if "as" in args.methods:
        pipeline = build_shield_pipeline(
            ("adaptive_plausibility", "statistical_bias"),
            statistical_config=StatisticalBiasConfig(
                threshold=0.011 if args.dataset == "pope" else 0.002,
                gain_per=0.5 if args.dataset == "pope" else 0.55,
            ),
            caption_token_injection=True,
        )
        caches["as"], caption_cache, _ = build_clean_shield_caches(
            model,
            processor,
            paths,
            raw,
            caption_features,
            captions,
            None,
            pipeline,
            device,
            dtype,
            args.progress,
        )
    evidence_captions: dict[str, torch.Tensor] = {}
    if required_caches & {"evidence", "evidence_visual", "gtp_positive", "gtp_control"}:
        caches["evidence"] = {}
        for path, diagnostic in zip(paths, diagnostics, strict=True):
            transformed, selected, joint = transport_features(
                raw[str(path)], caption_features[path.name], evidence_config
            )
            projected = model.multi_modal_projector(
                transformed.unsqueeze(0).to(device=device, dtype=dtype)
            ).squeeze(0)
            caches["evidence"][str(path)] = projected.detach()
            caption_ids = processor.tokenizer(
                process_input_caption(captions[path.name]), return_tensors="pt"
            )["input_ids"][0].to(device)
            embeds = model.get_input_embeddings()(caption_ids)
            evidence_captions[path.name] = ShieldStatisticalBias.map_text_segments(
                selected, caption_features[path.name].shape[0], embeds
            ).detach()
            diagnostic["selected_caption_tokens"] = int(selected.numel())
            diagnostic["transport_nonzero_fraction"] = (joint > 0).float().mean().item()
            diagnostic["transport_visual_relative_change"] = (
                (transformed.float() - raw[str(path)].float()).norm()
                / raw[str(path)].float().norm().clamp_min(1e-12)
            ).item()
        if "evidence_visual" in required_caches:
            caches["evidence_visual"] = caches["evidence"]
    write_json(args.output / "probe_diagnostics.json", diagnostics)
    del raw_variants, raw, caption_features
    gc.collect()
    torch.cuda.empty_cache()
    manifest["feature_and_load_seconds"] = time.perf_counter() - feature_start

    def inputs(batch: list[dict[str, Any]], name: str) -> dict[str, torch.Tensor]:
        return build_vulnerability_prompt_batch(
            model,
            processor,
            batch,
            images_root,
            caches[name],
            device,
            caption_embedding_cache=(
                caption_cache
                if name == "as"
                else evidence_captions
                if name == "evidence"
                or (args.gtp_caption_injection and name in {"gtp_positive", "gtp_control"})
                else None
            ),
            pope_answer_instruction=args.pope_answer_instruction,
        )

    if args.mode == "capture":
        captured: dict[str, list[torch.Tensor]] = {name: [] for name in caches}
        started = time.perf_counter()
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            for name in caches:
                packed = inputs(batch, name)
                out, logits = _forward_branch(
                    model,
                    {
                        "inputs_embeds": packed["inputs_embeds"],
                        "input_ids": None,
                        "attention_mask": packed["attention_mask"],
                        "position_ids": packed["position_ids"],
                        "past_key_values": None,
                        "cache_position": torch.arange(packed["input_ids"].shape[1], device=device),
                    },
                    state,
                    recipe,
                )
                captured[name].append(logits.float().cpu())
                del out, logits
            LOGGER.info("captured %d/%d", min(start + args.batch_size, len(rows)), len(rows))
        torch.save(
            {name: torch.cat(value) for name, value in captured.items()}, args.output / "logits.pt"
        )
        write_json(args.output / "rows.json", rows)
        write_json(args.output / "answer_token_ids.json", answer_token_ids(processor.tokenizer))
        manifest["capture_seconds"] = time.perf_counter() - started
    else:
        for method in args.methods:
            set_seed(args.seed)
            rule = rules[method]
            base_name, probe_name, reference_name = branch_names(method)
            records = []
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                base_inputs = inputs(batch, base_name)
                probe_inputs = inputs(batch, probe_name) if base_name != probe_name else base_inputs
                reference_inputs = (
                    inputs(batch, reference_name) if reference_name is not None else None
                )
                answers, generations = generate_contrastive_batch(
                    model,
                    processor,
                    base_inputs,
                    probe_inputs,
                    args,
                    state,
                    recipe,
                    rule,
                    reference_inputs=reference_inputs,
                )
                for row, answer, generation in zip(batch, answers, generations, strict=True):
                    record = {
                        "question_id": row["question_id"],
                        "image": row["image"],
                        "prompt": row["text"],
                        "answer": answer,
                        "decode_slots": generation["generated_tokens"],
                        "word_count": len(answer.split()),
                    }
                    if args.dataset == "pope":
                        record["label"] = int(row["label"].lower() == "yes")
                        record["prediction"] = canonical.answer_label(answer)
                    else:
                        record["image_id"] = canonical.image_id_from_name(row["image"])
                        record["caption"] = answer
                    records.append(record)
                LOGGER.info("%s generated %d/%d", method, len(records), len(rows))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            metrics = (
                canonical.compute_metrics(
                    [row["prediction"] for row in records], [row["label"] for row in records]
                )
                if args.dataset == "pope"
                else chair_score(records, args)
            )
            summary = {
                "status": completion_status,
                "method": method,
                "method_config": asdict(rule.config),
                "branch_routing": branch_names(method),
                "response_diagnostics": {
                    "decode_calls": rule.calls,
                    "semantic_delta_max": rule.semantic_delta_max,
                },
                "count": len(records),
                "protocol": manifest["protocol"],
                "metrics": metrics,
                "acceleration": acceleration(model, state),
                "generation_seconds": elapsed,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "mean_words": sum(row["word_count"] for row in records) / len(records),
                "caption_token_injection": method
                in ("as", "evidence_consensus", "evidence_shrink")
                or (method == "gtp" and args.gtp_caption_injection),
                "manifest": str(args.output / "manifest.json"),
                "model_revision": manifest["model_revision"],
                "model_config_sha256": manifest["model_config_sha256"],
                "data_sha256": manifest["data_sha256"],
                "source_hashes": manifest["source_hashes"],
            }
            (args.output / f"{method}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            write_json(args.output / f"{method}.metrics.json", summary)
            LOGGER.info("%s metrics: %s", method, metrics)
    manifest["acceleration"] = acceleration(model, state)
    manifest["status"] = completion_status
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
