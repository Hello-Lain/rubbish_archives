#!/usr/bin/env python3
"""Replay a paired first-token capture; never label replay as generated POPE."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from methods.shield_adaptive_plausibility import ShieldAdaptivePlausibility  # noqa: E402
from methods.vrsc import VRSC, VRSCConfig  # noqa: E402

LOGGER = logging.getLogger(__name__)
MIN_TEMPERATURE = 1e-4
MAX_TEMPERATURE = 1000.0


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(dim=-1)


def entropy_matched_distribution(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match entropy where attainable; keep tied maxima tied at the boundary."""
    if logits.ndim != 2 or logits.shape[1] == 0 or target.shape != logits.shape[:1]:
        raise ValueError("expected logits [rows, vocab] and target entropy [rows]")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(target).all()):
        raise ValueError("logits and target entropy must be finite")
    if bool((target < 0).any()):
        raise ValueError("target entropy must be nonnegative")
    logits = logits.float()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    target = target.to(logits)
    low = torch.full_like(target, MIN_TEMPERATURE)
    high = torch.full_like(target, MAX_TEMPERATURE)
    minimum = entropy((logits / MIN_TEMPERATURE).softmax(-1))
    maximum = entropy((logits / MAX_TEMPERATURE).softmax(-1))
    attainable_target = target.clamp(min=minimum, max=maximum)
    for _ in range(40):
        middle = (low * high).sqrt()
        current = entropy((logits / middle[:, None]).softmax(-1))
        low = torch.where(current < attainable_target, middle, low)
        high = torch.where(current < attainable_target, high, middle)
    temperature = (low * high).sqrt()
    temperature = torch.where(target <= minimum, MIN_TEMPERATURE, temperature)
    temperature = torch.where(target > maximum, MAX_TEMPERATURE, temperature)
    return (logits / temperature[:, None]).softmax(-1), temperature


def entropy_match_diagnostics(
    logits: torch.Tensor,
    target: torch.Tensor,
    matched: torch.Tensor,
    temperatures: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    centered = logits.float() - logits.float().max(dim=-1, keepdim=True).values
    minimum = entropy((centered / MIN_TEMPERATURE).softmax(-1))
    maximum = entropy((centered / MAX_TEMPERATURE).softmax(-1))
    reachable = (target >= minimum - 1e-6) & (target <= maximum + 1e-6)
    error = (entropy(matched) - target).abs()
    return {
        "temperature_bounds": [MIN_TEMPERATURE, MAX_TEMPERATURE],
        "temperature_mean": temperatures.mean().item(),
        "reachable_target_count": int(reachable.sum()),
        "unattainable_target_count": int((~reachable).sum()),
        "unattainable_row_indices": (~reachable).nonzero().flatten().tolist(),
        "max_absolute_error": error.max().item(),
        "max_absolute_error_on_reachable_targets": (
            error[reachable].max().item() if bool(reachable.any()) else None
        ),
    }, reachable


def image_bootstrap(
    values: np.ndarray,
    images: list[str],
    repetitions: int = 2000,
) -> dict[str, float]:
    """Resample image clusters, preserving all question rows in each drawn image."""
    unique = sorted(set(images))
    inverse = {name: index for index, name in enumerate(unique)}
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.float64)
    for image, value in zip(images, values, strict=True):
        sums[inverse[image]] += value
        counts[inverse[image]] += 1
    draws = np.random.default_rng(42).integers(0, len(unique), (repetitions, len(unique)))
    sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(sampled, 0.025)),
        "high": float(np.quantile(sampled, 0.975)),
    }


def correctness_probability(
    p: torch.Tensor,
    labels: torch.Tensor,
    answer_ids: dict[str, list[int]],
) -> torch.Tensor:
    yes, no = p[:, answer_ids["yes"]].sum(-1), p[:, answer_ids["no"]].sum(-1)
    return torch.where(labels, yes, no)


def replay(
    logits: dict[str, torch.Tensor],
    config: VRSCConfig,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    base, probe = logits["base"].float(), logits["semantic"].float()
    adaptive = ShieldAdaptivePlausibility()
    distributions = {
        "vanilla": base.softmax(-1),
        "adaptive": adaptive.apply(base, base).softmax(-1),
        "shrink": VRSC(replace(config, evidence_weight=0.0)).distribution(base, base),
        "vrsc": VRSC(config).distribution(base, probe),
        "shuffled": VRSC(config).distribution(base, logits["shuffled"]),
        "vrsc_residual": VRSC(
            replace(config, response_mode="orthogonal_residual")
        ).distribution(base, probe, logits["shuffled"]),
    }
    matched, temperatures = entropy_matched_distribution(base, entropy(distributions["vrsc"]))
    distributions["matched_entropy"] = matched
    greedy = torch.zeros_like(base).scatter(-1, base.argmax(-1, keepdim=True), 1.0)
    distributions["greedy"] = greedy
    if "as" in logits:
        distributions["as"] = adaptive.apply(logits["as"], logits["as"]).softmax(-1)
    return distributions, temperatures


@torch.inference_mode()
def analyze(directory: Path, output: Path | None = None) -> dict[str, Any]:
    output = output or directory
    for name in ("analysis.json", "analysis_rows.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "SMOKE" or manifest["mode"] != "capture":
        raise ValueError("a completed capture manifest is required")
    acceleration = manifest["acceleration"]
    if not (
        acceleration["effective_attention"] == "flash_attention_2"
        and acceleration["fp8_effective"] is True
        and acceleration["fp8_native_fallback_calls"] == 0
        and acceleration["fallback_events"] == []
    ):
        raise ValueError("capture failed canonical acceleration acceptance")
    logits = torch.load(directory / "logits.pt", map_location="cpu", weights_only=True)
    rows = json.loads((directory / "rows.json").read_text())
    answer_ids = json.loads((directory / "answer_token_ids.json").read_text())
    config = VRSCConfig(**manifest["method_config"])
    n = len(rows)
    if any(value.ndim != 2 or value.shape != logits["base"].shape for value in logits.values()):
        raise ValueError("capture branch shapes differ")
    if not n or logits["base"].shape[0] != n or manifest["count"] != n:
        raise ValueError("capture row count differs from metadata")
    if any(row["label"].lower() not in ("yes", "no") for row in rows):
        raise ValueError("capture requires Yes/No labels")
    labels = torch.tensor([row["label"].lower() == "yes" for row in rows])
    images = [row["image"] for row in rows]
    distributions, temperatures = replay(logits, config)
    entropy_diagnostics, reachable = entropy_match_diagnostics(
        logits["base"],
        entropy(distributions["vrsc"]),
        distributions["matched_entropy"],
        temperatures,
    )
    metrics: dict[str, Any] = {}
    correct = {}
    row_metrics: list[dict[str, Any]] = [
        {"question_id": row["question_id"], "image": row["image"], "label": row["label"]}
        for row in rows
    ]
    for name, p in distributions.items():
        yes, no = p[:, answer_ids["yes"]].sum(-1), p[:, answer_ids["no"]].sum(-1)
        good = correctness_probability(p, labels, answer_ids)
        correct[name] = good.numpy()
        argmax = p.argmax(-1)
        yes_top = torch.isin(argmax, torch.tensor(answer_ids["yes"]))
        no_top = torch.isin(argmax, torch.tensor(answer_ids["no"]))
        strict_correct = torch.where(labels, yes_top, no_top)
        metrics[name] = {
            "expected_correct_first_token": good.mean().item(),
            "argmax_exact_yes_no_accuracy": strict_correct.float().mean().item(),
            "non_answer_mass": (1 - yes - no).clamp_min(0).mean().item(),
            "mean_entropy": entropy(p).mean().item(),
            "mean_support": (p > 0).float().sum(-1).mean().item(),
            "yes_expected_correct": good[labels].mean().item(),
            "no_expected_correct": good[~labels].mean().item(),
        }
        for index, row in enumerate(row_metrics):
            row[name + "_correct_probability"] = good[index].item()
    comparisons = {}
    for anchor in ("vrsc", "vrsc_residual"):
        for comparator in (
            "vanilla",
            "adaptive",
            "shrink",
            "shuffled",
            "matched_entropy",
            "vrsc",
            "vrsc_residual",
            "as",
        ):
            if comparator in correct and comparator != anchor:
                comparisons[f"{anchor}_minus_{comparator}"] = image_bootstrap(
                    correct[anchor] - correct[comparator], images
                )
    if bool(reachable.any()):
        comparisons["vrsc_minus_matched_entropy_reachable_only"] = image_bootstrap(
            (correct["vrsc"] - correct["matched_entropy"])[reachable.numpy()],
            [image for image, keep in zip(images, reachable.tolist(), strict=True) if keep],
        )
    mechanism = {}
    base_log = logits["base"].float().log_softmax(-1)
    base_log_odds = torch.logsumexp(base_log[:, answer_ids["yes"]], -1) - torch.logsumexp(
        base_log[:, answer_ids["no"]], -1
    )
    signed_responses = {}
    for name in ("semantic", "shuffled"):
        branch_log = logits[name].float().log_softmax(-1)
        branch_log_odds = torch.logsumexp(branch_log[:, answer_ids["yes"]], -1)
        branch_log_odds -= torch.logsumexp(branch_log[:, answer_ids["no"]], -1)
        signed = (branch_log_odds - base_log_odds) * (labels.float() * 2 - 1)
        signed_responses[name] = signed.numpy()
        mechanism[name] = {
            "correct_answer_log_odds_change": image_bootstrap(signed.numpy(), images),
            "yes_mean": signed[labels].mean().item(),
            "no_mean": signed[~labels].mean().item(),
            "positive_response_fraction": (signed > 0).float().mean().item(),
            "mean_abs_logit_difference": (logits[name] - logits["base"]).abs().mean().item(),
        }
    mechanism["semantic_minus_shuffled"] = image_bootstrap(
        signed_responses["semantic"] - signed_responses["shuffled"], images
    )
    result = {
        "kind": "first_token_paired_replay_not_autoregressive_evaluation",
        "capture_directory": str(directory.resolve()),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture_manifest_sha256": hashlib.sha256(
            (directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "count": n,
        "image_clusters": len(set(images)),
        "method_config": manifest["method_config"],
        "methods": metrics,
        "paired_expected_correct_deltas": comparisons,
        "mechanism": mechanism,
        "answer_token_ids": answer_ids,
        "entropy_match": entropy_diagnostics,
        "acceleration": acceleration,
        "limitations": [
            "Expected first-token accuracy is not generated POPE Accuracy.",
            "Image-cluster bootstrap is exploratory; no multiplicity adjustment.",
            "Semantic and shuffled probe amplitudes must be inspected separately.",
            "Caption construction cost is excluded from runtime timing.",
            "Entropy-unattainable rows use a boundary, not an exact entropy match.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("analysis.json", result), ("analysis_rows.json", row_metrics)):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--output", type=Path, help="new analysis directory; preserve prior results"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = analyze(args.capture, args.output)
    LOGGER.info("methods: %s", json.dumps(result["methods"]))
    LOGGER.info("paired deltas: %s", json.dumps(result["paired_expected_correct_deltas"]))
    LOGGER.info("mechanism: %s", json.dumps(result["mechanism"]))
    LOGGER.info("entropy match: %s", json.dumps(result["entropy_match"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
