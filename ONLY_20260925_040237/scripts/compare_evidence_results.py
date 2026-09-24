#!/usr/bin/env python3
"""Audit existing predictions and compare paired images without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
BRANCH_ROUTING = {
    "vrsc_residual": ["base", "semantic", "shuffled"],
    "evidence_consensus": ["base", "evidence", None],
    "evidence_shrink": ["evidence", "evidence", None],
    "evidence_visual": ["base", "evidence_visual", None],
}


def load_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(path.read_text())
    if summary.get("status") not in {"PASS", "SMOKE"}:
        raise ValueError(f"incomplete result: {path}")
    prediction_path = Path(
        summary.get("predictions_path", str(path).replace(".metrics.json", ".jsonl"))
    )
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    if len(rows) != summary["count"]:
        raise ValueError(f"prediction count mismatch: {path}")
    if len({str(row["question_id"]) for row in rows}) != len(rows):
        raise ValueError(f"duplicate questions: {path}")
    method = summary["method"]
    if method in BRANCH_ROUTING and summary.get("branch_routing") != BRANCH_ROUTING[method]:
        raise ValueError(f"unverified branch routing: {path}")
    accel = summary["acceleration"]
    for key, expected in (
        ("effective_attention", "flash_attention_2"),
        ("fp8_effective", True),
        ("fp8_native_fallback_calls", 0),
        ("fallback_events", []),
        ("dtype", "bfloat16"),
        ("tf32", True),
        ("fp8_scope", "text_mlp"),
    ):
        if accel[key] != expected:
            raise ValueError(f"acceleration rejected: {path}: {key}")
    dataset = "chair" if rows and "chair" in rows[0] else "pope"
    if not rows or (
        summary["status"] == "PASS" and len(rows) != (500 if dataset == "chair" else 3000)
    ):
        raise ValueError(f"invalid completed dataset size: {path}")
    measured = aggregate(metric_arrays(rows, dataset).sum(0), dataset)
    reported = summary.get("metrics", summary.get("chair", {}))
    for key, value in measured.items():
        if key not in reported or not np.isclose(value, reported[key], atol=1e-8, rtol=0):
            raise ValueError(f"prediction/metrics mismatch: {path}: {key}")
    return summary, rows


def metric_arrays(rows: list[dict[str, Any]], dataset: str) -> np.ndarray:
    if dataset == "pope":
        label = np.array([row["label"] for row in rows], dtype=bool)
        pred = np.array([row["prediction"] for row in rows], dtype=bool)
        return np.stack((pred & label, ~pred & ~label, pred & ~label, ~pred & label), axis=-1)
    return np.array([
        [
            row["chair"]["CHAIRs"],
            round(row["chair"]["CHAIRi"] * len(row["objects"])),
            len(row["objects"]),
            row["chair"]["Recall"],
            1,
        ]
        for row in rows
    ], dtype=np.float64)


def aggregate(values: np.ndarray, dataset: str) -> dict[str, np.ndarray]:
    if dataset == "pope":
        tp, tn, fp, fn = np.moveaxis(values, -1, 0)
        return {
            "accuracy": (tp + tn) / np.maximum(tp + tn + fp + fn, 1),
            "precision": tp / np.maximum(tp + fp, 1),
            "recall": tp / np.maximum(tp + fn, 1),
            "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
        }
    halls, bad, objects, recalls, count = np.moveaxis(values, -1, 0)
    return {
        "CHAIRs": halls / np.maximum(count, 1),
        "CHAIRi": bad / np.maximum(objects, 1),
        "Recall": recalls / np.maximum(count, 1),
    }


def compare(candidate: Path, reference: Path, dataset: str, start: int = 0) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must be nonnegative")
    if dataset not in {"pope", "chair"}:
        raise ValueError("dataset must be pope or chair")
    cand_meta, cand_rows = load_result(candidate)
    ref_meta, ref_rows = load_result(reference)
    if any(("chair" in rows[0]) != (dataset == "chair") for rows in (cand_rows, ref_rows)):
        raise ValueError("prediction dataset does not match the requested dataset")
    cand_manifest = (
        json.loads(Path(cand_meta["manifest"]).read_text())
        if "manifest" in cand_meta
        else cand_meta
    )
    ref_manifest = (
        json.loads(Path(ref_meta["manifest"]).read_text())
        if "manifest" in ref_meta
        else ref_meta
    )
    for key in ("model_revision", "model_config_sha256"):
        if cand_manifest[key] != ref_manifest[key]:
            raise ValueError(f"mismatched {key}")
    data_key = "pope_sha256" if dataset == "pope" else "questions_sha256"
    if cand_manifest.get("data_sha256", cand_manifest.get(data_key)) != ref_manifest.get(
        "data_sha256", ref_manifest.get(data_key)
    ):
        raise ValueError("mismatched data hash")
    for key, value in ref_meta["protocol"].items():
        other = cand_meta["protocol"][key]
        # Older CHAIR artifacts record None for the disabled one-word suffix.
        equal = bool(other) == bool(value) if key == "pope_answer_instruction" else other == value
        if not equal:
            raise ValueError(f"mismatched protocol: {key}: {other} != {value}")
    candidate_vocab = cand_meta["protocol"].get("generation_vocab_size")
    reference_vocab = ref_meta["protocol"].get("generation_vocab_size")
    if candidate_vocab is None and reference_vocab is None:
        vocab_boundary = "UNRECORDED_BOTH"
    elif candidate_vocab is None:
        vocab_boundary = "CANDIDATE_UNRECORDED"
    elif reference_vocab is None:
        vocab_boundary = "REFERENCE_UNRECORDED"
    elif candidate_vocab != reference_vocab:
        vocab_boundary = "MISMATCH"
    else:
        vocab_boundary = "MATCH"
    indexed = {str(row["question_id"]): row for row in ref_rows}
    excluded_images = {row["image"] for row in cand_rows[:start]}
    cand_rows = cand_rows[start:]
    if not cand_rows:
        raise ValueError("empty comparison slice")
    if excluded_images & {row["image"] for row in cand_rows}:
        raise ValueError("start splits an image cluster; exclude complete images")
    paired = [indexed[str(row["question_id"])] for row in cand_rows]
    for a, b in zip(cand_rows, paired, strict=True):
        if (a["image"], a.get("prompt", a.get("question"))) != (
            b["image"], b.get("prompt", b.get("question"))
        ):
            raise ValueError("mismatched image or prompt")
        if dataset == "pope" and a["label"] != b["label"]:
            raise ValueError("mismatched labels")
    arrays = [metric_arrays(rows, dataset).astype(np.float64) for rows in (cand_rows, paired)]
    images = sorted({row["image"] for row in cand_rows})
    image_ids = {image: index for index, image in enumerate(images)}
    clusters = []
    for values in arrays:
        sums = np.zeros((len(images), values.shape[-1]))
        np.add.at(sums, [image_ids[row["image"]] for row in cand_rows], values)
        clusters.append(sums)
    draws = np.random.default_rng(42).integers(0, len(images), (2000, len(images)))
    totals = [aggregate(values.sum(0), dataset) for values in arrays]
    sampled = [aggregate(values[draws].sum(1), dataset) for values in clusters]
    deltas = {
        key: {
            "delta_pp": float((totals[0][key] - totals[1][key]) * 100),
            "image_cluster_95_interval_pp": (
                np.quantile(sampled[0][key] - sampled[1][key], [0.025, 0.975]) * 100
            ).tolist(),
        }
        for key in totals[0]
    }
    return {
        "candidate": str(candidate),
        "reference_reused": str(reference),
        "candidate_metrics_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "reference_metrics_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "dataset": dataset,
        "candidate_status": cand_meta["status"],
        "reference_status": ref_meta["status"],
        "generation_vocab_boundary_comparison": vocab_boundary,
        "strict_protocol_comparison": (
            "PASS" if vocab_boundary == "MATCH" else "INCOMPLETE"
        ),
        "full_dataset_comparison": (
            start == 0 and len(cand_rows) == (500 if dataset == "chair" else 3000)
        ),
        "count": len(cand_rows),
        "start_row_in_candidate": start,
        "excluded_image_clusters": len(excluded_images),
        "image_clusters": len(images),
        "candidate_metrics": {key: float(value) for key, value in totals[0].items()},
        "reference_metrics_on_same_rows": {key: float(value) for key, value in totals[1].items()},
        "candidate_minus_reference": deltas,
        "protocol_and_acceleration_audit": (
            "PASS" if vocab_boundary == "MATCH" else "PARTIAL"
        ),
        "limitations": [
            "Single-seed exploratory comparison, not independent replication.",
            "Bootstrap pairs images, not identical autoregressive random draws.",
            "Intervals are not adjusted for multiple comparisons.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dataset", choices=("pope", "chair"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.candidate, args.reference, args.dataset, args.start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    LOGGER.info("%s", json.dumps(result["candidate_minus_reference"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
