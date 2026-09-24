from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.compare_evidence_results import aggregate, compare, load_result, metric_arrays


def test_chair_micro_rate_is_not_mean_per_caption_rate() -> None:
    rows = [
        {"chair": {"CHAIRs": 1, "CHAIRi": 0.5, "Recall": 1.0}, "objects": ["a", "b"]},
        {"chair": {"CHAIRs": 0, "CHAIRi": 0.0, "Recall": 0.5}, "objects": ["c"] * 8},
    ]
    scores = aggregate(metric_arrays(rows, "chair").sum(0), "chair")
    assert scores["CHAIRs"] == pytest.approx(0.5)
    assert scores["CHAIRi"] == pytest.approx(0.1)
    assert scores["Recall"] == pytest.approx(0.75)


def test_pope_totals_and_vectorized_bootstrap_metrics() -> None:
    rows = [{"label": y, "prediction": p} for y, p in [(1, 1), (1, 0), (0, 0)]]
    counts = metric_arrays(rows, "pope").sum(0)
    result = aggregate(np.stack([counts, counts]), "pope")
    assert np.allclose(result["accuracy"], 2 / 3)
    assert np.allclose(result["precision"], 1)
    assert np.allclose(result["recall"], 0.5)
    assert np.allclose(result["f1"], 2 / 3)


@pytest.fixture
def pope_artifact(tmp_path: Path) -> Path:
    rows = [
        {
            "question_id": index,
            "image": f"image-{index // 2}",
            "prompt": f"question-{index}",
            "label": index % 2,
            "prediction": index % 2,
        }
        for index in range(4)
    ]
    (tmp_path / "result.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    meta = {
        "status": "SMOKE",
        "count": len(rows),
        "method": "evidence_consensus",
        "branch_routing": ["base", "evidence", None],
        "model_revision": "test-revision",
        "model_config_sha256": "test-config",
        "data_sha256": "test-data",
        "protocol": {"seed": 42},
        "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "acceleration": {
            "effective_attention": "flash_attention_2",
            "fp8_effective": True,
            "fp8_native_fallback_calls": 0,
            "fallback_events": [],
            "dtype": "bfloat16",
            "tf32": True,
            "fp8_scope": "text_mlp",
        },
    }
    path = tmp_path / "result.metrics.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"status": "RUNNING"}, "incomplete"),
        ({"status": "PASS"}, "dataset size"),
        ({"count": 3}, "count mismatch"),
        ({"branch_routing": ["base", "base", None]}, "branch routing"),
        ({"metrics": {"accuracy": 0.5}}, "metrics mismatch"),
    ],
)
def test_rejects_invalid_evidence(
    pope_artifact: Path, patch: dict[str, Any], message: str
) -> None:
    meta = json.loads(pope_artifact.read_text())
    meta.update(patch)
    pope_artifact.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        load_result(pope_artifact)


def test_cluster_exclusion_and_new_artifact_reference(pope_artifact: Path) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        compare(pope_artifact, pope_artifact, "pope", -1)
    with pytest.raises(ValueError, match="splits an image"):
        compare(pope_artifact, pope_artifact, "pope", 1)
    with pytest.raises(ValueError, match="dataset"):
        compare(pope_artifact, pope_artifact, "chair")
    result = compare(pope_artifact, pope_artifact, "pope", 2)
    assert result["count"] == 2
    assert result["image_clusters"] == 1
    assert result["excluded_image_clusters"] == 1
    assert not result["full_dataset_comparison"]
    assert result["candidate_minus_reference"]["accuracy"] == {
        "delta_pp": 0,
        "image_cluster_95_interval_pp": [0, 0],
    }


def test_generation_vocab_metadata_gap_is_not_reported_as_strict_match(
    pope_artifact: Path, tmp_path: Path
) -> None:
    candidate_meta = json.loads(pope_artifact.read_text())
    candidate_meta["protocol"]["generation_vocab_size"] = 4
    pope_artifact.write_text(json.dumps(candidate_meta))

    reference_rows = pope_artifact.with_name("reference.jsonl")
    reference_rows.write_text(pope_artifact.with_name("result.jsonl").read_text())
    reference_meta = {**candidate_meta, "protocol": {"seed": 42}}
    reference_path = tmp_path / "reference.metrics.json"
    reference_path.write_text(json.dumps(reference_meta))

    result = compare(pope_artifact, reference_path, "pope")

    assert result["generation_vocab_boundary_comparison"] == "REFERENCE_UNRECORDED"
    assert result["strict_protocol_comparison"] == "INCOMPLETE"
    assert result["protocol_and_acceleration_audit"] == "PARTIAL"

