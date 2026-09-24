from __future__ import annotations

from typing import Any

import pytest
import torch

from methods.gtp import (
    GTPConfig,
    GTPMethod,
    GroundedTokenProjection,
    counterfactual_patch_scores,
)


def test_counterfactual_patch_scores_prefers_relevant_positive_control_margin() -> None:
    scores = torch.tensor([0.1, 0.8, 0.5])
    control = torch.tensor([0.1, 0.2, 0.7])
    result = counterfactual_patch_scores(scores, control, temperature=0.05)
    assert torch.isfinite(result).all()
    assert torch.all(result >= 0)
    assert result[1] > result[0] > result[2]
    assert torch.allclose(result.mean(), torch.ones(()), atol=1e-5)


def test_counterfactual_patch_scores_rejects_bad_shapes_and_temperature() -> None:
    with pytest.raises(ValueError):
        counterfactual_patch_scores(torch.ones(2), torch.ones(1))
    with pytest.raises(ValueError):
        counterfactual_patch_scores(torch.ones(2), torch.ones(2), temperature=0)


def test_reliability_is_zero_for_no_signal_and_extreme_disagreement() -> None:
    plugin = GroundedTokenProjection(
        GTPConfig(signal_scale=0.02, disagreement_scale=0.08)
    )
    equal = torch.tensor([[0.6, 0.3, 0.1]])
    assert torch.allclose(plugin._reliability(equal, equal), torch.zeros((1, 1)))
    opposite = torch.tensor([[0.0, 1.0, 0.0]])
    reliability = plugin._reliability(equal, opposite)
    assert 0 < reliability.item() < 0.1


def test_distribution_is_normalized_and_clean_fallback_is_exact() -> None:
    plugin = GroundedTokenProjection()
    clean = torch.tensor([[3.0, 1.0, -1.0]])
    result = plugin.distribution(clean, clean)
    assert torch.allclose(result.sum(-1), torch.ones(1))
    assert torch.isfinite(result).all()
    assert torch.allclose(result, clean.softmax(-1), atol=1e-6)


def test_selective_projection_only_uses_positive_counterfactual_candidates() -> None:
    plugin = GroundedTokenProjection(
        GTPConfig(
            mode="selective",
            max_evidence_weight=1.0,
            candidate_top_k=2,
            support_margin=0.0,
        )
    )
    clean = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
    evidence = torch.tensor([[4.0, 3.0, 0.0, -1.0]])
    control = torch.tensor([[4.0, 1.0, 3.0, -1.0]])
    result = plugin.distribution(clean, evidence, control)

    assert torch.allclose(result.sum(-1), torch.ones(1))
    assert torch.isfinite(result).all()
    # Token 1 is supported by the evidence branch and preferred over control;
    # token 2 has the opposite counterfactual sign and must not be boosted.
    assert result[0, 1] > clean.softmax(-1)[0, 1]
    assert result[0, 2] <= clean.softmax(-1)[0, 2]


def test_legacy_projection_remains_available_for_reproduction() -> None:
    plugin = GroundedTokenProjection(GTPConfig(mode="legacy"))
    clean = torch.tensor([[2.0, 1.0, -1.0]])
    result = plugin.distribution(clean, clean)
    assert torch.allclose(result, clean.softmax(-1), atol=1e-6)


def test_method_contract_dispatches_gtp() -> None:
    class Backend:
        def generate_batch(
            self, records: list[dict[str, Any]], **kwargs: Any
        ) -> list[dict[str, Any]]:
            assert kwargs["method_name"] == "gtp"
            assert kwargs["method_config"]["max_evidence_weight"] == 0.75
            return [{"id": row["id"], "answer": "Yes"} for row in records]

    method = GTPMethod()
    method.setup(Backend(), torch.device("cpu"))
    assert method.generate({"records": [{"id": "sample"}]}) == [
        {"id": "sample", "answer": "Yes"}
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evidence_weight": 1.1},
        {"trust": 0.0},
        {"signal_scale": -1.0},
        {"disagreement_scale": float("nan")},
        {"sharpness": 0.0},
    ],
)
def test_invalid_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GTPConfig(**kwargs)
