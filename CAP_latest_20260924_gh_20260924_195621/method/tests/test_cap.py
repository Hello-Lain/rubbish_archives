from __future__ import annotations

from typing import Any

import pytest
import torch

from methods.base_method import BaseMethod
from methods.cap import (
    CAPConfig,
    CAPMethod,
    causal_attribution_weights,
    energy_preserving_probe,
)


def test_causal_attribution_basic_properties() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scores = causal_attribution_weights(features, torch.tensor([1.0, 0.0]))
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    assert torch.all(scores <= 1)
    assert torch.all(scores >= -1)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0, -1.0]))


def test_causal_attribution_aligned_patch_is_highest() -> None:
    features = torch.tensor([[0.1, 0.9], [1.0, 0.0], [0.6, 0.8]])
    scores = causal_attribution_weights(features, torch.tensor([0.6, 0.8]))
    assert scores.argmax() == 2


def test_causal_attribution_anti_aligned_patch_is_lowest() -> None:
    features = torch.tensor([[0.6, 0.8], [-0.6, -0.8], [0.8, -0.6]])
    scores = causal_attribution_weights(features, torch.tensor([0.6, 0.8]))
    assert scores.argmin() == 1


def test_causal_attribution_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="size"):
        causal_attribution_weights(torch.ones(2, 3), torch.ones(4))


@pytest.mark.parametrize(
    ("features", "hidden"),
    [
        (torch.ones(3), torch.ones(3)),
        (torch.ones(2, 3), torch.ones(1, 3)),
    ],
)
def test_causal_attribution_rejects_bad_shapes(
    features: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        causal_attribution_weights(features, hidden)


def test_causal_attribution_rejects_nonfinite() -> None:
    with pytest.raises(ValueError):
        causal_attribution_weights(
            torch.tensor([[1.0, float("nan")]]),
            torch.ones(2),
        )


def test_causal_attribution_handles_zero_norm_inputs() -> None:
    scores = causal_attribution_weights(torch.zeros(2, 3), torch.zeros(3))
    assert torch.equal(scores, torch.zeros(2))


def test_causal_attribution_handles_empty_patch_set() -> None:
    scores = causal_attribution_weights(torch.empty(0, 3), torch.ones(3))
    assert scores.shape == (0,)


def test_probe_zero_gain_is_identity() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    scores = torch.tensor([0.1, 0.8, -0.2, 0.3])
    assert torch.equal(energy_preserving_probe(features, scores, 0.0), features)


def test_probe_preserves_total_norm() -> None:
    features = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
    scores = torch.tensor([0.1, 0.8, -0.2, 0.3])
    probed = energy_preserving_probe(features, scores, 0.20)
    assert torch.allclose(probed.norm(), features.norm(), rtol=1e-4, atol=1e-5)


def test_probe_high_scores_get_enhanced() -> None:
    features = torch.ones(4, 3)
    scores = torch.tensor([-1.0, 1.0, -0.5, 0.0])
    probed = energy_preserving_probe(features, scores, 0.20)
    assert probed[1].norm() > probed[0].norm()


@pytest.mark.parametrize("gain", [-0.01, 1.0, float("nan"), float("inf")])
def test_probe_rejects_bad_eta(gain: float) -> None:
    with pytest.raises(ValueError):
        energy_preserving_probe(torch.ones(2, 3), torch.ones(2), gain)


def test_probe_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        energy_preserving_probe(torch.ones(2, 3), torch.ones(3), 0.20)


def test_config_default_is_valid() -> None:
    assert CAPConfig() == CAPConfig(probe_gain=0.20)


@pytest.mark.parametrize("gain", [-0.01, 1.0, float("nan")])
def test_config_rejects_invalid_gain(gain: float) -> None:
    with pytest.raises(ValueError):
        CAPConfig(probe_gain=gain)


def test_method_contract_dispatches_cap() -> None:
    class Backend:
        def generate_batch(
            self,
            records: list[dict[str, Any]],
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            assert kwargs["method_name"] == "cap"
            assert kwargs["method_config"]["probe_gain"] == 0.20
            return [{"id": row["id"], "answer": "Yes"} for row in records]

    method: BaseMethod = CAPMethod()
    method.setup(Backend(), torch.device("cpu"))
    assert method.generate({"records": [{"id": "sample"}]}) == [
        {"id": "sample", "answer": "Yes"}
    ]
