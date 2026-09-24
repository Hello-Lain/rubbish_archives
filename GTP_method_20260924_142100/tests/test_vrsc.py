from __future__ import annotations

from typing import Any

import pytest
import torch

from methods.vrsc import (
    VRSC,
    VRSCConfig,
    VRSCMethod,
    chi_square_projection,
    energy_preserving_probe,
    patch_caption_scores,
)


def test_projection_kkt_and_invariances() -> None:
    generator = torch.Generator().manual_seed(7)
    p = torch.randn(5, 71, generator=generator).softmax(-1)
    u = torch.randn(5, 71, generator=generator)
    q = chi_square_projection(p, u, 0.7)
    assert torch.all(q >= 0)
    assert torch.allclose(q.sum(-1), torch.ones(5))
    gradient = u - 0.7 * (q - p) / p
    for row in range(5):
        active = q[row] > 1e-7
        dual = gradient[row, active].mean()
        assert torch.allclose(
            gradient[row, active], dual.expand_as(gradient[row, active]), atol=2e-5
        )
        assert torch.all(gradient[row, ~active] <= dual + 2e-5)
    assert torch.allclose(q, chi_square_projection(p, u + 17, 0.7), atol=1e-6)
    order = torch.randperm(71, generator=generator)
    assert torch.allclose(
        q[:, order], chi_square_projection(p[:, order], u[:, order], 0.7), atol=1e-6
    )
    assert torch.allclose(p, chi_square_projection(p, torch.ones_like(p)), atol=1e-6)


def test_zero_support_and_extreme_logits() -> None:
    q = chi_square_projection(torch.tensor([0.0, 0.25, 0.75]), torch.tensor([100.0, -2.0, 1.0]))
    assert q[0] == 0
    result = VRSC().distribution(
        torch.tensor([[1000.0, -1000.0, 0.0]]), torch.tensor([[-1000.0, 1000.0, 0.0]])
    )
    assert torch.isfinite(result).all()
    assert torch.allclose(result.sum(-1), torch.ones(1))


def test_residual_response_requires_reference_and_is_finite() -> None:
    base = torch.tensor([[3.0, 1.0, -1.0]])
    probe = torch.tensor([[2.0, 2.0, -1.0]])
    reference = torch.tensor([[2.5, 1.5, -1.0]])
    method = VRSC(VRSCConfig(response_mode="orthogonal_residual"))
    result = method.distribution(base, probe, reference)
    assert torch.isfinite(result).all()
    assert torch.allclose(result.sum(-1), torch.ones(1))
    with pytest.raises(ValueError):
        method.distribution(base, probe)


def test_residual_projects_raw_response_before_clipping() -> None:
    base = torch.tensor([[0.0, -0.1, -0.3, -4.0]])
    probe = torch.tensor([[2.0, -0.5, 0.2, -3.0]])
    reference = torch.tensor([[0.3, 0.0, -0.6, -3.0]])
    log_p = base.log_softmax(-1)
    semantic = probe.log_softmax(-1) - log_p
    nuisance = reference.log_softmax(-1) - log_p
    coefficient = (semantic * nuisance).sum(-1, keepdim=True)
    coefficient /= nuisance.square().sum(-1, keepdim=True).clamp_min(1e-12)
    residual = (semantic - coefficient * nuisance).clamp(-1, 1)
    expected = chi_square_projection(log_p.exp(), log_p + 2 * residual)
    actual = VRSC(VRSCConfig(response_mode="orthogonal_residual")).distribution(
        base, probe, reference
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_visual_response_changes_rank_and_removes_tail() -> None:
    p = torch.tensor([0.5, 0.35, 0.145, 0.005])
    probe = torch.tensor([0.33, 0.51, 0.154, 0.006])
    q = VRSC().distribution(p.log(), probe.log())
    shrink = VRSC(VRSCConfig(evidence_weight=0)).distribution(p.log(), probe.log())
    assert q.argmax() == 1
    assert shrink.argmax() == p.argmax()
    assert q[-1] == 0
    assert torch.allclose(q, VRSC().distribution(p.log() + 12, probe.log() - 9), atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_energy_probe_norm_and_identity(dtype: torch.dtype) -> None:
    x = torch.arange(24).reshape(4, 6).to(dtype)
    s = torch.tensor([0.0, 1.0, 0.5, 0.25])
    probe = energy_preserving_probe(x, s, 0.25)
    assert probe.dtype == dtype
    assert torch.allclose(probe.float().norm(), x.float().norm(), rtol=0.002)
    assert torch.equal(energy_preserving_probe(x, s, 0), x)
    assert torch.equal(energy_preserving_probe(x, torch.ones(4), 0.25), x)
    assert torch.equal(energy_preserving_probe(torch.zeros_like(x), s, 0.25), torch.zeros_like(x))


def test_probe_tangent_and_score_shift_invariance() -> None:
    x = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
    scores = torch.tensor([0.1, 0.7, 0.2, 0.3])
    assert torch.allclose(
        energy_preserving_probe(x, scores, 0.25),
        energy_preserving_probe(x, scores + 2, 0.25),
        atol=1e-5,
    )
    eta = 1e-3
    direction = (energy_preserving_probe(x, scores, eta) - x) / eta
    cosine = (direction * x).sum() / (direction.norm() * x.norm())
    assert cosine.abs() < 0.002


def test_patch_score_empty_caption() -> None:
    assert torch.equal(
        patch_caption_scores(torch.ones(4, 1024), torch.empty(0, 768)), torch.zeros(4)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": 1},
        {"eta": -0.1},
        {"trust": 0},
        {"response_clip": float("nan")},
        {"evidence_weight": -1},
        {"response_mode": "unknown"},
    ],
)
def test_invalid_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        VRSCConfig(**kwargs)


def test_invalid_projection() -> None:
    with pytest.raises(ValueError):
        chi_square_projection(torch.tensor([0.1, 0.1]), torch.ones(2))
    with pytest.raises(ValueError):
        VRSC().distribution(torch.tensor([float("nan"), 1]), torch.ones(2))
    with pytest.raises(ValueError):
        energy_preserving_probe(torch.ones(3, 2), torch.ones(2), 0.2)


def test_method_contract() -> None:
    class Backend:
        def generate_batch(
            self,
            records: list[dict[str, Any]],
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            assert kwargs["method_name"] == "vrsc"
            assert kwargs["method_config"]["eta"] == 0.25
            return [{"id": row["id"], "answer": "Yes"} for row in records]

    method = VRSCMethod()
    method.setup(Backend(), torch.device("cpu"))
    assert method.generate({"records": [{"id": "sample"}]}) == [{"id": "sample", "answer": "Yes"}]
