from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from methods.evidence_consensus import (
    EvidenceConsensus,
    EvidenceConsensusConfig,
    EvidenceConsensusMethod,
    evidence_transport,
    transport_features,
)
from methods.vrsc import chi_square_projection


def test_two_reference_kkt() -> None:
    rng = torch.Generator().manual_seed(19)
    clean = torch.randn(3, 9, generator=rng)
    evidence = torch.randn(3, 9, generator=rng)
    weight, trust = 0.75, 1.0
    q = EvidenceConsensus().distribution(clean, evidence)
    p, g = clean.softmax(-1), evidence.softmax(-1)
    utility = (1 - weight) * clean.log_softmax(-1) + weight * evidence.log_softmax(-1)
    gradient = utility - trust * ((1 - weight) * (q / p - 1) + weight * (q / g - 1))
    for row in range(3):
        active = q[row] > 1e-7
        dual = gradient[row, active].mean()
        assert torch.allclose(
            gradient[row, active], dual.expand_as(gradient[row, active]), atol=1e-5
        )
        assert torch.all(gradient[row, ~active] <= dual + 1e-5)


def test_equal_branches_and_endpoints() -> None:
    clean = torch.tensor([[0.0, -0.3, -2.0]])
    evidence = torch.tensor([[-0.5, 0.0, -2.0]])
    expected = chi_square_projection(clean.softmax(-1), clean.log_softmax(-1))
    assert torch.allclose(EvidenceConsensus().distribution(clean, clean), expected)
    assert torch.allclose(
        EvidenceConsensus(EvidenceConsensusConfig(evidence_weight=0)).distribution(clean, evidence),
        expected,
    )
    swapped = EvidenceConsensus(
        EvidenceConsensusConfig(evidence_weight=0.25)
    ).distribution(evidence, clean)
    assert torch.allclose(EvidenceConsensus().distribution(clean, evidence), swapped, atol=1e-6)


def test_transport_marginals_permutation_and_flat_features() -> None:
    values = torch.tensor([[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    joint = evidence_transport(values)
    assert torch.allclose(joint.sum(), torch.tensor(1.0))
    assert joint[1].sum() > joint[0].sum()
    assert joint[:, 1].sum() > joint[:, 0].sum()
    order = torch.tensor([2, 0, 1])
    assert torch.allclose(evidence_transport(values[order]), joint[order])
    image = torch.ones(4, 1024)
    unchanged, selected, _ = transport_features(
        image, torch.ones(3, 768), EvidenceConsensusConfig()
    )
    assert torch.equal(unchanged, image)
    assert selected.numel() == 0


@pytest.mark.parametrize("kwargs", [{"trust": 0}, {"evidence_weight": 1.1}, {"feature_gain": -1}])
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EvidenceConsensusConfig(**kwargs)


def test_method_contract_and_extreme_branch_disagreement() -> None:
    class Backend:
        def generate_batch(
            self, records: list[dict[str, Any]], **kwargs: Any
        ) -> list[dict[str, Any]]:
            assert kwargs["method_name"] == "evidence_consensus"
            assert kwargs["method_config"]["evidence_weight"] == 0.75
            return [{"id": row["id"], "answer": "Yes"} for row in records]

    method = EvidenceConsensusMethod()
    method.setup(Backend(), torch.device("cpu"))
    assert method.generate({"records": [{"id": "sample"}]}) == [
        {"id": "sample", "answer": "Yes"}
    ]
    clean = torch.tensor([[1000.0, -1000.0]])
    q = method.plugin.distribution(clean, -clean)
    assert torch.isfinite(q).all()
    assert torch.all(q >= 0)
    assert torch.allclose(q.sum(-1), torch.ones(1))
    assert torch.allclose(q, torch.tensor([[0.25, 0.75]]), atol=1e-5)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.75, 1.0])
def test_normalization_fix_preserves_accepted_frozen_path(weight: float) -> None:
    generator = torch.Generator().manual_seed(23)
    clean = torch.randn((4, 257), generator=generator) * 12
    evidence = torch.randn((4, 257), generator=generator) * 12
    log_p, log_g = clean.log_softmax(-1), evidence.log_softmax(-1)
    if weight == 0:
        log_h = log_p
    elif weight == 1:
        log_h = log_g
    else:
        log_h = -torch.logaddexp(math.log1p(-weight) - log_p, math.log(weight) - log_g)
    log_agreement = log_h.logsumexp(-1, keepdim=True)
    prior = (log_h - log_agreement).exp()
    utility = (1 - weight) * log_p + weight * log_g
    expected = chi_square_projection(prior, log_agreement.exp() * utility)
    actual = EvidenceConsensus(
        EvidenceConsensusConfig(evidence_weight=weight)
    ).distribution(clean, evidence)
    assert torch.equal(actual, expected)
