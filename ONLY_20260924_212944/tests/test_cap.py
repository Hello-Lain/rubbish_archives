from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from methods.base_method import BaseMethod
from methods.cap import (
    CAPConfig,
    CAPMethod,
    attention_attribution_weights,
    causal_attribution_weights,
    contrastive_hidden_attribution_weights,
    eager_last_layer_attention,
    energy_preserving_probe,
    last_language_layer_index,
    zero_image_embeddings,
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


def test_attention_highest_patch_wins() -> None:
    heads, sequence = 4, 12
    attention = torch.full((heads, sequence, sequence), 1.0 / sequence)
    attention[:, 10, 3] = 0.5
    attention[:, 10] /= attention[:, 10].sum(dim=-1, keepdim=True)
    scores = attention_attribution_weights(attention, 10, list(range(10)))
    assert scores.argmax().item() == 3
    assert scores[3] > scores[0]


def test_attention_scores_are_nonnegative_and_bounded_by_one() -> None:
    attention = torch.rand(8, 20, 20).softmax(dim=-1)
    scores = attention_attribution_weights(attention, 19, list(range(10)))
    assert (scores >= 0).all()
    assert scores.sum().item() <= 1.0 + 1e-5


def test_attention_output_shape_is_one_score_per_patch() -> None:
    attention = torch.rand(32, 600, 600).softmax(dim=-1)
    scores = attention_attribution_weights(attention, 599, list(range(576)))
    assert scores.shape == (576,)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    ("attention", "position", "positions"),
    [
        (torch.ones(2, 3), 1, [0]),
        (torch.ones(2, 3, 4), 1, [0]),
        (torch.ones(2, 3, 3), 3, [0]),
        (torch.ones(2, 3, 3), 1, [3]),
    ],
)
def test_attention_rejects_invalid_shapes_and_positions(
    attention: torch.Tensor,
    position: int,
    positions: list[int],
) -> None:
    with pytest.raises(ValueError):
        attention_attribution_weights(attention, position, positions)


def test_attention_rejects_nonfinite_weights() -> None:
    attention = torch.ones(2, 3, 3)
    attention[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        attention_attribution_weights(attention, 1, [0, 1])


def test_contrastive_hidden_identical_states_are_zero() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hidden = torch.tensor([0.5, 0.5])
    scores = contrastive_hidden_attribution_weights(features, hidden, hidden)
    assert torch.equal(scores, torch.zeros(2))


def test_contrastive_hidden_aligned_visual_residual_is_highest() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visual_hidden = torch.tensor([1.0, 1.0])
    blank_hidden = torch.tensor([1.0, 0.0])
    scores = contrastive_hidden_attribution_weights(
        features,
        visual_hidden,
        blank_hidden,
    )
    assert scores.argmax().item() == 1
    assert torch.isfinite(scores).all()


def test_contrastive_hidden_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="visual_hidden"):
        contrastive_hidden_attribution_weights(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3))
    with pytest.raises(ValueError, match="blank_hidden"):
        contrastive_hidden_attribution_weights(torch.ones(2, 3), torch.ones(3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="equal shapes"):
        contrastive_hidden_attribution_weights(torch.ones(2, 3), torch.ones(3), torch.ones(4))


def test_zero_image_embeddings_only_changes_image_tokens() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    blank = zero_image_embeddings(inputs, [[1, 2], [0, 3]])
    assert torch.equal(blank[0, 1:3], torch.zeros(2, 3))
    assert torch.equal(blank[1, [0, 3]], torch.zeros(2, 3))
    assert torch.equal(blank[0, [0, 3]], inputs[0, [0, 3]])
    assert torch.equal(blank[1, 1:3], inputs[1, 1:3])
    assert torch.equal(inputs, torch.arange(24, dtype=torch.float32).reshape(2, 4, 3))


@pytest.mark.parametrize(
    ("inputs", "positions"),
    [
        (torch.ones(4, 3), [[1]]),
        (torch.ones(2, 4, 3), [[1]]),
        (torch.ones(2, 4, 3), [[4], [0]]),
        (torch.ones(2, 4, 3), [torch.tensor([[0]])] * 2),
    ],
)
def test_zero_image_embeddings_rejects_bad_inputs(
    inputs: torch.Tensor,
    positions: list[Any],
) -> None:
    with pytest.raises(ValueError):
        zero_image_embeddings(inputs, positions)


def test_eager_last_layer_attention_restores_backend() -> None:
    class Attention(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weights = torch.ones(1, 2, 2, 2)
            return hidden, weights

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            output, _ = self.self_attn(hidden)
            return output

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
            self.backends: list[str] = []

        def set_attn_implementation(self, implementation: str) -> None:
            self.backends.append(implementation)
            self.config._attn_implementation = implementation

    model = LanguageModel()
    with eager_last_layer_attention(model) as captured:
        assert model.config._attn_implementation == "eager"
        hidden = torch.ones(1, 2, 2)
        model.model.layers[-1](hidden)
        assert len(captured) == 1
        assert captured[0][1].shape == (1, 2, 2, 2)
    assert model.config._attn_implementation == "flash_attention_2"
    assert model.backends == ["eager", "flash_attention_2"]
    assert last_language_layer_index(model) == 0


def test_eager_last_layer_attention_restores_backend_after_error() -> None:
    class Attention(torch.nn.Module):
        pass

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")

        def set_attn_implementation(self, implementation: str) -> None:
            self.config._attn_implementation = implementation

    model = LanguageModel()
    with pytest.raises(RuntimeError, match="boom"), eager_last_layer_attention(model):
        assert model.config._attn_implementation == "eager"
        raise RuntimeError("boom")
    assert model.config._attn_implementation == "flash_attention_2"


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
