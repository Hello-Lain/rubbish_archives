from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from methods.base_method import BaseMethod
from methods.shield_statistical_bias import (
    ShieldStatisticalBias,
    ShieldStatisticalBiasMethod,
    StatisticalBiasConfig,
    maximize_weight_difference,
)
from models.base import BaseMLLMWrapper


class FakeStatisticalBiasModel(BaseMLLMWrapper):
    def __init__(self) -> None:
        super().__init__("mock://shield")
        self.calls: list[tuple[list[Mapping[str, Any]], str, Mapping[str, Any]]] = []

    def setup(self, device: torch.device) -> None:
        self.device = device

    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        return f"{prompt}|{image_path}"

    def generate_batch(
        self,
        records: list[Mapping[str, Any]],
        *,
        method_name: str,
        method_config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append((records, method_name, method_config))
        return [{"id": record["id"], "caption": "mock"} for record in records]


def test_maximize_weight_difference_is_bounded() -> None:
    weights = maximize_weight_difference(torch.tensor([2.0, 4.0, 8.0]), gamma=3.0)

    assert torch.allclose(weights, torch.tensor([0.0, 1.0 / 27.0, 1.0]))
    assert float(weights.min()) >= 0.0
    assert float(weights.max()) <= 1.0


def test_statistical_bias_enhance_preserves_patch_shape() -> None:
    module = ShieldStatisticalBias(
        StatisticalBiasConfig(threshold=2.0, gamma_gain=3.0, gain_per=0.55)
    )
    image_features = torch.randn(4, 1024)
    caption_features = torch.randn(6, 768)

    enhanced, selected = module.enhance(image_features, caption_features)

    assert enhanced.shape == image_features.shape
    assert enhanced.dtype == image_features.dtype
    assert selected.dtype == torch.long
    assert selected.numel() == 0


def test_statistical_bias_empty_selection_is_a_noop() -> None:
    module = ShieldStatisticalBias(
        StatisticalBiasConfig(threshold=2.0, gamma_gain=3.0, gain_per=0.55)
    )
    image_features = torch.randn(4, 1024)
    caption_features = torch.randn(6, 768)

    enhanced, selected = module.enhance(image_features, caption_features)

    assert selected.numel() == 0
    assert torch.equal(enhanced, image_features)


def test_statistical_bias_matches_upstream_weighting_formula() -> None:
    torch.manual_seed(7)
    module = ShieldStatisticalBias(
        StatisticalBiasConfig(threshold=-1.0, gamma_gain=2.0, gain_per=0.25)
    )
    image_features = torch.randn(5, 1024)
    caption_features = torch.randn(4, 768)

    enhanced, selected = module.enhance(image_features, caption_features)

    pooled = torch.nn.functional.adaptive_max_pool1d(
        image_features.unsqueeze(0),
        768,
    ).squeeze(0)
    image_normalized = torch.nn.functional.normalize(
        pooled,
        p=2,
        dim=-1,
    ).float()
    caption_normalized = torch.nn.functional.normalize(
        caption_features,
        p=2,
        dim=-1,
    ).float()
    cosine_similarity = image_normalized @ caption_normalized.transpose(0, 1)
    expected_selected = torch.nonzero(
        cosine_similarity.max(dim=0).values > -1.0,
        as_tuple=False,
    ).flatten()
    similarity_img = cosine_similarity[:, expected_selected].max(dim=1).values
    expected_weight = maximize_weight_difference(similarity_img, gamma=2.0)
    expected_gain = torch.where(
        expected_weight <= 0.25,
        torch.zeros_like(expected_weight),
        expected_weight,
    )
    expected = image_features + image_features * expected_gain.unsqueeze(-1)

    assert torch.equal(selected, expected_selected)
    assert torch.allclose(enhanced, expected)


def test_map_text_segments_keeps_contiguous_caption_spans() -> None:
    selected = torch.tensor([1, 2, 5], dtype=torch.long)
    input_caption_embeds = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    mapped = ShieldStatisticalBias.map_text_segments(
        selected,
        caption_token_count=6,
        input_caption_embeds=input_caption_embeds,
    )

    expected = torch.cat(
        (input_caption_embeds[1:3], input_caption_embeds[6:7]),
        dim=0,
    )
    assert torch.equal(mapped, expected)


def test_statistical_bias_method_contract_dispatches_backend() -> None:
    model = FakeStatisticalBiasModel()
    method: BaseMethod = ShieldStatisticalBiasMethod(
        threshold=0.011,
        gamma_gain=2.27,
        gain_per=0.5,
    )
    model.setup(torch.device("cpu"))
    method.setup(model, torch.device("cpu"))

    outputs = method.generate(
        {
            "records": [
                {"id": "one", "prompt": "describe", "image_path": "image.jpg"},
                {"id": "two", "prompt": "describe", "image_path": "image2.jpg"},
            ]
        }
    )

    assert [output["id"] for output in outputs] == ["one", "two"]
    assert len(model.calls) == 1
    _, method_name, method_config = model.calls[0]
    assert method_name == "shield_statistical_bias"
    assert method_config == {
        "threshold": 0.011,
        "gamma_gain": 2.27,
        "gain_per": 0.5,
    }


def test_statistical_bias_config_rejects_invalid_gain() -> None:
    with pytest.raises(ValueError, match="gain_per"):
        StatisticalBiasConfig(gain_per=1.1)
