from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.shield_cumulative_runtime import build_noise_bias_feature


class FakeVisionModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        self.seen_dtype: torch.dtype | None = None

    def vision_tower(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True
        self.seen_dtype = pixel_values.dtype
        features = pixel_values.flatten(2).transpose(1, 2)
        return SimpleNamespace(hidden_states=[features])


def test_noise_bias_uses_private_seeded_generator() -> None:
    model = FakeVisionModel()
    reference_pixels = torch.zeros(3, 4, 4)
    torch.manual_seed(123)
    expected_next = torch.rand(4)

    torch.manual_seed(123)
    bias = build_noise_bias_feature(
        model,
        reference_pixels,
        sample_num=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=42,
    )
    actual_next = torch.rand(4)

    assert torch.equal(actual_next, expected_next)
    assert model.seen_dtype == torch.float32
    assert bias.shape == (15, 3)


def test_noise_bias_is_reproducible_for_same_seed() -> None:
    reference_pixels = torch.zeros(3, 4, 4)
    first = build_noise_bias_feature(
        FakeVisionModel(),
        reference_pixels,
        sample_num=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=42,
    )
    second = build_noise_bias_feature(
        FakeVisionModel(),
        reference_pixels,
        sample_num=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=42,
    )

    assert torch.equal(first, second)
