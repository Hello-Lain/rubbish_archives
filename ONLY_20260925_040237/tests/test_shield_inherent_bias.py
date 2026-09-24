from __future__ import annotations

from types import SimpleNamespace

import torch

from methods.shield_inherent_bias import (
    InherentBiasConfig,
    ShieldInherentBias,
    ShieldInherentBiasMethod,
)


def test_inherent_bias_subtracts_only_the_configured_prior() -> None:
    module = ShieldInherentBias(InherentBiasConfig(bias_weight=0.25))
    image_features = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    bias_features = torch.ones_like(image_features)

    transformed = module.subtract(image_features, bias_features)

    assert torch.equal(transformed, image_features - 0.25)


def test_inherent_bias_estimate_uses_selected_vision_layer_and_is_reproducible() -> None:
    class FakeVisionModel:
        config = SimpleNamespace(
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )

        def vision_tower(
            self,
            pixel_values: torch.Tensor,
            *,
            output_hidden_states: bool,
        ) -> SimpleNamespace:
            assert output_hidden_states is True
            features = pixel_values.flatten(2).transpose(1, 2)
            return SimpleNamespace(hidden_states=[features])

    module = ShieldInherentBias(
        InherentBiasConfig(sample_num=3, seed=42),
    )
    reference = torch.zeros(3, 4, 4)

    first = module.estimate(
        FakeVisionModel(),
        reference,
        torch.device("cpu"),
        torch.float32,
    )
    second = module.estimate(
        FakeVisionModel(),
        reference,
        torch.device("cpu"),
        torch.float32,
    )

    assert first.shape == (15, 3)
    assert torch.equal(first, second)


def test_inherent_bias_method_exposes_plugin_configuration() -> None:
    method = ShieldInherentBiasMethod(bias_weight=0.2, sample_num=8, seed=7)

    assert method.plugin.config == InherentBiasConfig(
        bias_weight=0.2,
        sample_num=8,
        seed=7,
    )
