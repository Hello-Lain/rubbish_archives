"""Inherent Bias estimation and subtraction as an independent plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from methods.shield_method_base import ShieldBackendMethod


@dataclass(frozen=True)
class InherentBiasConfig:
    """Configuration for SHIELD's noise-derived visual prior subtraction."""

    bias_weight: float = 0.01
    sample_num: int = 32
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.bias_weight < 0:
            raise ValueError("bias_weight must be non-negative")
        if self.sample_num < 1:
            raise ValueError("sample_num must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative when provided")


def select_vision_features(model: Any, outputs: Any) -> torch.Tensor:
    """Select the configured LLaVA vision layer and remove its CLS token."""

    layer = model.config.vision_feature_layer
    if not isinstance(layer, int):
        raise ValueError("SHIELD expects an integer vision_feature_layer")
    features = outputs.hidden_states[layer]
    if model.config.vision_feature_select_strategy == "default":
        features = features[:, 1:]
    return features


class ShieldInherentBias:
    """Estimate a noise-induced visual prior and subtract it from features."""

    def __init__(self, config: InherentBiasConfig | None = None) -> None:
        self.config = config or InherentBiasConfig()

    @torch.inference_mode()
    def estimate(
        self,
        model: Any,
        reference_pixels: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if reference_pixels.ndim != 3:
            raise ValueError("reference_pixels must have shape [channels, height, width]")
        generator: torch.Generator | None = None
        if self.config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.seed)
        noise_dtype = torch.float16 if device.type == "cuda" else dtype
        random_pixels = torch.rand(
            (self.config.sample_num, *reference_pixels.shape),
            device=device,
            dtype=noise_dtype,
            generator=generator,
        )
        outputs = model.vision_tower(
            random_pixels,
            output_hidden_states=True,
        )
        return (
            select_vision_features(model, outputs)
            .mean(dim=0)
            .detach()
            .to(device=device, dtype=dtype)
        )

    def subtract(
        self,
        image_features: torch.Tensor,
        bias_features: torch.Tensor,
    ) -> torch.Tensor:
        if image_features.shape != bias_features.shape:
            raise ValueError(
                "bias_features must have the same shape as image_features"
            )
        return image_features - self.config.bias_weight * bias_features


class ShieldInherentBiasMethod(ShieldBackendMethod):
    """Hydra/BaseMethod adapter for the Inherent Bias plugin."""

    def __init__(
        self,
        bias_weight: float = 0.01,
        sample_num: int = 32,
        seed: int | None = 42,
    ) -> None:
        config = InherentBiasConfig(
            bias_weight=bias_weight,
            sample_num=sample_num,
            seed=seed,
        )
        self.plugin = ShieldInherentBias(config)
        super().__init__(
            "shield_inherent_bias",
            {
                "bias_weight": config.bias_weight,
                "sample_num": config.sample_num,
                "seed": config.seed,
            },
        )
