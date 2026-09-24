"""Causal Attribution Probe (CAP) visual feature intervention."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from methods.shield_method_base import ShieldBackendMethod
from methods.vrsc import energy_preserving_probe


def causal_attribution_weights(
    projected_features: torch.Tensor,
    generation_hidden: torch.Tensor,
) -> torch.Tensor:
    """Score each projected image patch against the generation-position state.

    The score is a cosine attribution proxy: patches aligned with the hidden
    state used to predict the next token receive positive scores, while
    anti-aligned patches receive negative scores. Zero-norm inputs are treated
    as having zero attribution rather than producing NaNs.
    """

    if projected_features.ndim != 2:
        raise ValueError("projected_features must have shape [patches, hidden_dim]")
    if generation_hidden.ndim != 1:
        raise ValueError("generation_hidden must have shape [hidden_dim]")
    if projected_features.shape[-1] != generation_hidden.shape[0]:
        raise ValueError("projected features and generation hidden size must match")

    features = projected_features.float()
    hidden = generation_hidden.float().to(device=features.device)
    if not bool(torch.isfinite(features).all() and torch.isfinite(hidden).all()):
        raise ValueError("projected features and generation hidden must be finite")
    if projected_features.shape[0] == 0:
        return features.new_empty((0,))

    feature_norms = features.norm(dim=-1)
    hidden_norm = hidden.norm()
    denominator = (feature_norms * hidden_norm).clamp_min(torch.finfo(features.dtype).tiny)
    scores = (features @ hidden) / denominator
    return scores.clamp(-1.0, 1.0)


@dataclass(frozen=True)
class CAPConfig:
    """Configuration for the CAP visual feature probe."""

    probe_gain: float = 0.20

    def __post_init__(self) -> None:
        if not math.isfinite(self.probe_gain):
            raise ValueError("probe_gain must be finite")
        if not 0 <= self.probe_gain < 1:
            raise ValueError("probe_gain must be in [0, 1)")


class CAP:
    """Hold the validated CAP configuration for adapter and runtime metadata."""

    def __init__(self, config: CAPConfig | None = None) -> None:
        self.config = config or CAPConfig()


class CAPMethod(ShieldBackendMethod):
    """Backend adapter exposing CAP through the canonical method contract."""

    def __init__(self, probe_gain: float = 0.20) -> None:
        config = CAPConfig(probe_gain=probe_gain)
        self.plugin = CAP(config)
        super().__init__("cap", asdict(self.plugin.config))


__all__ = [
    "CAP",
    "CAPConfig",
    "CAPMethod",
    "causal_attribution_weights",
    "energy_preserving_probe",
]
