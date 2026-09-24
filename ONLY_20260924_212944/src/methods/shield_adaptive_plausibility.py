"""Adaptive plausibility constraint as an independent SHIELD plugin."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from methods.shield_method_base import ShieldBackendMethod


@dataclass(frozen=True)
class AdaptivePlausibilityConfig:
    """Configuration for SHIELD's candidate plausibility cutoff."""

    beta: float = 0.35

    def __post_init__(self) -> None:
        if not 0 < self.beta <= 1:
            raise ValueError("beta must be in (0, 1]")


class ShieldAdaptivePlausibility:
    """Filter contrastive candidates below a clean-logit probability floor."""

    def __init__(self, config: AdaptivePlausibilityConfig | None = None) -> None:
        self.config = config or AdaptivePlausibilityConfig()

    def apply(
        self,
        clean_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
    ) -> torch.Tensor:
        if clean_logits.shape != candidate_logits.shape:
            raise ValueError("clean and candidate logits must have the same shape")
        clean = clean_logits.float()
        candidates = candidate_logits.float()
        cutoff = torch.log(
            torch.tensor(self.config.beta, device=clean.device, dtype=clean.dtype)
        )
        cutoff = cutoff + clean.max(dim=-1, keepdim=True).values
        return candidates.masked_fill(clean < cutoff, -float("inf"))


class ShieldAdaptivePlausibilityMethod(ShieldBackendMethod):
    """Hydra/BaseMethod adapter for the adaptive plausibility plugin."""

    def __init__(self, beta: float = 0.35) -> None:
        config = AdaptivePlausibilityConfig(beta=beta)
        self.plugin = ShieldAdaptivePlausibility(config)
        super().__init__(
            "shield_adaptive_plausibility",
            {"beta": config.beta},
        )
