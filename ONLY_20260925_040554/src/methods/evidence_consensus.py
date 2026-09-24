"""Sparse evidence transport and a two-reference chi-square decision."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from methods.shield_method_base import ShieldBackendMethod
from methods.vrsc import chi_square_projection


@dataclass(frozen=True)
class EvidenceConsensusConfig:
    evidence_weight: float = 0.75
    trust: float = 1.0
    transport_trust: float = 1.0
    feature_gain: float = 1.0

    def __post_init__(self) -> None:
        values = (self.evidence_weight, self.trust, self.transport_trust, self.feature_gain)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("parameters must be finite")
        if not 0 <= self.evidence_weight <= 1:
            raise ValueError("evidence_weight must be in [0, 1]")
        if min(self.trust, self.transport_trust) <= 0 or self.feature_gain < 0:
            raise ValueError("trust must be positive; feature_gain must be nonnegative")


def evidence_transport(
    similarities: torch.Tensor, trust: float = 1.0
) -> torch.Tensor:
    """A single sparse joint measure over patch-caption pairs, not two gates."""
    if similarities.ndim != 2 or not similarities.numel():
        raise ValueError("expected a nonempty [patches, caption tokens] matrix")
    values = similarities.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("similarities must be finite")
    scale = values.std(correction=0).clamp_min(1e-6)
    utility = (values - values.mean()) / scale
    prior = torch.full_like(utility, 1.0 / utility.numel())
    return chi_square_projection(prior.flatten(), utility.flatten(), trust).reshape_as(values)


def transport_features(
    image: torch.Tensor,
    caption: torch.Tensor,
    config: EvidenceConsensusConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the two marginals of one measure for visual gain and caption memory."""
    if image.ndim != 2 or caption.ndim != 2 or caption.shape[-1] != 768:
        raise ValueError("expected image [patches, channels], caption [tokens, 768]")
    if not bool(torch.isfinite(image).all() and torch.isfinite(caption).all()):
        raise ValueError("features must be finite")
    selected = torch.empty(0, dtype=torch.long, device=image.device)
    if not image.shape[0] or not caption.shape[0]:
        return image, selected, image.new_zeros((image.shape[0], caption.shape[0]))
    pooled = F.adaptive_max_pool1d(image.float().unsqueeze(0), 768).squeeze(0)
    similarities = F.normalize(pooled, dim=-1) @ F.normalize(
        caption.float().to(image.device), dim=-1
    ).T
    joint = evidence_transport(similarities, config.transport_trust)
    if float(similarities.std(correction=0)) < 1e-6:
        return image, selected, joint
    patch_density = joint.sum(-1) * image.shape[0]
    gain = (patch_density - 1).clamp_min(0) / (patch_density + 1)
    enhanced = image * (1 + config.feature_gain * gain[:, None]).to(image.dtype)
    selected = (joint.sum(0) > 0).nonzero().flatten()
    return enhanced, selected, joint


class EvidenceConsensus:
    def __init__(self, config: EvidenceConsensusConfig | None = None) -> None:
        self.config = config or EvidenceConsensusConfig()

    def distribution(
        self, clean_logits: torch.Tensor, evidence_logits: torch.Tensor
    ) -> torch.Tensor:
        """Maximize <q,u> minus the weighted chi-square cost to both branches.

        h = 1 / ((1-a)/p + a/g), H = sum(h), b = h/H.
        The objective is equivalent to projection(b, H*u, trust).
        Agreement H<=1 makes contradictory evidence more conservative.
        """
        if clean_logits.shape != evidence_logits.shape:
            raise ValueError("branch logits must have the same shape")
        if not bool(torch.isfinite(clean_logits).all() and torch.isfinite(evidence_logits).all()):
            raise ValueError("branch logits must be finite")
        log_p, log_g = clean_logits.float().log_softmax(-1), evidence_logits.float().log_softmax(-1)
        weight = self.config.evidence_weight
        if weight == 0:
            log_h = log_p
        elif weight == 1:
            log_h = log_g
        else:
            log_h = -torch.logaddexp(
                math.log1p(-weight) - log_p, math.log(weight) - log_g
            )
        log_agreement = log_h.logsumexp(-1, keepdim=True)
        prior = (log_h - log_agreement).exp()
        # Extreme disagreement can lose precision in the logsumexp subtraction.
        # Preserve the frozen-run arithmetic whenever the solver accepts it.
        if not torch.allclose(
            prior.sum(-1), torch.ones_like(prior[..., 0]), atol=1e-5, rtol=1e-5
        ):
            prior = log_h.softmax(-1)
        utility = (1 - weight) * log_p + weight * log_g
        return chi_square_projection(
            prior, log_agreement.exp() * utility, self.config.trust
        )

    def combine_logits(
        self, clean_logits: torch.Tensor, evidence_logits: torch.Tensor
    ) -> torch.Tensor:
        return self.distribution(clean_logits, evidence_logits).log()


class EvidenceConsensusMethod(ShieldBackendMethod):
    def __init__(
        self,
        evidence_weight: float = 0.75,
        trust: float = 1.0,
        transport_trust: float = 1.0,
        feature_gain: float = 1.0,
    ) -> None:
        self.plugin = EvidenceConsensus(
            EvidenceConsensusConfig(evidence_weight, trust, transport_trust, feature_gain)
        )
        super().__init__("evidence_consensus", asdict(self.plugin.config))
