"""Experimental visual-response calibration with a weighted simplex projection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from methods.shield_method_base import ShieldBackendMethod


@dataclass(frozen=True)
class VRSCConfig:
    eta: float = 0.25
    evidence_weight: float = 2.0
    trust: float = 1.0
    response_clip: float = 1.0
    response_mode: str = "direct"

    def __post_init__(self) -> None:
        numeric_values = (
            self.eta,
            self.evidence_weight,
            self.trust,
            self.response_clip,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("VRSC parameters must be finite")
        if not 0 <= self.eta < 1:
            raise ValueError("eta must be in [0, 1)")
        if self.evidence_weight < 0 or self.trust <= 0 or self.response_clip <= 0:
            raise ValueError("weight must be nonnegative; trust and clip must be positive")
        if self.response_mode not in {"direct", "residual", "orthogonal_residual"}:
            raise ValueError(
                "response_mode must be direct, residual, or orthogonal_residual"
            )


def _finite_float(value: torch.Tensor, name: str) -> torch.Tensor:
    result = value.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite in float32")
    return result


def patch_caption_scores(
    image_features: torch.Tensor,
    caption_features: torch.Tensor,
) -> torch.Tensor:
    """Reuse S's score proposal, not its enhancement or caption injection."""
    if image_features.ndim != 2 or caption_features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if image_features.shape[-1] == 0 or caption_features.shape[-1] != 768:
        raise ValueError("expected nonempty image channels and 768 caption channels")
    image = _finite_float(image_features, "image features")
    caption = _finite_float(caption_features, "caption features").to(image.device)
    if not image.shape[0] or not caption.shape[0]:
        return image.new_zeros(image.shape[0])
    pooled = F.adaptive_max_pool1d(image.unsqueeze(0), 768).squeeze(0)
    similarity = F.normalize(pooled, dim=-1) @ F.normalize(caption, dim=-1).T
    selected = similarity.max(dim=0).values > 0.002
    if not bool(selected.any()):
        return image.new_zeros(image.shape[0])
    return similarity[:, selected].max(dim=1).values


def energy_preserving_probe(
    image_features: torch.Tensor,
    scores: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Redistribute patch energy along a tangent direction, retaining total norm."""
    if not math.isfinite(eta) or not 0 <= eta < 1:
        raise ValueError("eta must be finite and in [0, 1)")
    if image_features.ndim != 2 or scores.shape != image_features.shape[:1]:
        raise ValueError("expected features [patches, dim] and scores [patches]")
    image = _finite_float(image_features, "image features")
    scores = _finite_float(scores, "scores").to(image.device)
    if eta == 0 or image.numel() == 0:
        return image_features
    energy = image.square().sum(dim=-1)
    total = energy.sum()
    if float(total) == 0.0 or bool(scores.max() == scores.min()):
        return image_features
    direction = scores - (scores * (energy / total)).sum()
    direction = direction / direction.abs().max().clamp_min(1e-12)
    probe = image * (1.0 + eta * direction[:, None])
    probe *= torch.sqrt(total / probe.square().sum())
    return probe.to(image_features.dtype)


def chi_square_projection(
    probabilities: torch.Tensor,
    utilities: torch.Tensor,
    trust: float = 1.0,
) -> torch.Tensor:
    """Maximize <q,u> - trust/2 * sum((q-p)^2/p) over the simplex.

    The finite-precision zero support of p remains zero. Sorting utility gives
    an exact active-set solution; the dual threshold is shared by all tokens.
    """
    if probabilities.shape != utilities.shape or probabilities.ndim < 1:
        raise ValueError("probabilities and utilities must have the same non-scalar shape")
    if probabilities.shape[-1] == 0 or not math.isfinite(trust) or trust <= 0:
        raise ValueError("nonempty vocabulary and finite positive trust required")
    p = _finite_float(probabilities, "probabilities")
    u = _finite_float(utilities, "utilities")
    if p.device != u.device:
        raise ValueError("probabilities and utilities must share a device")
    if bool((p < 0).any()) or not torch.allclose(
        p.sum(dim=-1), torch.ones_like(p[..., 0]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("probabilities must be nonnegative and sum to one")
    p = p / p.sum(dim=-1, keepdim=True)
    # Center before cumulative sums; positive-support maximum guarantees a root.
    u = u - u.masked_fill(p == 0, -torch.inf).max(dim=-1, keepdim=True).values
    a = 1.0 + u / trust
    sorted_a, order = a.sort(dim=-1, descending=True)
    sorted_p = p.gather(-1, order)
    cumulative_p = sorted_p.cumsum(dim=-1)
    thresholds = ((sorted_p * sorted_a).cumsum(dim=-1) - 1.0) / cumulative_p.clamp_min(
        torch.finfo(p.dtype).tiny
    )
    indices = torch.arange(p.shape[-1], device=p.device)
    active = (sorted_a > thresholds) & (sorted_p > 0)
    last = torch.where(active, indices, -1).max(dim=-1, keepdim=True).values
    threshold = thresholds.gather(-1, last.clamp_min(0))
    q = p * (a - threshold).clamp_min(0)
    return q / q.sum(dim=-1, keepdim=True)


class VRSC:
    def __init__(self, config: VRSCConfig | None = None) -> None:
        self.config = config or VRSCConfig()

    def distribution(
        self,
        clean_logits: torch.Tensor,
        probe_logits: torch.Tensor,
        reference_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clean_logits.shape != probe_logits.shape:
            raise ValueError("branch logits must have the same shape")
        if reference_logits is not None and reference_logits.shape != clean_logits.shape:
            raise ValueError("reference branch logits must have the same shape")
        base = _finite_float(clean_logits, "clean logits").log_softmax(dim=-1)
        probe = _finite_float(probe_logits, "probe logits").log_softmax(dim=-1)
        response = probe - base
        if self.config.response_mode != "direct":
            if reference_logits is None:
                raise ValueError("a reference branch is required for residual responses")
            reference = _finite_float(reference_logits, "reference logits").log_softmax(dim=-1)
            nuisance = reference - base
            if self.config.response_mode == "residual":
                response = response - nuisance
            else:
                coefficient = (response * nuisance).sum(dim=-1, keepdim=True)
                coefficient = coefficient / nuisance.square().sum(dim=-1, keepdim=True).clamp_min(
                    1e-12
                )
                response = response - coefficient * nuisance
        response = response.clamp(-self.config.response_clip, self.config.response_clip)
        utility = base - base.max(dim=-1, keepdim=True).values
        utility = utility + self.config.evidence_weight * response
        return chi_square_projection(base.exp(), utility, self.config.trust)

    def combine_logits(
        self,
        clean_logits: torch.Tensor,
        probe_logits: torch.Tensor,
        reference_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution(clean_logits, probe_logits, reference_logits).log()


class VRSCMethod(ShieldBackendMethod):
    """Adapter for backends implementing VRSC; the research CLI is separate."""

    def __init__(
        self,
        eta: float = 0.25,
        evidence_weight: float = 2.0,
        trust: float = 1.0,
        response_clip: float = 1.0,
        response_mode: str = "direct",
    ) -> None:
        self.plugin = VRSC(
            VRSCConfig(
                eta=eta,
                evidence_weight=evidence_weight,
                trust=trust,
                response_clip=response_clip,
                response_mode=response_mode,
            )
        )
        super().__init__("vrsc", asdict(self.plugin.config))
