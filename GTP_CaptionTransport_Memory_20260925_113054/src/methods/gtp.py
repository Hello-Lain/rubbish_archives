from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper


@dataclass(frozen=True)
class GroundingMap:
    """Sparse patch-claim mass and patch amplification derived from it."""

    joint_mass: torch.Tensor
    patch_mass: torch.Tensor
    claim_mass: torch.Tensor
    patch_gain: torch.Tensor
    valid: bool
    diagnostics: dict[str, float]
    fallback_reason: str | None = None


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits onto the probability simplex with an exact active set."""

    if logits.ndim == 0:
        raise ValueError("sparsemax requires at least one tensor dimension")
    shifted = logits.float() - logits.float().amax(dim=dim, keepdim=True)
    sorted_logits = shifted.sort(dim=dim, descending=True).values
    cumulative = sorted_logits.cumsum(dim=dim)
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = logits.shape[dim]
    ranks = torch.arange(
        1,
        logits.shape[dim] + 1,
        dtype=shifted.dtype,
        device=logits.device,
    ).view(ranks_shape)
    support = 1.0 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(dim, support_size.to(torch.long) - 1) - 1.0) / support_size.to(
        shifted.dtype
    )
    return (shifted - threshold).clamp_min(0.0)


def _normalized_entropy(mass: torch.Tensor) -> float:
    if mass.numel() <= 1:
        return 0.0
    normalized = mass.float().clamp_min(0.0)
    total = normalized.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return 0.0
    normalized = normalized / total
    positive = normalized > 0
    entropy = -(normalized[positive] * normalized[positive].log()).sum()
    return float((entropy / math.log(mass.numel())).item())


def build_sparse_grounding_map(
    patch_features: torch.Tensor,
    claim_features: torch.Tensor,
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> GroundingMap:
    """Build the globally standardized sparsemax patch-claim grounding map."""

    if patch_features.ndim != 2 or claim_features.ndim != 2:
        raise ValueError("patch_features and claim_features must both be rank-2")
    if patch_features.shape[1] != claim_features.shape[1]:
        raise ValueError("patch and claim feature dimensions must match")
    if patch_features.shape[0] < 1 or claim_features.shape[0] < 1:
        raise ValueError("grounding requires at least one patch and one claim")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    patches = patch_features.float()
    claims = claim_features.float()
    if not bool(torch.isfinite(patches).all() and torch.isfinite(claims).all()):
        raise ValueError("grounding features must be finite")

    similarities = F.normalize(patches, dim=-1) @ F.normalize(claims, dim=-1).T
    mean = similarities.mean()
    std = similarities.std(unbiased=False)
    patch_count, claim_count = similarities.shape
    empty_joint = torch.zeros_like(similarities)
    empty_patch = torch.zeros(
        patch_count,
        dtype=similarities.dtype,
        device=similarities.device,
    )
    empty_claim = torch.zeros(
        claim_count,
        dtype=similarities.dtype,
        device=similarities.device,
    )
    empty_gain = torch.zeros_like(empty_patch)

    if not bool(torch.isfinite(std)) or float(std) <= epsilon:
        return GroundingMap(
            joint_mass=empty_joint,
            patch_mass=empty_patch,
            claim_mass=empty_claim,
            patch_gain=empty_gain,
            valid=False,
            diagnostics={
                "support_ratio": 0.0,
                "patch_entropy": 0.0,
                "claim_entropy": 0.0,
                "patch_gain_mean": 0.0,
                "patch_gain_std": 0.0,
                "saturated_patch_ratio": 0.0,
                "similarity_std": float(std.item()) if torch.isfinite(std) else 0.0,
            },
            fallback_reason="similarity_variance_too_small",
        )

    standardized = (similarities - mean) / std
    rho = 1.0 / (patch_count * claim_count)
    joint = sparsemax(
        (rho * (1.0 + standardized / temperature)).flatten(),
        dim=0,
    ).reshape(patch_count, claim_count)
    patch_mass = joint.sum(dim=1)
    claim_mass = joint.sum(dim=0)
    relative_mass = patch_count * patch_mass
    patch_gain = (relative_mass - 1.0).clamp_min(0.0) / (relative_mass + 1.0)
    diagnostics = {
        "support_ratio": float((joint > 0).float().mean().item()),
        "patch_entropy": _normalized_entropy(patch_mass),
        "claim_entropy": _normalized_entropy(claim_mass),
        "patch_gain_mean": float(patch_gain.mean().item()),
        "patch_gain_std": float(patch_gain.std(unbiased=False).item()),
        "saturated_patch_ratio": float((relative_mass >= 0.5 * patch_count).float().mean().item()),
        "similarity_std": float(std.item()),
    }
    return GroundingMap(
        joint_mass=joint,
        patch_mass=patch_mass,
        claim_mass=claim_mass,
        patch_gain=patch_gain,
        valid=True,
        diagnostics=diagnostics,
    )


def amplify_patch_features(
    patch_features: torch.Tensor,
    patch_gain: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    if patch_features.ndim != 2 or patch_gain.ndim != 1:
        raise ValueError("expected [patch, dim] features and [patch] gains")
    if patch_features.shape[0] != patch_gain.shape[0]:
        raise ValueError("patch gain count must match patch feature count")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    scale = 1.0 + strength * patch_gain.to(
        device=patch_features.device,
        dtype=patch_features.dtype,
    )
    return patch_features * scale.unsqueeze(-1)


def log_harmonic_gate(
    log_p: torch.Tensor,
    log_g: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Compute log weighted-harmonic trust without exponentiating probabilities."""

    if log_p.shape != log_g.shape:
        raise ValueError("clean and grounded log probabilities must have equal shape")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    clean = log_p.float()
    grounded = log_g.float()
    if weight == 0.0:
        return clean
    if weight == 1.0:
        return grounded

    log_clean_weight = math.log1p(-weight)
    log_grounded_weight = math.log(weight)
    reciprocal_log_terms = torch.stack(
        (
            log_clean_weight - clean,
            log_grounded_weight - grounded,
        ),
        dim=0,
    )
    return -torch.logsumexp(reciprocal_log_terms, dim=0)


def dual_reference_projection(
    log_p: torch.Tensor,
    log_g: torch.Tensor,
    *,
    utility_weight: float,
    gate_weight: float,
    temperature: float,
) -> torch.Tensor:
    """Solve the GTP dual-reference Pearson-chi-square projection."""

    if log_p.shape != log_g.shape or log_p.ndim < 1:
        raise ValueError("log_p and log_g must have equal non-scalar shapes")
    if not 0.0 <= utility_weight <= 1.0:
        raise ValueError("utility_weight must be in [0, 1]")
    if not 0.0 <= gate_weight <= 1.0:
        raise ValueError("gate_weight must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if bool(torch.isnan(log_p).any() or torch.isnan(log_g).any()):
        raise ValueError("log probabilities cannot contain NaN")

    clean = log_p.float()
    grounded = log_g.float()
    utility = (1.0 - utility_weight) * clean + utility_weight * grounded
    log_gate = log_harmonic_gate(clean, grounded, weight=gate_weight)

    # Scale gates before exponentiation so their relative mass survives when
    # every harmonic probability is below the float64 normal range.
    utility64 = utility.double()
    log_gate64 = log_gate.double()
    sorted_utility, order = utility64.sort(dim=-1, descending=True)
    sorted_log_gate = log_gate64.gather(dim=-1, index=order)
    gate_scale = log_gate64.amax(dim=-1, keepdim=True)
    scaled_gate = (sorted_log_gate - gate_scale).exp()
    cumulative_scaled_gate = scaled_gate.cumsum(dim=-1)
    cumulative_weighted_utility = (scaled_gate * sorted_utility).cumsum(dim=-1)
    prefix_mean = cumulative_weighted_utility / cumulative_scaled_gate
    log_prefix_mass = cumulative_scaled_gate.log() + gate_scale
    log_offset = math.log(temperature) - log_prefix_mass
    max_log = math.log(torch.finfo(torch.float64).max)
    offset = log_offset.clamp_max(max_log).exp()
    offset = torch.where(
        log_offset > max_log,
        torch.full_like(offset, torch.inf),
        offset,
    )

    ranks_shape = [1] * utility.ndim
    ranks_shape[-1] = utility.shape[-1]
    ranks = torch.arange(
        1,
        utility.shape[-1] + 1,
        dtype=torch.long,
        device=utility.device,
    ).view(ranks_shape)
    active_candidate = sorted_utility - prefix_mean >= -offset
    active_count = torch.where(active_candidate, ranks, 0).amax(dim=-1, keepdim=True)
    if bool((active_count == 0).any()):
        raise FloatingPointError("could not find a non-empty GTP projection active set")

    active = ranks <= active_count
    active_scaled_gate = scaled_gate * active
    active_mass_scaled = active_scaled_gate.sum(dim=-1, keepdim=True)
    active_mean = (active_scaled_gate * sorted_utility).sum(
        dim=-1, keepdim=True
    ) / active_mass_scaled
    normalized_gate = active_scaled_gate / active_mass_scaled
    log_active_mass = active_mass_scaled.log() + gate_scale
    active_mass = log_active_mass.exp()
    projected_sorted = normalized_gate * (
        1.0 + active_mass * (sorted_utility - active_mean) / temperature
    )
    projected_sorted = projected_sorted.clamp_min(0.0)
    projected = torch.zeros_like(projected_sorted).scatter(
        dim=-1,
        index=order,
        src=projected_sorted,
    )
    normalizer = projected.sum(dim=-1, keepdim=True)
    if bool((~torch.isfinite(normalizer) | (normalizer <= 0)).any()):
        raise FloatingPointError("GTP projection produced an invalid normalizer")
    projected = projected / normalizer
    return projected.to(dtype=log_p.dtype if log_p.is_floating_point() else torch.float32)


class GTPMethod(BaseMethod):
    """GTP strategy core, with model-family generation delegated to a backend."""

    def __init__(
        self,
        a_u: float = 0.5,
        a_h: float = 0.5,
        tau: float = 1.0,
        tau_e: float = 1.0,
        kappa: float = 1.0,
        grounding_source: str = "coco_claims",
        inject_caption_memory: bool = False,
        caption_memory_slots: int = 32,
    ) -> None:
        if not 0.0 <= a_u <= 1.0:
            raise ValueError("a_u must be in [0, 1]")
        if not 0.0 <= a_h <= 1.0:
            raise ValueError("a_h must be in [0, 1]")
        if tau <= 0 or tau_e <= 0:
            raise ValueError("tau and tau_e must be positive")
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        if grounding_source not in {"coco_claims", "caption_tokens"}:
            raise ValueError("grounding_source must be 'coco_claims' or 'caption_tokens'")
        if caption_memory_slots < 1:
            raise ValueError("caption_memory_slots must be positive")
        if inject_caption_memory and grounding_source != "caption_tokens":
            raise ValueError("caption memory injection requires grounding_source='caption_tokens'")
        self.a_u = a_u
        self.a_h = a_h
        self.tau = tau
        self.tau_e = tau_e
        self.kappa = kappa
        self.grounding_source = grounding_source
        self.inject_caption_memory = inject_caption_memory
        self.caption_memory_slots = caption_memory_slots
        self.model: BaseMLLMWrapper | None = None
        self.device: torch.device | None = None

    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        if not callable(getattr(model, "generate_gtp_batch", None)):
            raise TypeError("GTP requires a model-family adapter implementing generate_gtp_batch.")
        self.model = model
        self.device = device

    def generate(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.model is None or self.device is None:
            raise RuntimeError("GTPMethod.setup(model, device) is required.")
        records = batch.get("records")
        if not isinstance(records, list):
            raise TypeError("batch['records'] must be a list")
        outputs = self.model.generate_gtp_batch(batch, method=self)  # type: ignore[attr-defined]
        if len(outputs) != len(records):
            raise RuntimeError(
                f"GTP backend returned {len(outputs)} outputs for {len(records)} records"
            )
        return outputs

    def project_logits(
        self,
        clean_logits: torch.Tensor,
        grounded_logits: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        log_p = torch.log_softmax(clean_logits.float() / temperature, dim=-1)
        log_g = torch.log_softmax(grounded_logits.float() / temperature, dim=-1)
        return dual_reference_projection(
            log_p,
            log_g,
            utility_weight=self.a_u,
            gate_weight=self.a_h,
            temperature=self.tau,
        )
