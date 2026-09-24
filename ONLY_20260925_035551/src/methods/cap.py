"""Causal Attribution Probe (CAP) visual feature intervention."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

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


def attention_attribution_weights(
    attention_weights: torch.Tensor,
    generation_position: int,
    image_positions: list[int] | torch.Tensor,
) -> torch.Tensor:
    """Return the generation token's mean-head attention over image patches."""

    if attention_weights.ndim != 3:
        raise ValueError("attention_weights must have shape [heads, seq, seq]")
    if attention_weights.shape[-1] != attention_weights.shape[-2]:
        raise ValueError("attention_weights must be square in its sequence axes")
    if not 0 <= generation_position < attention_weights.shape[-2]:
        raise ValueError("generation_position is outside the sequence")
    positions = torch.as_tensor(
        image_positions,
        dtype=torch.long,
        device=attention_weights.device,
    )
    if positions.ndim != 1:
        raise ValueError("image_positions must be one-dimensional")
    if positions.numel() == 0:
        return attention_weights.new_empty((0,), dtype=torch.float32)
    if bool(((positions < 0) | (positions >= attention_weights.shape[-1])).any()):
        raise ValueError("image_positions contain an out-of-range index")

    attention = attention_weights.float()
    if not bool(torch.isfinite(attention).all()):
        raise ValueError("attention_weights must be finite")
    scores = attention.mean(dim=0)[generation_position, positions]
    return scores.clamp_min(0.0)


def contrastive_hidden_attribution_weights(
    projected_features: torch.Tensor,
    visual_hidden: torch.Tensor,
    blank_hidden: torch.Tensor,
) -> torch.Tensor:
    """Score patches by visual-minus-language-prior hidden alignment."""

    if visual_hidden.ndim != 1:
        raise ValueError("visual_hidden must have shape [hidden_dim]")
    if blank_hidden.ndim != 1:
        raise ValueError("blank_hidden must have shape [hidden_dim]")
    if visual_hidden.shape != blank_hidden.shape:
        raise ValueError("visual_hidden and blank_hidden must have equal shapes")
    visual_scores = causal_attribution_weights(projected_features, visual_hidden)
    blank_scores = causal_attribution_weights(projected_features, blank_hidden)
    return visual_scores - blank_scores


def zero_image_embeddings(
    inputs_embeds: torch.Tensor,
    image_positions: list[list[int] | torch.Tensor],
) -> torch.Tensor:
    """Clone prompt embeddings and zero only the image-token positions."""

    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [batch, seq, hidden_dim]")
    if len(image_positions) != inputs_embeds.shape[0]:
        raise ValueError("image_positions must contain one entry per batch row")

    blank = inputs_embeds.clone()
    sequence_length = inputs_embeds.shape[1]
    for row_index, positions in enumerate(image_positions):
        row_positions = torch.as_tensor(
            positions,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        if row_positions.ndim != 1:
            raise ValueError("each image_positions entry must be one-dimensional")
        if row_positions.numel() and bool(
            ((row_positions < 0) | (row_positions >= sequence_length)).any()
        ):
            raise ValueError("image_positions contain an out-of-range index")
        blank[row_index, row_positions] = 0
    return blank


@contextmanager
def eager_last_layer_attention(
    model: Any,
) -> Iterator[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Capture last-layer attention with one temporary eager attention island.

    ``output_attentions=True`` is unsupported by FlashAttention in the mounted
    Transformers release. This context switches the language model to eager
    attention for a single attribution forward and restores the prior backend
    even when the forward raises.
    """

    language_model = getattr(model, "language_model", model)
    base_model = getattr(language_model, "model", language_model)
    layers = getattr(base_model, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise ValueError("model language stack must expose a nonempty ModuleList")
    last_attention = getattr(layers[-1], "self_attn", None)
    if not isinstance(last_attention, nn.Module):
        raise ValueError("last language layer must expose self_attn")

    replacement = language_model if hasattr(language_model, "set_attn_implementation") else model
    if not hasattr(replacement, "set_attn_implementation"):
        raise ValueError("model must support set_attn_implementation")
    previous = getattr(replacement.config, "_attn_implementation", None)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            captured.append((output[0], output[1]))

    handle = last_attention.register_forward_hook(capture)
    replacement.set_attn_implementation("eager")
    try:
        yield captured
    finally:
        handle.remove()
        replacement.set_attn_implementation(previous)


def last_language_layer_index(model: Any) -> int:
    """Return the decoder stack's last layer index across wrapper layouts."""

    language_model = getattr(model, "language_model", model)
    base_model = getattr(language_model, "model", language_model)
    layers = getattr(base_model, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise ValueError("model language stack must expose a nonempty ModuleList")
    return len(layers) - 1


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


@dataclass(frozen=True)
class CAPRefinedConfig:
    """Controls the reliability-gated CAP-R decoding correction."""

    probe_gain: float = 0.20
    max_weight: float = 0.75
    jsd_threshold: float = 0.005
    jsd_temperature: float = 0.01
    entropy_temperature: float = 0.01
    plausibility_beta: float = 0.10
    plausibility_temperature: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.probe_gain,
            self.max_weight,
            self.jsd_threshold,
            self.jsd_temperature,
            self.entropy_temperature,
            self.plausibility_beta,
            self.plausibility_temperature,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("CAP-R parameters must be finite")
        if not 0 <= self.probe_gain < 1:
            raise ValueError("probe_gain must be in [0, 1)")
        if not 0 <= self.max_weight <= 1:
            raise ValueError("max_weight must be in [0, 1]")
        if self.jsd_threshold < 0:
            raise ValueError("jsd_threshold must be nonnegative")
        if self.jsd_temperature <= 0 or self.entropy_temperature <= 0:
            raise ValueError("CAP-R temperatures must be positive")
        if not 0 < self.plausibility_beta <= 1:
            raise ValueError("plausibility_beta must be in (0, 1]")
        if self.plausibility_temperature <= 0:
            raise ValueError("plausibility_temperature must be positive")


def adaptive_cap_logits(
    base_logits: torch.Tensor,
    probe_logits: torch.Tensor,
    config: CAPRefinedConfig | None = None,
) -> torch.Tensor:
    """Apply CAP-R's soft, reliability-gated visual correction.

    The probe is used only when it both disagrees with the base branch and
    lowers predictive entropy. A continuous plausibility gate keeps all finite
    vocabulary entries alive, so this correction does not turn sampling into a
    hard greedy/APC decoder.
    """

    if base_logits.shape != probe_logits.shape or base_logits.ndim < 1:
        raise ValueError("base and probe logits must have the same non-scalar shape")
    config = config or CAPRefinedConfig()
    base = base_logits.float()
    probe = probe_logits.float().to(device=base.device)
    if not bool(torch.isfinite(base).all() and torch.isfinite(probe).all()):
        raise ValueError("base and probe logits must be finite")
    if base.shape[-1] < 2:
        raise ValueError("logits must have at least two vocabulary entries")

    log_base = F.log_softmax(base, dim=-1)
    log_probe = F.log_softmax(probe, dim=-1)
    base_prob = log_base.exp()
    probe_prob = log_probe.exp()
    midpoint = 0.5 * (base_prob + probe_prob)
    log_midpoint = midpoint.clamp_min(torch.finfo(base.dtype).tiny).log()
    jsd = 0.5 * (
        (base_prob * (log_base - log_midpoint)).sum(dim=-1)
        + (probe_prob * (log_probe - log_midpoint)).sum(dim=-1)
    )
    base_entropy = -(base_prob * log_base).sum(dim=-1)
    probe_entropy = -(probe_prob * log_probe).sum(dim=-1)
    entropy_gain = (base_entropy - probe_entropy) / math.log(base.shape[-1])
    disagreement_gate = torch.sigmoid(
        (jsd - config.jsd_threshold) / config.jsd_temperature
    )
    confidence_gate = torch.sigmoid(entropy_gain / config.entropy_temperature)
    weight = config.max_weight * disagreement_gate * confidence_gate

    relative_base = log_base - log_base.max(dim=-1, keepdim=True).values
    plausibility_boundary = math.log(config.plausibility_beta)
    plausibility_gate = torch.sigmoid(
        (relative_base - plausibility_boundary) / config.plausibility_temperature
    )
    mixed = log_base + weight.unsqueeze(-1) * plausibility_gate * (
        log_probe - log_base
    )
    return mixed - mixed.logsumexp(dim=-1, keepdim=True)


class CAPMethod(ShieldBackendMethod):
    """Backend adapter exposing CAP through the canonical method contract."""

    def __init__(self, probe_gain: float = 0.20) -> None:
        config = CAPConfig(probe_gain=probe_gain)
        self.plugin = CAP(config)
        super().__init__("cap", asdict(self.plugin.config))


class CAPRefinedMethod(ShieldBackendMethod):
    """Backend adapter exposing the reliability-gated CAP-R variant."""

    def __init__(
        self,
        probe_gain: float = 0.20,
        max_weight: float = 0.75,
        jsd_threshold: float = 0.005,
        jsd_temperature: float = 0.01,
        entropy_temperature: float = 0.01,
        plausibility_beta: float = 0.10,
        plausibility_temperature: float = 0.05,
    ) -> None:
        cap_config = CAPConfig(probe_gain=probe_gain)
        refined_config = CAPRefinedConfig(
            probe_gain=probe_gain,
            max_weight=max_weight,
            jsd_threshold=jsd_threshold,
            jsd_temperature=jsd_temperature,
            entropy_temperature=entropy_temperature,
            plausibility_beta=plausibility_beta,
            plausibility_temperature=plausibility_temperature,
        )
        self.plugin = (cap_config, refined_config)
        super().__init__(
            "cap_refined",
            asdict(refined_config),
        )


__all__ = [
    "CAP",
    "CAPConfig",
    "CAPMethod",
    "CAPRefinedConfig",
    "CAPRefinedMethod",
    "adaptive_cap_logits",
    "attention_attribution_weights",
    "causal_attribution_weights",
    "contrastive_hidden_attribution_weights",
    "eager_last_layer_attention",
    "energy_preserving_probe",
    "last_language_layer_index",
    "zero_image_embeddings",
]
