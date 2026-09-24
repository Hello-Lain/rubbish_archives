"""Reliability-calibrated Grounded Token Projection (GTP)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from methods.shield_method_base import ShieldBackendMethod
from methods.vrsc import chi_square_projection


def counterfactual_patch_scores(
    scores: torch.Tensor,
    control_scores: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Fuse visual relevance with a shuffled counterfactual control.

    A patch is trusted only when it is both relevant to the caption and more
    relevant than the shuffled-control counterpart. The output is nonnegative
    and remains patch-shaped, so it can be used without changing attention or
    injecting generated text.
    """
    if scores.shape != control_scores.shape or scores.ndim != 1:
        raise ValueError("patch scores and controls must be matching vectors")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = scores.float()
    control_scores = control_scores.float().to(scores.device)
    if not bool(torch.isfinite(scores).all() and torch.isfinite(control_scores).all()):
        raise ValueError("patch scores must be finite")
    if scores.numel() == 0:
        return scores
    relevance = torch.sigmoid(
        (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(temperature)
    )
    counterfactual = torch.sigmoid((scores - control_scores) / temperature)
    consensus = relevance * counterfactual
    return consensus / consensus.mean().clamp_min(torch.finfo(consensus.dtype).tiny)


@dataclass(frozen=True)
class GTPConfig:
    """Controls for counterfactual evidence-specific token projection."""

    mode: str = "legacy"
    max_evidence_weight: float = 0.75
    minimum_evidence_fraction: float = 0.0
    trust: float = 1.0
    signal_scale: float = 0.000001
    disagreement_scale: float = 0.08
    sharpness: float = 1.0
    specificity_temperature: float = 0.5
    support_margin: float = 0.0
    correction_clip: float = 4.0
    candidate_top_k: int = 128

    def __post_init__(self) -> None:
        values = (
            self.max_evidence_weight,
            self.minimum_evidence_fraction,
            self.trust,
            self.signal_scale,
            self.disagreement_scale,
            self.sharpness,
            self.specificity_temperature,
            self.support_margin,
            self.correction_clip,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("GTP parameters must be finite")
        if not 0 <= self.max_evidence_weight <= 1:
            raise ValueError("max_evidence_weight must be in [0, 1]")
        if not 0 <= self.minimum_evidence_fraction <= 1:
            raise ValueError("minimum_evidence_fraction must be in [0, 1]")
        if self.mode not in {"legacy", "selective"}:
            raise ValueError("mode must be legacy or selective")
        if self.candidate_top_k < 1:
            raise ValueError("candidate_top_k must be positive")
        if min(
            self.trust,
            self.signal_scale,
            self.disagreement_scale,
            self.sharpness,
            self.specificity_temperature,
            self.correction_clip,
        ) <= 0:
            raise ValueError("trust and scales must be positive")


class GroundedTokenProjection:
    """Project a clean/evidence pair using a bounded evidence reliability gate.

    The gate is deliberately zero for both uninformative evidence (no branch
    movement) and highly contradictory evidence. This prevents a single noisy
    claim from receiving the same authority as corroborated visual evidence.
    """

    def __init__(self, config: GTPConfig | None = None) -> None:
        self.config = config or GTPConfig()

    def _reliability(
        self,
        clean_probabilities: torch.Tensor,
        evidence_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        total_variation = 0.5 * (
            clean_probabilities - evidence_probabilities
        ).abs().sum(dim=-1, keepdim=True)
        signal = torch.where(
            total_variation > self.config.signal_scale,
            torch.ones_like(total_variation),
            torch.zeros_like(total_variation),
        )
        excess = (total_variation - self.config.disagreement_scale).clamp_min(0.0)
        corroboration = torch.where(
            total_variation <= self.config.disagreement_scale,
            torch.ones_like(total_variation),
            torch.exp(-excess / self.config.signal_scale),
        )
        return (signal * corroboration).pow(self.config.sharpness).clamp(0.0, 1.0)

    def distribution(
        self,
        clean_logits: torch.Tensor,
        evidence_logits: torch.Tensor,
        control_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            clean_logits.shape != evidence_logits.shape
            or clean_logits.ndim < 1
            or (control_logits is not None and control_logits.shape != clean_logits.shape)
        ):
            raise ValueError("branch logits must have the same non-scalar shape")
        if not bool(
            torch.isfinite(clean_logits).all()
            and torch.isfinite(evidence_logits).all()
            and (control_logits is None or torch.isfinite(control_logits).all())
        ):
            raise ValueError("branch logits must be finite")

        log_p = clean_logits.float().log_softmax(dim=-1)
        log_g = evidence_logits.float().log_softmax(dim=-1)
        if control_logits is not None:
            log_c = control_logits.float().log_softmax(dim=-1)
            specificity = log_g - log_c
            visual_shift = log_g - log_p
            if self.config.mode == "legacy":
                specificity = specificity - specificity.mean(dim=-1, keepdim=True)
                support_gate = torch.sigmoid(
                    (visual_shift.abs() - self.config.support_margin)
                    / self.config.specificity_temperature
                )
                correction = specificity * support_gate
            else:
                # GTP is a counterfactual consensus rule, not a global
                # contrastive decoder. Only boost tokens supported by both
                # the clean-to-visual shift and the visual-to-control margin.
                k = min(self.config.candidate_top_k, log_g.shape[-1])
                candidate = torch.zeros_like(log_g, dtype=torch.bool)
                candidate.scatter_(
                    -1,
                    log_g.topk(k, dim=-1).indices,
                    True,
                )
                support_gate = torch.sigmoid(
                    (visual_shift - self.config.support_margin)
                    / self.config.specificity_temperature
                )
                counterfactual_gate = torch.sigmoid(
                    specificity / self.config.specificity_temperature
                )
                correction = (
                    specificity.clamp_min(0.0)
                    * support_gate
                    * counterfactual_gate
                    * candidate
                )
            correction = correction.clamp(
                -self.config.correction_clip,
                self.config.correction_clip,
            )
            utility = log_p + (
                self.config.max_evidence_weight / self.config.trust
            ) * correction
            if self.config.mode == "selective":
                # Keep every clean-supported token alive. The chi-square
                # active-set solver is useful for broad projections, but its
                # hard support truncation is unsafe for autoregressive
                # captioning where a low-probability noun can be valid.
                return utility.softmax(dim=-1)
            return chi_square_projection(
                log_p.exp(),
                utility,
                self.config.trust,
            )

        p, g = log_p.exp(), log_g.exp()
        total_variation = 0.5 * (p - g).abs().sum(dim=-1, keepdim=True)
        reliability = self._reliability(p, g)
        weight = self.config.max_evidence_weight * (
            self.config.minimum_evidence_fraction
            + (1.0 - self.config.minimum_evidence_fraction) * reliability
        )
        zero_signal = total_variation.le(1e-7)
        weight = torch.where(zero_signal, torch.zeros_like(weight), weight)

        # Each row has its own evidence weight, while the KKT projection
        # remains exact because the weight is fixed before solving q.
        log_h = -torch.logaddexp(
            torch.log1p(-weight).clamp_min(-80.0) - log_p,
            torch.log(weight.clamp_min(torch.finfo(log_p.dtype).tiny)) - log_g,
        )
        zero_weight = weight.squeeze(-1).eq(0)
        if bool(zero_weight.any()):
            log_h = torch.where(zero_weight.unsqueeze(-1), log_p, log_h)
        log_agreement = log_h.logsumexp(dim=-1, keepdim=True)
        prior = (log_h - log_agreement).exp()
        utility = (1.0 - weight) * log_p + weight * log_g
        projected = chi_square_projection(
            prior,
            log_agreement.exp() * utility,
            self.config.trust,
        )
        return torch.where(zero_weight.unsqueeze(-1), p, projected)

    def combine_logits(
        self,
        clean_logits: torch.Tensor,
        evidence_logits: torch.Tensor,
        control_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution(clean_logits, evidence_logits, control_logits).clamp_min(
            torch.finfo(torch.float32).tiny
        ).log()


class GTPMethod(ShieldBackendMethod):
    """Generic backend adapter for the GTP method contract."""

    def __init__(
        self,
        mode: str = "legacy",
        max_evidence_weight: float = 0.75,
        minimum_evidence_fraction: float = 0.0,
        trust: float = 1.0,
        signal_scale: float = 0.000001,
        disagreement_scale: float = 0.08,
        sharpness: float = 1.0,
        specificity_temperature: float = 0.5,
        support_margin: float = 0.0,
        correction_clip: float = 4.0,
        candidate_top_k: int = 128,
    ) -> None:
        self.plugin = GroundedTokenProjection(
            GTPConfig(
                mode=mode,
                max_evidence_weight=max_evidence_weight,
                minimum_evidence_fraction=minimum_evidence_fraction,
                trust=trust,
                signal_scale=signal_scale,
                disagreement_scale=disagreement_scale,
                sharpness=sharpness,
                specificity_temperature=specificity_temperature,
                support_margin=support_margin,
                correction_clip=correction_clip,
                candidate_top_k=candidate_top_k,
            )
        )
        super().__init__("gtp", asdict(self.plugin.config))
