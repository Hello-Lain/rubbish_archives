"""Composable SHIELD component pipeline and cumulative method plugin."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from methods.shield_adaptive_plausibility import (
    AdaptivePlausibilityConfig,
    ShieldAdaptivePlausibility,
)
from methods.shield_inherent_bias import InherentBiasConfig, ShieldInherentBias
from methods.shield_method_base import ShieldBackendMethod
from methods.shield_statistical_bias import (
    ShieldStatisticalBias,
    StatisticalBiasConfig,
)
from methods.shield_vulnerability_defense import (
    ShieldVulnerabilityDefense,
    VulnerabilityDefenseConfig,
)

SHIELD_COMPONENT_NAMES = (
    "adaptive_plausibility",
    "vulnerability_defense",
    "statistical_bias",
    "inherent_bias",
)


@dataclass(frozen=True)
class ShieldStageSpec:
    """Canonical cumulative component selection for one ablation stage."""

    name: str
    components: tuple[str, ...]
    caption_token_injection: bool

    @property
    def use_adaptive_plausibility(self) -> bool:
        return "adaptive_plausibility" in self.components

    @property
    def use_vulnerability_defense(self) -> bool:
        return "vulnerability_defense" in self.components

    @property
    def use_statistical_bias(self) -> bool:
        return "statistical_bias" in self.components

    @property
    def use_inherent_bias(self) -> bool:
        return "inherent_bias" in self.components


_STAGE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "adaptive": ("adaptive_plausibility",),
    "vulnerability": ("adaptive_plausibility", "vulnerability_defense"),
    "statistical": (
        "adaptive_plausibility",
        "vulnerability_defense",
        "statistical_bias",
    ),
    "inherent": (
        "adaptive_plausibility",
        "vulnerability_defense",
        "statistical_bias",
        "inherent_bias",
    ),
    "full": (
        "adaptive_plausibility",
        "vulnerability_defense",
        "statistical_bias",
        "inherent_bias",
    ),
}


def stage_spec(stage: str) -> ShieldStageSpec:
    """Resolve a named ablation stage without embedding it in an evaluator."""

    try:
        components = _STAGE_COMPONENTS[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported SHIELD ablation stage: {stage}") from exc
    return ShieldStageSpec(
        name=stage,
        components=components,
        caption_token_injection="statistical_bias" in components,
    )


def custom_stage_spec(
    components: Sequence[str],
    *,
    name: str = "custom",
) -> ShieldStageSpec:
    """Build an auditable stage from an explicit component subset."""

    names = normalize_component_names(components)
    if not names:
        raise ValueError("At least one SHIELD component is required")
    suffix = "_".join(names)
    return ShieldStageSpec(
        name=f"{name}_{suffix}",
        components=names,
        caption_token_injection="statistical_bias" in names,
    )


def normalize_component_names(components: Sequence[str]) -> tuple[str, ...]:
    """Validate and canonicalize a user-provided plugin list."""

    names = tuple(str(component) for component in components)
    unknown = sorted(set(names) - set(SHIELD_COMPONENT_NAMES))
    if unknown:
        raise ValueError(f"Unknown SHIELD components: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("SHIELD component names must be unique")
    return tuple(name for name in SHIELD_COMPONENT_NAMES if name in set(names))


class ShieldComponentPipeline:
    """Compose feature and logit transformations in the SHIELD order."""

    def __init__(
        self,
        components: Sequence[str],
        *,
        adaptive: ShieldAdaptivePlausibility | None = None,
        vulnerability: ShieldVulnerabilityDefense | None = None,
        statistical: ShieldStatisticalBias | None = None,
        inherent: ShieldInherentBias | None = None,
        caption_token_injection: bool = False,
    ) -> None:
        self.components = normalize_component_names(components)
        self.adaptive = adaptive
        self.vulnerability = vulnerability
        self.statistical = statistical
        self.inherent = inherent
        self.caption_token_injection = caption_token_injection
        if caption_token_injection and statistical is None:
            raise ValueError(
                "caption_token_injection requires the statistical_bias component"
            )
        self._validate_component_instances()

    def _validate_component_instances(self) -> None:
        required = {
            "adaptive_plausibility": self.adaptive,
            "vulnerability_defense": self.vulnerability,
            "statistical_bias": self.statistical,
            "inherent_bias": self.inherent,
        }
        for name, component in required.items():
            if name in self.components and component is None:
                raise ValueError(f"Missing implementation for SHIELD component {name}")
            if name not in self.components and component is not None:
                raise ValueError(f"Unexpected implementation for SHIELD component {name}")

    @property
    def requires_caption(self) -> bool:
        return self.statistical is not None or self.vulnerability is not None

    @property
    def requires_clip(self) -> bool:
        return self.requires_caption

    @property
    def requires_adversarial_branch(self) -> bool:
        return self.vulnerability is not None

    @property
    def requires_feature_transform(self) -> bool:
        return self.statistical is not None or self.inherent is not None

    def transform_clean_features(
        self,
        image_features: torch.Tensor,
        *,
        caption_features: torch.Tensor | None = None,
        bias_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Statistical Bias first, then Inherent Bias subtraction."""

        transformed = image_features
        selected = torch.empty(
            (0,),
            dtype=torch.long,
            device=image_features.device,
        )
        if self.statistical is not None:
            if caption_features is None:
                raise ValueError("Statistical Bias requires caption features")
            transformed, selected = self.statistical.enhance(
                transformed,
                caption_features,
            )
        if self.inherent is not None:
            if bias_features is None:
                raise ValueError("Inherent Bias requires bias features")
            transformed = self.inherent.subtract(transformed, bias_features)
        return transformed, selected

    def map_caption_tokens(
        self,
        selected_indices: torch.Tensor,
        caption_token_count: int,
        input_caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.statistical is None or not self.caption_token_injection:
            return input_caption_embeds[:0]
        return self.statistical.map_text_segments(
            selected_indices,
            caption_token_count,
            input_caption_embeds,
        )

    def combine_logits(
        self,
        clean_logits: torch.Tensor,
        adversarial_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Vulnerability Defense, then Adaptive Plausibility."""

        combined = clean_logits
        if self.vulnerability is not None:
            combined = self.vulnerability.combine_logits(
                clean_logits,
                adversarial_logits,
            )
        if self.adaptive is not None:
            combined = self.adaptive.apply(clean_logits, combined)
        return combined

    def metadata(self) -> dict[str, Any]:
        """Return a serializable component manifest for metrics files."""

        metadata: dict[str, Any] = {
            "components": list(self.components),
            "caption_token_injection": self.caption_token_injection,
        }
        if self.adaptive is not None:
            metadata["adaptive_plausibility"] = {
                "beta": self.adaptive.config.beta,
            }
        if self.vulnerability is not None:
            metadata["vulnerability_defense"] = {
                "cd_alpha": self.vulnerability.config.cd_alpha,
                "cd_beta": self.vulnerability.config.cd_beta,
                "epsilon": self.vulnerability.config.epsilon,
                "attack_steps": self.vulnerability.config.attack_steps,
                "attack_c": self.vulnerability.config.attack_c,
                "attack_lr": self.vulnerability.config.attack_lr,
            }
        if self.statistical is not None:
            metadata["statistical_bias"] = {
                "threshold": self.statistical.config.threshold,
                "gamma_gain": self.statistical.config.gamma_gain,
                "gain_per": self.statistical.config.gain_per,
            }
        if self.inherent is not None:
            metadata["inherent_bias"] = {
                "bias_weight": self.inherent.config.bias_weight,
                "sample_num": self.inherent.config.sample_num,
                "seed": self.inherent.config.seed,
            }
        return metadata


def build_shield_pipeline(
    components: Sequence[str],
    *,
    adaptive_config: AdaptivePlausibilityConfig | None = None,
    vulnerability_config: VulnerabilityDefenseConfig | None = None,
    statistical_config: StatisticalBiasConfig | None = None,
    inherent_config: InherentBiasConfig | None = None,
    caption_token_injection: bool = False,
) -> ShieldComponentPipeline:
    """Build a validated pipeline from independently configurable plugins."""

    names = normalize_component_names(components)
    return ShieldComponentPipeline(
        names,
        adaptive=(
            ShieldAdaptivePlausibility(adaptive_config)
            if "adaptive_plausibility" in names
            else None
        ),
        vulnerability=(
            ShieldVulnerabilityDefense(vulnerability_config)
            if "vulnerability_defense" in names
            else None
        ),
        statistical=(
            ShieldStatisticalBias(statistical_config)
            if "statistical_bias" in names
            else None
        ),
        inherent=(
            ShieldInherentBias(inherent_config)
            if "inherent_bias" in names
            else None
        ),
        caption_token_injection=caption_token_injection,
    )


class ShieldCumulativeMethod(ShieldBackendMethod):
    """Hydra/BaseMethod adapter for an arbitrary SHIELD plugin composition."""

    def __init__(
        self,
        components: Sequence[str],
        caption_token_injection: bool = True,
        cd_alpha: float = 2.0,
        cd_beta: float = 0.35,
        threshold: float = 0.002,
        gamma_gain: float = 3.0,
        gain_per: float = 0.55,
        bias_weight: float = 0.01,
        bias_sample_num: int = 32,
        bias_seed: int | None = 42,
        epsilon: float = 0.14,
        attack_steps: int = 30,
        attack_c: float = 12.0,
        attack_lr: float = 0.14,
    ) -> None:
        names = normalize_component_names(components)
        if caption_token_injection and "statistical_bias" not in names:
            raise ValueError(
                "caption_token_injection requires the statistical_bias component"
            )
        self.pipeline = build_shield_pipeline(
            names,
            adaptive_config=AdaptivePlausibilityConfig(beta=cd_beta),
            vulnerability_config=VulnerabilityDefenseConfig(
                cd_alpha=cd_alpha,
                cd_beta=cd_beta,
                epsilon=epsilon,
                attack_steps=attack_steps,
                attack_c=attack_c,
                attack_lr=attack_lr,
            ),
            statistical_config=StatisticalBiasConfig(
                threshold=threshold,
                gamma_gain=gamma_gain,
                gain_per=gain_per,
            ),
            inherent_config=InherentBiasConfig(
                bias_weight=bias_weight,
                sample_num=bias_sample_num,
                seed=bias_seed,
            ),
            caption_token_injection=caption_token_injection,
        )
        super().__init__(
            "shield_cumulative",
            {
                "components": list(names),
                "caption_token_injection": caption_token_injection,
                "cd_alpha": cd_alpha,
                "cd_beta": cd_beta,
                "threshold": threshold,
                "gamma_gain": gamma_gain,
                "gain_per": gain_per,
                "bias_weight": bias_weight,
                "bias_sample_num": bias_sample_num,
                "bias_seed": bias_seed,
                "epsilon": epsilon,
                "attack_steps": attack_steps,
                "attack_c": attack_c,
                "attack_lr": attack_lr,
            },
        )
