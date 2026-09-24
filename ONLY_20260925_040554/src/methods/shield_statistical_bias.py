"""SHIELD Statistical Bias mitigation for LLaVA visual features.

The feature transformation follows ``hukcc/SHIELD`` (Apache-2.0, commit
``0a60786da6a6e52c15babc9501984df7eb05a166``), while the method adapter keeps
the project's strategy contract independent from the model-family runner.
Noise-derived Inherent Bias subtraction and adversarial contrastive decoding
are separate method plugins composed by ``ShieldComponentPipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from methods.shield_method_base import ShieldBackendMethod


@dataclass(frozen=True)
class StatisticalBiasConfig:
    """Hyper-parameters for SHIELD's Statistical Bias module."""

    threshold: float = 0.002
    gamma_gain: float = 3.0
    gain_per: float = 0.55

    def __post_init__(self) -> None:
        if self.gamma_gain <= 0:
            raise ValueError("gamma_gain must be positive")
        if not 0 <= self.gain_per <= 1:
            raise ValueError("gain_per must be in [0, 1]")


def maximize_weight_difference(weights: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Stretch normalized weights while preserving a bounded [0, 1] range."""

    if weights.numel() == 0:
        return weights
    weights_min = weights.min()
    weights_max = weights.max()
    weights_range = weights_max - weights_min
    if bool(weights_range < 1e-6):
        weights_norm = torch.zeros_like(weights)
    else:
        weights_norm = (weights - weights_min) / weights_range

    weights_adjusted = weights_norm.pow(gamma)
    adjusted_range = weights_adjusted.max() - weights_adjusted.min()
    if bool(adjusted_range < 1e-6):
        return torch.zeros_like(weights_adjusted)
    return (weights_adjusted - weights_adjusted.min()) / adjusted_range


class ShieldStatisticalBias:
    """Apply SHIELD token re-weighting before the LLaVA multimodal projector."""

    def __init__(self, config: StatisticalBiasConfig | None = None) -> None:
        self.config = config or StatisticalBiasConfig()

    def enhance(
        self,
        image_features: torch.Tensor,
        caption_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhance one image's patch features and return selected caption indices.

        Args:
            image_features: Vision-tower features with shape ``[patches, dim]``.
            caption_features: CLIP token features with shape ``[tokens, 768]``.
        """

        if image_features.ndim != 2:
            raise ValueError(
                f"image_features must have shape [patches, dim], got {tuple(image_features.shape)}"
            )
        if caption_features.ndim != 2:
            raise ValueError(
                "caption_features must have shape [tokens, 768], "
                f"got {tuple(caption_features.shape)}"
            )
        if image_features.shape[0] == 0 or caption_features.shape[0] == 0:
            return image_features, torch.empty(
                (0,),
                dtype=torch.long,
                device=image_features.device,
            )

        # SHIELD pools each patch vector to CLIP's 768-dimensional space before
        # calculating patch-caption cosine similarity.
        pooled = F.adaptive_max_pool1d(
            image_features.unsqueeze(0),
            768,
        ).squeeze(0)
        image_normalized = F.normalize(pooled, p=2, dim=-1).float()
        caption_normalized = F.normalize(
            caption_features.to(image_normalized.device),
            p=2,
            dim=-1,
        ).float()
        cosine_similarity = image_normalized @ caption_normalized.transpose(0, 1)

        similarity_text_max = cosine_similarity.max(dim=0).values
        top_k_indices = torch.nonzero(
            similarity_text_max > self.config.threshold,
            as_tuple=False,
        ).flatten()
        # No caption token passed the plausibility threshold.  The upstream
        # implementation assumes at least one match, but falling back to all
        # caption tokens silently applies an unrelated visual reweighting.
        if top_k_indices.numel() == 0:
            return image_features, top_k_indices
        selected_similarity = cosine_similarity[:, top_k_indices]
        similarity_img = selected_similarity.max(dim=1).values
        patch_weight = maximize_weight_difference(
            similarity_img,
            gamma=self.config.gamma_gain,
        )
        patch_weight_gain = torch.where(
            patch_weight <= self.config.gain_per,
            torch.zeros_like(patch_weight),
            patch_weight,
        ).to(dtype=image_features.dtype)
        enhanced = image_features + image_features * patch_weight_gain.unsqueeze(-1)
        return enhanced, top_k_indices

    @staticmethod
    def map_text_segments(
        top_k_indices: torch.Tensor,
        caption_token_count: int,
        input_caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Map CLIP token spans to LLaVA tokenizer embedding spans."""

        if top_k_indices.numel() == 0 or input_caption_embeds.shape[0] == 0:
            return input_caption_embeds[:0]
        if caption_token_count <= 0:
            raise ValueError("caption_token_count must be positive")

        indices = top_k_indices.flatten().to(dtype=torch.long)
        diffs = torch.diff(indices)
        breaks = torch.nonzero(diffs > 1, as_tuple=False).flatten().tolist()
        break_points = breaks + [len(indices) - 1]
        segment_lengths = [
            break_points[index] - break_points[index - 1]
            if index > 0
            else break_points[0] + 1
            for index in range(len(break_points))
        ]

        selected_segments: list[torch.Tensor] = []
        for segment in torch.split(indices, segment_lengths):
            start_index = int(segment[0].item())
            end_index = int(segment[-1].item())
            start_in_other = int(
                start_index / caption_token_count * input_caption_embeds.shape[0]
            )
            end_in_other = int(
                end_index / caption_token_count * input_caption_embeds.shape[0]
            )
            selected_segments.append(
                input_caption_embeds[start_in_other : end_in_other + 1]
            )
        return torch.cat(selected_segments, dim=0)


class ShieldStatisticalBiasMethod(ShieldBackendMethod):
    """Strategy adapter exposing the Statistical Bias method as a plugin."""

    method_name = "shield_statistical_bias"

    def __init__(
        self,
        threshold: float = 0.002,
        gamma_gain: float = 3.0,
        gain_per: float = 0.55,
    ) -> None:
        self.module = ShieldStatisticalBias(
            StatisticalBiasConfig(
                threshold=threshold,
                gamma_gain=gamma_gain,
                gain_per=gain_per,
            )
        )
        super().__init__(
            self.method_name,
            {
                "threshold": self.module.config.threshold,
                "gamma_gain": self.module.config.gamma_gain,
                "gain_per": self.module.config.gain_per,
            },
        )
