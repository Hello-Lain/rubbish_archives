from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from methods.shield_pipeline import (
    ShieldComponentPipeline,
    ShieldCumulativeMethod,
    custom_stage_spec,
    normalize_component_names,
    stage_spec,
)
from models.base import BaseMLLMWrapper


class RecordingStatistical:
    config = SimpleNamespace(threshold=0.002, gamma_gain=3.0, gain_per=0.55)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def enhance(
        self,
        image_features: torch.Tensor,
        caption_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del caption_features
        self.events.append("statistical")
        return image_features + 2.0, torch.tensor([1], dtype=torch.long)

    @staticmethod
    def map_text_segments(
        selected_indices: torch.Tensor,
        caption_token_count: int,
        input_caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        del selected_indices, caption_token_count
        return input_caption_embeds[:1]


class RecordingInherent:
    config = SimpleNamespace(bias_weight=0.01, sample_num=32, seed=42)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def subtract(
        self,
        image_features: torch.Tensor,
        bias_features: torch.Tensor,
    ) -> torch.Tensor:
        del bias_features
        self.events.append("inherent")
        return image_features - 0.5


class RecordingVulnerability:
    config = SimpleNamespace(
        cd_alpha=2.0,
        cd_beta=0.35,
        epsilon=0.14,
        attack_steps=30,
        attack_c=12.0,
        attack_lr=0.14,
    )

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def combine_logits(
        self,
        clean_logits: torch.Tensor,
        adversarial_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.events.append("vulnerability")
        return clean_logits + adversarial_logits


class RecordingAdaptive:
    config = SimpleNamespace(beta=0.35)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def apply(
        self,
        clean_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
    ) -> torch.Tensor:
        del clean_logits
        self.events.append("adaptive")
        return candidate_logits * 3.0


class FakeCumulativeModel(BaseMLLMWrapper):
    def __init__(self) -> None:
        super().__init__("mock://shield-pipeline")
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def setup(self, device: torch.device) -> None:
        self.device = device

    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        return f"{prompt}|{image_path}"

    def generate_batch(
        self,
        records: list[Mapping[str, Any]],
        *,
        method_name: str,
        method_config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append((method_name, method_config))
        return [{"id": record["id"]} for record in records]


def test_stage_spec_defines_cumulative_component_selection() -> None:
    assert stage_spec("adaptive").components == ("adaptive_plausibility",)
    assert stage_spec("vulnerability").components == (
        "adaptive_plausibility",
        "vulnerability_defense",
    )
    assert stage_spec("statistical").caption_token_injection is True
    assert stage_spec("full").use_inherent_bias is True


def test_custom_stage_spec_canonicalizes_explicit_component_subsets() -> None:
    spec = custom_stage_spec(
        ("statistical_bias", "adaptive_plausibility"),
        name="loo",
    )

    assert spec.name == "loo_adaptive_plausibility_statistical_bias"
    assert spec.components == (
        "adaptive_plausibility",
        "statistical_bias",
    )
    assert spec.caption_token_injection is True


def test_component_names_reject_unknown_and_duplicate_plugins() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        normalize_component_names(("adaptive_plausibility", "unknown"))
    with pytest.raises(ValueError, match="unique"):
        normalize_component_names(("adaptive_plausibility", "adaptive_plausibility"))


def test_clean_feature_pipeline_applies_statistical_before_inherent() -> None:
    events: list[str] = []
    pipeline = ShieldComponentPipeline(
        ("inherent_bias", "statistical_bias"),
        statistical=RecordingStatistical(events),
        inherent=RecordingInherent(events),
    )

    transformed, selected = pipeline.transform_clean_features(
        torch.ones(2, 3),
        caption_features=torch.ones(4, 5),
        bias_features=torch.ones(2, 3),
    )

    assert events == ["statistical", "inherent"]
    assert torch.equal(transformed, torch.full((2, 3), 2.5))
    assert torch.equal(selected, torch.tensor([1]))


def test_logit_pipeline_applies_vulnerability_before_adaptive() -> None:
    events: list[str] = []
    pipeline = ShieldComponentPipeline(
        ("adaptive_plausibility", "vulnerability_defense"),
        adaptive=RecordingAdaptive(events),
        vulnerability=RecordingVulnerability(events),
    )

    combined = pipeline.combine_logits(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    )

    assert events == ["vulnerability", "adaptive"]
    assert torch.equal(combined, torch.tensor([[12.0, 18.0]]))


def test_caption_injection_requires_statistical_bias() -> None:
    with pytest.raises(ValueError, match="statistical_bias"):
        ShieldComponentPipeline(
            ("adaptive_plausibility",),
            adaptive=RecordingAdaptive([]),
            caption_token_injection=True,
        )


def test_cumulative_method_dispatches_the_composed_plugin_manifest() -> None:
    method = ShieldCumulativeMethod(
        components=("adaptive_plausibility", "vulnerability_defense"),
        caption_token_injection=False,
        cd_alpha=1.5,
        cd_beta=0.2,
    )
    model = FakeCumulativeModel()
    device = torch.device("cpu")
    model.setup(device)
    method.setup(model, device)

    outputs = method.generate({"records": [{"id": "one"}]})

    assert outputs == [{"id": "one"}]
    assert method.pipeline.components == (
        "adaptive_plausibility",
        "vulnerability_defense",
    )
    method_name, method_config = model.calls[0]
    assert method_name == "shield_cumulative"
    assert method_config["components"] == [
        "adaptive_plausibility",
        "vulnerability_defense",
    ]
