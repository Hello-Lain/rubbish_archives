from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from adapters.vlmeval_adapter import VLMEvalAdapter
from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper


class FakeModel(BaseMLLMWrapper):
    def setup(self, device: torch.device) -> None:
        self.device = device

    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        return f"{prompt}|{image_path}"


class FakeMethod(BaseMethod):
    def __init__(self) -> None:
        self.last_batch: Mapping[str, Any] | None = None

    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        self.model = model
        self.device = device

    def generate(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.last_batch = batch
        record = batch["records"][0]
        return [{"prediction": f"answer:{record['prompt']}:{record['image_path']}"}]


def test_vlmeval_adapter_translates_message() -> None:
    model = FakeModel(path="mock://fake")
    method = FakeMethod()
    model.setup(torch.device("cpu"))
    method.setup(model, torch.device("cpu"))
    adapter = VLMEvalAdapter(model_wrapper=model, method=method, device=torch.device("cpu"))

    answer = adapter.generate_inner(
        [
            {"type": "image", "value": "x.jpg"},
            {"type": "text", "value": "question"},
        ],
        dataset="MMBench_DEV_EN",
    )

    assert answer == "answer:question:x.jpg"
    assert method.last_batch is not None
    assert method.last_batch["records"][0]["prompt"] == "question"


def test_vlmeval_adapter_rejects_video() -> None:
    adapter = VLMEvalAdapter(
        model_wrapper=FakeModel(path="mock://fake"),
        method=FakeMethod(),
        device=torch.device("cpu"),
    )
    try:
        adapter.generate_inner(
            [{"type": "video", "value": "x.mp4"}],
            dataset="VideoBench",
        )
    except NotImplementedError as exc:
        assert "image benchmarks only" in str(exc)
    else:
        raise AssertionError("Expected video input to be rejected")
