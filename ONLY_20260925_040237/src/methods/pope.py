from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper


class PopeBatchBackend(Protocol):
    """Optional backend implemented by a model-family-specific POPE adapter."""

    def generate_batch(
        self,
        records: list[Mapping[str, Any]],
        *,
        method_name: str,
        method_config: Mapping[str, Any],
    ) -> list[dict[str, Any]]: ...


class PopeGenerationMethod(BaseMethod):
    """Dispatch POPE generation to a model backend without embedding its logic."""

    def __init__(
        self,
        method_name: str,
        method_config: Mapping[str, Any] | None = None,
    ) -> None:
        if method_name not in {"vanilla", "only"}:
            raise ValueError("method_name must be 'vanilla' or 'only'")
        self.method_name = method_name
        self.method_config = dict(method_config or {})
        self.model: BaseMLLMWrapper | None = None
        self.device: torch.device | None = None

    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        self.model = model
        self.device = device

    def generate(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.model is None or self.device is None:
            raise RuntimeError("PopeGenerationMethod.setup(model, device) is required.")
        records = batch.get("records")
        if not isinstance(records, list):
            raise TypeError("batch['records'] must be a list")
        backend = getattr(self.model, "generate_batch", None)
        if backend is None:
            if self.method_name == "vanilla":
                return self._generate_vanilla_records(records)
            raise NotImplementedError(
                "The model has no batched POPE backend. "
                "Use adapters.pope_runner for the verified POPE runner "
                "or implement generate_batch on a model-family adapter."
            )
        outputs = backend(
            records,
            method_name=self.method_name,
            method_config=self.method_config,
        )
        if len(outputs) != len(records):
            raise RuntimeError(
                f"POPE backend returned {len(outputs)} outputs for {len(records)} records"
            )
        return outputs

    def _generate_vanilla_records(
        self,
        records: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("PopeGenerationMethod.setup(model, device) is required.")
        outputs: list[dict[str, Any]] = []
        for record in records:
            prompt = str(record["prompt"])
            image_path = record.get("image_path")
            prediction = self.model.generate_response(
                prompt=prompt,
                image_path=str(image_path) if image_path is not None else None,
            )
            outputs.append(
                {
                    "id": record["id"],
                    "prompt": prompt,
                    "image_path": image_path,
                    "prediction": prediction,
                    "method": "vanilla",
                    "backend": "generic_generate_response",
                }
            )
        return outputs
