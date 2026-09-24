"""Shared backend adapter for composable SHIELD method plugins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper


class ShieldBackendMethod(BaseMethod):
    """Dispatch a SHIELD plugin to a model-family-specific batch backend."""

    def __init__(
        self,
        method_name: str,
        method_config: Mapping[str, Any],
    ) -> None:
        self.method_name = method_name
        self.method_config = dict(method_config)
        self.model: BaseMLLMWrapper | None = None
        self.device: torch.device | None = None

    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        self.model = model
        self.device = device

    def generate(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.model is None or self.device is None:
            raise RuntimeError(f"{self.__class__.__name__}.setup is required")
        records = batch.get("records")
        if not isinstance(records, list):
            raise TypeError("batch['records'] must be a list")
        backend = getattr(self.model, "generate_batch", None)
        if backend is None:
            raise NotImplementedError(
                f"The model must expose generate_batch for {self.method_name}"
            )
        outputs = backend(
            records,
            method_name=self.method_name,
            method_config=self.method_config,
        )
        if len(outputs) != len(records):
            raise RuntimeError(
                f"{self.method_name} backend returned {len(outputs)} outputs for "
                f"{len(records)} records"
            )
        return outputs
