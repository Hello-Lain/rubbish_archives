from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper


class StandardInferMethod(BaseMethod):
    """Model-agnostic autoregressive method for smoke and single-record use."""

    def __init__(
        self,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=true")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.model: BaseMLLMWrapper | None = None
        self.device: torch.device | None = None

    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        self.model = model
        self.device = device

    def generate(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.model is None or self.device is None:
            raise RuntimeError("StandardInferMethod.setup(model, device) is required.")
        records = batch.get("records")
        if not isinstance(records, list):
            raise TypeError("batch['records'] must be a list")

        outputs: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("every batch record must be a mapping")
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
                    "method": self.__class__.__name__,
                    "generation": {
                        "max_new_tokens": self.max_new_tokens,
                        "do_sample": self.do_sample,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    },
                }
            )
        return outputs
