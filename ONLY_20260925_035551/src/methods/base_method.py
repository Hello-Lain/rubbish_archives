from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from models.base import BaseMLLMWrapper


class BaseMethod(ABC):
    """Strategy interface shared by all canonical inference methods."""

    @abstractmethod
    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        """Bind the model wrapper and process-local device."""

    @abstractmethod
    def generate(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Generate one output record for every input record in ``batch``."""
