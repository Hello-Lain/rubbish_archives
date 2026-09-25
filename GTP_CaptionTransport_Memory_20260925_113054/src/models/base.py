from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseMLLMWrapper(ABC):
    """Bind configuration in ``__init__`` and load device resources in setup."""

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.kwargs = dict(kwargs)
        self.device: torch.device | None = None

    @abstractmethod
    def setup(self, device: torch.device) -> None:
        """Load model, processor, and device-bound resources lazily."""

    @abstractmethod
    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        """Generate one response using the model-specific processor contract."""
