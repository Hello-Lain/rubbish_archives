from __future__ import annotations

import torch

from models.base import BaseMLLMWrapper


class MockMLLMWrapper(BaseMLLMWrapper):
    """Model wrapper that makes the canonical pipeline runnable without weights."""

    def __init__(
        self,
        path: str,
        dtype: str,
        attn_implementation: str,
        low_cpu_mem_usage: bool,
    ) -> None:
        super().__init__(
            path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.is_ready = False

    def setup(self, device: torch.device) -> None:
        self.device = device
        self.is_ready = True

    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        if not self.is_ready or self.device is None:
            raise RuntimeError("MockMLLMWrapper.setup(device) must be called first.")
        return (
            f"mock_response(device={self.device.type}, dtype={self.dtype}, "
            f"attn={self.attn_implementation}, image={image_path}, prompt={prompt})"
        )
