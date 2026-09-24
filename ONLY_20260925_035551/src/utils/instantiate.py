from __future__ import annotations

from typing import TypeVar

from hydra.utils import instantiate

T = TypeVar("T")


def instantiate_from_config(config: object) -> T:
    """Instantiate a Hydra object while keeping the call site typed."""
    return instantiate(config)
