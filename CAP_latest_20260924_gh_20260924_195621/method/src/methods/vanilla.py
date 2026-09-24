from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from methods.pope import PopeGenerationMethod


class VanillaPopeMethod(PopeGenerationMethod):
    """Canonical plugin name for the unmodified model POPE baseline."""

    def __init__(self, method_config: Mapping[str, Any] | None = None) -> None:
        super().__init__("vanilla", method_config)
