from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from methods.pope import PopeGenerationMethod


class OnlyPopeMethod(PopeGenerationMethod):
    """Canonical plugin name for ONLY-compatible model backends."""

    def __init__(self, method_config: Mapping[str, Any] | None = None) -> None:
        super().__init__("only", method_config)
