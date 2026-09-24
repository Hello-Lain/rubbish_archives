from __future__ import annotations

import logging
from typing import Any

from utils import dist_utils


class RankedLogger(logging.LoggerAdapter):
    """Logger adapter that annotates non-main-rank messages."""

    def __init__(self, name: str, *, rank_zero_only: bool = False) -> None:
        super().__init__(logging.getLogger(name), {})
        self.rank_zero_only = rank_zero_only

    def process(
        self,
        msg: object,
        kwargs: dict[str, Any],
    ) -> tuple[object, dict[str, Any]]:
        if self.rank_zero_only:
            return msg, kwargs
        return f"[rank={dist_utils.get_rank()}] {msg}", kwargs

    def log(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
        if self.rank_zero_only and not dist_utils.is_main_process():
            return
        super().log(level, msg, *args, **kwargs)


def get_ranked_logger(
    name: str,
    *,
    rank_zero_only: bool = True,
) -> RankedLogger:
    return RankedLogger(name, rank_zero_only=rank_zero_only)
