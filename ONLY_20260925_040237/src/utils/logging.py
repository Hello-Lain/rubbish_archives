from __future__ import annotations

import logging

from utils import dist_utils


def setup_logging() -> None:
    level = logging.INFO if dist_utils.is_main_process() else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(level)
