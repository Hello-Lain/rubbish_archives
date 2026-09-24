from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TypeVar

import torch
from omegaconf import DictConfig, OmegaConf

from utils import dist_utils, rich_utils

T = TypeVar("T")
log = logging.getLogger(__name__)


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_extras(cfg: DictConfig) -> None:
    extras = cfg.get("extras")
    if not extras:
        return
    if bool(extras.get("ignore_warnings", False)):
        warnings.filterwarnings("ignore")
    if bool(extras.get("enforce_tags", False)) and not cfg.get("tags"):
        raise ValueError("tags must be set when extras.enforce_tags=true.")
    if bool(extras.get("print_config", False)):
        rich_utils.print_config_tree(
            cfg,
            resolve=True,
            save_to_file=bool(extras.get("save_config_tree", False)),
        )


def log_output_dir(cfg: DictConfig) -> None:
    if dist_utils.is_main_process() and "paths" in cfg and "output_dir" in cfg.paths:
        log.info("Output dir: %s", cfg.paths.output_dir)


def save_resolved_config(cfg: DictConfig) -> None:
    if not dist_utils.is_main_process() or "paths" not in cfg or "output_dir" not in cfg.paths:
        return
    output_dir = Path(str(cfg.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )


def prepare_task(cfg: DictConfig) -> None:
    apply_extras(cfg)
    save_resolved_config(cfg)
    log_output_dir(cfg)


def task_wrapper(
    fn: Callable[[DictConfig], T],
) -> Callable[[DictConfig], T]:
    @wraps(fn)
    def wrapped(cfg: DictConfig) -> T:
        try:
            return fn(cfg)
        except Exception:
            log.exception("Task failed.")
            raise
        finally:
            log_output_dir(cfg)

    return wrapped
