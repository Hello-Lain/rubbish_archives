from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from utils import dist_utils


def print_config_tree(
    cfg: DictConfig,
    *,
    print_order: Sequence[str] = (
        "data",
        "model",
        "method",
        "eval_tasks",
        "env",
        "paths",
        "logger",
        "extras",
    ),
    resolve: bool = True,
    save_to_file: bool = False,
) -> None:
    if not dist_utils.is_main_process():
        return

    tree = rich.tree.Tree("CONFIG", style="dim", guide_style="dim")
    queued = [key for key in print_order if key in cfg]
    queued.extend(key for key in cfg if key not in queued)
    for key in queued:
        branch = tree.add(str(key), style="dim", guide_style="dim")
        value = cfg[key]
        content = (
            OmegaConf.to_yaml(value, resolve=resolve)
            if isinstance(value, DictConfig)
            else str(value)
        )
        branch.add(rich.syntax.Syntax(content, "yaml"))

    rich.print(tree)
    if save_to_file:
        output_dir = Path(str(cfg.paths.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config_tree.log").open("w", encoding="utf-8") as handle:
            rich.print(tree, file=handle)
