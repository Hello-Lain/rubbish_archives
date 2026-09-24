from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PopeRunnerResult:
    """Paths and summary produced by one POPE runner invocation."""

    output_prefix: Path
    predictions_path: Path
    metrics_path: Path
    summary: dict[str, Any]


def _plain_mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping, got {type(value).__name__}")
    return dict(value)


def build_pope_command(
    cfg: DictConfig,
    *,
    method: str,
    output_prefix: Path,
) -> list[str]:
    """Build a reproducible command for the configured POPE runner."""
    if method not in {"vanilla", "only"}:
        raise ValueError("method must be 'vanilla' or 'only'")
    model_cfg = _plain_mapping(cfg.model)
    data_cfg = _plain_mapping(cfg.data)
    eval_cfg = _plain_mapping(cfg.eval_tasks)
    method_cfg = _plain_mapping(cfg.method)
    nested_method_cfg = method_cfg.get("method_config")
    if isinstance(nested_method_cfg, dict):
        method_cfg = {**method_cfg, **nested_method_cfg}
    env_cfg = _plain_mapping(cfg.env)
    script = Path(str(eval_cfg["runner_script"])).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    command = [
        sys.executable,
        str(script),
        "--method",
        method,
        "--model",
        str(model_cfg["path"]),
        "--pope-path",
        str(data_cfg["path"]),
        "--images-root",
        str(data_cfg["images_root"]),
        "--output",
        str(output_prefix),
        "--device",
        str(env_cfg.get("device", "cuda:0")),
        "--dtype",
        str(model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16"))),
        "--attention",
        str(model_cfg.get("attention", "auto")),
        "--fp8-scope",
        str(model_cfg.get("fp8_scope", "text_mlp")),
        "--batch-size",
        str(cfg.get("batch_size", 1)),
        "--image-batch-size",
        str(cfg.get("image_batch_size", 1)),
        "--num-workers",
        str(cfg.get("num_workers", 0)),
        "--prefetch-factor",
        str(cfg.get("prefetch_factor", 1)),
        "--max-new-tokens",
        str(cfg.get("max_new_tokens", 8)),
        "--decoding",
        str(cfg.get("decoding", "sample")),
        "--temperature",
        str(cfg.get("temperature", 1.0)),
        "--top-p",
        str(cfg.get("top_p", 1.0)),
        "--seed",
        str(cfg.get("seed", 42)),
        "--only-enhance-layer",
        str(method_cfg.get("enhance_layer", 0)),
        "--only-alpha-pos",
        str(method_cfg.get("alpha_pos", 3.0)),
        "--only-alpha-neg",
        str(method_cfg.get("alpha_neg", 1.0)),
        "--only-beta",
        str(method_cfg.get("beta", 0.1)),
        "--only-tvd-gamma",
        str(method_cfg.get("tvd_gamma", 0.2)),
    ]
    top_k = cfg.get("top_k")
    if top_k is not None:
        command.extend(["--top-k", str(top_k)])
    for flag, enabled in (
        ("--fp8", bool(model_cfg.get("fp8", True))),
        ("--tf32", bool(model_cfg.get("tf32", True))),
        ("--fallback", bool(model_cfg.get("fallback", True))),
        ("--progress", bool(cfg.get("progress", False))),
    ):
        command.append(flag if enabled else f"--no-{flag[2:]}")
    if str(eval_cfg.get("model_family", "")).lower() == "llava":
        command.extend(
            [
                "--image-feature-cache"
                if bool(cfg.get("image_feature_cache", True))
                else "--no-image-feature-cache"
            ]
        )
    start = cfg.get("start", 0)
    if start is not None:
        command.extend(["--start", str(start)])
    limit = cfg.get("limit")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def run_pope(
    cfg: DictConfig,
    *,
    method: str,
) -> PopeRunnerResult:
    output_dir = Path(str(cfg.paths.eval_output_dir)).expanduser().resolve()
    output_prefix = output_dir / method
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = build_pope_command(
        cfg,
        method=method,
        output_prefix=output_prefix,
    )
    log.info("Running POPE runner: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=Path(str(cfg.paths.root_dir)).expanduser().resolve(),
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        log.info("POPE runner stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        log.info("POPE runner stderr:\n%s", completed.stderr.rstrip())
    predictions_path = output_prefix.with_suffix(".jsonl")
    metrics_path = output_prefix.with_suffix(".metrics.json")
    if not predictions_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"POPE runner did not produce expected outputs: {predictions_path}, {metrics_path}"
        )
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    return PopeRunnerResult(
        output_prefix=output_prefix,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        summary=summary,
    )
