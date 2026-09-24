from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from utils import dist_utils


class ExperimentLogger(Protocol):
    def log_hyperparams(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None: ...

    def finish(self) -> None: ...


class NullExperimentLogger:
    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        del params

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        del metrics, step

    def finish(self) -> None:
        return


class JsonlExperimentLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {"event": event, **_to_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._write("hyperparams", params)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        payload: dict[str, Any] = dict(metrics)
        if step is not None:
            payload["step"] = step
        self._write("metrics", payload)

    def finish(self) -> None:
        return


class WandbExperimentLogger:
    def __init__(
        self,
        project: str,
        name: str | None = None,
        group: str | None = None,
        tags: Sequence[str] | ListConfig | None = None,
        save_dir: str | None = None,
        mode: str = "online",
    ) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=name,
            group=group,
            tags=list(tags or []),
            dir=save_dir,
            mode=mode,
        )

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.run.config.update(_to_jsonable(params), allow_val_change=True)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        self.wandb.log(_to_jsonable(metrics), step=step)

    def finish(self) -> None:
        self.wandb.finish()


def _to_jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


def create_experiment_logger(cfg: DictConfig) -> ExperimentLogger:
    if not dist_utils.is_main_process():
        return NullExperimentLogger()
    logger_cfg = cfg.get("logger")
    if not logger_cfg:
        return NullExperimentLogger()
    return instantiate(logger_cfg)
