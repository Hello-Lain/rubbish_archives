from __future__ import annotations

# ruff: noqa: E402
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper
from utils import dist_utils
from utils.experiment_logger import NullExperimentLogger, create_experiment_logger
from utils.instantiate import instantiate_from_config
from utils.logging import setup_logging
from utils.task_utils import prepare_task, seed_everything

log = logging.getLogger(__name__)


class StridedDistributedSampler(Sampler[int]):
    """Shard indexes by rank without padding or duplicated samples."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        rank: int,
        world_size: int,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def collate_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {"records": [dict(record) for record in records]}


def write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _sort_outputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: str(item.get("id", "")))


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    seed = None if cfg.get("seed") is None else int(cfg.seed)
    seed_everything(seed)
    dist_utils.init_distributed(cfg.env)
    setup_logging()
    prepare_task(cfg)

    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    device = dist_utils.get_device(str(cfg.env.get("device", "auto")))
    experiment_logger = NullExperimentLogger()
    try:
        experiment_logger = create_experiment_logger(cfg)
        experiment_logger.log_hyperparams(
            {
                "task_name": cfg.get("task_name"),
                "tags": cfg.get("tags"),
                "seed": cfg.get("seed"),
                "data": cfg.get("data"),
                "model": cfg.get("model"),
                "method": cfg.get("method"),
                "env": cfg.get("env"),
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
            }
        )

        dataset: Dataset[Any] = instantiate_from_config(cfg.data)
        sampler = StridedDistributedSampler(
            dataset,
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            sampler=sampler,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.env.get("pin_memory", device.type == "cuda")),
            collate_fn=collate_records,
        )

        model: BaseMLLMWrapper = instantiate_from_config(cfg.model)
        method: BaseMethod = instantiate_from_config(cfg.method)
        model.setup(device)
        method.setup(model, device)

        if dist_utils.is_main_process():
            log.info(
                "Running %d samples across %d process(es) on %s.",
                len(dataset),
                world_size,
                device,
            )

        local_outputs: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in loader:
                local_outputs.extend(method.generate(batch))

        merged_outputs = _sort_outputs(dist_utils.all_gather_list(local_outputs))
        if dist_utils.is_main_process():
            output_path = Path(str(cfg.output_path)).expanduser().resolve()
            write_jsonl(output_path, merged_outputs)
            log.info("Wrote %d predictions to %s.", len(merged_outputs), output_path)
            experiment_logger.log_metrics(
                {
                    "prediction_count": len(merged_outputs),
                    "dataset_size": len(dataset),
                    "world_size": world_size,
                }
            )
    finally:
        experiment_logger.finish()
        dist_utils.cleanup_distributed()


if __name__ == "__main__":
    main()
