from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

import eval as eval_entry
from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper


class FakeModel(BaseMLLMWrapper):
    def setup(self, device: torch.device) -> None:
        self.device = device

    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        return f"{prompt}|{image_path}"


class FakeMethod(BaseMethod):
    def setup(self, model: BaseMLLMWrapper, device: torch.device) -> None:
        self.model = model
        self.device = device

    def generate(self, batch: Any) -> list[dict[str, Any]]:
        record = batch["records"][0]
        return [{"prediction": f"answer:{record['prompt']}:{record['image_path']}"}]


def test_eval_orchestration_calls_vlmeval_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MagicMock()
    dataset.evaluate.return_value = {"score": 1.0}
    infer_job = MagicMock()
    result_file = tmp_path / "custom_mllm_MMBench_DEV_EN.xlsx"

    build_dataset = MagicMock(return_value=dataset)
    get_pred_file_path = MagicMock(return_value=str(result_file))
    monkeypatch.setattr(eval_entry, "build_dataset", build_dataset)
    monkeypatch.setattr(eval_entry, "infer_data_job", infer_job)
    monkeypatch.setattr(eval_entry, "get_pred_file_path", get_pred_file_path)

    cfg = OmegaConf.create(
        {
            "model": {"_target_": "test_eval_orchestration.FakeModel", "path": "mock://fake"},
            "method": {"_target_": "test_eval_orchestration.FakeMethod"},
            "env": {
                "distributed_backend": "nccl",
                "cpu_backend": "gloo",
                "timeout_seconds": 30,
                "device": "cpu",
            },
            "paths": {"eval_output_dir": str(tmp_path), "output_dir": str(tmp_path)},
            "eval_tasks": {
                "backend": "vlmeval",
                "datasets": ["MMBench_DEV_EN"],
                "dataset_kwargs": {"skip_noimg": True},
                "judge": {"model": "exact_matching"},
            },
            "model_name": "custom_mllm",
            "api_nproc": 1,
            "retry_failed": True,
            "verbose": False,
            "mode": "eval",
            "optimized_metric": None,
        }
    )

    eval_entry.evaluate_once(cfg)

    build_dataset.assert_called_once_with("MMBench_DEV_EN", skip_noimg=True)
    infer_job.assert_called_once()
    call_kwargs = infer_job.call_args.kwargs
    assert call_kwargs["model_name"] == "custom_mllm"
    assert call_kwargs["dataset"] is dataset
    dataset.evaluate.assert_called_once_with(str(result_file), model="exact_matching")
