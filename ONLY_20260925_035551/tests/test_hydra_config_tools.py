from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_hydra_config_composes_without_local_file() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "src/infer.py", "--cfg", "job"],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "task_name: infer" in result.stdout


def test_jsonl_logger_and_resolved_config_are_written(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    output_path = tmp_path / "predictions.jsonl"
    run_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "src/infer.py",
            "experiment=mock_smoke",
            "logger=jsonl",
            f"paths.output_dir={run_dir}",
            f"output_path={output_path}",
        ],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    assert (run_dir / "config_tree.log").exists()
    assert (run_dir / "resolved_config.yaml").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["hyperparams", "metrics"]
    assert events[-1]["prediction_count"] == 5
