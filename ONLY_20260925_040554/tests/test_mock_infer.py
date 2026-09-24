from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_mock_infer_writes_jsonl(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "predictions.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    subprocess.run(
        [
            sys.executable,
            "src/infer.py",
            "experiment=mock_smoke",
            f"output_path={output_path}",
        ],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert [row["id"] for row in rows] == [
        "sample-001",
        "sample-002",
        "sample-003",
        "sample-004",
        "sample-005",
    ]
