from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from adapters.pope_runner import build_pope_command, run_pope


def test_pope_command_is_config_driven(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    script.write_text("print('placeholder')\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "data": {
                "path": "/cache/pope.json",
                "images_root": "/cache/images",
            },
            "model": {
                "path": "/cache/model",
                "dtype": "bfloat16",
                "attention": "auto",
                "fp8": True,
                "fp8_scope": "text_mlp",
                "tf32": True,
                "fallback": True,
            },
            "method": {
                "enhance_layer": 2,
                "alpha_pos": 3.0,
                "alpha_neg": 1.0,
                "beta": 0.1,
                "tvd_gamma": 0.2,
            },
            "eval_tasks": {
                "runner_script": str(script),
                "model_family": "qwen25_vl",
            },
            "env": {"device": "cuda:1"},
            "batch_size": 8,
            "image_batch_size": 16,
            "num_workers": 4,
            "prefetch_factor": 4,
            "max_new_tokens": 8,
            "decoding": "sample",
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 42,
            "progress": False,
            "start": 0,
            "limit": 16,
        }
    )
    command = build_pope_command(
        cfg,
        method="only",
        output_prefix=tmp_path / "only",
    )

    assert command[:2] == [sys.executable, str(script)]
    assert command[command.index("--method") + 1] == "only"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--only-enhance-layer") + 1] == "2"
    assert "--no-progress" in command


def test_pope_runner_reads_subprocess_artifacts(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    script.write_text(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "output.with_suffix('.jsonl').write_text('{\"id\": \"1\"}\\n')\n"
        "output.with_suffix('.metrics.json').write_text(\n"
        "    json.dumps({'metrics': {'accuracy': 1.0}})\n"
        ")\n",
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "data": {"path": "pope.json", "images_root": "images"},
            "model": {"path": "model", "torch_dtype": "bfloat16"},
            "method": {"method_config": {}},
            "eval_tasks": {
                "runner_script": str(script),
                "model_family": "qwen25_vl",
            },
            "env": {"device": "cpu"},
            "paths": {"root_dir": str(tmp_path), "eval_output_dir": str(tmp_path / "out")},
            "method_name": "vanilla",
        }
    )

    result = run_pope(cfg, method="vanilla")

    assert result.predictions_path.is_file()
    assert result.metrics_path.is_file()
    assert result.summary["metrics"]["accuracy"] == 1.0
