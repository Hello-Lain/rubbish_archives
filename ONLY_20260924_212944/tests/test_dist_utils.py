from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import utils.dist_utils as dist_utils


def test_cpu_device_request_uses_cpu_backend_even_when_cuda_exists() -> None:
    captured: dict[str, object] = {}
    env_cfg = SimpleNamespace(
        device="cpu",
        distributed_backend="nccl",
        cpu_backend="gloo",
        timeout_seconds=30,
    )

    with (
        patch.dict("os.environ", {"WORLD_SIZE": "2"}, clear=False),
        patch.object(dist_utils.dist, "is_initialized", return_value=False),
        patch.object(dist_utils.dist, "is_available", return_value=True),
        patch.object(
            dist_utils.dist,
            "init_process_group",
            side_effect=lambda **kwargs: captured.update(kwargs),
        ),
    ):
        dist_utils.init_distributed(env_cfg)

    assert captured["backend"] == "gloo"
