#!/usr/bin/env bash
set -euo pipefail

exec uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 src/infer.py "$@"
