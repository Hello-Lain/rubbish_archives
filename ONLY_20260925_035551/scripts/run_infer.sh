#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
exec uv run torchrun --standalone --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" src/infer.py "$@"
