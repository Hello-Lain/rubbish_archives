#!/usr/bin/env bash
set -euo pipefail

# Docker is used only as a wheel source. Inference runs from .venv afterward.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${QWEN_ACCEL_ENV:-${ROOT_DIR}/.venv}"
WHEELHOUSE="${UV_CACHE_DIR:-${HOME}/.cache/uv}/wheelhouse"
IMAGE="${QWEN_ACCEL_IMAGE:-only-qwen25-accel:local}"
PYTHON="${QWEN_ACCEL_PYTHON:-3.11}"
INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
REPLACE=0

if [[ $# -gt 0 && "$1" == "--replace" ]]; then
    REPLACE=1
    shift
fi
if [[ $# -gt 0 ]]; then
    echo "usage: $0 [--replace]" >&2
    exit 2
fi
if [[ -e "$ENV_DIR" && "$REPLACE" -ne 1 ]]; then
    echo "$ENV_DIR already exists; pass --replace to recreate it" >&2
    exit 2
fi

FLASH_WHEEL="${WHEELHOUSE}/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl"
TE_TORCH_WHEEL="${WHEELHOUSE}/transformer_engine_torch-2.8.0-cp311-cp311-linux_x86_64.whl"
mkdir -p "$WHEELHOUSE"

if [[ ! -f "$FLASH_WHEEL" ]]; then
    docker run --rm \
        --entrypoint bash \
        -v "${WHEELHOUSE}:/out" \
        "$IMAGE" \
        -lc "cp /wheelhouse/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl /out/"
fi

if [[ ! -f "$TE_TORCH_WHEEL" ]]; then
    echo "Missing local Transformer Engine wheel: $TE_TORCH_WHEEL" >&2
    exit 1
fi

# Docker may create the exported wheel as root. Restore ownership before uv
# writes metadata next to it.
docker run --rm \
    --entrypoint bash \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    -v "${WHEELHOUSE}:/out" \
    "$IMAGE" \
    -lc 'chown "$HOST_UID:$HOST_GID" /out/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl /out/transformer_engine_torch-2.8.0-cp311-cp311-linux_x86_64.whl'

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy
export UV_DEFAULT_INDEX="$INDEX"

uv venv --python "$PYTHON" --clear "$ENV_DIR"
if ! "$ENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    "$ENV_DIR/bin/python" -m ensurepip --upgrade
fi

# Install the exact torch/CUDA ABI first. CUDA extension wheels are installed
# with --no-deps so uv cannot silently replace the tested torch stack.
uv pip install \
    --python "$ENV_DIR/bin/python" \
    "torch==2.8.0" \
    "torchvision==0.23.0"

uv pip install \
    --python "$ENV_DIR/bin/python" \
    --no-deps \
    "$FLASH_WHEEL" \
    "$TE_TORCH_WHEEL"

uv pip install \
    --python "$ENV_DIR/bin/python" \
    --no-deps \
    "transformer-engine==2.8.0" \
    "transformer-engine-cu12==2.8.0"

# transformer-engine-cu12 2.8.0 is published with a cp310 wheel tag even
# though this artifact is the Python 3.11 CUDA package. Repair only that
# known metadata defect so uv can validate the otherwise working install.
repair_transformer_engine_metadata() {
    "$ENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import base64
import hashlib
import sysconfig
from pathlib import Path

site = Path(sysconfig.get_path("purelib"))
dist = site / "transformer_engine_cu12-2.8.0.dist-info"
wheel = dist / "WHEEL"
record = dist / "RECORD"
old = "Tag: cp310-cp310-manylinux_2_28_x86_64"
new = "Tag: cp311-cp311-manylinux_2_28_x86_64"

if not wheel.is_file() or not record.is_file():
    raise SystemExit(f"Transformer Engine metadata is incomplete under {dist}")

wheel_text = wheel.read_text()
if old in wheel_text:
    wheel.write_text(wheel_text.replace(old, new))
elif new not in wheel_text:
    raise SystemExit(f"Unexpected Transformer Engine wheel tag in {wheel}")

wheel_bytes = wheel.read_bytes()
digest = base64.urlsafe_b64encode(
    hashlib.sha256(wheel_bytes).digest()
).rstrip(b"=").decode("ascii")
record_lines = record.read_text().splitlines()
entry = "transformer_engine_cu12-2.8.0.dist-info/WHEEL,"
updated = []
found = False
for line in record_lines:
    if line.startswith(entry):
        updated.append(f"{entry}sha256={digest},{len(wheel_bytes)}")
        found = True
    else:
        updated.append(line)
if not found:
    raise SystemExit(f"Missing {entry} in {record}")
record.write_text("\n".join(updated) + "\n")
print(f"Repaired {wheel}")
PY
}

repair_transformer_engine_metadata

uv pip install \
    --python "$ENV_DIR/bin/python" \
    "transformers==4.57.6" \
    "qwen-vl-utils==0.0.14" \
    accelerate \
    pillow \
    numpy \
    einops \
    onnx \
    onnxscript \
    pydantic \
    packaging \
    importlib-metadata

"$ENV_DIR/bin/python" -m pip check
uv pip check --python "$ENV_DIR/bin/python"
"$ENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import flash_attn
import transformer_engine.pytorch as te

assert torch.__version__.startswith("2.8.0")
assert transformers.__version__ == "4.57.6"
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
print("transformer_engine", te.__file__)
PY

echo "Environment ready: $ENV_DIR"
echo "Run inference with: $ENV_DIR/bin/python scripts/infer_qwen25_vl.py ..."
