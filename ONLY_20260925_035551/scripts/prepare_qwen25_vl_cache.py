#!/usr/bin/env python3
"""Preprocess Qwen2.5-VL JSONL inputs into memory-mappable tensor files.

Each input line must contain ``image`` and ``prompt`` and may contain ``id``.
The generated index keeps the original metadata while tensor fields are
stored as append-only binary files with offsets, shapes, and dtypes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from infer_qwen25_vl import DEFAULT_MODEL, build_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, type=Path, help="Output cache directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            if "image" not in record or "prompt" not in record:
                raise ValueError(f"Line {line_number} must contain image and prompt")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def append_tensor(handle, tensor: torch.Tensor) -> dict:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    offset = handle.tell()
    handle.write(array.tobytes(order="C"))
    return {
        "offset": offset,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite to replace its files"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(input_path)
    model_path = Path(args.model).expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )

    handles: dict[str, object] = {}
    output_records = []
    try:
        for index, record in enumerate(records):
            image = Path(record["image"]).expanduser().resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            tensors = build_inputs(processor, image, str(record["prompt"]), torch.device("cpu"))
            tensor_specs = {}
            for key, value in tensors.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key not in handles:
                    path = output_dir / f"{key}.bin"
                    handles[key] = path.open("wb")
                tensor_specs[key] = append_tensor(handles[key], value)
            output_records.append(
                {
                    "id": record.get("id", index),
                    "image": str(image),
                    "prompt": str(record["prompt"]),
                    "tensors": tensor_specs,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "format": "qwen25-vl-mmap-v1",
        "model": str(model_path),
        "source": str(input_path),
        "records": output_records,
    }
    (output_dir / "index.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "records": len(output_records),
                "tensor_fields": sorted(handles),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
