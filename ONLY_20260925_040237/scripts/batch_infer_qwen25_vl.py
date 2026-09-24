#!/usr/bin/env python3
"""Batch Qwen2.5-VL inference over a prepared mmap cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from infer_qwen25_vl import (
    DEFAULT_MODEL,
    configure_cuda,
    generate_once,
    get_fp8_call_counts,
    load_runtime,
    set_fp8_enabled,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="Prepared mmap cache directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--attention",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default="auto",
    )
    parser.add_argument("--cache-implementation", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("reduce-overhead", "max-autotune", "default"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--fp8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive Transformer Engine FP8 when the local stack supports it.",
    )
    parser.add_argument("--fp8-scope", choices=("text_mlp", "text_all"), default="text_mlp")
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--output", type=Path, help="Output JSONL file")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


class MMapTensorDataset(Dataset):
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        self.records = self.index["records"]
        self._maps: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_map(self, key: str, dtype: np.dtype) -> np.memmap:
        if key not in self._maps:
            self._maps[key] = np.memmap(
                self.cache_dir / f"{key}.bin",
                # Copy-on-write keeps the mmap-backed read path while giving
                # torch a writable view without ever modifying the cache.
                mode="c",
                dtype=dtype,
            )
        return self._maps[key]

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        tensors = {}
        for key, spec in record["tensors"].items():
            dtype = np.dtype(spec["dtype"])
            shape = tuple(spec["shape"])
            count = int(np.prod(shape, dtype=np.int64))
            start = int(spec["offset"]) // dtype.itemsize
            array = self._get_map(key, dtype)[start : start + count].reshape(shape)
            tensors[key] = torch.from_numpy(array)
        return {"id": record["id"], "tensors": tensors}


def collate_records(batch: list[dict], pad_token_id: int, padding_side: str) -> dict:
    tensor_rows = [item["tensors"] for item in batch]
    lengths = [row["input_ids"].shape[-1] for row in tensor_rows]
    max_length = max(lengths)
    input_ids = torch.full(
        (len(batch), max_length),
        pad_token_id,
        dtype=tensor_rows[0]["input_ids"].dtype,
    )
    attention_mask = torch.zeros(
        (len(batch), max_length),
        dtype=tensor_rows[0]["attention_mask"].dtype,
    )
    for row_index, (row, length) in enumerate(zip(tensor_rows, lengths)):
        if padding_side == "left":
            input_ids[row_index, -length:] = row["input_ids"].reshape(-1)
            attention_mask[row_index, -length:] = row["attention_mask"].reshape(-1)
        else:
            input_ids[row_index, :length] = row["input_ids"].reshape(-1)
            attention_mask[row_index, :length] = row["attention_mask"].reshape(-1)

    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.cat([row["pixel_values"] for row in tensor_rows], dim=0),
        "image_grid_thw": torch.cat([row["image_grid_thw"] for row in tensor_rows], dim=0),
        "ids": [item["id"] for item in batch],
    }
    if all("pixel_values_videos" in row for row in tensor_rows):
        output["pixel_values_videos"] = torch.cat(
            [row["pixel_values_videos"] for row in tensor_rows],
            dim=0,
        )
    if all("video_grid_thw" in row for row in tensor_rows):
        output["video_grid_thw"] = torch.cat(
            [row["video_grid_thw"] for row in tensor_rows],
            dim=0,
        )
    return output


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if key != "ids"
    }


def build_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        attention=args.attention,
        cache=args.cache_implementation,
        compile=args.compile,
        compile_mode=args.compile_mode,
        fp8=args.fp8,
        fp8_scope=args.fp8_scope,
        tf32=args.tf32,
        fallback=args.fallback,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    cache_dir = args.cache.expanduser().resolve()
    if not (cache_dir / "index.json").is_file():
        raise FileNotFoundError(cache_dir / "index.json")
    if args.batch_size < 1 or args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("batch-size must be positive, num-workers non-negative, prefetch-factor positive")

    device = resolve_device(args.device)
    configure_cuda(device, args.tf32)
    model_path = Path(args.model).expanduser().resolve()
    runtime_args = build_runtime_args(args)
    model, state, fp8_recipe = load_runtime(runtime_args, model_path, device)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    dataset = MMapTensorDataset(cache_dir)
    collate = partial(
        collate_records,
        pad_token_id=processor.tokenizer.pad_token_id or 0,
        # Decoder-only generation must left-pad a batch, regardless of the
        # tokenizer's preprocessing default.
        padding_side="left",
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": args.drop_last,
        "collate_fn": collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)

    output_handle = None
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    started = time.perf_counter()
    sample_count = 0
    token_count = 0
    batch_seconds = []
    try:
        for batch in loader:
            ids = batch.pop("ids")
            batch_inputs = move_batch_to_device(batch, device)
            batch_started = time.perf_counter()
            try:
                generated, _ = generate_once(
                    model,
                    batch_inputs,
                    runtime_args,
                    state,
                    fp8_recipe,
                )
            except Exception as exc:
                if not args.fallback:
                    raise
                state.fallback_events.append(
                    f"Optional batch path failed: {type(exc).__name__}: {exc}"
                )
                state.compile_effective = False
                state.cache = "dynamic"
                state.fp8_effective = False
                set_fp8_enabled(model, False)
                fp8_recipe = None
                generated, _ = generate_once(
                    model,
                    batch_inputs,
                    runtime_args,
                    state,
                    fp8_recipe,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_seconds.append(time.perf_counter() - batch_started)
            prompt_length = batch_inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for item_id, answer in zip(ids, decoded):
                result = {"id": item_id, "answer": answer.strip()}
                if output_handle:
                    output_handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                else:
                    print(json.dumps(result, ensure_ascii=True))
            sample_count += len(ids)
            token_count += int(generated.shape[1] - prompt_length) * len(ids)
    finally:
        if output_handle:
            output_handle.close()

    elapsed = time.perf_counter() - started
    fp8_calls, native_calls = get_fp8_call_counts(model)
    summary = {
        "cache": str(cache_dir),
        "records": len(dataset),
        "processed": sample_count,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers else None,
        "runtime": state.as_dict(),
        "fp8_te_calls": fp8_calls if args.fp8 else 0,
        "fp8_native_fallback_calls": native_calls if args.fp8 else 0,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / elapsed if elapsed else 0.0,
        "generated_tokens": token_count,
        "generated_tokens_per_second": token_count / elapsed if elapsed else 0.0,
        "batch_seconds": batch_seconds,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=True), file=sys.stderr)


if __name__ == "__main__":
    main()
