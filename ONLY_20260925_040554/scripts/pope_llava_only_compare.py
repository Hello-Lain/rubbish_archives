#!/usr/bin/env python3
"""Compare Vanilla and ONLY on the cached COCO POPE adversarial split.

This entry point targets the cached Hugging Face LLaVA-1.5 7B checkpoint.
The normal path uses BF16, FlashAttention-2 when available, a dynamic KV
cache, image-feature reuse, pinned image batches, and Hopper FP8 MLPs when
Transformer Engine can run them safely. ONLY keeps the same backend for all
non-intervention layers and uses a small eager attention island only where
the head-selection signal is required.

The default prompt and sampling path follow the official LLaVA-v1 evaluator.
Use ``--decoding greedy`` for deterministic ablations. Generated answers and
the exact effective acceleration state are written outside the Hugging Face
cache.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, set_seed
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast


HUB_ROOT = Path("/data/lcq/.cache/huggingface/hub").resolve()
MODEL_CACHE_ROOT = HUB_ROOT / "models--llava-hf--llava-1.5-7b-hf"
DEFAULT_POPE = HUB_ROOT / "pope/coco/coco_pope_adversarial.json"
DEFAULT_IMAGES = HUB_ROOT / "coco2014/val2014"
DEFAULT_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's "
    "questions."
)
POPE_ANSWER_INSTRUCTION = " Please answer this question with one word."


def format_pope_prompt(question: str, *, one_word: bool = True) -> str:
    """Build the official LLaVA POPE prompt with an explicit answer format."""

    instruction = POPE_ANSWER_INSTRUCTION if one_word else ""
    return f"{DEFAULT_SYSTEM_PROMPT} USER: <image>\n{question}{instruction} ASSISTANT:"


def resolve_default_model() -> Path:
    """Resolve a complete local snapshot without consulting the network."""
    refs_main = MODEL_CACHE_ROOT / "refs/main"
    candidates: list[Path] = []
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(MODEL_CACHE_ROOT / "snapshots" / revision)
    candidates.extend(sorted((MODEL_CACHE_ROOT / "snapshots").glob("*")))
    for candidate in candidates:
        if (
            (candidate / "config.json").is_file()
            and (candidate / "processor_config.json").is_file()
            and (candidate / "model.safetensors.index.json").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        f"No complete cached LLaVA snapshot found under {MODEL_CACHE_ROOT}"
    )


DEFAULT_MODEL = resolve_default_model()


@dataclass
class RuntimeState:
    requested_attention: str
    effective_attention: str
    flash_attn_version: Optional[str]
    dtype: str
    tf32: bool
    fp8_requested: bool
    fp8_capability_ready: bool
    fp8_effective: bool
    fp8_scope: Optional[str]
    fp8_linear_count: int
    fp8_calls: int
    fp8_native_fallback_calls: int
    fallback_events: list[str]
    image_feature_cache: bool
    image_feature_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_attention": self.requested_attention,
            "effective_attention": self.effective_attention,
            "flash_attn_version": self.flash_attn_version,
            "dtype": self.dtype,
            "tf32": self.tf32,
            "fp8_requested": self.fp8_requested,
            "fp8_capability_ready": self.fp8_capability_ready,
            "fp8_effective": self.fp8_effective,
            "fp8_scope": self.fp8_scope,
            "fp8_linear_count": self.fp8_linear_count,
            "fp8_calls": self.fp8_calls,
            "fp8_native_fallback_calls": self.fp8_native_fallback_calls,
            "fallback_events": self.fallback_events,
            "image_feature_cache": self.image_feature_cache,
            "image_feature_count": self.image_feature_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vanilla", "only"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pope-path", type=Path, default=DEFAULT_POPE)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="auto",
        help="auto selects BF16 on CUDA devices with BF16 support.",
    )
    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument(
        "--fp8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive Transformer Engine FP8 for text MLP projections.",
    )
    parser.add_argument(
        "--fp8-scope",
        choices=("text_mlp", "text_all"),
        default="text_mlp",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fall back to SDPA/BF16 if an optional acceleration path fails.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--pope-answer-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append SHIELD/LLaVA's official one-word POPE answer instruction.",
    )
    parser.add_argument(
        "--decoding",
        choices=("sample", "greedy"),
        default="sample",
        help="Official ONLY uses sampling; greedy is retained for ablations.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--top-k", dest="top_k", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-enhance-layer", type=int, default=0)
    parser.add_argument("--only-alpha-pos", type=float, default=3.0)
    parser.add_argument("--only-alpha-neg", type=float, default=1.0)
    parser.add_argument("--only-beta", type=float, default=0.1)
    parser.add_argument("--only-tvd-gamma", type=float, default=0.2)
    parser.add_argument(
        "--image-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute each unique COCO image feature once and reuse it.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def require_hub_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != HUB_ROOT and HUB_ROOT not in resolved.parents:
        raise ValueError(f"{label} must be under {HUB_ROOT}: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> Optional[str]:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except Exception:
        return None


def load_rows(path: Path, start: int, limit: Optional[int]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    end = None if limit is None else start + limit
    selected = rows[start:end]
    if not selected:
        raise ValueError(f"No POPE rows selected from {path}")
    return selected


def answer_label(text: str) -> int:
    first_line = text.splitlines()[0] if text.splitlines() else text
    normalized = first_line.replace(".", " ").replace(",", " ").lower()
    return 0 if re.search(r"\b(no|not)\b|n't", normalized) else 1


def compute_metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": sum(predictions) / total if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def filter_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, -float("inf"))


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = torch.zeros_like(sorted_remove).scatter(
        -1,
        sorted_indices,
        sorted_remove,
    )
    return logits.masked_fill(remove, -float("inf"))


def decode_logits(
    logits: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Apply the official sampling order or deterministic greedy decoding."""
    logits = logits.float()
    if args.decoding == "greedy":
        return logits.argmax(dim=-1)
    logits = logits / args.temperature
    logits = filter_top_k(logits, args.top_k)
    logits = filter_top_p(logits, args.top_p)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else torch.float32


def configure_cuda(device: torch.device, tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def detect_flash_attention() -> tuple[bool, Optional[str]]:
    try:
        import flash_attn

        return True, str(getattr(flash_attn, "__version__", "unknown"))
    except Exception:
        return False, None


def choose_attention(
    requested: str, device: torch.device
) -> tuple[str, Optional[str]]:
    available, version = detect_flash_attention()
    if requested == "auto":
        if device.type == "cuda" and available:
            return "flash_attention_2", version
        return "sdpa", version
    if requested == "flash_attention_2" and not available:
        raise RuntimeError("flash_attention_2 was requested but flash-attn is unavailable")
    return requested, version


def is_torch_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except AttributeError:
        return torch._dynamo.is_compiling()


class AdaptiveTELinear(nn.Module):
    """Use TE with internal token padding and retain a native fallback."""

    def __init__(self, native: nn.Linear, te_linear: nn.Module) -> None:
        super().__init__()
        self.native = native
        self.te_linear = te_linear
        self.fp8_enabled = True
        self.fp8_calls = 0
        self.native_calls = 0
        self.in_features = native.in_features
        self.out_features = native.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        token_count = inputs.numel() // inputs.shape[-1]
        can_use_fp8 = (
            self.fp8_enabled
            and inputs.shape[-1] % 16 == 0
            and token_count > 0
        )
        if can_use_fp8:
            if not is_torch_compiling():
                self.fp8_calls += 1
            padding = (-token_count) % 8
            if padding == 0:
                return self.te_linear(inputs)
            # TE's FP8 kernels require an 8-token multiple.  Pad only the
            # transient flattened activation so short final batches do not
            # alter batch composition or consume a native fallback call.
            flat_inputs = inputs.reshape(token_count, inputs.shape[-1])
            padded_inputs = torch.cat(
                (
                    flat_inputs,
                    torch.zeros(
                        (padding, inputs.shape[-1]),
                        dtype=inputs.dtype,
                        device=inputs.device,
                    ),
                ),
                dim=0,
            )
            padded_outputs = self.te_linear(padded_inputs)
            return padded_outputs[:token_count].reshape(
                *inputs.shape[:-1],
                self.out_features,
            )
        if not is_torch_compiling():
            self.native_calls += 1
        return self.native(inputs)


def _share_linear_parameters(source: nn.Linear, target: nn.Module) -> None:
    # Keep one BF16 parameter storage for both paths. Duplicating all MLP
    # weights in a 7B model can consume most of an 80 GB H100.
    target.weight = source.weight
    if source.bias is not None:
        target.bias = source.bias
    target.weight.requires_grad_(source.weight.requires_grad)
    if source.bias is not None:
        target.bias.requires_grad_(source.bias.requires_grad)


def enable_te_fp8(
    model: LlavaForConditionalGeneration,
    device: torch.device,
    scope: str,
) -> tuple[object, int]:
    if device.type != "cuda":
        raise RuntimeError("Transformer Engine FP8 requires CUDA")
    from transformer_engine import pytorch as te
    from transformer_engine.common import recipe

    language_model = model.language_model
    converted = 0

    def should_convert(path: str) -> bool:
        if ".mlp." in path:
            return True
        return scope == "text_all" and ".self_attn." in path

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and should_convert(path):
                if child.weight.device.type != "cuda":
                    raise RuntimeError(f"Cannot convert non-CUDA layer: {path}")
                target = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                    name=path,
                )
                _share_linear_parameters(child, target)
                setattr(parent, name, AdaptiveTELinear(child, target))
                converted += 1
            elif not isinstance(child, AdaptiveTELinear):
                visit(child, path)

    visit(language_model)
    if converted == 0:
        raise RuntimeError(f"No Linear layers matched Transformer Engine scope {scope!r}")
    gc.collect()
    torch.cuda.empty_cache()
    return recipe.Float8CurrentScaling(fp8_format=recipe.Format.HYBRID), converted


@contextlib.contextmanager
def fp8_context(state: RuntimeState, recipe: object | None) -> Iterator[None]:
    if not state.fp8_effective or recipe is None:
        yield
        return
    from transformer_engine.pytorch import fp8_autocast

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        yield


def fp8_call_counts(model: nn.Module) -> tuple[int, int]:
    fp8_calls = 0
    native_calls = 0
    for module in model.modules():
        if isinstance(module, AdaptiveTELinear):
            fp8_calls += module.fp8_calls
            native_calls += module.native_calls
    return fp8_calls, native_calls


def set_fp8_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, AdaptiveTELinear):
            module.fp8_enabled = enabled


def remove_te_fp8(model: nn.Module) -> int:
    """Restore native Linear modules and release Transformer Engine state."""
    removed = 0

    def visit(parent: nn.Module) -> None:
        nonlocal removed
        for name, child in list(parent.named_children()):
            if isinstance(child, AdaptiveTELinear):
                setattr(parent, name, child.native)
                removed += 1
            else:
                visit(child)

    visit(model)
    return removed


def disable_fp8(
    model: nn.Module,
    state: RuntimeState,
    reason: str,
) -> None:
    removed = remove_te_fp8(model)
    state.fp8_effective = False
    state.fp8_scope = None
    state.fallback_events.append(
        f"FP8 runtime fallback ({removed} layers restored): {reason}"
    )
    gc.collect()
    if next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()


class ImageDataset(Dataset[tuple[Image.Image, str]]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            return image.convert("RGB"), str(path)


def image_collate(
    batch: list[tuple[Image.Image, str]],
) -> tuple[list[Image.Image], list[str]]:
    return [item[0] for item in batch], [item[1] for item in batch]


def build_image_feature_cache(
    model: LlavaForConditionalGeneration,
    processor: Any,
    image_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    progress: bool,
) -> dict[str, torch.Tensor]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": image_collate,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(ImageDataset(image_paths), **loader_kwargs)
    image_cache: dict[str, torch.Tensor] = {}
    iterator = tqdm(loader, desc="image-features", disable=not progress)
    for images, paths in iterator:
        pixel_values = processor.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        if device.type == "cuda":
            pixel_values = pixel_values.pin_memory()
        pixel_values = pixel_values.to(
            device=device,
            dtype=dtype,
            non_blocking=device.type == "cuda",
        )
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
        for path, feature in zip(paths, features):
            image_cache[path] = feature.detach()
    state_count = len(image_cache)
    if state_count != len(image_paths):
        raise RuntimeError(
            f"Image feature cache incomplete: {state_count}/{len(image_paths)}"
        )
    return image_cache


def image_token_count(processor: Any) -> int:
    crop_size = processor.image_processor.crop_size
    height = int(crop_size["height"])
    width = int(crop_size["width"])
    count = (height // int(processor.patch_size)) * (
        width // int(processor.patch_size)
    )
    count += int(processor.num_additional_image_tokens)
    if processor.vision_feature_select_strategy == "default":
        count -= 1
    return count


def tokenize_legacy_image_prompt(
    tokenizer: Any,
    prompt: str,
    image_token_id: int,
    image_count: int,
) -> list[int]:
    """Tokenize text chunks separately, matching legacy image expansion."""
    chunks = prompt.split("<image>")
    if len(chunks) != 2:
        raise ValueError("POPE prompt must contain exactly one <image> token")
    prefix = tokenizer(chunks[0], add_special_tokens=True)["input_ids"]
    suffix = tokenizer(chunks[1], add_special_tokens=True)["input_ids"]
    if suffix and suffix[0] == tokenizer.bos_token_id:
        suffix = suffix[1:]
    return prefix + [image_token_id] * image_count + suffix


@torch.inference_mode()
def build_prompt_batch(
    model: LlavaForConditionalGeneration,
    processor: Any,
    rows: list[dict[str, Any]],
    image_root: Path,
    image_cache: dict[str, torch.Tensor],
    device: torch.device,
    pope_answer_instruction: bool = True,
) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
    count = image_token_count(processor)
    prompts = [
        format_pope_prompt(
            str(row["text"]),
            one_word=pope_answer_instruction,
        )
        for row in rows
    ]
    input_id_rows = [
        tokenize_legacy_image_prompt(
            processor.tokenizer,
            prompt,
            int(model.config.image_token_index),
            count,
        )
        for prompt in prompts
    ]
    tokenized = processor.tokenizer.pad(
        [
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
            for input_ids in input_id_rows
        ],
        padding=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    text_embeds = model.get_input_embeddings()(input_ids)
    features: list[torch.Tensor] = []
    for row in rows:
        path = str((image_root / row["image"]).resolve())
        if path not in image_cache:
            raise KeyError(f"Missing image feature for {path}")
        features.append(image_cache[path])
    image_features = torch.stack(features, dim=0).to(
        device=device,
        dtype=text_embeds.dtype,
        non_blocking=device.type == "cuda",
    )
    image_mask = model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=text_embeds,
        image_features=image_features,
    )
    inputs_embeds = text_embeds.masked_scatter(image_mask, image_features)
    image_positions = [
        (input_ids[index] == model.config.image_token_index)
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
        for index in range(input_ids.shape[0])
    ]
    return {
        "inputs_embeds": inputs_embeds,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, image_positions


def dense_causal_mask(
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.LongTensor,
    past_key_values: Optional[Cache],
    query_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = (
        attention_mask.shape[0]
        if attention_mask is not None
        else 1
    )
    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    key_length = past_length + query_length
    key_positions = torch.arange(key_length, device=cache_position.device)
    causal = key_positions[None, :] <= cache_position[:, None]
    causal = causal.unsqueeze(0).expand(batch_size, -1, -1)
    if attention_mask is not None:
        valid_keys = attention_mask[:, :key_length].bool()
        causal = causal & valid_keys[:, None, :]
    mask = torch.zeros(
        (batch_size, 1, query_length, key_length),
        dtype=dtype,
        device=cache_position.device,
    )
    return mask.masked_fill(~causal[:, None, :, :], torch.finfo(dtype).min)


def manual_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Optional[Cache],
    cache_position: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )
    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            attention.layer_idx,
            cache_kwargs,
        )
    repeated_keys = repeat_kv(key_states, attention.num_key_value_groups)
    repeated_values = repeat_kv(value_states, attention.num_key_value_groups)
    attn_weights = torch.matmul(
        query_states,
        repeated_keys.transpose(2, 3),
    ) * attention.scaling
    attn_weights = attn_weights + attention_mask[:, :, :, : repeated_keys.shape[-2]]
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, repeated_values)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attention.o_proj(
        attn_output.reshape(*input_shape, -1)
    )
    return attn_output, attn_weights, repeated_values


def suppress_group(
    weights: torch.Tensor,
    group_mask: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    mask = group_mask[:, None, None, :]
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    selected = weights.masked_fill(~mask, 0.0)
    mean = selected.sum(dim=-1, keepdim=True) / count
    variance = (
        ((selected - mean).masked_fill(~mask, 0.0) ** 2).sum(
            dim=-1, keepdim=True
        )
        / (count - 1).clamp_min(1)
    )
    threshold = mean + torch.sqrt(variance)
    selected = torch.where(
        mask & (weights > threshold),
        torch.zeros_like(weights),
        selected,
    )
    if normalize:
        selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return selected


def auxiliary_attention(
    weights: torch.Tensor,
    repeated_values: torch.Tensor,
    attention: nn.Module,
    image_positions: list[list[int]],
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    batch_size, _, _, key_length = weights.shape
    valid_mask = attention_mask[:, :key_length].bool()
    image_mask = torch.zeros(
        (batch_size, key_length),
        dtype=torch.bool,
        device=weights.device,
    )
    for row_index, positions in enumerate(image_positions):
        valid = [pos for pos in positions if 0 <= pos < key_length]
        if valid:
            image_mask[row_index, torch.tensor(valid, device=weights.device)] = True
    image_mask &= valid_mask

    # The official implementation excludes the first valid token (BOS) from
    # TVER statistics, but leaves it in the auxiliary attention output.
    first_valid = valid_mask.long().argmax(dim=-1)
    row_indices = torch.arange(batch_size, device=weights.device)
    bos_mask = torch.zeros_like(valid_mask)
    bos_mask[row_indices, first_valid] = valid_mask[row_indices, first_valid]
    text_mask = valid_mask & ~image_mask & ~bos_mask

    if not bool(image_mask.any()):
        details = [
            {
                "kept_heads": list(range(attention.num_heads)),
                "removed_heads": [],
                "entropy_ratio": [1.0] * attention.num_heads,
            }
            for _ in range(batch_size)
        ]
        auxiliary_weights = weights * valid_mask[:, None, None, :].to(weights.dtype)
        auxiliary_context = torch.matmul(auxiliary_weights, repeated_values)
        auxiliary_context = auxiliary_context.transpose(1, 2).contiguous()
        return attention.o_proj(
            auxiliary_context.reshape(batch_size, weights.shape[2], -1)
        ), details

    text_clean = suppress_group(weights, text_mask)
    image_clean = suppress_group(weights, image_mask)
    text_norm = (
        text_clean / text_clean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    )[:, :, -1, :]
    image_norm = (
        image_clean / image_clean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    )[:, :, -1, :]
    text_entropy = -torch.sum(
        text_norm * torch.log(text_norm + 1e-6),
        dim=-1,
    )
    image_entropy = -torch.sum(
        image_norm * torch.log(image_norm + 1e-6),
        dim=-1,
    )
    ratio = torch.nan_to_num(
        text_entropy / (image_entropy + 1e-6),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    keep = ratio >= ratio.mean(dim=-1, keepdim=True)
    # The official path writes the cleaned text/image groups back into the
    # auxiliary attention map, retains BOS, and then removes rejected heads.
    bos_weights = weights * bos_mask[:, None, None, :].to(weights.dtype)
    auxiliary_weights = (
        bos_weights + text_clean + image_clean
    ) * keep[:, :, None, None].to(weights.dtype)
    auxiliary_context = torch.matmul(auxiliary_weights, repeated_values)
    auxiliary_context = auxiliary_context.transpose(1, 2).contiguous()
    auxiliary_output = attention.o_proj(
        auxiliary_context.reshape(batch_size, weights.shape[2], -1)
    )
    details = []
    for row_index in range(batch_size):
        keep_row = keep[row_index].detach().cpu().tolist()
        details.append(
            {
                "kept_heads": [
                    head for head, value in enumerate(keep_row) if value
                ],
                "removed_heads": [
                    head for head, value in enumerate(keep_row) if not value
                ],
                "entropy_ratio": ratio[row_index].detach().float().cpu().tolist(),
            }
        )
    return auxiliary_output, details


def only_language_forward(
    language_model: nn.Module,
    input_ids: Optional[torch.LongTensor],
    inputs_embeds: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache],
    cache_position: Optional[torch.LongTensor],
    image_positions: list[list[int]],
    enhance_layer: int,
) -> tuple[BaseModelOutputWithPast, torch.Tensor, list[dict[str, Any]]]:
    base = getattr(language_model, "model", language_model)
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = base.embed_tokens(input_ids)
    use_cache = True
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=base.config)
    query_length = inputs_embeds.shape[1]
    if cache_position is None:
        past_seen = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen,
            past_seen + query_length,
            device=inputs_embeds.device,
        )
    if cache_position.numel() != query_length:
        raise ValueError("cache_position must match the query length")
    if attention_mask is None:
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], cache_position[-1].item() + 1),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    position_ids = cache_position.unsqueeze(0)
    backend_mask = create_causal_mask(
        config=base.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    eager_mask = dense_causal_mask(
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        query_length=query_length,
        dtype=inputs_embeds.dtype,
    )
    hidden_states = inputs_embeds
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    auxiliary_state: Optional[torch.Tensor] = None
    details: list[dict[str, Any]] = [
        {
            "kept_heads": [],
            "removed_heads": [],
            "entropy_ratio": [],
        }
        for _ in range(inputs_embeds.shape[0])
    ]
    last_layer = len(base.layers) - 1
    if not 0 <= enhance_layer < last_layer:
        raise ValueError(
            f"enhance_layer must be in [0, {last_layer - 1}], got {enhance_layer}"
        )
    for layer_index, decoder_layer in enumerate(base.layers):
        if layer_index == enhance_layer:
            residual = hidden_states
            normalized = decoder_layer.input_layernorm(hidden_states)
            normal_attn, weights, repeated_values = manual_attention(
                decoder_layer.self_attn,
                normalized,
                position_embeddings,
                eager_mask,
                past_key_values,
                cache_position,
            )
            auxiliary_attn, details = auxiliary_attention(
                weights,
                repeated_values,
                decoder_layer.self_attn,
                image_positions,
                attention_mask,
            )
            hidden_states = residual + normal_attn
            hidden_states = hidden_states + decoder_layer.mlp(
                decoder_layer.post_attention_layernorm(hidden_states)
            )
            auxiliary_state = auxiliary_attn
        elif layer_index == last_layer:
            if auxiliary_state is None:
                raise RuntimeError("ONLY auxiliary state was not captured")
            residual = hidden_states
            normalized = decoder_layer.input_layernorm(hidden_states)
            normal_attn, _, _ = manual_attention(
                decoder_layer.self_attn,
                normalized,
                position_embeddings,
                eager_mask,
                past_key_values,
                cache_position,
            )
            hidden_states = residual + normal_attn
            hidden_states = hidden_states + decoder_layer.mlp(
                decoder_layer.post_attention_layernorm(hidden_states)
            )
            auxiliary_state = decoder_layer.input_layernorm(auxiliary_state)
            auxiliary_state = 0.2 * residual + auxiliary_state
            auxiliary_residual = auxiliary_state
            auxiliary_state = auxiliary_residual + decoder_layer.mlp(
                decoder_layer.post_attention_layernorm(auxiliary_state)
            )
        else:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=backend_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
    normal_hidden = base.norm(hidden_states)
    if auxiliary_state is None:
        raise RuntimeError("ONLY auxiliary state is missing at final normalization")
    auxiliary_hidden = base.norm(auxiliary_state)
    output = BaseModelOutputWithPast(
        last_hidden_state=normal_hidden,
        past_key_values=past_key_values,
    )
    return output, auxiliary_hidden + 0.5 * normal_hidden, details


def choose_tokens(
    base_logits: torch.Tensor,
    auxiliary_logits: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if auxiliary_logits is None:
        return decode_logits(base_logits, args), [
            {
                "branch": "vanilla",
                "tvd": None,
                "max_abs_logit_delta": None,
                "decoding": args.decoding,
            }
            for _ in range(base_logits.shape[0])
        ]
    base_float = base_logits.float()
    auxiliary_float = auxiliary_logits.float()
    base_probs = torch.softmax(base_float, dim=-1)
    auxiliary_probs = torch.softmax(auxiliary_float, dim=-1)
    tvd = torch.sum(torch.abs(base_probs - auxiliary_probs), dim=-1)
    positive = tvd < args.only_tvd_gamma
    positive_logits = base_float + args.only_alpha_pos * auxiliary_float
    negative_logits = (
        (1.0 + args.only_alpha_neg) * base_float
        - args.only_alpha_neg * auxiliary_float
    )
    combined = torch.where(positive[:, None], positive_logits, negative_logits)
    cutoff = math.log(args.only_beta) + base_float.max(dim=-1, keepdim=True).values
    combined = combined.masked_fill(base_float < cutoff, -float("inf"))
    tokens = decode_logits(combined, args)
    decisions = []
    max_delta = torch.max(torch.abs(base_float - auxiliary_float), dim=-1).values
    for index in range(base_logits.shape[0]):
        decisions.append(
            {
                "branch": "positive" if bool(positive[index]) else "negative",
                "tvd": float(tvd[index].item()),
                "max_abs_logit_delta": float(max_delta[index].item()),
                "decoding": args.decoding,
            }
        )
    return tokens, decisions


@torch.inference_mode()
def generate_batch(
    model: LlavaForConditionalGeneration,
    processor: Any,
    inputs: dict[str, torch.Tensor],
    image_positions: list[list[int]],
    method: str,
    args: argparse.Namespace,
    state: RuntimeState,
    fp8_recipe: object | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    past_key_values: Optional[Cache] = None
    current_token: Optional[torch.Tensor] = None
    cache_position = torch.arange(
        prompt_length,
        device=input_ids.device,
        dtype=torch.long,
    )
    generated: list[torch.Tensor] = []
    decisions = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    for step in range(args.max_new_tokens):
        if step == 0:
            call_kwargs = {
                "inputs_embeds": inputs["inputs_embeds"],
                "input_ids": None,
                "attention_mask": attention_mask,
                "position_ids": inputs.get("position_ids"),
                "past_key_values": None,
                "cache_position": cache_position,
            }
        else:
            position_ids = inputs.get("position_ids")
            call_kwargs = {
                "inputs_embeds": None,
                "input_ids": current_token,
                "attention_mask": attention_mask,
                "position_ids": (
                    attention_mask.long().sum(dim=-1, keepdim=True)
                    if position_ids is not None
                    else None
                ),
                "past_key_values": past_key_values,
                "cache_position": cache_position,
            }
        def forward_step() -> tuple[
            BaseModelOutputWithPast,
            torch.Tensor,
            Optional[torch.Tensor],
            list[dict[str, Any]],
        ]:
            with fp8_context(state, fp8_recipe):
                if method == "only":
                    outputs, combined_hidden, head_details = only_language_forward(
                        model.language_model,
                        image_positions=image_positions,
                        enhance_layer=args.only_enhance_layer,
                        **call_kwargs,
                    )
                    base_logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
                    auxiliary_logits = model.lm_head(combined_hidden[:, -1, :])
                else:
                    outputs = model.language_model(
                        **call_kwargs,
                        use_cache=True,
                    )
                    base_logits = model.lm_head(
                        outputs.last_hidden_state[:, -1, :]
                    )
                    auxiliary_logits = None
                    head_details = [
                        {
                            "kept_heads": [],
                            "removed_heads": [],
                            "entropy_ratio": [],
                        }
                        for _ in range(batch_size)
                    ]
            return outputs, base_logits, auxiliary_logits, head_details

        try:
            outputs, base_logits, auxiliary_logits, head_details = forward_step()
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            # A failed prefill has not exposed its newly-created cache, so it
            # is safe to retry once with native BF16. Remove TE wrappers
            # instead of merely toggling a flag so their workspaces are
            # released before the retry. Decode failures are not retried
            # because a partially updated cache must not be reused.
            if (
                step != 0
                or not state.fp8_effective
                or not args.fallback
            ):
                raise
            disable_fp8(model, state, f"{type(exc).__name__}: {exc}")
            outputs, base_logits, auxiliary_logits, head_details = forward_step()
        next_tokens, token_decisions = choose_tokens(
            base_logits,
            auxiliary_logits,
            args,
        )
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, int(pad_id)),
            next_tokens,
        )
        generated.append(next_tokens)
        for row_index in range(batch_size):
            decision = dict(token_decisions[row_index])
            decision["token_id"] = int(next_tokens[row_index].item())
            if method == "only":
                decision.update(head_details[row_index])
            decisions[row_index].append(decision)
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            finished = finished | next_tokens.eq(int(eos_id))
        if bool(finished.all()):
            past_key_values = outputs.past_key_values
            break
        past_key_values = outputs.past_key_values
        current_token = next_tokens[:, None]
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        cache_position = torch.tensor(
            [prompt_length + step],
            dtype=torch.long,
            device=input_ids.device,
        )

    generated_tensor = torch.stack(generated, dim=1)
    answers = processor.batch_decode(
        generated_tensor,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [answer.strip() for answer in answers], [
        {
            "generated_tokens": len(decisions[index]),
            "steps": decisions[index],
        }
        for index in range(batch_size)
    ]


def load_runtime(
    args: argparse.Namespace,
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LlavaForConditionalGeneration, RuntimeState, object | None]:
    effective_attention, flash_version = choose_attention(args.attention, device)
    state = RuntimeState(
        requested_attention=args.attention,
        effective_attention=effective_attention,
        flash_attn_version=flash_version,
        dtype=str(dtype).replace("torch.", ""),
        tf32=args.tf32,
        fp8_requested=args.fp8,
        fp8_capability_ready=False,
        fp8_effective=False,
        fp8_scope=args.fp8_scope if args.fp8 else None,
        fp8_linear_count=0,
        fp8_calls=0,
        fp8_native_fallback_calls=0,
        fallback_events=[],
        image_feature_cache=args.image_feature_cache,
    )
    if device.type == "cuda":
        state.fp8_capability_ready = (
            torch.cuda.get_device_capability(device)[0] >= 9
            and args.fp8
        )
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map={"": device.index if device.index is not None else 0},
            attn_implementation=effective_attention,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        if not args.fallback or effective_attention == "eager":
            raise
        state.fallback_events.append(
            f"Attention fallback from {effective_attention}: "
            f"{type(exc).__name__}: {exc}"
        )
        effective_attention = "sdpa"
        state.effective_attention = effective_attention
        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map={"": device.index if device.index is not None else 0},
            attn_implementation=effective_attention,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    model.eval()

    fp8_recipe = None
    if args.fp8 and state.fp8_capability_ready:
        try:
            fp8_recipe, converted = enable_te_fp8(
                model,
                device,
                args.fp8_scope,
            )
            state.fp8_effective = True
            state.fp8_linear_count = converted
        except Exception as exc:
            if not args.fallback:
                raise
            state.fallback_events.append(
                f"FP8 disabled: {type(exc).__name__}: {exc}"
            )
            state.fp8_scope = None
    elif args.fp8:
        state.fallback_events.append(
            "FP8 disabled: visible device is not Hopper/Blackwell or CUDA is unavailable"
        )
    return model, state, fp8_recipe


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("num_workers must be >= 0 and prefetch_factor must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be positive when provided")
    if args.only_beta <= 0:
        raise ValueError("--only-beta must be positive")
    if args.method == "only" and args.only_enhance_layer < 0:
        raise ValueError("--only-enhance-layer must be non-negative")

    os.environ.setdefault("HF_HOME", str(HUB_ROOT.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HUB_ROOT))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HUB_ROOT))
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    configure_cuda(device, args.tf32)
    model_path = require_hub_path(args.model, "model")
    pope_path = require_hub_path(args.pope_path, "POPE file")
    images_root = require_hub_path(args.images_root, "image root")
    output_prefix = args.output.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model snapshot is incomplete: {model_path}")
    if not pope_path.is_file():
        raise FileNotFoundError(pope_path)
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    rows = load_rows(pope_path, args.start, args.limit)
    image_paths = sorted(
        {
            (images_root / row["image"]).resolve()
            for row in rows
        }
    )
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, first={missing[0]}")

    model, state, fp8_recipe = load_runtime(args, model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    if args.batch_size > 1 and processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    feature_started = time.perf_counter()
    if args.image_feature_cache:
        image_cache = build_image_feature_cache(
            model=model,
            processor=processor,
            image_paths=image_paths,
            device=device,
            dtype=dtype,
            batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            progress=args.progress,
        )
    else:
        raise ValueError(
            "This evaluator requires --image-feature-cache for the reproducible "
            "POPE protocol; disablement is intentionally unsupported."
        )
    feature_elapsed = time.perf_counter() - feature_started
    state.image_feature_count = len(image_cache)

    predictions: list[int] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    iterator = range(0, len(rows), args.batch_size)
    if args.progress:
        iterator = tqdm(iterator, desc=args.method)
    for start in iterator:
        batch_rows = rows[start : start + args.batch_size]
        inputs, image_positions = build_prompt_batch(
            model=model,
            processor=processor,
            rows=batch_rows,
            image_root=images_root,
            image_cache=image_cache,
            device=device,
            pope_answer_instruction=args.pope_answer_instruction,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        answers, generation = generate_batch(
            model=model,
            processor=processor,
            inputs=inputs,
            image_positions=image_positions,
            method=args.method,
            args=args,
            state=state,
            fp8_recipe=fp8_recipe,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for row, answer, generation_row in zip(batch_rows, answers, generation):
            prediction = answer_label(answer)
            label = 1 if row["label"].lower() == "yes" else 0
            predictions.append(prediction)
            labels.append(label)
            records.append(
                {
                    "question_id": row.get("question_id"),
                    "image": row["image"],
                    "question": row["text"],
                    "label": label,
                    "prediction": prediction,
                    "answer": answer,
                    "correct": prediction == label,
                    **generation_row,
                }
            )
        if args.progress:
            iterator.set_postfix(  # type: ignore[union-attr]
                acc=f"{compute_metrics(predictions, labels)['accuracy']:.4f}"
            )
    generation_elapsed = time.perf_counter() - generation_started

    if state.fp8_requested:
        state.fp8_calls, state.fp8_native_fallback_calls = fp8_call_counts(model)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_prefix.with_suffix(".jsonl")
    metrics_path = output_prefix.with_suffix(".metrics.json")
    predictions_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    all_steps = [
        step
        for record in records
        for step in record["steps"]
    ]
    summary = {
        "status": "PASS",
        "method": args.method,
        "model": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_revision": model_path.name,
        "pope_path": str(pope_path),
        "pope_sha256": sha256_file(pope_path),
        "images_root": str(images_root),
        "count": len(records),
        "start": args.start,
        "limit": args.limit,
        "protocol": {
            "dtype": state.dtype,
            "attention": state.effective_attention,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "prompt_template": (
                "{system_prompt} USER: <image>\\n{question}"
                " Please answer this question with one word. ASSISTANT:"
                if args.pope_answer_instruction
                else "{system_prompt} USER: <image>\\n{question} ASSISTANT:"
            ),
            "pope_answer_instruction": args.pope_answer_instruction,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "image_batch_size": args.image_batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
        },
        "acceleration": state.as_dict(),
        "only": (
            {
                "enhance_layer": args.only_enhance_layer,
                "alpha_pos": args.only_alpha_pos,
                "alpha_neg": args.only_alpha_neg,
                "beta": args.only_beta,
                "tvd_gamma": args.only_tvd_gamma,
            }
            if args.method == "only"
            else None
        ),
        "metrics": compute_metrics(predictions, labels),
        "runtime": {
            "feature_cache_seconds": feature_elapsed,
            "generation_seconds": generation_elapsed,
            "total_seconds": feature_elapsed + generation_elapsed,
            "rows_per_second": (
                len(records) / generation_elapsed if generation_elapsed else 0.0
            ),
            "generated_tokens": sum(
                record["generated_tokens"] for record in records
            ),
            "only_changed_steps": sum(
                step.get("max_abs_logit_delta") is not None
                and step.get("max_abs_logit_delta", 0.0) > 1e-5
                for step in all_steps
            ),
            "only_positive_steps": sum(
                step.get("branch") == "positive" for step in all_steps
            ),
            "only_negative_steps": sum(
                step.get("branch") == "negative" for step in all_steps
            ),
        },
        "git_sha": git_sha(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "predictions_path": str(predictions_path),
    }
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
