#!/usr/bin/env python3
"""Compare accelerated Vanilla and ONLY on the COCO POPE adversarial split.

The runner ports the original ONLY mechanism to Qwen2.5-VL while sharing the
project acceleration protocol:

* BF16, FlashAttention-2, and adaptive Transformer Engine FP8 text MLPs;
* batched multimodal prefill and cached incremental decoding;
* per-row image/text attention head selection in the ONLY branch;
* the original auxiliary-state and TVD-gated logit intervention.

The enhance layer and the final layer keep a small manual-attention island
because ONLY needs attention weights. All other decoder layers use the
selected native attention backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Optional

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen25

try:
    from infer_qwen25_vl import (
        configure_cuda,
        fp8_context,
        get_fp8_call_counts,
        load_runtime,
    )
except ModuleNotFoundError:
    from scripts.infer_qwen25_vl import (
        configure_cuda,
        fp8_context,
        get_fp8_call_counts,
        load_runtime,
    )


DEFAULT_MODEL = (
    "/data/lcq/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
    "/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)
DEFAULT_POPE = (
    "/data/lcq/.cache/huggingface/hub/pope/coco/coco_pope_adversarial.json"
)
DEFAULT_IMAGES = "/data/lcq/.cache/huggingface/hub/coco2014/val2014"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vanilla", "only"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pope-path", default=DEFAULT_POPE)
    parser.add_argument("--images-root", default=DEFAULT_IMAGES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attention",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default="auto",
    )
    parser.add_argument(
        "--fp8",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--decoding",
        choices=("sample", "greedy"),
        default="sample",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-enhance-layer", type=int, default=0)
    parser.add_argument("--only-alpha-pos", type=float, default=3.0)
    parser.add_argument("--only-alpha-neg", type=float, default=1.0)
    parser.add_argument("--only-beta", type=float, default=0.1)
    parser.add_argument("--only-tvd-gamma", type=float, default=0.1)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except Exception:
        return None


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_rows(path: Path, start: int, limit: Optional[int]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    end = None if limit is None else start + limit
    selected = rows[start:end]
    if not selected:
        raise ValueError(f"No POPE rows selected from {path} with start={start}, limit={limit}")
    return selected


def answer_label(text: str) -> int:
    """Match the legacy recorder: any explicit no/not on the first answer line is no."""
    first_line = text.splitlines()[0] if text.splitlines() else text
    normalized = first_line.replace(".", " ").replace(",", " ").lower()
    if re.search(r"\b(no|not)\b|n't", normalized):
        return 0
    return 1


def metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
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


def _attention_states(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Optional[Cache],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    image_positions: list[list[int]],
    valid_token_mask: Optional[torch.Tensor],
    capture_auxiliary: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[dict[str, Any]]]]:
    """Run the modern Qwen attention and optionally build the ONLY branch."""
    batch_size, query_length, _ = hidden_states.shape
    query_states = attention.q_proj(hidden_states)
    key_states = attention.k_proj(hidden_states)
    value_states = attention.v_proj(hidden_states)

    query_states = query_states.view(
        batch_size, query_length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, attention.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, attention.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen25.apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        attention.rope_scaling["mrope_section"],
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

    repeated_keys = qwen25.repeat_kv(key_states, attention.num_key_value_groups)
    repeated_values = qwen25.repeat_kv(value_states, attention.num_key_value_groups)
    attention_weights = torch.matmul(
        query_states,
        repeated_keys.transpose(2, 3),
    ) * attention.scaling
    key_length = repeated_keys.shape[-2]
    if attention_mask is not None and attention_mask.ndim == 2:
        valid_keys = attention_mask[:, :key_length].bool()
        if valid_keys.shape[-1] < key_length:
            valid_keys = torch.nn.functional.pad(
                valid_keys,
                (0, key_length - valid_keys.shape[-1]),
                value=True,
            )
        query_positions = cache_position
        if query_positions is None or query_positions.numel() != query_length:
            query_positions = torch.arange(
                key_length - query_length,
                key_length,
                device=hidden_states.device,
                dtype=torch.long,
            )
        key_positions = torch.arange(
            key_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
        allowed = (
            key_positions[None, None, :] <= query_positions[None, :, None]
        ) & valid_keys[:, None, :]
        manual_mask = torch.zeros(
            (batch_size, 1, query_length, key_length),
            dtype=attention_weights.dtype,
            device=attention_weights.device,
        )
        manual_mask.masked_fill_(
            ~allowed[:, None, :, :],
            torch.finfo(attention_weights.dtype).min,
        )
        attention_weights = attention_weights + manual_mask
    elif attention_mask is not None:
        attention_weights = attention_weights + attention_mask[
            :, :, :, :key_length
        ]
    else:
        query_positions = cache_position
        if query_positions is None or query_positions.numel() != query_length:
            query_positions = torch.arange(
                key_length - query_length,
                key_length,
                device=hidden_states.device,
                dtype=torch.long,
            )
        key_positions = torch.arange(
            key_length,
            device=hidden_states.device,
            dtype=torch.long,
        )
        causal = key_positions[None, :] <= query_positions[:, None]
        manual_mask = torch.zeros(
            (1, 1, query_length, key_length),
            dtype=attention_weights.dtype,
            device=attention_weights.device,
        )
        manual_mask.masked_fill_(
            ~causal[None, None, :, :],
            torch.finfo(attention_weights.dtype).min,
        )
        attention_weights = attention_weights + manual_mask
    attention_weights = torch.softmax(
        attention_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query_states.dtype)
    normal_context = torch.matmul(attention_weights, repeated_values)
    normal_context = normal_context.transpose(1, 2).contiguous()
    normal_output = attention.o_proj(
        normal_context.reshape(batch_size, query_length, -1)
    )

    if not capture_auxiliary:
        return normal_output, None, None

    if valid_token_mask is None:
        valid_mask = torch.ones(
            (batch_size, key_length),
            dtype=torch.bool,
            device=hidden_states.device,
        )
    else:
        valid_mask = valid_token_mask[:, :key_length].bool()
    image_mask = torch.zeros(
        (batch_size, key_length),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    for row_index in range(batch_size):
        row_positions = (
            image_positions[row_index]
            if row_index < len(image_positions)
            else []
        )
        valid_positions = [
            pos for pos in row_positions if 0 <= pos < key_length
        ]
        if valid_positions:
            image_mask[row_index, torch.tensor(
                valid_positions,
                device=image_mask.device,
            )] = True
    image_mask &= valid_mask
    text_mask = valid_mask & ~image_mask
    has_image = image_mask.any(dim=-1)

    def suppress_group(
        weights: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = group_mask[:, None, None, :]
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        selected = weights.masked_fill(~mask, 0.0)
        mean = selected.sum(dim=-1, keepdim=True) / count
        variance = (
            ((selected - mean).masked_fill(~mask, 0.0) ** 2).sum(
                dim=-1,
                keepdim=True,
            )
            / (count - 1).clamp_min(1)
        )
        threshold = mean + torch.sqrt(variance)
        return torch.where(
            mask & (weights > threshold),
            torch.zeros_like(weights),
            selected,
        )

    text_clean = suppress_group(attention_weights, text_mask)
    image_clean = suppress_group(attention_weights, image_mask)
    text_last = text_clean[:, :, -1, :]
    image_last = image_clean[:, :, -1, :]
    text_norm = text_last / text_last.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    image_norm = image_last / image_last.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    text_entropy = -torch.sum(text_norm * torch.log(text_norm + 1e-6), dim=-1)
    image_entropy = -torch.sum(image_norm * torch.log(image_norm + 1e-6), dim=-1)
    ratio = torch.nan_to_num(
        text_entropy / (image_entropy + 1e-6),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ratio = torch.where(has_image[:, None], ratio, torch.ones_like(ratio))
    keep = ratio >= ratio.mean(dim=-1, keepdim=True)
    keep = torch.where(
        has_image[:, None],
        keep,
        torch.ones_like(keep, dtype=torch.bool),
    )
    auxiliary_weights = (
        attention_weights
        * valid_mask[:, None, None, :].to(attention_weights.dtype)
        * keep[:, :, None, None].to(attention_weights.dtype)
    )
    auxiliary_context = torch.matmul(auxiliary_weights, repeated_values)
    auxiliary_context = auxiliary_context.transpose(1, 2).contiguous()
    auxiliary_output = attention.o_proj(
        auxiliary_context.reshape(batch_size, query_length, -1)
    )
    details = []
    for row_index in range(batch_size):
        keep_row = keep[row_index].detach().cpu().tolist()
        ratio_row = ratio[row_index].detach().float().cpu().tolist()
        details.append(
            {
                "kept_heads": [
                    i for i, value in enumerate(keep_row) if value
                ],
                "removed_heads": [
                    i for i, value in enumerate(keep_row) if not value
                ],
                "entropy_ratio": ratio_row,
            }
        )
    return normal_output, auxiliary_output, details


def _decoder_layer_with_auxiliary(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_values: Optional[Cache],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    image_positions: list[list[int]],
    valid_token_mask: Optional[torch.Tensor],
    capture_auxiliary: bool,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[list[dict[str, Any]]],
    Optional[torch.Tensor],
]:
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    normal_attention, auxiliary_attention, details = _attention_states(
        layer.self_attn,
        normalized,
        attention_mask,
        position_embeddings,
        past_key_values,
        use_cache,
        cache_position,
        image_positions,
        valid_token_mask,
        capture_auxiliary=capture_auxiliary,
    )
    hidden_states = residual + normal_attention
    residual = hidden_states
    normalized = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(normalized)
    attention_weights = None
    return hidden_states, auxiliary_attention, details, attention_weights


def _last_decoder_layer_with_auxiliary(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    auxiliary_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_values: Optional[Cache],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    valid_token_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_ids
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    normal_attention, _, _ = _attention_states(
        layer.self_attn,
        normalized,
        attention_mask,
        position_embeddings,
        past_key_values,
        use_cache,
        cache_position,
        [],
        valid_token_mask,
        capture_auxiliary=False,
    )
    hidden_states = residual + normal_attention
    residual = hidden_states
    normalized = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(normalized)

    auxiliary_state = layer.input_layernorm(auxiliary_state)
    auxiliary_state = 0.2 * normal_attention + auxiliary_state
    auxiliary_residual = auxiliary_state
    auxiliary_state = layer.post_attention_layernorm(auxiliary_state)
    auxiliary_state = auxiliary_residual + layer.mlp(auxiliary_state)
    return hidden_states, auxiliary_state


def _only_text_forward(
    self: torch.nn.Module,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Any,
) -> BaseModelOutputWithPast:
    """Modern Qwen2.5-VL text forward with the legacy ONLY auxiliary branch."""
    del input_ids, kwargs
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        raise ValueError("The ONLY text forward expects inputs_embeds from Qwen2.5-VL")
    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    valid_token_mask = (
        attention_mask.bool()
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2
        else None
    )
    if isinstance(attention_mask, dict):
        causal_mask_mapping = attention_mask
    else:
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": qwen25.create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = qwen25.create_sliding_window_causal_mask(
                **mask_kwargs
            )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    auxiliary_state: Optional[torch.Tensor] = None
    layer_details: Optional[list[dict[str, Any]]] = None
    enhance_layer = int(getattr(self, "_only_enhance_layer", 0))
    last_layer = len(self.layers) - 1
    image_positions = list(getattr(self, "_only_image_positions", []))

    for layer_index, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_mask = causal_mask_mapping[decoder_layer.attention_type]
        if layer_index == enhance_layer:
            hidden_states, auxiliary_state, layer_details, _ = _decoder_layer_with_auxiliary(
                decoder_layer,
                hidden_states,
                layer_mask,
                text_position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings,
                image_positions,
                valid_token_mask,
                capture_auxiliary=True,
            )
        elif layer_index == last_layer:
            if auxiliary_state is None:
                raise RuntimeError("ONLY auxiliary state was not captured before the last layer")
            hidden_states, auxiliary_state = _last_decoder_layer_with_auxiliary(
                decoder_layer,
                hidden_states,
                auxiliary_state,
                layer_mask,
                text_position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings,
                valid_token_mask,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            continue

        if output_attentions:
            all_self_attns += (None,)

    normal_hidden = self.norm(hidden_states)
    if auxiliary_state is None:
        raise RuntimeError("ONLY auxiliary state is missing at final normalization")
    auxiliary_hidden = self.norm(auxiliary_state)
    combined_hidden = auxiliary_hidden + 0.5 * normal_hidden
    self._only_last_hidden_state = combined_hidden
    self._only_last_layer_details = layer_details or [
        {
            "kept_heads": [],
            "removed_heads": [],
            "entropy_ratio": [],
        }
        for _ in range(hidden_states.shape[0])
    ]

    if output_hidden_states:
        all_hidden_states += (normal_hidden,)
    if not return_dict:
        return (
            normal_hidden,
            past_key_values,
            all_hidden_states,
            all_self_attns,
        )
    return BaseModelOutputWithPast(
        last_hidden_state=normal_hidden,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def install_only(
    model: Qwen2_5_VLForConditionalGeneration,
    image_positions: list[list[int]],
    enhance_layer: int,
) -> torch.nn.Module:
    language_model = model.model.language_model
    language_model._only_image_positions = image_positions
    language_model._only_enhance_layer = enhance_layer
    language_model._only_last_hidden_state = None
    language_model._only_last_layer_details = []
    language_model.forward = MethodType(_only_text_forward, language_model)
    return language_model


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
    logits = logits.float()
    if args.decoding == "greedy":
        return logits.argmax(dim=-1)
    logits = logits / args.temperature
    logits = filter_top_k(logits, args.top_k)
    logits = filter_top_p(logits, args.top_p)
    return torch.multinomial(
        torch.softmax(logits, dim=-1),
        num_samples=1,
    ).squeeze(-1)


def select_next_tokens(
    base_logits: torch.Tensor,
    only_logits: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if only_logits is None:
        tokens = decode_logits(base_logits, args)
        return tokens, [
            {
                "tvd": None,
                "branch": "vanilla",
                "max_abs_logit_delta": None,
                "decoding": args.decoding,
            }
            for _ in range(base_logits.shape[0])
        ]

    base_float = base_logits.float()
    only_float = only_logits.float()
    base_probs = torch.softmax(base_float, dim=-1)
    only_probs = torch.softmax(only_float, dim=-1)
    tvd = torch.sum(torch.abs(base_probs - only_probs), dim=-1)
    positive = tvd < args.only_tvd_gamma
    positive_logits = base_float + args.only_alpha_pos * only_float
    negative_logits = (
        (1.0 + args.only_alpha_neg) * base_float
        - args.only_alpha_neg * only_float
    )
    combined = torch.where(positive[:, None], positive_logits, negative_logits)
    cutoff = math.log(args.only_beta) + base_float.max(dim=-1, keepdim=True).values
    combined = combined.masked_fill(base_float < cutoff, -float("inf"))
    tokens = decode_logits(combined, args)
    max_delta = torch.max(torch.abs(base_float - only_float), dim=-1).values
    return tokens, [
        {
            "tvd": float(tvd[index].item()),
            "branch": "positive" if bool(positive[index]) else "negative",
            "max_abs_logit_delta": float(max_delta[index].item()),
            "decoding": args.decoding,
        }
        for index in range(base_logits.shape[0])
    ]


class PopeDataset(Dataset[tuple[dict[str, Any], Image.Image]]):
    def __init__(self, rows: list[dict[str, Any]], images_root: Path) -> None:
        self.rows = rows
        self.images_root = images_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], Image.Image]:
        row = self.rows[index]
        image_path = self.images_root / row["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            return row, image.convert("RGB")


def pope_collate(
    batch: list[tuple[dict[str, Any], Image.Image]],
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    return [item[0] for item in batch], [item[1] for item in batch]


def move_inputs(
    inputs: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if device.type == "cuda":
            value = value.pin_memory()
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value.to(device)
    return moved


def build_inputs_batch(
    processor: AutoProcessor,
    rows: list[dict[str, Any]],
    images: list[Image.Image],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"{row['text']} Answer:",
                    },
                ],
            }
        ]
        for row in rows
    ]
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return move_inputs(inputs, device)


def generate_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    inputs: dict[str, torch.Tensor],
    method: str,
    args: argparse.Namespace,
    language_model: Optional[torch.nn.Module],
    state: Any,
    fp8_recipe: object | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    model_kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
    model_kwargs["use_cache"] = True
    model_kwargs["logits_to_keep"] = 1
    model_kwargs["cache_position"] = torch.arange(
        prompt_length,
        device=input_ids.device,
        dtype=torch.long,
    )
    eos_id = model.generation_config.eos_token_id
    eos_ids = {int(eos_id)} if isinstance(eos_id, int) else {int(value) for value in (eos_id or [])}
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    batch_size = input_ids.shape[0]
    generated_steps = [[] for _ in range(batch_size)]
    generated_tokens: list[torch.Tensor] = []
    current_ids = input_ids
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )

    for _ in range(args.max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(current_ids, **model_kwargs)
        with fp8_context(state, fp8_recipe):
            outputs = model(**model_inputs, return_dict=True)
        base_logits = outputs.logits[:, -1, :]
        only_logits = None
        if method == "only":
            if language_model is None or language_model._only_last_hidden_state is None:
                raise RuntimeError("ONLY forward did not produce auxiliary hidden states")
            auxiliary_hidden = language_model._only_last_hidden_state
            only_logits = model.lm_head(auxiliary_hidden[:, -1, :])
        next_tokens, decisions = select_next_tokens(
            base_logits,
            only_logits,
            args,
        )
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, int(pad_id)),
            next_tokens,
        )
        generated_tokens.append(next_tokens)
        for row_index, decision in enumerate(decisions):
            decision["token_id"] = int(next_tokens[row_index].item())
            if method == "only":
                details = language_model._only_last_layer_details[row_index]
                decision.update(details)
            generated_steps[row_index].append(decision)
        current_ids = torch.cat([current_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=False,
        )
        for eos in eos_ids:
            finished = finished | next_tokens.eq(eos)
        if bool(finished.all()):
            break

    answers = processor.batch_decode(
        torch.stack(generated_tokens, dim=1),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [answer.strip() for answer in answers], [
        {
            "generated_tokens": len(generated_steps[index]),
            "steps": generated_steps[index],
        }
        for index in range(batch_size)
    ]


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.method == "only" and args.only_beta <= 0:
        raise ValueError("--only-beta must be positive")
    if not (0 <= args.only_enhance_layer):
        raise ValueError("--only-enhance-layer must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.image_batch_size < 1:
        raise ValueError("--image-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.decoding == "sample" and args.temperature <= 0:
        raise ValueError("--temperature must be positive for sampling")
    if not (0 < args.top_p <= 1):
        raise ValueError("--top-p must be in (0, 1]")

    os.environ.setdefault("HF_HOME", "/data/lcq/.cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/data/lcq/.cache/huggingface/hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/lcq/.cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data/lcq/.cache/huggingface/hub")
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_cuda(device, args.tf32)
    model_path = Path(args.model).expanduser().resolve()
    pope_path = Path(args.pope_path).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model snapshot is incomplete: {model_path}")
    if not pope_path.is_file():
        raise FileNotFoundError(pope_path)
    rows = load_rows(pope_path, args.start, args.limit)
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)

    # The shared runtime loader owns the acceleration capability/effectiveness
    # record. This evaluator intentionally keeps compile disabled because POPE
    # uses variable prompt lengths and the ONLY branch has a manual island.
    args.compile = False
    args.cache = "dynamic"
    args.compile_mode = "reduce-overhead"
    model, state, fp8_recipe = load_runtime(args, model_path, device)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "left"

    predictions: list[int] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": pope_collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(PopeDataset(rows, images_root), **loader_kwargs)
    iterable = tqdm(loader, desc=args.method, disable=not args.progress)
    for batch_rows, batch_images in iterable:
        inputs = build_inputs_batch(processor, batch_rows, batch_images, device)
        language_model: Optional[torch.nn.Module] = None
        image_positions: list[list[int]] = []
        if args.method == "only":
            image_token_id = int(model.config.image_token_id)
            valid_mask = inputs.get("attention_mask")
            for row_index in range(len(batch_rows)):
                row_ids = inputs["input_ids"][row_index]
                positions = (row_ids == image_token_id).nonzero(
                    as_tuple=False
                ).flatten().tolist()
                if torch.is_tensor(valid_mask) and valid_mask.ndim == 2:
                    positions = [
                        position
                        for position in positions
                        if bool(valid_mask[row_index, position].item())
                    ]
                image_positions.append(positions)
            language_model = install_only(
                model,
                image_positions=image_positions,
                enhance_layer=args.only_enhance_layer,
            )

        with torch.inference_mode():
            answers, generations = generate_batch(
                model,
                processor,
                inputs,
                args.method,
                args,
                language_model,
                state,
                fp8_recipe,
            )
        for row, answer, generation in zip(batch_rows, answers, generations):
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
                    **generation,
                }
            )
        if iterable is not None:
            iterable.set_postfix(acc=f"{metrics(predictions, labels)['accuracy']:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path.with_suffix(".jsonl")
    metrics_path = output_path.with_suffix(".metrics.json")
    predictions_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
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
            "dtype": args.dtype,
            "attention": state.effective_attention,
            "requested_attention": state.requested_attention,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "prompt_suffix": " Answer:",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "image_batch_size": args.image_batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "cache": state.cache,
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
        "metrics": metrics(predictions, labels),
        "elapsed_seconds": elapsed_seconds,
        "rows_per_second": len(records) / elapsed_seconds if elapsed_seconds else 0.0,
        "git_sha": git_sha(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "predictions_path": str(predictions_path),
        "image_feature_cache": {
            "enabled": False,
            "reason": "Qwen2.5-VL native multimodal path; batch image prefetch only",
        },
    }
    if state.fp8_requested:
        fp8_calls, native_calls = get_fp8_call_counts(model)
        summary["acceleration"]["fp8_calls"] = fp8_calls
        summary["acceleration"]["fp8_native_fallback_calls"] = native_calls
    if args.method == "only":
        all_steps = [step for record in records for step in record["steps"]]
        summary["only_runtime"] = {
            "changed_first_token": sum(
                step.get("max_abs_logit_delta", 0.0) is not None
                and step.get("max_abs_logit_delta", 0.0) > 1e-5
                for step in all_steps
            ),
            "positive_steps": sum(step.get("branch") == "positive" for step in all_steps),
            "negative_steps": sum(step.get("branch") == "negative" for step in all_steps),
            "max_abs_logit_delta": max(
                (step.get("max_abs_logit_delta") or 0.0 for step in all_steps),
                default=0.0,
            ),
        }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
