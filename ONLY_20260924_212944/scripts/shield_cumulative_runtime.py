"""Shared feature preparation for cumulative SHIELD ablations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from shield_vulnerability_runtime import (  # noqa: E402
    build_attacked_pixel_cache,
    build_llava_pixel_cache,
    build_vulnerability_prompt_batch,
    load_caption_lookup,
)

from methods.shield_inherent_bias import (  # noqa: E402
    InherentBiasConfig,
    ShieldInherentBias,
    select_vision_features,
)
from methods.shield_pipeline import ShieldComponentPipeline  # noqa: E402


def process_input_caption(caption: str) -> str:
    """Match SHIELD's caption string used for token injection."""

    processed = ".".join(caption.replace("\n\n", "").split(".")[:-1])
    return processed + ". "


@torch.inference_mode()
def build_vision_feature_cache(
    model: Any,
    image_paths: list[Path],
    pixel_values_by_path: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    progress: bool,
    desc: str,
) -> dict[str, torch.Tensor]:
    """Extract raw LLaVA vision-tower features before the multimodal projector."""

    cache: dict[str, torch.Tensor] = {}
    iterator = range(0, len(image_paths), batch_size)
    if progress:
        iterator = tqdm(iterator, desc=desc)
    for start in iterator:
        batch_paths = image_paths[start : start + batch_size]
        pixel_values = torch.stack(
            [pixel_values_by_path[str(path)] for path in batch_paths],
            dim=0,
        ).to(device=device, dtype=dtype)
        outputs = model.vision_tower(
            pixel_values,
            output_hidden_states=True,
        )
        features = select_vision_features(model, outputs)
        for path, feature in zip(batch_paths, features, strict=True):
            cache[str(path)] = feature.detach()
    if len(cache) != len(image_paths):
        raise RuntimeError(f"{desc} cache incomplete: {len(cache)}/{len(image_paths)}")
    return cache


@torch.inference_mode()
def build_noise_bias_feature(
    model: Any,
    reference_pixels: torch.Tensor,
    sample_num: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    """Estimate SHIELD's inherent bias through the Inherent Bias plugin."""

    return ShieldInherentBias(
        InherentBiasConfig(sample_num=sample_num, seed=seed)
    ).estimate(
        model,
        reference_pixels,
        device,
        dtype,
    )


@torch.inference_mode()
def build_clip_feature_cache(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    rows: list[dict[str, Any]],
    captions: dict[str, str],
    device: torch.device,
    batch_size: int,
    progress: bool,
) -> dict[str, torch.Tensor]:
    """Cache CLIP token features for the caption associated with each image."""

    cache: dict[str, torch.Tensor] = {}
    iterator = range(0, len(rows), batch_size)
    if progress:
        iterator = tqdm(iterator, desc="clip-text-features")
    for start in iterator:
        batch_rows = rows[start : start + batch_size]
        texts = [
            process_input_caption(captions[str(row["image"])])
            for row in batch_rows
        ]
        inputs = clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = clip_model.text_model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
        hidden = outputs.last_hidden_state[:, 1:]
        lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        for row, feature, length in zip(batch_rows, hidden, lengths, strict=True):
            token_count = max(int(length) - 1, 0)
            cache[str(row["image"])] = feature[:token_count].detach()
    expected = {str(row["image"]) for row in rows}
    if set(cache) != expected:
        raise RuntimeError(
            f"CLIP feature cache incomplete: {len(cache)}/{len(expected)}"
        )
    return cache


@torch.inference_mode()
def build_projected_feature_cache(
    model: Any,
    image_paths: list[Path],
    raw_features_by_path: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    progress: bool,
    desc: str,
) -> dict[str, torch.Tensor]:
    """Project raw vision features into the language-model image-token space."""

    cache: dict[str, torch.Tensor] = {}
    iterator = range(0, len(image_paths), batch_size)
    if progress:
        iterator = tqdm(iterator, desc=desc)
    for start in iterator:
        batch_paths = image_paths[start : start + batch_size]
        raw_features = torch.stack(
            [raw_features_by_path[str(path)] for path in batch_paths],
            dim=0,
        ).to(device=device, dtype=dtype)
        projected = model.multi_modal_projector(raw_features)
        for path, feature in zip(batch_paths, projected, strict=True):
            cache[str(path)] = feature.detach()
    if len(cache) != len(image_paths):
        raise RuntimeError(f"{desc} cache incomplete: {len(cache)}/{len(image_paths)}")
    return cache


@torch.inference_mode()
def build_clean_shield_caches(
    model: Any,
    processor: Any,
    image_paths: list[Path],
    raw_features_by_path: dict[str, torch.Tensor],
    clip_features_by_image: dict[str, torch.Tensor],
    captions: dict[str, str],
    bias_features: torch.Tensor | None,
    pipeline: ShieldComponentPipeline,
    device: torch.device,
    dtype: torch.dtype,
    progress: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    """Build clean-branch features and optional caption-token embeddings."""

    if not pipeline.requires_feature_transform:
        raise ValueError(
            "Clean SHIELD cache requires a statistical_bias or inherent_bias plugin"
        )
    if pipeline.statistical is not None and not clip_features_by_image:
        raise ValueError("Statistical Bias clean preparation requires CLIP features")
    if pipeline.caption_token_injection and not clip_features_by_image:
        raise ValueError("Caption injection requires Statistical Bias features")
    projected_cache: dict[str, torch.Tensor] = {}
    caption_cache: dict[str, torch.Tensor] = {}
    selected_counts: dict[str, int] = {}
    iterator: Any = image_paths
    if progress:
        iterator = tqdm(image_paths, desc="clean-shield-features")
    for path in iterator:
        image_name = path.name
        raw = raw_features_by_path[str(path)]
        clip_features = clip_features_by_image.get(image_name)
        enhanced, selected = pipeline.transform_clean_features(
            raw,
            caption_features=clip_features,
            bias_features=bias_features,
        )
        projected = model.multi_modal_projector(
            enhanced.unsqueeze(0).to(device=device, dtype=dtype)
        ).squeeze(0)
        projected_cache[str(path)] = projected.detach()
        selected_counts[image_name] = int(selected.numel())

        if pipeline.caption_token_injection:
            if clip_features is None:
                raise ValueError(
                    f"Missing Statistical Bias features for {image_name}"
                )
            caption_text = process_input_caption(captions[image_name])
            caption_ids = processor.tokenizer(
                caption_text,
                return_tensors="pt",
            )["input_ids"][0].to(device)
            caption_embeds = model.get_input_embeddings()(caption_ids)
            selected_embeds = pipeline.map_caption_tokens(
                selected,
                int(clip_features.shape[0]),
                caption_embeds,
            )
            caption_cache[image_name] = selected_embeds.detach()
    return projected_cache, caption_cache, selected_counts


__all__ = [
    "build_attacked_pixel_cache",
    "build_clean_shield_caches",
    "build_clip_feature_cache",
    "build_llava_pixel_cache",
    "build_noise_bias_feature",
    "build_projected_feature_cache",
    "build_vulnerability_prompt_batch",
    "build_vision_feature_cache",
    "load_caption_lookup",
]
