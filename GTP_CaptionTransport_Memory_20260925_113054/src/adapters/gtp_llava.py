from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from transformers import LlavaForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from adapters.chair_metrics import SYNONYMS_TEXT
from methods.gtp import (
    GroundingMap,
    GTPMethod,
    amplify_patch_features,
    build_sparse_grounding_map,
)
from models.base import BaseMLLMWrapper
from utils.chair_singularize import singularize


def _build_claim_aliases() -> tuple[dict[tuple[str, ...], str], int]:
    aliases: dict[tuple[str, ...], str] = {}
    max_length = 1
    for line in SYNONYMS_TEXT.splitlines():
        values = [value.strip().lower() for value in line.split(",") if value.strip()]
        if not values:
            continue
        canonical = values[0]
        for value in values:
            tokens = tuple(singularize(word) for word in re.findall(r"[a-z]+", value))
            if tokens:
                aliases[tokens] = canonical
                max_length = max(max_length, len(tokens))
    return aliases, max_length


_CLAIM_ALIASES, _MAX_ALIAS_LENGTH = _build_claim_aliases()


def extract_coco_claims(text: str) -> list[str]:
    """Extract COCO-object claims without consulting image annotations."""

    tokens = [singularize(word) for word in re.findall(r"[a-z]+", text.lower())]
    claims: list[str] = []
    seen: set[str] = set()
    cursor = 0
    while cursor < len(tokens):
        matched = False
        for length in range(min(_MAX_ALIAS_LENGTH, len(tokens) - cursor), 0, -1):
            phrase = tuple(tokens[cursor : cursor + length])
            canonical = _CLAIM_ALIASES.get(phrase)
            if canonical is not None:
                if canonical not in seen:
                    claims.append(canonical)
                    seen.add(canonical)
                cursor += length
                matched = True
                break
        if not matched:
            cursor += 1
    return claims


def inject_caption_memory(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    image_positions: list[list[int]],
    memories: list[torch.Tensor],
    *,
    memory_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Insert a fixed-width, masked caption-memory block after image patches."""

    batch_size, prompt_length, hidden_size = inputs_embeds.shape
    if attention_mask.shape != (batch_size, prompt_length):
        raise ValueError("attention_mask must match the prompt batch dimensions")
    if len(image_positions) != batch_size or len(memories) != batch_size:
        raise ValueError("image_positions and memories must match the batch size")
    if memory_slots < 1:
        raise ValueError("memory_slots must be positive")

    valid_lengths = attention_mask.sum(dim=-1).tolist()
    output_length = max(int(length) for length in valid_lengths) + memory_slots
    output_embeds = inputs_embeds.new_zeros((batch_size, output_length, hidden_size))
    output_mask = attention_mask.new_zeros((batch_size, output_length))
    memory_counts: list[int] = []

    for index, memory in enumerate(memories):
        valid_positions = attention_mask[index].nonzero(as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError("caption memory cannot be inserted into an empty prompt")
        if not image_positions[index]:
            raise ValueError("caption memory requires at least one image position")
        if memory.ndim != 2 or memory.shape[1] != hidden_size:
            raise ValueError("each caption memory must have shape [tokens, hidden_size]")
        if memory.shape[0] > memory_slots:
            raise ValueError("caption memory exceeds the configured slot capacity")

        valid_start = int(valid_positions[0].item())
        image_end = max(image_positions[index])
        if image_end < valid_start or image_end >= prompt_length:
            raise ValueError("image positions must lie inside the unpadded prompt")

        row_embeds = inputs_embeds[index, valid_start:]
        row_mask = attention_mask[index, valid_start:]
        local_image_end = image_end - valid_start
        memory = memory.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        memory_count = int(memory.shape[0])
        memory_padding = inputs_embeds.new_zeros((memory_slots - memory_count, hidden_size))
        memory_mask = attention_mask.new_zeros(memory_slots)
        memory_mask[:memory_count] = 1
        fused_embeds = torch.cat(
            (
                row_embeds[: local_image_end + 1],
                memory,
                memory_padding,
                row_embeds[local_image_end + 1 :],
            ),
            dim=0,
        )
        fused_mask = torch.cat(
            (
                row_mask[: local_image_end + 1],
                memory_mask,
                row_mask[local_image_end + 1 :],
            ),
            dim=0,
        )
        left_padding = output_length - fused_embeds.shape[0]
        if left_padding < 0:
            raise RuntimeError("caption-memory padding calculation became negative")
        output_embeds[index, left_padding:] = fused_embeds
        output_mask[index, left_padding:] = fused_mask
        memory_counts.append(memory_count)

    return output_embeds, output_mask, memory_counts


class LlavaGTPBatchBackend(BaseMLLMWrapper):
    """LLaVA-specific feature insertion and dual-cache GTP decoder."""

    def __init__(
        self,
        *,
        model_path: str,
        model: LlavaForConditionalGeneration,
        processor: Any,
        image_cache: Mapping[str, torch.Tensor],
        images_root: Path,
        device: torch.device,
        max_new_tokens: int,
        decoding: str,
        temperature: float,
        top_p: float,
        top_k: int | None,
        forward_context: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        super().__init__(model_path)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if decoding not in {"sample", "greedy"}:
            raise ValueError("decoding must be 'sample' or 'greedy'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when provided")
        self.model = model
        self.processor = processor
        self.image_cache = image_cache
        self.images_root = images_root
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.decoding = decoding
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.forward_context = forward_context

    def setup(self, device: torch.device) -> None:
        if device != self.device:
            raise ValueError(f"backend was initialized on {self.device}, not {device}")

    def generate_response(self, *, prompt: str, image_path: str | None) -> str:
        raise NotImplementedError("LlavaGTPBatchBackend is only used through generate_gtp_batch.")

    @torch.inference_mode()
    def generate_gtp_batch(
        self,
        batch: Mapping[str, Any],
        *,
        method: GTPMethod,
    ) -> list[dict[str, Any]]:
        records = batch.get("records")
        inputs = batch.get("inputs")
        image_positions = batch.get("image_positions")
        if not isinstance(records, list) or not isinstance(inputs, Mapping):
            raise TypeError("GTP batch requires records and prepared inputs")
        if not isinstance(image_positions, list) or len(image_positions) != len(records):
            raise ValueError("image_positions must have one entry per record")

        clean_embeds = inputs["inputs_embeds"]
        grounded_embeds = clean_embeds.clone()
        grounding_maps: list[GroundingMap] = []
        caption_memories: list[torch.Tensor] = []
        for index, record in enumerate(records):
            image_path = str((self.images_root / str(record["image"])).resolve())
            try:
                patch_features = self.image_cache[image_path]
            except KeyError as exc:
                raise KeyError(f"Missing cached image features for {image_path}") from exc

            claims = [str(claim) for claim in record.get("gtp_claims", [])]
            grounding_labels = claims
            if method.grounding_source == "caption_tokens":
                draft = str(record.get("gtp_claim_draft") or "")
                encoded = self.processor.tokenizer(
                    draft,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                token_ids = encoded["input_ids"][0].to(self.device)
                grounding_labels = self.processor.tokenizer.convert_ids_to_tokens(
                    token_ids.tolist()
                )
                if token_ids.numel():
                    claim_features = self.model.get_input_embeddings()(token_ids).float()
                else:
                    claim_features = patch_features.new_empty((0, patch_features.shape[-1]))
            elif claims:
                claim_features = self._encode_claims(claims)
            else:
                claim_features = patch_features.new_empty((0, patch_features.shape[-1]))

            if claim_features.shape[0] == 0:
                grounding = self._invalid_grounding(
                    patch_count=patch_features.shape[0],
                    reason="empty_claim_set",
                )
            else:
                if not bool(
                    torch.isfinite(patch_features).all() and torch.isfinite(claim_features).all()
                ):
                    grounding = self._invalid_grounding(
                        patch_count=patch_features.shape[0],
                        reason="non_finite_grounding_features",
                    )
                else:
                    grounding = build_sparse_grounding_map(
                        patch_features,
                        claim_features,
                        temperature=method.tau_e,
                    )
            grounding_maps.append(grounding)
            selected_tokens = (
                (grounding.claim_mass > 0).nonzero(as_tuple=False).flatten()
                if grounding.valid
                else torch.empty(0, dtype=torch.long, device=self.device)
            )
            if selected_tokens.numel() > method.caption_memory_slots:
                selected_tokens = (
                    torch.topk(
                        grounding.claim_mass,
                        k=method.caption_memory_slots,
                    )
                    .indices.sort()
                    .values
                )
            caption_memories.append(
                claim_features.index_select(0, selected_tokens)
                if method.grounding_source == "caption_tokens" and method.inject_caption_memory
                else claim_features[:0]
            )
            record["gtp_grounding_labels"] = grounding_labels
            record["gtp_supported_token_count"] = int(selected_tokens.numel())

            positions = image_positions[index]
            if len(positions) != patch_features.shape[0]:
                raise RuntimeError(
                    "LLaVA image token count does not match projected patch count: "
                    f"{len(positions)} != {patch_features.shape[0]}"
                )
            if grounding.valid:
                enhanced = amplify_patch_features(
                    patch_features,
                    grounding.patch_gain,
                    strength=method.kappa,
                )
            else:
                enhanced = patch_features
            position_tensor = torch.as_tensor(
                positions,
                dtype=torch.long,
                device=grounded_embeds.device,
            )
            grounded_embeds[index, position_tensor] = enhanced.to(
                device=grounded_embeds.device,
                dtype=grounded_embeds.dtype,
            )

        grounded_attention_mask = inputs["attention_mask"]
        injected_token_counts = [0] * len(records)
        if method.inject_caption_memory:
            (
                grounded_embeds,
                grounded_attention_mask,
                injected_token_counts,
            ) = inject_caption_memory(
                grounded_embeds,
                inputs["attention_mask"],
                image_positions,
                caption_memories,
                memory_slots=method.caption_memory_slots,
            )

        return self._decode(
            records=records,
            method=method,
            clean_embeds=clean_embeds,
            grounded_embeds=grounded_embeds,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            grounded_attention_mask=grounded_attention_mask,
            grounding_maps=grounding_maps,
            injected_token_counts=injected_token_counts,
        )

    def _encode_claims(self, claims: list[str]) -> torch.Tensor:
        embedding_layer = self.model.get_input_embeddings()
        vectors: list[torch.Tensor] = []
        for claim in claims:
            encoded = self.processor.tokenizer(
                claim,
                add_special_tokens=False,
                return_tensors="pt",
            )
            token_ids = encoded["input_ids"].to(self.device)
            if token_ids.numel() == 0:
                continue
            token_embeddings = embedding_layer(token_ids).float()
            vectors.append(token_embeddings.mean(dim=1).squeeze(0))
        if not vectors:
            return torch.empty(
                (0, embedding_layer.weight.shape[-1]),
                device=self.device,
                dtype=torch.float32,
            )
        return torch.stack(vectors)

    @staticmethod
    def _invalid_grounding(*, patch_count: int, reason: str) -> GroundingMap:
        zeros = torch.zeros(patch_count, dtype=torch.float32)
        return GroundingMap(
            joint_mass=torch.zeros((patch_count, 0), dtype=torch.float32),
            patch_mass=zeros,
            claim_mass=torch.zeros(0, dtype=torch.float32),
            patch_gain=zeros,
            valid=False,
            diagnostics={
                "support_ratio": 0.0,
                "patch_entropy": 0.0,
                "claim_entropy": 0.0,
                "patch_gain_mean": 0.0,
                "patch_gain_std": 0.0,
                "saturated_patch_ratio": 0.0,
                "similarity_std": 0.0,
            },
            fallback_reason=reason,
        )

    def _forward(
        self,
        *,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "use_cache": True,
        }
        if self.forward_context is None:
            outputs = self.model.language_model(**kwargs)
        else:
            with self.forward_context():
                outputs = self.model.language_model(**kwargs)
        return outputs

    @staticmethod
    def _filter_distribution(
        probabilities: torch.Tensor,
        *,
        top_k: int | None,
        top_p: float,
    ) -> torch.Tensor:
        filtered = probabilities.clone()
        if top_k is not None and top_k < filtered.shape[-1]:
            threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1:]
            filtered = filtered.masked_fill(filtered < threshold, 0.0)
        if top_p < 1.0:
            sorted_probabilities, sorted_indices = filtered.sort(dim=-1, descending=True)
            cumulative = sorted_probabilities.cumsum(dim=-1)
            remove = (cumulative - sorted_probabilities) >= top_p
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
            filtered = torch.zeros_like(filtered).scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_probabilities,
            )
        normalizer = filtered.sum(dim=-1, keepdim=True)
        if bool((~torch.isfinite(normalizer) | (normalizer <= 0)).any()):
            raise FloatingPointError("sampling filters removed all GTP probability mass")
        return filtered / normalizer

    @torch.inference_mode()
    def _decode(
        self,
        *,
        records: list[Mapping[str, Any]],
        method: GTPMethod,
        clean_embeds: torch.Tensor,
        grounded_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        grounded_attention_mask: torch.Tensor,
        grounding_maps: list[GroundingMap],
        injected_token_counts: list[int],
    ) -> list[dict[str, Any]]:
        batch_size, clean_prompt_length = input_ids.shape
        grounded_prompt_length = grounded_embeds.shape[1]
        clean_cache_position = torch.arange(
            clean_prompt_length,
            dtype=torch.long,
            device=self.device,
        )
        grounded_cache_position = torch.arange(
            grounded_prompt_length,
            dtype=torch.long,
            device=self.device,
        )
        clean_attention_mask = attention_mask
        clean_cache: Cache | None = None
        grounded_cache: Cache | None = None
        current_tokens: torch.Tensor | None = None
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        excluded_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        support_sums = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        decision_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        pad_id = self.processor.tokenizer.pad_token_id
        eos_id = self.processor.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = eos_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

        for step in range(self.max_new_tokens):
            if step == 0:
                clean_output = self._forward(
                    attention_mask=clean_attention_mask,
                    cache_position=clean_cache_position,
                    past_key_values=None,
                    inputs_embeds=clean_embeds,
                )
                grounded_output = self._forward(
                    attention_mask=grounded_attention_mask,
                    cache_position=grounded_cache_position,
                    past_key_values=None,
                    inputs_embeds=grounded_embeds,
                )
            else:
                if current_tokens is None:
                    raise RuntimeError("decode token is missing after prompt prefill")
                clean_output = self._forward(
                    attention_mask=clean_attention_mask,
                    cache_position=clean_cache_position,
                    past_key_values=clean_cache,
                    input_ids=current_tokens,
                )
                grounded_output = self._forward(
                    attention_mask=grounded_attention_mask,
                    cache_position=grounded_cache_position,
                    past_key_values=grounded_cache,
                    input_ids=current_tokens,
                )

            clean_logits = self.model.lm_head(clean_output.last_hidden_state[:, -1, :])
            grounded_logits = self.model.lm_head(grounded_output.last_hidden_state[:, -1, :])
            clean_finite = torch.isfinite(clean_logits).all(dim=-1)
            if not bool(clean_finite.all()):
                raise FloatingPointError("GTP clean branch produced non-finite logits")
            grounded_finite = torch.isfinite(grounded_logits).all(dim=-1)
            for index in (~grounded_finite).nonzero(as_tuple=False).flatten().tolist():
                grounding_maps[index] = self._invalid_grounding(
                    patch_count=grounding_maps[index].patch_gain.numel(),
                    reason="non_finite_grounded_logits",
                )
            grounded_logits = torch.where(
                grounded_finite.unsqueeze(-1),
                grounded_logits,
                clean_logits,
            )
            projected = method.project_logits(
                clean_logits,
                grounded_logits,
                temperature=self.temperature,
            )
            clean_distribution = torch.softmax(
                clean_logits.float() / self.temperature,
                dim=-1,
            )
            grounding_valid = torch.tensor(
                [grounding.valid for grounding in grounding_maps],
                dtype=torch.bool,
                device=projected.device,
            )
            projected = torch.where(
                grounding_valid.unsqueeze(-1),
                projected,
                clean_distribution,
            )
            sampling_distribution = self._filter_distribution(
                projected,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            if self.decoding == "greedy":
                next_tokens = sampling_distribution.argmax(dim=-1)
            else:
                next_tokens = torch.multinomial(sampling_distribution, num_samples=1).squeeze(-1)

            was_finished = finished.clone()
            next_tokens = torch.where(
                was_finished,
                torch.full_like(next_tokens, int(pad_id)),
                next_tokens,
            )
            generated.append(next_tokens)
            clean_top = clean_logits.argmax(dim=-1)
            excluded = projected.gather(-1, clean_top.unsqueeze(-1)).squeeze(-1) == 0
            support = (projected > 0).sum(dim=-1)
            active = ~was_finished
            excluded_counts += (excluded & active).to(excluded_counts.dtype)
            support_sums += support.to(torch.float64) * active.to(torch.float64)
            decision_counts += active.to(decision_counts.dtype)

            if eos_id is not None:
                finished |= next_tokens.eq(int(eos_id))
            clean_cache = clean_output.past_key_values
            grounded_cache = grounded_output.past_key_values
            if bool(finished.all()):
                break
            current_tokens = next_tokens[:, None]
            clean_attention_mask = torch.cat(
                (
                    clean_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=clean_attention_mask.dtype,
                        device=self.device,
                    ),
                ),
                dim=1,
            )
            grounded_attention_mask = torch.cat(
                (
                    grounded_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=grounded_attention_mask.dtype,
                        device=self.device,
                    ),
                ),
                dim=1,
            )
            clean_cache_position = torch.tensor(
                [clean_prompt_length + step],
                dtype=torch.long,
                device=self.device,
            )
            grounded_cache_position = torch.tensor(
                [grounded_prompt_length + step],
                dtype=torch.long,
                device=self.device,
            )

        generated_tensor = torch.stack(generated, dim=1)
        decoded = self.processor.batch_decode(
            generated_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results: list[dict[str, Any]] = []
        for index, (record, answer, grounding) in enumerate(
            zip(records, decoded, grounding_maps, strict=True)
        ):
            token_ids = generated_tensor[index].tolist()
            if eos_id is not None and int(eos_id) in token_ids:
                token_count = token_ids.index(int(eos_id)) + 1
            else:
                token_count = len(token_ids)
            claim_names = [str(claim) for claim in record.get("gtp_claims", [])]
            claim_mass = {
                claim: float(grounding.claim_mass[item].item())
                for item, claim in enumerate(claim_names)
                if method.grounding_source == "coco_claims" and item < grounding.claim_mass.numel()
            }
            token_labels = [str(label) for label in record.get("gtp_grounding_labels", [])]
            count = int(decision_counts[index].item())
            results.append(
                {
                    "id": record.get("id", record.get("question_id")),
                    "image": record["image"],
                    "prompt": str(record["text"]),
                    "answer": answer.strip(),
                    "prediction": answer.strip(),
                    "generated_tokens": token_count,
                    "gtp_claims": claim_names,
                    "gtp_claim_draft": record.get("gtp_claim_draft"),
                    "claim_grounding_mass": claim_mass,
                    "gtp_grounding_source": method.grounding_source,
                    "gtp_token_grounding_mass": (
                        [
                            {
                                "token": label,
                                "mass": float(grounding.claim_mass[item].item()),
                            }
                            for item, label in enumerate(token_labels)
                            if item < grounding.claim_mass.numel()
                        ]
                        if method.grounding_source == "caption_tokens"
                        else None
                    ),
                    "gtp_supported_token_count": int(record.get("gtp_supported_token_count", 0)),
                    "gtp_injected_caption_tokens": int(injected_token_counts[index]),
                    "gtp_grounding_valid": grounding.valid,
                    "gtp_fallback_reason": grounding.fallback_reason,
                    "gtp_grounding_diagnostics": grounding.diagnostics,
                    "gtp_clean_top1_excluded_steps": int(excluded_counts[index].item()),
                    "gtp_clean_top1_excluded_rate": (
                        float(excluded_counts[index].item() / count) if count else 0.0
                    ),
                    "gtp_mean_projected_support": (
                        float(support_sums[index].item() / count) if count else 0.0
                    ),
                    "gtp_method": {
                        "a_u": method.a_u,
                        "a_h": method.a_h,
                        "tau": method.tau,
                        "tau_e": method.tau_e,
                        "kappa": method.kappa,
                        "grounding_source": method.grounding_source,
                        "inject_caption_memory": method.inject_caption_memory,
                        "caption_memory_slots": method.caption_memory_slots,
                    },
                }
            )
        return results
