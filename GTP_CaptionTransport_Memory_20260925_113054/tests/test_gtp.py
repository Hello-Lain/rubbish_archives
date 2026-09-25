from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from adapters.gtp_llava import (
    LlavaGTPBatchBackend,
    extract_coco_claims,
    inject_caption_memory,
)
from methods.gtp import (
    GTPMethod,
    amplify_patch_features,
    build_sparse_grounding_map,
    dual_reference_projection,
    log_harmonic_gate,
    sparsemax,
)
from models.base import BaseMLLMWrapper
from scripts.gtp_llava_eval import _build_chair_dev_records, _build_claim_rows


def test_sparsemax_is_normalized_sparse_and_translation_invariant() -> None:
    logits = torch.tensor([[4.0, 1.0, -2.0], [0.2, 0.1, 0.0]])
    projected = sparsemax(logits)

    assert projected.sum(dim=-1) == pytest.approx(torch.ones(2))
    assert projected[0, 2] == 0
    torch.testing.assert_close(projected, sparsemax(logits + 9.0))


def test_grounding_map_normalizes_and_bounds_patch_gain() -> None:
    patches = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    claims = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = build_sparse_grounding_map(patches, claims, temperature=0.5)

    assert result.valid
    assert result.joint_mass.sum() == pytest.approx(1.0)
    assert result.patch_mass.sum() == pytest.approx(1.0)
    assert result.claim_mass.sum() == pytest.approx(1.0)
    assert torch.all(result.patch_gain >= 0)
    assert torch.all(result.patch_gain <= 3.0 / 5.0)

    amplified = amplify_patch_features(patches, result.patch_gain, strength=1.7)
    original_norm = torch.linalg.vector_norm(patches, dim=-1)
    relative_shift = torch.linalg.vector_norm(amplified - patches, dim=-1) / original_norm
    assert torch.allclose(relative_shift, 1.7 * result.patch_gain)


def test_grounding_map_invalidates_zero_similarity_variance() -> None:
    patches = torch.ones(3, 4)
    claims = torch.ones(2, 4)

    result = build_sparse_grounding_map(patches, claims)

    assert not result.valid
    assert result.fallback_reason == "similarity_variance_too_small"
    assert result.patch_gain.sum() == 0


def test_log_harmonic_gate_is_stable_at_extreme_logits_and_endpoints() -> None:
    log_p = torch.tensor([-1000.0, -1.0])
    log_g = torch.tensor([-2.0, -900.0])
    gate = log_harmonic_gate(log_p, log_g, weight=0.3)
    expected = -torch.logaddexp(
        torch.log(torch.tensor(0.7)) - log_p,
        torch.log(torch.tensor(0.3)) - log_g,
    )

    assert torch.isfinite(gate).all()
    assert torch.allclose(gate, expected)
    assert torch.equal(log_harmonic_gate(log_p, log_g, weight=0.0), log_p)
    assert torch.equal(log_harmonic_gate(log_p, log_g, weight=1.0), log_g)


def test_projection_is_simplex_and_satisfies_active_set_kkt() -> None:
    torch.manual_seed(7)
    clean_logits = torch.randn(4, 13)
    grounded_logits = torch.randn(4, 13)
    log_p = torch.log_softmax(clean_logits, dim=-1)
    log_g = torch.log_softmax(grounded_logits, dim=-1)
    a_u = 0.35
    a_h = 0.6
    tau = 0.7
    utility = (1 - a_u) * log_p + a_u * log_g
    log_gate = log_harmonic_gate(log_p, log_g, weight=a_h)
    gate = log_gate.exp()
    projected = dual_reference_projection(
        log_p,
        log_g,
        utility_weight=a_u,
        gate_weight=a_h,
        temperature=tau,
    )

    assert torch.all(projected >= 0)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(4), atol=1e-6)
    active = projected > 0
    inferred_threshold = utility - tau * (projected / gate - 1.0)
    for row in range(projected.shape[0]):
        active_threshold = inferred_threshold[row, active[row]]
        assert torch.allclose(
            active_threshold,
            active_threshold[0].expand_as(active_threshold),
            atol=2e-5,
        )
        threshold = active_threshold[0]
        assert torch.all(utility[row, ~active[row]] <= threshold - tau + 2e-5)


def test_projection_remains_finite_when_reference_mass_underflows() -> None:
    log_p = torch.log_softmax(torch.tensor([[0.0, -1000.0, -2000.0]]), dim=-1)
    log_g = torch.log_softmax(torch.tensor([[-1000.0, 0.0, -2000.0]]), dim=-1)

    projected = dual_reference_projection(
        log_p,
        log_g,
        utility_weight=0.5,
        gate_weight=0.5,
        temperature=1.0,
    )

    assert torch.isfinite(projected).all()
    assert torch.allclose(projected, torch.tensor([[0.5, 0.5, 0.0]]), atol=1e-6)


def test_projection_temperature_limits() -> None:
    log_p = torch.log(torch.tensor([[0.65, 0.25, 0.10]]))
    log_g = torch.log(torch.tensor([[0.10, 0.30, 0.60]]))
    utility = 0.5 * (log_p + log_g)
    log_gate = log_harmonic_gate(log_p, log_g, weight=0.5)
    gate = log_gate.exp()

    low_temperature = dual_reference_projection(
        log_p,
        log_g,
        utility_weight=0.5,
        gate_weight=0.5,
        temperature=1e-5,
    )
    high_temperature = dual_reference_projection(
        log_p,
        log_g,
        utility_weight=0.5,
        gate_weight=0.5,
        temperature=1e5,
    )
    expected_high = gate / gate.sum(dim=-1, keepdim=True)

    assert int(low_temperature.argmax(dim=-1).item()) == int(utility.argmax(dim=-1).item())
    assert low_temperature.max() > 0.999
    assert torch.allclose(high_temperature, expected_high, atol=2e-5)


def test_claim_extraction_uses_longest_coco_alias_and_deduplicates() -> None:
    claims = extract_coco_claims("Two traffic lights stand near a cell phone, a phone, and a dog.")

    assert claims == ["traffic light", "cell phone", "dog"]


def test_caption_memory_injection_preserves_image_text_order_and_batch_mask() -> None:
    embeds = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    memories = [
        torch.tensor([[90.0, 91.0]]),
        torch.tensor([[80.0, 81.0], [82.0, 83.0]]),
    ]

    fused, fused_mask, memory_counts = inject_caption_memory(
        embeds,
        attention_mask,
        [[1, 2], [0, 1]],
        memories,
        memory_slots=2,
    )

    assert fused.shape == (2, 7, 2)
    assert memory_counts == [1, 2]
    assert fused_mask.tolist() == [
        [0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    torch.testing.assert_close(fused[0, 1:3], embeds[0, 1:3])
    torch.testing.assert_close(fused[0, 3], memories[0][0])
    torch.testing.assert_close(fused[0, 4], torch.zeros(2))
    torch.testing.assert_close(fused[0, 5:7], embeds[0, 3:5])
    torch.testing.assert_close(fused[1, 0:2], embeds[1, 0:2])
    torch.testing.assert_close(fused[1, 2:4], memories[1])
    torch.testing.assert_close(fused[1, 4:], embeds[1, 2:])


def test_gtp_caption_memory_requires_token_transport() -> None:
    with pytest.raises(ValueError, match="caption_tokens"):
        GTPMethod(inject_caption_memory=True)
    with pytest.raises(ValueError, match="caption_memory_slots"):
        GTPMethod(caption_memory_slots=0)


def test_chair_dev_records_are_deterministic_and_exclude_fixed_images(
    tmp_path: Path,
) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    image_rows = []
    for image_id in range(1, 8):
        filename = f"COCO_val2014_{image_id:012d}.jpg"
        (images_root / filename).touch()
        image_rows.append({"id": image_id, "file_name": filename})
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(
        json.dumps({"images": image_rows}),
        encoding="utf-8",
    )

    first = _build_chair_dev_records(
        annotations_path=annotations_path,
        images_root=images_root,
        excluded_image_ids={2, 5},
        start=0,
        count=3,
        seed=9,
    )
    repeated = _build_chair_dev_records(
        annotations_path=annotations_path,
        images_root=images_root,
        excluded_image_ids={2, 5},
        start=0,
        count=3,
        seed=9,
    )

    assert first == repeated
    assert len(first) == 3
    assert {row["image_id"] for row in first}.isdisjoint({2, 5})
    assert all(Path(row["image_path"]).is_file() for row in first)


def test_claim_rows_support_explicit_visible_object_prompt() -> None:
    rows = [{"id": "one", "image": "one.jpg", "text": "describe this image"}]
    prompt = "List only directly visible objects."

    claim_rows, source = _build_claim_rows(
        rows,
        mode="visible_object_list",
        prompt=prompt,
    )

    assert source == "greedy_visible_object_list"
    assert claim_rows[0]["text"] == prompt
    assert rows[0]["text"] == "describe this image"
    assert claim_rows[0]["image"] == rows[0]["image"]


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        del skip_special_tokens, clean_up_tokenization_spaces
        return [" ".join(str(int(token)) for token in row) for row in token_ids]


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.created_caches: list[object] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        hidden_input = kwargs["inputs_embeds"]
        if hidden_input is None:
            hidden_input = torch.zeros(
                kwargs["input_ids"].shape[0],
                kwargs["input_ids"].shape[1],
                4,
            )
        cache = object()
        self.calls.append(
            {
                "past_key_values": kwargs["past_key_values"],
                "inputs_embeds": kwargs["inputs_embeds"],
                "input_ids": kwargs["input_ids"],
                "cache": cache,
            }
        )
        self.created_caches.append(cache)
        return SimpleNamespace(
            last_hidden_state=torch.zeros_like(hidden_input),
            past_key_values=cache,
        )


class _FakeLMHead:
    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*hidden.shape[:-1], 4), -5.0)
        logits[..., 1] = 5.0
        return logits


class _FakeLlavaModel:
    def __init__(self) -> None:
        self.language_model = _FakeLanguageModel()
        self.lm_head = _FakeLMHead()


def test_llava_backend_keeps_clean_and_grounded_kv_caches_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = (tmp_path / "sample.jpg").resolve()
    backend = LlavaGTPBatchBackend(
        model_path="mock",
        model=_FakeLlavaModel(),  # type: ignore[arg-type]
        processor=_FakeProcessor(),
        image_cache={str(image_path): torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])},
        images_root=tmp_path,
        device=torch.device("cpu"),
        max_new_tokens=3,
        decoding="greedy",
        temperature=1.0,
        top_p=1.0,
        top_k=None,
    )
    method = GTPMethod()
    method.setup(backend, torch.device("cpu"))

    def force_grounded_choice(
        clean_logits: torch.Tensor,
        grounded_logits: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        del grounded_logits, temperature
        forced = torch.zeros_like(clean_logits)
        forced[:, 2] = 1.0
        return forced

    monkeypatch.setattr(method, "project_logits", force_grounded_choice)
    results = method.generate(
        {
            "records": [
                {
                    "id": "sample",
                    "image": image_path.name,
                    "text": "question",
                    "gtp_claims": [],
                }
            ],
            "inputs": {
                "inputs_embeds": torch.zeros(1, 4, 4),
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            "image_positions": [[1, 2]],
        }
    )

    calls = backend.model.language_model.calls  # type: ignore[attr-defined]
    assert len(calls) == 6
    assert calls[0]["past_key_values"] is None
    assert calls[1]["past_key_values"] is None
    assert calls[2]["past_key_values"] is calls[0]["cache"]
    assert calls[3]["past_key_values"] is calls[1]["cache"]
    assert calls[2]["past_key_values"] is not calls[3]["past_key_values"]
    assert calls[4]["past_key_values"] is calls[2]["cache"]
    assert calls[5]["past_key_values"] is calls[3]["cache"]
    assert results[0]["generated_tokens"] == 3
    assert results[0]["answer"] == "1 1 1"


class _FakeGTPBackend(BaseMLLMWrapper):
    def __init__(self) -> None:
        super().__init__("mock")
        self.received_method: GTPMethod | None = None

    def setup(self, device: torch.device) -> None:
        self.device = device

    def generate_response(self, *, prompt: str, image_path: str | None) -> str:
        return prompt

    def generate_gtp_batch(
        self,
        batch: dict[str, Any],
        *,
        method: GTPMethod,
    ) -> list[dict[str, Any]]:
        self.received_method = method
        records = batch["records"]
        return [{"id": record["id"], "prediction": "ok"} for record in records]


def test_gtp_method_contract_dispatches_to_model_backend() -> None:
    backend = _FakeGTPBackend()
    method = GTPMethod()
    method.setup(backend, torch.device("cpu"))

    results = method.generate({"records": [{"id": "sample"}]})

    assert results == [{"id": "sample", "prediction": "ok"}]
    assert backend.received_method is method


def test_gtp_method_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="a_u"):
        GTPMethod(a_u=1.1)
    with pytest.raises(ValueError, match="tau"):
        GTPMethod(tau=0.0)
