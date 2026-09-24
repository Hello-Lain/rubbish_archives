from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts import shield_vulnerability_runtime as runtime  # noqa: E402


def test_decode_excludes_unregistered_model_output_rows() -> None:
    class Tokenizer:
        vocab_size = 3

        def __len__(self) -> int:
            return 4

    logits = torch.tensor([[1.0, 2.0, 3.0, 10.0, 100.0]])
    args = SimpleNamespace(decoding="greedy")

    tokens = runtime.decode_valid_logits(logits, args, Tokenizer())

    assert tokens.tolist() == [3]


def test_decode_rejects_tokenizer_larger_than_model_output() -> None:
    class Tokenizer:
        def __len__(self) -> int:
            return 4

    with pytest.raises(ValueError, match="incompatible"):
        runtime.decode_valid_logits(
            torch.zeros((1, 3)),
            SimpleNamespace(decoding="greedy"),
            Tokenizer(),
        )


@pytest.mark.parametrize("use_reference", [False, True])
def test_generation_preserves_branch_identity_and_shared_prefix(
    monkeypatch: pytest.MonkeyPatch, use_reference: bool
) -> None:
    calls: list[tuple[int, int, list[int] | None]] = []
    decoded: list[torch.Tensor] = []

    def packed(branch: int) -> dict[str, torch.Tensor]:
        length = branch + 2
        return {
            "input_ids": torch.zeros((2, length), dtype=torch.long),
            "inputs_embeds": torch.full((2, length, 1), float(branch)),
            "attention_mask": torch.ones((2, length), dtype=torch.long),
            "position_ids": torch.arange(length).repeat(2, 1),
        }

    def forward(
        model: Any, kwargs: dict[str, Any], state: Any, recipe: Any
    ) -> tuple[SimpleNamespace, torch.Tensor]:
        del model, state, recipe
        if kwargs["past_key_values"] is None:
            branch = int(kwargs["inputs_embeds"][0, 0, 0])
            step = 0
            assert kwargs["input_ids"] is None
        else:
            branch, previous = kwargs["past_key_values"]
            step = previous + 1
        prefix = kwargs["input_ids"].flatten().tolist() if step else None
        calls.append((branch, step, prefix))
        assert kwargs["attention_mask"].shape[1] == branch + 2 + step
        return SimpleNamespace(past_key_values=(branch, step)), torch.full(
            (2, 3), float(branch)
        )

    class Rule:
        def combine_logits(self, *logits: torch.Tensor) -> torch.Tensor:
            assert len(logits) == (3 if use_reference else 2)
            for branch, tensor in enumerate(logits):
                assert torch.equal(tensor, torch.full((2, 3), float(branch)))
            return logits[0]

    def decode(logits: torch.Tensor, args: Any) -> torch.Tensor:
        del logits, args
        return torch.tensor([1, 2]) if not decoded else torch.tensor([1, 1])

    def record_decode(logits: torch.Tensor, args: Any) -> torch.Tensor:
        value = decode(logits, args)
        decoded.append(value)
        return value

    class Tokenizer:
        pad_token_id = 3
        eos_token_id = 1

        def __len__(self) -> int:
            return 3

    class Processor:
        tokenizer = Tokenizer()

        def batch_decode(self, tokens: torch.Tensor, **kwargs: Any) -> list[str]:
            assert tokens.tolist() == [[1, 3], [2, 1]]
            return ["", "word"]

    monkeypatch.setattr(runtime, "_forward_branch", forward)
    monkeypatch.setattr(runtime, "decode_logits", record_decode)
    answers, generations = runtime.generate_contrastive_batch(
        None, Processor(), packed(0), packed(1),
        SimpleNamespace(max_new_tokens=4), None, None, Rule(),
        reference_inputs=packed(2) if use_reference else None,
    )
    count = 3 if use_reference else 2
    assert len(calls) == count * 2
    assert all(prefix == [1, 2] for _, step, prefix in calls if step == 1)
    assert answers == ["", "word"]
    assert generations == [{"generated_tokens": 2}, {"generated_tokens": 2}]
