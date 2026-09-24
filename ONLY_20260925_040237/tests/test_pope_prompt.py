from __future__ import annotations

from scripts.pope_llava_only_compare import format_pope_prompt
from scripts.shield_cumulative import resolve_pope_answer_instruction


def test_official_pope_prompt_requests_one_word_answer() -> None:
    prompt = format_pope_prompt("Is there a cat in the image?")

    assert prompt.endswith(
        "Is there a cat in the image? Please answer this question with one word. "
        "ASSISTANT:"
    )


def test_legacy_pope_prompt_remains_explicitly_available() -> None:
    prompt = format_pope_prompt(
        "Is there a cat in the image?",
        one_word=False,
    )

    assert prompt.endswith("Is there a cat in the image? ASSISTANT:")


def test_cumulative_prompt_defaults_follow_dataset_contract() -> None:
    assert resolve_pope_answer_instruction("pope", None) is True
    assert resolve_pope_answer_instruction("chair", None) is False


def test_cumulative_prompt_rejects_one_word_suffix_for_chair() -> None:
    try:
        resolve_pope_answer_instruction("chair", True)
    except ValueError as exc:
        assert "CHAIR" in str(exc)
    else:
        raise AssertionError("CHAIR must reject the POPE one-word suffix")
