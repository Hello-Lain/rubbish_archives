from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vrsc_probe import (  # noqa: E402
    analyze,
    correctness_probability,
    entropy,
    entropy_match_diagnostics,
    entropy_matched_distribution,
    image_bootstrap,
    replay,
)
from vrsc_probe import (  # noqa: E402
    DecodeRule,
    answer_token_ids,
    branch_names,
    cap_row_key,
    parse_args,
    shuffled_scores,
)

from methods.cap import CAPConfig, CAPRefinedConfig  # noqa: E402
from methods.vrsc import VRSCConfig  # noqa: E402


def test_dataset_defaults_and_bad_options() -> None:
    pope = parse_args(["--output", "unused"])
    assert (pope.batch_size, pope.max_new_tokens, pope.pope_answer_instruction) == (8, 8, True)
    assert "vrsc_residual" in pope.methods
    chair = parse_args(["--output", "unused", "--mode", "generate", "--dataset", "chair"])
    assert (chair.batch_size, chair.max_new_tokens, chair.pope_answer_instruction) == (
        32,
        512,
        False,
    )
    with pytest.raises(SystemExit):
        parse_args(["--output", "unused", "--dataset", "chair"])
    with pytest.raises(SystemExit):
        parse_args(["--output", "unused", "--methods", "vrsc,vrsc"])


def test_cap_cli_and_routing() -> None:
    args = parse_args(
        [
            "--output",
            "unused",
            "--mode",
            "generate",
            "--methods",
            "vanilla,cap",
            "--cap-probe-gain",
            "0.2",
        ]
    )
    assert args.methods == ["vanilla", "cap"]
    assert args.cap_probe_gain == pytest.approx(0.2)
    assert args.cap_attribution == "attention"
    assert branch_names("cap") == ("base", "cap", None)
    cosine = parse_args(
        [
            "--output",
            "unused",
            "--mode",
            "generate",
            "--methods",
            "cap",
            "--cap-attribution",
            "cosine",
        ]
    )
    assert cosine.cap_attribution == "cosine"
    contrastive = parse_args(
        [
            "--output",
            "unused",
            "--mode",
            "generate",
            "--methods",
            "cap",
            "--cap-attribution",
            "contrastive_hidden",
        ]
    )
    assert contrastive.cap_attribution == "contrastive_hidden"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output",
                "unused",
                "--mode",
                "generate",
                "--methods",
                "cap",
                "--cap-attribution",
                "invalid",
            ]
        )

    refined = parse_args(
        [
            "--output",
            "unused",
            "--mode",
            "generate",
            "--methods",
            "cap_refined",
        ]
    )
    assert refined.cap_attribution == "contrastive_hidden"
    assert branch_names("cap_refined") == ("base", "cap_refined", None)
    assert refined.cap_refined_max_weight == pytest.approx(0.75)


def test_cap_decode_rule_uses_probe_logits() -> None:
    rule = DecodeRule("cap", VRSCConfig(), cap_config=CAPConfig(probe_gain=0.2))
    base = torch.tensor([[2.0, 0.0, -1.0]])
    probe = torch.tensor([[-1.0, 3.0, 0.0]])
    assert torch.equal(rule.combine_logits(base, probe), probe)
    assert rule.config == CAPConfig(probe_gain=0.2)


def test_cap_refined_decode_rule_returns_probability_distribution() -> None:
    rule = DecodeRule(
        "cap_refined",
        VRSCConfig(),
        cap_refined_config=CAPRefinedConfig(max_weight=0.5),
    )
    base = torch.tensor([[2.0, 0.0, -1.0]])
    probe = torch.tensor([[1.5, 1.5, -1.0]])
    combined = rule.combine_logits(base, probe)
    assert torch.isfinite(combined).all()
    assert torch.allclose(combined.exp().sum(dim=-1), torch.ones(1), atol=1e-6)


def test_cap_row_key_separates_questions_on_the_same_image() -> None:
    first = {"image": "COCO_val2014_000000000001.jpg", "question_id": 10, "text": "Is there a dog?"}
    second = {
        "image": "COCO_val2014_000000000001.jpg",
        "question_id": 11,
        "text": "Is there a person?",
    }
    assert cap_row_key(first) != cap_row_key(second)
    assert cap_row_key(first) == cap_row_key(dict(first))


def test_shuffle_preserves_generation_rng() -> None:
    before = torch.random.get_rng_state().clone()
    x = torch.arange(100)
    result = shuffled_scores(x, torch.Generator().manual_seed(42))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(result.sort().values, x)
    assert not torch.equal(result, x)


def test_entropy_match() -> None:
    x = torch.tensor([[0.0, 1.0, -2.0], [1.0, 0.0, -4.0]])
    target = entropy((x / 0.5).softmax(-1))
    p, temperature = entropy_matched_distribution(x, target)
    assert torch.allclose(entropy(p), target, atol=1e-6)
    assert torch.allclose(temperature, torch.full((2,), 0.5), atol=1e-5)


def test_entropy_match_boundaries_and_shift_invariance() -> None:
    logits = torch.tensor([[4.0, 4.0, -2.0], [1.0, 0.0, -4.0]])
    target = torch.tensor([0.1, 2.0])
    p, temperature = entropy_matched_distribution(logits, target)
    assert torch.allclose(p[0], torch.tensor([0.5, 0.5, 0.0]))
    assert torch.allclose(temperature, torch.tensor([1e-4, 1000.0]))
    shifted, _ = entropy_matched_distribution(logits + 1000, target)
    assert torch.allclose(shifted, p)
    diagnostics, reachable = entropy_match_diagnostics(logits, target, p, temperature)
    assert diagnostics["unattainable_row_indices"] == [0, 1]
    assert diagnostics["max_absolute_error_on_reachable_targets"] is None
    assert not reachable.any()


def test_entropy_match_uniform_and_invalid_inputs() -> None:
    logits = torch.zeros(1, 3)
    p, _ = entropy_matched_distribution(logits, torch.tensor([0.0]))
    assert torch.allclose(p, torch.full_like(p, 1 / 3))
    for invalid in (torch.tensor([-1.0]), torch.tensor([float("nan")]), torch.zeros(2)):
        with pytest.raises(ValueError):
            entropy_matched_distribution(logits, invalid)


def test_analysis_preserves_capture_and_reports_unreachable_rows(tmp_path: Path) -> None:
    capture, output = tmp_path / "capture", tmp_path / "analysis"
    capture.mkdir()
    manifest = {
        "status": "SMOKE",
        "mode": "capture",
        "count": 2,
        "method_config": {},
        "acceleration": {
            "effective_attention": "flash_attention_2",
            "fp8_effective": True,
            "fp8_native_fallback_calls": 0,
            "fallback_events": [],
        },
    }
    rows = [
        {"question_id": i, "image": f"image{i}", "label": label}
        for i, label in enumerate(("yes", "no"))
    ]
    for name, payload in (
        ("manifest.json", manifest),
        ("rows.json", rows),
        ("answer_token_ids.json", {"yes": [0], "no": [1]}),
    ):
        (capture / name).write_text(json.dumps(payload))
    base = torch.tensor([[4.0, 4.0, -2.0], [1.0, 0.0, -4.0]])
    torch.save(
        {"base": base, "semantic": base + torch.tensor([0.5, 0.0, 0.0]), "shuffled": base},
        capture / "logits.pt",
    )
    result = analyze(capture, output)
    assert result["entropy_match"]["unattainable_row_indices"] == [0]
    assert result["entropy_match"]["reachable_target_count"] == 1
    assert result["entropy_match"]["max_absolute_error_on_reachable_targets"] < 1e-6
    assert "vrsc_minus_matched_entropy_reachable_only" in result["paired_expected_correct_deltas"]
    assert (output / "analysis.json").is_file()
    assert len(json.loads((output / "analysis_rows.json").read_text())) == 2
    assert not (capture / "analysis.json").exists()
    with pytest.raises(FileExistsError):
        analyze(capture, output)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "analysis_rows.json").write_text("[]")
    with pytest.raises(FileExistsError):
        analyze(capture, partial)
    assert not (partial / "analysis.json").exists()


def test_replay_and_strict_answer_mass() -> None:
    base = torch.tensor([[1.0, 0.0, -2.0], [0.0, 1.0, -2.0]])
    distributions, _ = replay({"base": base, "semantic": base + 1, "shuffled": base}, VRSCConfig())
    assert torch.allclose(distributions["shrink"], distributions["vrsc"], atol=1e-6)
    p = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]])
    assert torch.allclose(
        correctness_probability(p, torch.tensor([True, False]), {"yes": [0], "no": [1]}),
        torch.tensor([0.2, 0.5]),
    )


def test_residual_decode_rule_requires_reference_branch() -> None:
    rule = DecodeRule("vrsc_residual", VRSCConfig())
    base = torch.tensor([[2.0, 1.0, -1.0]])
    probe = torch.tensor([[1.5, 1.5, -1.0]])
    reference = torch.tensor([[1.8, 1.2, -1.0]])
    with pytest.raises(ValueError, match="reference"):
        rule.combine_logits(base, probe)
    combined = rule.combine_logits(base, probe, reference)
    assert combined.shape == base.shape
    probabilities = combined.exp()
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(-1), torch.ones(1))


def test_residual_runtime_routing_does_not_collapse_to_shrink() -> None:
    branches = {
        "base": torch.tensor([[0.0, -0.2, -0.4]]),
        "semantic": torch.tensor([[-0.5, 0.3, -0.4]]),
        "shuffled": torch.tensor([[0.1, -0.1, -0.9]]),
    }
    base, probe, reference = branch_names("vrsc_residual")
    assert (base, probe, reference) == ("base", "semantic", "shuffled")
    rule = DecodeRule("vrsc_residual", VRSCConfig())
    q = rule.combine_logits(branches[base], branches[probe], branches[reference]).exp()
    shrink = DecodeRule("shrink", VRSCConfig()).combine_logits(
        branches[base], branches[base]
    ).exp()
    assert not torch.allclose(q, shrink)
    assert rule.semantic_delta_max > 0


def test_cluster_bootstrap_constant() -> None:
    result = image_bootstrap(np.ones(5) * 0.2, ["a", "a", "b", "c", "c"])
    assert result["mean"] == pytest.approx(0.2)
    assert result["low"] == pytest.approx(0.2)
    assert result["high"] == pytest.approx(0.2)


def test_answer_tokens_are_complete_words() -> None:
    class Tokenizer:
        def __len__(self) -> int:
            return 5

        def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
            return [" Yes", "No", "yes", "Nothing", "y"][ids[0]]

    assert answer_token_ids(Tokenizer()) == {"yes": [0, 2], "no": [1]}
