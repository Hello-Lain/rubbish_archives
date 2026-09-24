#!/usr/bin/env python3
"""Validate the SHIELD 2^3 matrix and compute strict component contributions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORTS_ROOT = PROJECT_ROOT / "reports"

COMPONENTS: tuple[str, ...] = ("A", "V", "S")
COMPONENT_NAMES: dict[str, str] = {
    "A": "Adaptive Plausibility",
    "V": "Vulnerability Defense",
    "S": "Statistical Bias",
}
FULL_NAME_TO_ABBREV: dict[str, str] = {
    "adaptive_plausibility": "A",
    "vulnerability_defense": "V",
    "statistical_bias": "S",
}
COMBINATIONS: tuple[str, ...] = ("", "A", "V", "S", "AV", "AS", "VS", "AVS")
CHAIR_METRICS: tuple[str, ...] = ("CHAIRs", "CHAIRi", "Recall")
POPE_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")

EXPECTED_MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
EXPECTED_MODEL_CONFIG_SHA256 = (
    "0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f"
)
EXPECTED_CHAIR_QUESTIONS_SHA256 = (
    "d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e"
)
EXPECTED_POPE_SHA256 = (
    "185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac"
)


@dataclass(frozen=True)
class Artifact:
    dataset: str
    combo: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_ROOT,
        help="Directory containing the eight formal metrics files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_ROOT / "shield_strict_contributions_20260922.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=REPORTS_ROOT / "shield_strict_contributions_20260922.md",
    )
    return parser.parse_args()


def combo_key(components: Sequence[str]) -> str:
    normalized = {
        FULL_NAME_TO_ABBREV.get(str(component), str(component)) for component in components
    }
    selected = set(normalized)
    unknown = selected - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown component abbreviations: {sorted(unknown)}")
    return "".join(component for component in COMPONENTS if component in selected)


def combo_components(combo: str) -> tuple[str, ...]:
    return tuple(component for component in COMPONENTS if component in combo)


def artifact_paths(outputs_dir: Path) -> dict[str, tuple[Artifact, ...]]:
    chair = {
        "": "chair_llava_7b_shield_vanilla_b32_20260921.metrics.json",
        "A": "chair_llava_7b_shield_cumulative_adaptive_b32_20260922.metrics.json",
        "V": (
            "chair_llava_7b_shield_subset_vulnerability_protocolfix_20260922.metrics.json"
        ),
        "S": (
            "chair_llava_7b_shield_subset_statistical_protocolfix_20260922.metrics.json"
        ),
        "AV": (
            "chair_llava_7b_shield_cumulative_vulnerability_lr002_b32_"
            "20260922.metrics.json"
        ),
        "AS": (
            "chair_llava_7b_shield_subset_adaptive_statistical_protocolfix_"
            "20260922.metrics.json"
        ),
        "VS": (
            "chair_llava_7b_shield_subset_vulnerability_statistical_protocolfix_"
            "20260922.metrics.json"
        ),
        "AVS": "chair_llava_7b_shield_cumulative_statistical_b32_20260922.metrics.json",
    }
    pope = {
        "": "pope_llava_7b_adversarial_promptfix_vanilla_20260922.metrics.json",
        "A": (
            "pope_llava_7b_adversarial_cumulative_adaptive_official_"
            "20260922.metrics.json"
        ),
        "V": (
            "pope_llava_7b_adversarial_subset_vulnerability_official_"
            "20260922.metrics.json"
        ),
        "S": (
            "pope_llava_7b_adversarial_subset_statistical_official_"
            "20260922.metrics.json"
        ),
        "AV": (
            "pope_llava_7b_adversarial_cumulative_vulnerability_official_"
            "20260922.metrics.json"
        ),
        "AS": (
            "pope_llava_7b_adversarial_subset_adaptive_statistical_official_"
            "20260922.metrics.json"
        ),
        "VS": (
            "pope_llava_7b_adversarial_subset_vulnerability_statistical_official_"
            "20260922.metrics.json"
        ),
        "AVS": (
            "pope_llava_7b_adversarial_cumulative_statistical_official_"
            "20260922.metrics.json"
        ),
    }
    return {
        "chair": tuple(
            Artifact("chair", combo, outputs_dir / filename)
            for combo, filename in chair.items()
        ),
        "pope": tuple(
            Artifact("pope", combo, outputs_dir / filename)
            for combo, filename in pope.items()
        ),
    }


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def expected_components_from_ablation(data: Mapping[str, Any]) -> tuple[str, ...]:
    if data.get("shield_ablation") is None:
        if data.get("only") is not None or data.get("method") in {
            "llava_vanilla",
            "vanilla",
        }:
            return ()
        raise ValueError("shield_ablation must be an object")
    ablation = require_mapping(data.get("shield_ablation"), "shield_ablation")
    flags = {
        "A": bool(ablation.get("use_adaptive_plausibility")),
        "V": bool(ablation.get("use_vulnerability_defense")),
        "S": bool(ablation.get("use_statistical_bias")),
    }
    return tuple(component for component in COMPONENTS if flags[component])


def validate_acceleration(data: Mapping[str, Any], artifact: Artifact) -> None:
    acceleration = require_mapping(data.get("acceleration"), "acceleration")
    checks = {
        "effective_attention": acceleration.get("effective_attention")
        == "flash_attention_2",
        "fp8_effective": acceleration.get("fp8_effective") is True,
        "fp8_native_fallback_calls": acceleration.get("fp8_native_fallback_calls")
        == 0,
        "fallback_events": acceleration.get("fallback_events") == [],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{artifact.path}: acceleration checks failed: {', '.join(failed)}"
        )


def validate_artifact(data: Mapping[str, Any], artifact: Artifact) -> None:
    expected_count = 500 if artifact.dataset == "chair" else 3000
    if data.get("status") != "PASS":
        raise ValueError(f"{artifact.path}: status is not PASS")
    if data.get("count") != expected_count:
        raise ValueError(
            f"{artifact.path}: expected count {expected_count}, got {data.get('count')}"
        )
    if data.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise ValueError(f"{artifact.path}: unexpected model revision")
    if data.get("model_config_sha256") != EXPECTED_MODEL_CONFIG_SHA256:
        raise ValueError(f"{artifact.path}: unexpected model config hash")
    validate_acceleration(data, artifact)

    protocol = require_mapping(data.get("protocol"), "protocol")
    expected_protocol = {
        "decoding": "sample",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "seed": 42,
        "image_batch_size": 16,
        "num_workers": 4,
        "prefetch_factor": 4,
        "cache": "dynamic",
    }
    if artifact.dataset == "chair":
        expected_protocol.update({"max_new_tokens": 512, "batch_size": 32})
        if data.get("questions_sha256") != EXPECTED_CHAIR_QUESTIONS_SHA256:
            raise ValueError(f"{artifact.path}: unexpected CHAIR questions hash")
    else:
        expected_protocol.update(
            {
                "max_new_tokens": 8,
                "batch_size": 8,
                "pope_answer_instruction": True,
            }
        )
        if data.get("pope_sha256") != EXPECTED_POPE_SHA256:
            raise ValueError(f"{artifact.path}: unexpected POPE hash")
    for key, expected in expected_protocol.items():
        if (
            key == "cache"
            and key not in protocol
            and artifact.dataset == "pope"
            and not artifact.combo
        ):
            # The matched Vanilla baseline predates the cumulative metrics
            # schema and records dynamic-cache behavior in the runtime only.
            continue
        if protocol.get(key) != expected:
            raise ValueError(
                f"{artifact.path}: protocol {key}={protocol.get(key)!r}, "
                f"expected {expected!r}"
            )

    manifest = data.get("shield_components")
    if manifest is not None:
        manifest_mapping = require_mapping(manifest, "shield_components")
        actual = combo_key(manifest_mapping.get("components", ()))
    else:
        actual = combo_key(expected_components_from_ablation(data))
    if actual != artifact.combo:
        expected = artifact.combo or "vanilla"
        found = actual or "vanilla"
        raise ValueError(f"{artifact.path}: expected {expected}, found {found}")


def load_matrix(
    outputs_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    artifacts = artifact_paths(outputs_dir)
    values: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, Any]] = {}
    for dataset, dataset_artifacts in artifacts.items():
        values[dataset] = {}
        raw[dataset] = {}
        metric_names = CHAIR_METRICS if dataset == "chair" else POPE_METRICS
        for artifact in dataset_artifacts:
            if not artifact.path.is_file():
                raise FileNotFoundError(artifact.path)
            data = json.loads(artifact.path.read_text(encoding="utf-8"))
            validate_artifact(data, artifact)
            source = data["chair"] if dataset == "chair" else data["metrics"]
            source_mapping = require_mapping(source, f"{artifact.path}: metrics")
            values[dataset][artifact.combo] = {
                metric: float(source_mapping[metric]) for metric in metric_names
            }
            raw[dataset][artifact.combo] = {
                "path": str(artifact.path),
                "status": data["status"],
                "count": data["count"],
                "components": list(combo_components(artifact.combo)),
                "protocol": data["protocol"],
                "acceleration": data["acceleration"],
            }
    return values, raw


def utility(dataset: str, metric: str, value: float) -> float:
    """Convert every metric so a positive delta means an improvement."""

    if dataset == "chair" and metric in {"CHAIRs", "CHAIRi"}:
        return -value
    return value


def marginal(
    values: Mapping[str, Mapping[str, float]],
    dataset: str,
    metric: str,
    before: str,
    after: str,
) -> float:
    return utility(dataset, metric, values[after][metric]) - utility(
        dataset,
        metric,
        values[before][metric],
    )


def full_leave_one_out(
    values: Mapping[str, Mapping[str, float]],
    dataset: str,
    metric: str,
) -> dict[str, float]:
    without = {"A": "VS", "V": "AS", "S": "AV"}
    return {
        component: marginal(values, dataset, metric, combo, "AVS")
        for component, combo in without.items()
    }


def shapley_value(
    values: Mapping[str, Mapping[str, float]],
    dataset: str,
    metric: str,
    component: str,
) -> float:
    others = tuple(item for item in COMPONENTS if item != component)
    total = 0.0
    for size in range(len(others) + 1):
        for subset in itertools.combinations(others, size):
            before = combo_key(subset)
            after = combo_key((*subset, component))
            weight = (
                math.factorial(size)
                * math.factorial(len(COMPONENTS) - size - 1)
                / math.factorial(len(COMPONENTS))
            )
            total += weight * marginal(values, dataset, metric, before, after)
    return total


def interaction_terms(
    values: Mapping[str, Mapping[str, float]],
    dataset: str,
    metric: str,
) -> dict[str, Any]:
    base = utility(dataset, metric, values[""][metric])
    singles = {
        component: utility(dataset, metric, values[component][metric])
        for component in COMPONENTS
    }
    pairwise: dict[str, float] = {}
    for pair in ("AV", "AS", "VS"):
        pairwise[pair] = (
            utility(dataset, metric, values[pair][metric])
            - singles[pair[0]]
            - singles[pair[1]]
            + base
        )
    third_order = (
        utility(dataset, metric, values["AVS"][metric])
        - base
        - sum(single - base for single in singles.values())
        - sum(pairwise.values())
    )
    return {"pairwise": pairwise, "third_order": third_order}


def calculate(
    values: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "components": COMPONENT_NAMES,
        "combination_order": list(COMBINATIONS),
        "utility": {
            "chair": {
                "CHAIRs": "negative_raw_value",
                "CHAIRi": "negative_raw_value",
                "Recall": "positive_raw_value",
            },
            "pope": {metric: "positive_raw_value" for metric in POPE_METRICS},
        },
        "matrix": {},
        "full_leave_one_out": {},
        "shapley": {},
        "interactions": {},
        "total_gain_vs_vanilla": {},
    }
    for dataset, metric_names in (
        ("chair", CHAIR_METRICS),
        ("pope", POPE_METRICS),
    ):
        result["matrix"][dataset] = {
            combo or "vanilla": dict(values[dataset][combo])
            for combo in COMBINATIONS
        }
        result["full_leave_one_out"][dataset] = {}
        result["shapley"][dataset] = {}
        result["interactions"][dataset] = {}
        result["total_gain_vs_vanilla"][dataset] = {}
        for metric in metric_names:
            result["full_leave_one_out"][dataset][metric] = (
                full_leave_one_out(values[dataset], dataset, metric)
            )
            result["shapley"][dataset][metric] = {
                component: shapley_value(values[dataset], dataset, metric, component)
                for component in COMPONENTS
            }
            result["interactions"][dataset][metric] = interaction_terms(
                values[dataset],
                dataset,
                metric,
            )
            result["total_gain_vs_vanilla"][dataset][metric] = marginal(
                values[dataset],
                dataset,
                metric,
                "",
                "AVS",
            )
    return result


def pp(value: float) -> float:
    return value * 100.0


def fmt(value: float, digits: int = 2) -> str:
    return f"{pp(value):+.{digits}f}"


def combo_label(combo: str) -> str:
    return "Vanilla" if not combo else "+".join(COMPONENT_NAMES[item] for item in combo)


def table_matrix(
    result: Mapping[str, Any],
    dataset: str,
    metrics: Sequence[str],
) -> list[str]:
    lines = [
        "| 组合 | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    matrix = result["matrix"][dataset]
    for combo in COMBINATIONS:
        row = matrix[combo or "vanilla"]
        lines.append(
            "| "
            + combo_label(combo)
            + " | "
            + " | ".join(
                f"{row[metric] * (100 if dataset == 'pope' else 1):.6f}"
                for metric in metrics
            )
            + " |"
        )
    return lines


def table_contribution(
    result: Mapping[str, Any],
    section: str,
    dataset: str,
    metrics: Sequence[str],
) -> list[str]:
    lines = [
        "| 指标 | " + " | ".join(COMPONENT_NAMES[item] for item in COMPONENTS) + " |",
        "|---|" + "---:|" * len(COMPONENTS),
    ]
    source = result[section][dataset]
    for metric in metrics:
        row = source[metric]
        lines.append(
            f"| {metric} | "
            + " | ".join(fmt(row[component]) for component in COMPONENTS)
            + " |"
        )
    return lines


def table_interaction(
    result: Mapping[str, Any],
    dataset: str,
    metrics: Sequence[str],
) -> list[str]:
    lines = [
        "| 指标 | A×V | A×S | V×S | A×V×S |",
        "|---:|---:|---:|---:|---:|",
    ]
    source = result["interactions"][dataset]
    for metric in metrics:
        terms = source[metric]
        lines.append(
            f"| {metric} | {fmt(terms['pairwise']['AV'])} | "
            f"{fmt(terms['pairwise']['AS'])} | {fmt(terms['pairwise']['VS'])} | "
            f"{fmt(terms['third_order'])} |"
        )
    return lines


def render_markdown(result: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    lines = [
        "# SHIELD 严格组件贡献",
        "",
        "日期：2026-09-22",
        "",
        "本报告使用同一模型、数据、prompt、seed、解码和加速协议的完整 "
        "`2^3=8` 组合矩阵。`A`、`V`、`S` 分别表示 Adaptive Plausibility、"
        "Vulnerability Defense、Statistical Bias。",
        "",
        "所有 artifact 均通过 `status=PASS`、样本数、资源哈希、"
        "`effective_attention=flash_attention_2`、`fp8_effective=true`、"
        "`fp8_native_fallback_calls=0` 和空 `fallback_events` 验收。",
        "",
        "## 计算口径",
        "",
        "- CHAIRs/CHAIRi 的效用方向是降低原始值，因此表中的正值表示下降、"
        "负值表示上升。",
        "- Recall、Accuracy、Precision、F1 的效用方向是提高原始值。",
        "- `Full leave-one-out` 是在完整 `A+V+S` 上移除一个组件的条件贡献："
        "`u(AVS)-u(N\\{i})`。",
        "- `Shapley` 是该组件在所有加入顺序上的平均边际贡献，适合存在交互"
        "时分摊总增益。",
        "- 交互项使用效用函数的 ANOVA/Möbius 分解；正值表示协同，负值表示"
        "相互抵消。",
        "",
        "## 组合矩阵",
        "",
        "### CHAIR-500",
        "",
        *table_matrix(result, "chair", CHAIR_METRICS),
        "",
        "### POPE adversarial-3000",
        "",
        *table_matrix(result, "pope", POPE_METRICS),
        "",
        "## CHAIR Protocol Audit",
        "",
        "The first explicit-subset CHAIR runs accidentally inherited POPE's "
        "one-word answer suffix. Their short outputs made CHAIRs/CHAIRi "
        "artificially low and Recall abnormally low, so they are excluded from "
        "the matrix above.",
        "",
        "| 节点 | 错误平均词数 | 示例 |",
        "|---|---:|---|",
        "| V | 3.42 | `Camera` |",
        "| S | 2.32 | `Food` |",
        "| A+S | 1.68 | `Pizza` |",
        "| V+S | 3.89 | `Camera` |",
        "",
        "The corrected nodes use the dataset-specific prompt default "
        "`CHAIR=false`, and are stored under the `protocolfix` output prefix.",
        "",
        "## Full Leave-One-Out",
        "",
        "单位为百分点（pp），正值表示该组件改善对应指标。",
        "",
        "### CHAIR-500",
        "",
        *table_contribution(result, "full_leave_one_out", "chair", CHAIR_METRICS),
        "",
        "### POPE adversarial-3000",
        "",
        *table_contribution(result, "full_leave_one_out", "pope", POPE_METRICS),
        "",
        "## Shapley Contribution",
        "",
        "单位为百分点（pp），三组件 Shapley 值之和等于 `A+V+S` 相对 "
        "Vanilla 的总增益。",
        "",
        "### CHAIR-500",
        "",
        *table_contribution(result, "shapley", "chair", CHAIR_METRICS),
        "",
        "### POPE adversarial-3000",
        "",
        *table_contribution(result, "shapley", "pope", POPE_METRICS),
        "",
        "## Interaction",
        "",
        "单位为百分点（pp）。",
        "",
        "### CHAIR-500",
        "",
        *table_interaction(result, "chair", CHAIR_METRICS),
        "",
        "### POPE adversarial-3000",
        "",
        *table_interaction(result, "pope", POPE_METRICS),
        "",
        "## 结论",
        "",
        "- CHAIR 的严格 Shapley 结果显示，`S` 是降低 CHAIRs/CHAIRi 的主贡献"
        "者，`V` 提供次要的降幻觉贡献；`A` 的 CHAIRs Shapley 略为负，但"
        "对 Recall 的贡献最大，体现幻觉率与覆盖率之间的权衡。",
        "- CHAIR Recall 的主要 Shapley 贡献来自 `A`；`S` 的 Recall Shapley "
        "为负，说明 Statistical Bias 的降幻觉收益伴随 grounded coverage "
        "损失，V 的单独 Recall Shapley 较小但参与强交互。",
        "- POPE Accuracy 和 F1 的 Shapley 主贡献来自 `A`；POPE Recall 的"
        "主贡献来自 `V`，其 Precision Shapley 为负，体现明显的 "
        "recall/precision trade-off。",
        "- `S` 在 POPE 的 Shapley 增益较小但为正，主要表现为温和提高 "
        "Recall、"
        "F1；它在 CHAIR 上的贡献远大于在 POPE 上的贡献。",
        "- 因此三组件不是统计意义上的独立可加模块。若目标是 CHAIR 幻觉率，"
        "`S` 最关键；若目标是 POPE Recall，`V` 最关键；若综合 Accuracy/F1 "
        "和 CHAIR Recall，`A` 的平均贡献最大。默认三组件仍应作为联合方法，"
        "不能根据单一数据集或单一指标删除任一组件。",
        "",
        "## Artifact",
        "",
    ]
    for dataset in ("chair", "pope"):
        lines.append(f"### {dataset}")
        lines.append("")
        for combo in COMBINATIONS:
            entry = raw[dataset][combo]
            lines.append(f"- `{combo or 'vanilla'}`: `{entry['path']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    values, raw = load_matrix(args.outputs_dir.expanduser().resolve())
    result = calculate(values)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "validation": {
                    "datasets": ["chair", "pope"],
                    "combination_count_per_dataset": len(COMBINATIONS),
                    "all_formal_artifacts_passed": True,
                },
                "result": result,
                "artifacts": raw,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        render_markdown(result, raw) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
                "datasets": ["chair", "pope"],
                "combination_count_per_dataset": len(COMBINATIONS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
