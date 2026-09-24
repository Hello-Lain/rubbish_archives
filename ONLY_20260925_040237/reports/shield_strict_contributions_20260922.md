# SHIELD 严格组件贡献

日期：2026-09-22

本报告使用同一模型、数据、prompt、seed、解码和加速协议的完整 `2^3=8` 组合矩阵。`A`、`V`、`S` 分别表示 Adaptive Plausibility、Vulnerability Defense、Statistical Bias。

所有 artifact 均通过 `status=PASS`、样本数、资源哈希、`effective_attention=flash_attention_2`、`fp8_effective=true`、`fp8_native_fallback_calls=0` 和空 `fallback_events` 验收。

## 计算口径

- CHAIRs/CHAIRi 的效用方向是降低原始值，因此表中的正值表示下降、负值表示上升。
- Recall、Accuracy、Precision、F1 的效用方向是提高原始值。
- `Full leave-one-out` 是在完整 `A+V+S` 上移除一个组件的条件贡献：`u(AVS)-u(N\{i})`。
- `Shapley` 是该组件在所有加入顺序上的平均边际贡献，适合存在交互时分摊总增益。
- 交互项使用效用函数的 ANOVA/Möbius 分解；正值表示协同，负值表示相互抵消。

## 组合矩阵

### CHAIR-500

| 组合 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.518000 | 0.163492 | 0.808083 |
| Adaptive Plausibility | 0.496000 | 0.135772 | 0.840866 |
| Vulnerability Defense | 0.460000 | 0.143551 | 0.797748 |
| Statistical Bias | 0.454000 | 0.142385 | 0.795050 |
| Adaptive Plausibility+Vulnerability Defense | 0.472000 | 0.139740 | 0.836637 |
| Adaptive Plausibility+Statistical Bias | 0.432000 | 0.129335 | 0.828539 |
| Vulnerability Defense+Statistical Bias | 0.278000 | 0.109549 | 0.705422 |
| Adaptive Plausibility+Vulnerability Defense+Statistical Bias | 0.358000 | 0.100977 | 0.809490 |

### POPE adversarial-3000

| 组合 | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| Vanilla | 79.066667 | 84.439179 | 71.266667 | 77.295734 |
| Adaptive Plausibility | 82.366667 | 88.138256 | 74.800000 | 80.923188 |
| Vulnerability Defense | 77.466667 | 71.845175 | 90.333333 | 80.035440 |
| Statistical Bias | 79.666667 | 83.610272 | 73.800000 | 78.399433 |
| Adaptive Plausibility+Vulnerability Defense | 83.233333 | 83.211193 | 83.266667 | 83.238920 |
| Adaptive Plausibility+Statistical Bias | 82.700000 | 87.528692 | 76.266667 | 81.510509 |
| Vulnerability Defense+Statistical Bias | 76.300000 | 69.954476 | 92.200000 | 79.551337 |
| Adaptive Plausibility+Vulnerability Defense+Statistical Bias | 83.633333 | 82.781027 | 84.933333 | 83.843370 |

## CHAIR Protocol Audit

The first explicit-subset CHAIR runs accidentally inherited POPE's one-word answer suffix. Their short outputs made CHAIRs/CHAIRi artificially low and Recall abnormally low, so they are excluded from the matrix above.

| 节点 | 错误平均词数 | 示例 |
|---|---:|---|
| V | 3.42 | `Camera` |
| S | 2.32 | `Food` |
| A+S | 1.68 | `Pizza` |
| V+S | 3.89 | `Camera` |

The corrected nodes use the dataset-specific prompt default `CHAIR=false`, and are stored under the `protocolfix` output prefix.

## Full Leave-One-Out

单位为百分点（pp），正值表示该组件改善对应指标。

### CHAIR-500

| 指标 | Adaptive Plausibility | Vulnerability Defense | Statistical Bias |
|---|---:|---:|---:|
| CHAIRs | -8.00 | +7.40 | +11.40 |
| CHAIRi | +0.86 | +2.84 | +3.88 |
| Recall | +10.41 | -1.90 | -2.71 |

### POPE adversarial-3000

| 指标 | Adaptive Plausibility | Vulnerability Defense | Statistical Bias |
|---|---:|---:|---:|
| accuracy | +7.33 | +0.93 | +0.40 |
| precision | +12.83 | -4.75 | -0.43 |
| recall | -7.27 | +8.67 | +1.67 |
| f1 | +4.29 | +2.33 | +0.60 |

## Shapley Contribution

单位为百分点（pp），三组件 Shapley 值之和等于 `A+V+S` 相对 Vanilla 的总增益。

### CHAIR-500

| 指标 | Adaptive Plausibility | Vulnerability Defense | Statistical Bias |
|---|---:|---:|---:|
| CHAIRs | -1.77 | +7.73 | +10.03 |
| CHAIRi | +1.49 | +2.09 | +2.67 |
| Recall | +5.77 | -2.54 | -3.08 |

### POPE adversarial-3000

| 指标 | Adaptive Plausibility | Vulnerability Defense | Statistical Bias |
|---|---:|---:|---:|
| accuracy | +5.01 | -0.64 | +0.19 |
| precision | +8.06 | -8.88 | -0.84 |
| recall | -2.01 | +13.72 | +1.96 |
| f1 | +3.69 | +2.27 | +0.59 |

## Interaction

单位为百分点（pp）。

### CHAIR-500

| 指标 | A×V | A×S | V×S | A×V×S |
|---:|---:|---:|---:|---:|
| CHAIRs | -3.40 | +0.00 | +11.80 | -6.80 |
| CHAIRi | -2.39 | -1.47 | +1.29 | +1.94 |
| Recall | +0.61 | +0.07 | -7.93 | +6.45 |

### POPE adversarial-3000

| 指标 | A×V | A×S | V×S | A×V×S |
|---:|---:|---:|---:|---:|
| accuracy | +2.47 | -0.27 | -1.77 | +1.83 |
| precision | +7.67 | +0.22 | -1.06 | +1.24 |
| recall | -10.60 | -1.07 | -0.67 | +0.87 |
| f1 | -0.42 | -0.52 | -1.59 | +1.60 |

## 结论

- CHAIR 的严格 Shapley 结果显示，`S` 是降低 CHAIRs/CHAIRi 的主贡献者，`V` 提供次要的降幻觉贡献；`A` 的 CHAIRs Shapley 略为负，但对 Recall 的贡献最大，体现幻觉率与覆盖率之间的权衡。
- CHAIR Recall 的主要 Shapley 贡献来自 `A`；`S` 的 Recall Shapley 为负，说明 Statistical Bias 的降幻觉收益伴随 grounded coverage 损失，V 的单独 Recall Shapley 较小但参与强交互。
- POPE Accuracy 和 F1 的 Shapley 主贡献来自 `A`；POPE Recall 的主贡献来自 `V`，其 Precision Shapley 为负，体现明显的 recall/precision trade-off。
- `S` 在 POPE 的 Shapley 增益较小但为正，主要表现为温和提高 Recall、F1；它在 CHAIR 上的贡献远大于在 POPE 上的贡献。
- 因此三组件不是统计意义上的独立可加模块。若目标是 CHAIR 幻觉率，`S` 最关键；若目标是 POPE Recall，`V` 最关键；若综合 Accuracy/F1 和 CHAIR Recall，`A` 的平均贡献最大。默认三组件仍应作为联合方法，不能根据单一数据集或单一指标删除任一组件。

## Artifact

### chair

- `vanilla`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_vanilla_b32_20260921.metrics.json`
- `A`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_cumulative_adaptive_b32_20260922.metrics.json`
- `V`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_subset_vulnerability_protocolfix_20260922.metrics.json`
- `S`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_subset_statistical_protocolfix_20260922.metrics.json`
- `AV`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_cumulative_vulnerability_lr002_b32_20260922.metrics.json`
- `AS`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_subset_adaptive_statistical_protocolfix_20260922.metrics.json`
- `VS`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_subset_vulnerability_statistical_protocolfix_20260922.metrics.json`
- `AVS`: `/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_cumulative_statistical_b32_20260922.metrics.json`

### pope

- `vanilla`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_promptfix_vanilla_20260922.metrics.json`
- `A`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_cumulative_adaptive_official_20260922.metrics.json`
- `V`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_subset_vulnerability_official_20260922.metrics.json`
- `S`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_subset_statistical_official_20260922.metrics.json`
- `AV`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_cumulative_vulnerability_official_20260922.metrics.json`
- `AS`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_subset_adaptive_statistical_official_20260922.metrics.json`
- `VS`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_subset_vulnerability_statistical_official_20260922.metrics.json`
- `AVS`: `/data/lcq/Downloads/ONLY/outputs/pope_llava_7b_adversarial_cumulative_statistical_official_20260922.metrics.json`
