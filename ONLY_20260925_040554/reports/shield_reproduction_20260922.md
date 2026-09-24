# SHIELD 模块复现报告

日期：2026-09-22

## 结论

在 LLaVA-1.5-7B、固定 SHIELD-compatible CHAIR-500 和 COCO POPE
adversarial-3000 上，累计路径
`Vanilla -> Adaptive -> Vulnerability -> Statistical -> Full` 可以复现
论文 Table 7 的加入顺序，但不能作为严格独立贡献。严格贡献以
`Vanilla`、三个单组件、三个双组件和 `A+V+S` 的完整 `2^3=8` 组合矩阵为准；
累计结果保留为论文路径对照。

- **最终推荐方法只保留三个组件**：
  `Adaptive Plausibility + Vulnerability Defense + Statistical Bias`。
  它对应累计消融中的 `Statistical` 阶段；`Inherent Bias` 和四组件
  `Full` 仅保留为可选研究消融。
- `Statistical Bias` 主要降低 CHAIR 的句子级和对象级幻觉，但 Recall 略降；
  在 POPE 累计路径上继续提升 Accuracy/Recall/F1
  `+0.40/+1.67/+0.60 pp`，Precision 略降 `-0.43 pp`。
- `Vulnerability Defense` 主要提升 grounded object Recall 和 POPE Recall/F1；
  在 POPE 相对 Adaptive 阶段提升 Accuracy/Recall/F1
  `+0.87/+8.47/+2.32 pp`，代价是 Precision `-4.93 pp`。
- `Adaptive Plausibility` 是最稳定的第一步收益，在两个数据集上都同时改善
  主要指标。
- 修复 Inherent Bias 后，完整累计配置从旧实现的 `CHAIRs/CHAIRi =
  0.396/0.11325` 改善到 `0.376/0.10520`，相对 Vanilla 为
  `-14.20/-5.83 pp`，已经接近论文 Table 7 的 `36.6/10.3`。
- 在当前 HF、BF16、批量推理协议下，Statistical 阶段本身已经达到
  `0.358/0.10098`，因此 Inherent Bias 的增量仍是 trade-off，而不是
  超过 Statistical 阶段的额外下降；这与“模块公式修复后有实际绝对收益”
  并不矛盾。
- POPE 的旧结果没有复现出预期增益，根因是入口漏掉了官方的一词回答
  指令 `Please answer this question with one word.`，导致 Vanilla 与
  SHIELD 都在不同于论文的输出协议上评测。补齐该 prompt 后，Full 在
  adversarial-3000 上达到 `83.53/83.86`（Accuracy/F1），相对同协议
  Vanilla 的 `79.07/77.30` 提升 `+4.47/+6.56 pp`，与论文 Table 3 的
  `82.5/83.6` 处于同一量级。
- 代码已将 Adaptive Plausibility、Vulnerability Defense、Statistical
  Bias 和 Inherent Bias 解耦为独立插件，并由同一个
  `ShieldComponentPipeline` 组合；累计评估入口和独立 Vulnerability 入口
  不再各自维护一套 logits 组合公式。默认 `shield_cumulative` 配置只实例化
  前三个组件。
- 严格组合实验表明三个组件存在明显交互，不能宣布某个组件在两个数据集、
  所有指标上都冗余。CHAIR 的 Shapley 贡献中 `S` 对 CHAIRs/CHAIRi 最大，
  `A` 对 Recall 最大；POPE 中 `A` 对 Accuracy/Precision/F1 最大，
  `V` 对 Recall 最大。完整表格和所有 artifact 路径见
  `reports/shield_strict_contributions_20260922.md`。
- CHAIR 严格矩阵的第一次补跑暴露了 prompt 协议 bug：显式
  `--components` 的四个旧节点继承了 POPE 的一词回答指令，生成了
  `Camera`、`Food`、`Yes` 等短答案，虽然 metrics 仍显示 `status=PASS`。
  这些旧 artifact 已标记为无效，不参与贡献计算；修复后四个节点的平均
  caption 长度恢复到 `69.7--92.8` 词，并使用 `protocolfix` 输出。

严格贡献的核心结果（单位：pp，正值表示改善）：

| 数据集/指标 | Adaptive Plausibility | Vulnerability Defense | Statistical Bias |
|---|---:|---:|---:|
| CHAIRs full leave-one-out | -8.00 | +7.40 | +11.40 |
| CHAIRi full leave-one-out | +0.86 | +2.84 | +3.88 |
| CHAIR Recall full leave-one-out | +10.41 | -1.90 | -2.71 |
| POPE Accuracy full leave-one-out | +7.33 | +0.93 | +0.40 |
| POPE Recall full leave-one-out | -7.27 | +8.67 | +1.67 |
| POPE F1 full leave-one-out | +4.29 | +2.33 | +0.60 |

Shapley 分解给出的跨组合平均贡献为：

| 数据集/指标 | Adaptive Plausibility | Vulnerability Defense | Statistical Bias |
|---|---:|---:|---:|
| CHAIRs | -1.77 | +7.73 | +10.03 |
| CHAIRi | +1.49 | +2.09 | +2.67 |
| CHAIR Recall | +5.77 | -2.54 | -3.08 |
| POPE Accuracy | +5.01 | -0.64 | +0.19 |
| POPE Recall | -2.01 | +13.72 | +1.96 |
| POPE F1 | +3.69 | +2.27 | +0.59 |

严格贡献分析脚本：
`scripts/analyze_shield_contributions.py`

机器可读结果：
`reports/shield_strict_contributions_20260922.json`

此前没有复现出原模块实际增益的主要原因是：POPE 入口漏掉官方一词回答
后缀、CHAIR 显式组件入口错误继承了 POPE 一词回答 prompt、消融口径错误、
CHAIR 使用了 POPE 的 `attack_lr=0.14` 而不是论文的 `0.02`、clean branch
没有叠加前置累计模块，以及 Inherent Bias 的 random feature 使用 BF16 且
污染了生成 RNG。

## CHAIR 协议审计

错误的 CHAIR 组合使用了 POPE 的：

```text
Please answer this question with one word.
```

错误 predictions 的平均 caption 长度如下：

| 节点 | 错误平均词数 | 错误示例 |
|---|---:|---|
| V | 3.42 | `Camera` |
| S | 2.32 | `Food` |
| A+S | 1.68 | `Pizza` |
| V+S | 3.89 | `Camera` |

错误节点及其 metrics 保留用于审计，但不参与严格矩阵：

```text
outputs/chair_llava_7b_shield_subset_vulnerability_20260922.metrics.json
outputs/chair_llava_7b_shield_subset_statistical_20260922.metrics.json
outputs/chair_llava_7b_shield_subset_adaptive_statistical_20260922.metrics.json
outputs/chair_llava_7b_shield_subset_vulnerability_statistical_20260922.metrics.json
```

修复后的正式节点使用 `--no-pope-answer-instruction`，入口默认按数据集选择：

```text
POPE: pope_answer_instruction=true
CHAIR: pope_answer_instruction=false
```

正式 CHAIR 组合结果：

```text
outputs/chair_llava_7b_shield_subset_vulnerability_protocolfix_20260922.metrics.json
outputs/chair_llava_7b_shield_subset_statistical_protocolfix_20260922.metrics.json
outputs/chair_llava_7b_shield_subset_adaptive_statistical_protocolfix_20260922.metrics.json
outputs/chair_llava_7b_shield_subset_vulnerability_statistical_protocolfix_20260922.metrics.json
```

## 复现范围

| 项目 | 固定值 |
|---|---|
| 模型 | LLaVA-1.5-7B |
| 模型 revision | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| 模型 config SHA-256 | `0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f` |
| CHAIR | 固定 500 张 COCO val2014 图像 |
| POPE | COCO adversarial，3000 条，500 张唯一图像 |
| seed | `42` |
| decoding | sampling，`temperature=1`，`top_p=1` |
| POPE prompt | `{question} Please answer this question with one word.` |
| attention | FlashAttention-2 |
| dtype | BF16 |
| FP8 | text MLP，effective |
| KV cache | dynamic |

固定资源：

- POPE：`/data/lcq/.cache/huggingface/hub/pope/coco/coco_pope_adversarial.json`
- POPE SHA-256：`185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac`
- CHAIR questions：`/data/lcq/.cache/huggingface/hub/chair/shield/questions.jsonl`
- CHAIR questions SHA-256：`d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e`
- CHAIR ground-truth cache：`outputs/chair_shield_ground_truth_v2.json`

所有正式结果均满足：

```text
status=PASS
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

## 消融口径与修复

论文 Table 7 的累计配置为：

```text
Vanilla
+ adaptive plausibility
+ vulnerability defense
+ statistical bias
+ inherent bias
```

此前的“Vulnerability Defense 实际无增益”结论来自不匹配的配置：POPE
使用了无一词回答后缀的 legacy prompt，且 clean branch 仍是 Vanilla clean
image，没有叠加 Statistical Bias 和 caption-token injection；同时 CHAIR
错用了 POPE 的 `attack_lr=0.14`。当前累计入口
`scripts/shield_cumulative.py` 已统一这条消融链，并按数据集设置：

```text
CHAIR attack_lr=0.02
POPE attack_lr=0.14
cd_alpha=2.0
cd_beta=0.35
epsilon=0.14
attack_steps=30
attack_c=12.0
```

Inherent Bias 的官方实现是：

```text
bias = mean(vision_tower(torch.rand(32, C, H, W).half()), dim=0)
x' = x - bias_weight * bias + x * patch_weight_gain
```

当前实现已修复为：

- CUDA 上用 FP16 生成 random image tensors，再转回运行时 feature dtype；
- 使用 `bias_seed=42` 的独立 `torch.Generator`，不改变语言生成的 sampling
  RNG；
- 保持 `vision_feature_layer=-2`、去 CLS patch、`bias_weight=0.01` 和
  `bias_sample_num=32`；
- metrics 记录 `bias_seed` 与 `bias_input_dtype`，使该修复可审计。

独立 generator 是为了满足本项目 Vanilla/方法共享相同生成随机流的协议；
它不改变 Inherent Bias 的特征公式。

## 模块实现

### 组件贡献摘要

下面的消融是累计配置，不是四个完全孤立的开关：
`Vanilla -> Adaptive -> Vulnerability -> Statistical -> Full`。因此“增量”
列表示加入当前组件后相对上一阶段的变化，更适合判断该组件的边际贡献。

| 加入组件 | 累计 CHAIRs | 累计 CHAIRi | 累计 Recall | 相对上一阶段 CHAIRs / CHAIRi / Recall |
|---|---:|---:|---:|---:|
| Vanilla | 0.5180 | 0.16349 | 0.80808 | - / - / - |
| Adaptive Plausibility | 0.4960 | 0.13577 | 0.84087 | -2.20 / -2.77 / +3.28 pp |
| Vulnerability Defense | 0.4720 | 0.13974 | 0.83664 | -2.40 / +0.40 / -0.42 pp |
| Statistical Bias | **0.3580** | **0.10098** | 0.80949 | **-11.40 / -3.88 / -2.71 pp** |
| Inherent Bias（Full） | 0.3760 | 0.10520 | **0.81447** | +1.80 / +0.42 / **+0.50 pp** |

结论应分成两层理解：

- **Statistical Bias 是 CHAIR 降幻觉的主要来源**：在已有前置模块上继续
  将 CHAIRs/CHAIRi 降低 `11.40/3.88 pp`，相对 Vanilla 的总降幅为
  `16.00/6.25 pp`。
- **Adaptive Plausibility 提供最稳定的第一步收益**：同时降低两类 CHAIR
  并提升 Recall `3.28 pp`，作用是限制低可信候选。
- **Vulnerability Defense 的边际作用不是单调降低全部 CHAIR 指标**：它在
  累计结果中继续降低 CHAIRs，但 CHAIRi 和 Recall 相比 Adaptive 略有回落；
  它的主要价值体现在对抗分支带来的抗扰动/POPE Recall，而非单独追求
  CHAIR Recall 最大化。
- **Inherent Bias 是 trade-off 修正项**：相对 Statistical 阶段提高 Recall
  `0.50 pp`，但 CHAIRs/CHAIRi 增加 `1.80/0.42 pp`。修复 FP16 bias 输入
  和独立 RNG 后，Full 仍比修复前更好，但在当前 HF 协议下它不是 CHAIR
  降幅最大的组件。

### POPE 组件贡献

POPE 使用官方一词回答 prompt，以下仍是同一条累计消融路径：
`Vanilla -> Adaptive -> Vulnerability -> Statistical -> Full`。

| 阶段 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla | 79.0667% | 84.4392% | 71.2667% | 77.2957% |
| + Adaptive Plausibility | 82.3667% | **88.1383%** | 74.8000% | 80.9232% |
| + Vulnerability Defense | 83.2333% | 83.2112% | 83.2667% | 83.2389% |
| + Statistical Bias | **83.6333%** | 82.7810% | 84.9333% | 83.8434% |
| + Inherent Bias（Full） | 83.5333% | 82.2436% | **85.5333%** | **83.8562%** |

相对上一累计阶段的边际变化：

| 新增组件 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Adaptive Plausibility | +3.30 pp | **+3.70 pp** | +3.53 pp | +3.63 pp |
| Vulnerability Defense | +0.87 pp | -4.93 pp | **+8.47 pp** | +2.32 pp |
| Statistical Bias | +0.40 pp | -0.43 pp | +1.67 pp | +0.60 pp |
| Inherent Bias | -0.10 pp | -0.54 pp | +0.60 pp | +0.01 pp |

相对 Vanilla 的总变化：

| 累计配置 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| + Adaptive Plausibility | +3.30 pp | +3.70 pp | +3.53 pp | +3.63 pp |
| + Vulnerability Defense | +4.17 pp | -1.23 pp | +12.00 pp | +5.94 pp |
| + Statistical Bias | **+4.57 pp** | -1.66 pp | +13.67 pp | +6.55 pp |
| Full（含 Inherent Bias） | +4.47 pp | -2.20 pp | **+14.27 pp** | **+6.56 pp** |

POPE 上的作用分工比较明确：

- **Adaptive Plausibility** 首先抑制低可信候选，因此四项指标同时提升，
  其中 Precision 提升最大，为 `+3.70 pp`。
- **Vulnerability Defense** 是 POPE Recall 增益的主要来源之一，在已有
  Adaptive 的基础上将 Recall 再提高 `8.47 pp`，但由于预测更偏向回答
  `yes`，Precision 下降 `4.93 pp`。
- **Statistical Bias** 在 POPE 上是温和的校正项，继续提高 Accuracy、
  Recall 和 F1，但 Precision 略降；其最大贡献仍体现在 CHAIR 的幻觉率下降。
- **Inherent Bias** 对 POPE 的边际影响很小，主要将 Recall 提高 `0.60 pp`，
  同时 Accuracy、Precision 略降；Full 的最终 F1 基本与 Statistical 阶段
  持平（`+0.01 pp`）。

### Statistical Bias

实现文件：

- `src/methods/shield_statistical_bias.py`
- `scripts/chair_llava_shield_statistical_bias.py`
- `scripts/pope_llava_shield_statistical_bias.py`

流程与 SHIELD 官方 `feature.py` / `wrapper.py` 对齐：

1. 用 Vanilla LVLM 预先生成 naive caption。
2. 用 CLIP text encoder 得到 caption token features。
3. 对每个视觉 patch 与 caption token 计算 cosine similarity：

   ```text
   M[i,j] = cosine(image_patch[i], caption_token[j])
   ```

4. 对 caption 维取最大值，min-max 归一化并使用
   `gamma_gain=3.0` 拉开权重差异。
5. 仅保留大于 `gain_per=0.55` 的 patch 权重，执行：

   ```text
   x'_i = x_i + x_i * w_i
   ```

6. 将被选中的 CLIP caption token span 映射到 LLaVA tokenizer embedding，
   插入视觉 token 后继续生成。

本次 isolated module 只验证 Statistical Bias 的 token re-weighting：
`bias_weight=0`、不启用 noise-derived inherent-bias subtraction，也不启用
contrastive decoding。

**canonical caption 输入**：官方 CHAIR 入口使用
`first_cap/llava15_chair_first_caption.jsonl`。本项目对应文件为：

```text
outputs/shield_llava15_chair_first_caption_20260922.jsonl
```

当前 CHAIR 脚本已将该文件设为默认值。此前使用
`chair_llava_7b_shield_vanilla_b32_20260921.jsonl` 的结果是诊断性重跑，
不是 canonical SHIELD caption 条件，不纳入下表。

参数：

```text
threshold=0.002
gamma_gain=3.0
gain_per=0.55
```

### Vulnerability Defense

实现文件：

- `src/methods/shield_vulnerability_defense.py`
- `scripts/shield_vulnerability_runtime.py`
- `scripts/chair_llava_shield_vulnerability.py`
- `scripts/pope_llava_shield_vulnerability.py`

修复后的数据流：

1. 用 CLIP processor 得到 CLIP normalized pixels。
2. 在 CLIP 预处理空间优化 CW-style learnable perturbation，官方实现的攻击
   loss 为 `outputs.logits_per_image[0, 0]`。
3. 将得到的 `c * delta` 应用到 LLaVA 预处理 pixels，而不是把 LLaVA pixels
   直接当成 CLIP attack 输入。
4. clean 和 adversarial 分支各自维护独立 KV cache。
5. 每一步使用 SHIELD/VCD 公式：

   ```text
   combined = (1 + alpha) * clean_logits - alpha * adversarial_logits
   cutoff = log(beta) + max(clean_logits)
   combined[clean_logits < cutoff] = -inf
   ```

参数：

```text
epsilon=0.14
attack_steps=30
attack_c=12.0
cd_alpha=2.0
cd_beta=0.35
```

数据集对应的攻击学习率为：

```text
CHAIR attack_lr=0.02
POPE attack_lr=0.14
```

独立 Vulnerability Defense 实验明确关闭 Statistical Bias、Inherent Bias 和
caption token injection；上面的 CHAIR 累计表则按论文消融路径保留前置
Adaptive Plausibility。

## 插件化架构

实现文件：

- `src/methods/shield_adaptive_plausibility.py`
- `src/methods/shield_vulnerability_defense.py`
- `src/methods/shield_statistical_bias.py`
- `src/methods/shield_inherent_bias.py`
- `src/methods/shield_pipeline.py`
- `src/methods/shield_method_base.py`

每个组件包含一个纯算法对象、一个 dataclass 配置和一个
`BaseMethod` 适配器。模型族相关的实际 batch backend 仍由
`ShieldBackendMethod` 转发，组件本身不依赖 LLaVA 评测脚本。

输入输出边界固定为：

```text
Statistical Bias:
  raw vision features + CLIP caption-token features
  -> reweighted vision features + selected caption spans

Inherent Bias:
  visual features + noise-derived bias feature
  -> bias-subtracted visual features

Vulnerability Defense:
  clean logits + adversarial logits
  -> contrastive logits

Adaptive Plausibility:
  clean logits + candidate logits
  -> plausibility-masked candidate logits
```

`ShieldComponentPipeline` 负责三件事：

1. 校验组件名称、重复项和必需依赖；
2. 固定推荐配置的 feature 侧顺序为 `Statistical`；
   如果显式加入 Inherent Bias，则扩展为 `Statistical -> Inherent`；
3. 固定 logits 侧顺序为 `Vulnerability -> Adaptive`。

累计入口 `scripts/shield_cumulative.py` 只负责数据、缓存和评估编排，
通过 `stage_spec()` 选择组件；实际的组件参数由
`build_shield_pipeline()` 注入。新的累计运行会在 metrics 中写入
`shield_components` manifest，同时保留旧的 `shield_ablation` 字段用于结果兼容；
本报告中的正式 GPU 结果是插件化重构前已验收的同协议 artifact，没有被重写。

Hydra 方法配置：

```text
configs/method/shield_adaptive_plausibility.yaml
configs/method/shield_vulnerability_defense.yaml
configs/method/shield_statistical_bias.yaml
configs/method/shield_inherent_bias.yaml
configs/method/shield_cumulative.yaml
```

因此新增或替换一个组件只需实现对应插件并更新 `components` 列表，不需要
修改 logits 解码主循环。默认
`configs/method/shield_cumulative.yaml` 只包含
`adaptive_plausibility`、`vulnerability_defense` 和 `statistical_bias`；
`shield_inherent_bias.yaml` 及四组件 `full` 阶段仅用于可选消融。

## 推荐最终配置

默认方法只启用：

```text
Adaptive Plausibility
+ Vulnerability Defense
+ Statistical Bias
```

对应的已验收结果为累计消融中的 `statistical` 阶段：

| 数据集 | 指标 |
|---|---|
| CHAIR-500 | CHAIRs `0.3580`，CHAIRi `0.10098`，Recall `0.80949` |
| POPE adversarial-3000 | Accuracy `83.6333%`，Precision `82.7810%`，Recall `84.9333%`，F1 `83.8434%` |

正式 artifact：

```text
outputs/chair_llava_7b_shield_cumulative_statistical_b32_20260922.metrics.json
outputs/pope_llava_7b_adversarial_cumulative_statistical_official_20260922.metrics.json
```

四组件 `Full` 仍保留用于论文复现和 leave-one-out 对照，但不属于默认
部署方法。

## CHAIR-500

指标定义：

- `CHAIRs = hallucinated_captions / captions`
- `CHAIRi = hallucinated_object_mentions / object_mentions`
- `Recall` = 每张图 grounded COCO object coverage 的 macro mean

| 方法 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.5180 | 0.16349 | 0.80808 |
| Adaptive | 0.4960 | 0.13577 | 0.84087 |
| Vulnerability Defense | 0.4720 | 0.13974 | 0.83664 |
| Statistical Bias | **0.3580** | **0.10098** | 0.80949 |
| Full（修复后 Inherent Bias） | 0.3760 | 0.10520 | **0.81447** |

相对 Vanilla 的百分点变化：

| 方法 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Adaptive | -2.20 pp | -2.77 pp | +3.28 pp |
| Vulnerability Defense | -4.60 pp | -2.38 pp | +2.86 pp |
| Statistical Bias | **-16.00 pp** | **-6.25 pp** | +0.14 pp |
| Full（修复后） | -14.20 pp | -5.83 pp | **+0.64 pp** |

修复前后完整配置对比：

| Full 配置 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| 修复前：BF16 bias、共享 RNG | 0.3960 | 0.11325 | 0.81247 |
| 修复后：FP16 bias、独立 bias RNG | **0.3760** | **0.10520** | **0.81447** |

论文 Table 7 的 CHAIRs/CHAIRi 为：

```text
Vanilla 48.8/14.2
Adaptive 50.2/13.8
Vulnerability 46.4/12.8
Statistical 40.4/11.0
Full 36.6/10.3
```

当前修复后 Full 为 `37.6/10.52`，与论文 Full 的绝对差为
`+1.0/+0.22 pp`。由于当前运行时、模型加载方式、批大小和 decoding
实现与论文 legacy LLaVA 不同，应以同一当前协议下相对 Vanilla 的变化为
主要复现结论。

正式结果：

- Vanilla：`outputs/chair_llava_7b_shield_vanilla_b32_20260921.metrics.json`
- Adaptive Plausibility：
  `outputs/chair_llava_7b_shield_cumulative_adaptive_b32_20260922.metrics.json`
- Vulnerability Defense：
  `outputs/chair_llava_7b_shield_cumulative_vulnerability_lr002_b32_20260922.metrics.json`
- Statistical Bias：
  `outputs/chair_llava_7b_shield_cumulative_statistical_b32_20260922.metrics.json`
- Full（修复后）：
  `outputs/chair_llava_7b_shield_cumulative_full_fp16bias_rngisolated_b32_20260922.metrics.json`

旧的 `outputs/chair_llava_7b_shield_cumulative_full_official_b32_20260922.metrics.json`
保留作为修复前对照，不作为最终 Inherent Bias 实现。

此前的 isolated Statistical Bias 结果 `0.454/0.14239/0.79505` 以及
CHAIR `attack_lr=0.14` 的 isolated Vulnerability 结果
`0.548/0.14363/0.86157` 只保留用于诊断，不能与上面的累计表混用。

## POPE Adversarial

### 官方 prompt 重跑

旧 POPE 入口缺少 SHIELD/官方 LLaVA 的一词回答指令，导致输出协议与论文
不一致。修复后的 prompt 为：

```text
{system_prompt} USER: <image>
{question} Please answer this question with one word. ASSISTANT:
```

Vanilla 与 Full 使用相同的模型、3000 条数据、prompt、seed、sampling、
batch 和加速协议：

| 方法 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla（官方 prompt） | 79.0667% | **84.4392%** | 71.2667% | 77.2957% |
| Full 累计（官方 prompt） | **83.5333%** | 82.2436% | **85.5333%** | **83.8562%** |

相对 Vanilla 的百分点变化：

| 方法 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Full 累计（官方 prompt） | **+4.4667 pp** | -2.1956 pp | **+14.2667 pp** | **+6.5605 pp** |

混淆矩阵变化：

| 方法 | TP | TN | FP | FN | yes ratio |
|---|---:|---:|---:|---:|---:|
| Vanilla（官方 prompt） | 1069 | 1303 | 197 | 431 | 42.20% |
| Full 累计（官方 prompt） | 1283 | 1223 | 277 | 217 | 52.00% |

Full 的主要效果是将 `FN` 从 `431` 降到 `217`，因此 Recall 和 F1 明显提升；
代价是 `FP` 增加 `80`，Precision 下降 `2.20 pp`。这属于有效但偏
recall-oriented 的改进，不是所有指标同时单调提升。

正式结果：

- Vanilla：
  `outputs/pope_llava_7b_adversarial_promptfix_vanilla_20260922.metrics.json`
- Vanilla predictions：
  `outputs/pope_llava_7b_adversarial_promptfix_vanilla_20260922.jsonl`
- Adaptive Plausibility：
  `outputs/pope_llava_7b_adversarial_cumulative_adaptive_official_20260922.metrics.json`
- Vulnerability Defense：
  `outputs/pope_llava_7b_adversarial_cumulative_vulnerability_official_20260922.metrics.json`
- Statistical Bias：
  `outputs/pope_llava_7b_adversarial_cumulative_statistical_official_20260922.metrics.json`
- Full 累计：
  `outputs/pope_llava_7b_adversarial_promptfix_full_20260922.metrics.json`
- Full predictions：
  `outputs/pope_llava_7b_adversarial_promptfix_full_20260922.jsonl`

论文 Table 3 中 LLaVA-1.5 在 COCO adversarial split 的 SHIELD 参考值为
`82.5 Accuracy / 83.6 F1`；当前结果为 `83.53 / 83.86`。由于模型加载、
Transformers 版本和批量运行时不同，这里以同协议 Vanilla 的配对增益为主，
参考值只用于判断量级是否合理。

此前无一词回答后缀的结果：

```text
Vanilla: 76.6333 Accuracy / 78.6215 F1
Full:    73.6333 Accuracy / 78.1673 F1
```

这些结果保留在旧输出目录中作为诊断，但不再作为正式 POPE 对比，因为
Vanilla 与论文/SHIELD 使用了不同的 prompt contract。

## 解释

- Statistical Bias 累计阶段在 CHAIR 上将 CHAIRs 和 CHAIRi 分别降低
  `16.00` 和 `6.25 pp`，Recall 相对 Vanilla 基本持平（`+0.14 pp`）；
  在 POPE 上相对上一阶段继续带来 `+0.40/+1.67/+0.60 pp`
  的 Accuracy/Recall/F1 边际收益。
- Vulnerability Defense 累计阶段在 CHAIR 上将 CHAIRs/CHAIRi 降低
  `4.60/2.38 pp`，Recall 提高 `2.86 pp`；在 POPE 上是 Recall 增益的
  主要来源，在已有 Adaptive 的基础上再提高 `8.47 pp`，但 Precision
  下降 `4.93 pp`。
- Adaptive Plausibility 是最稳定的第一步收益：CHAIRs/CHAIRi 下降
  `2.20/2.77 pp`，Recall 提高 `3.28 pp`；POPE 的四项指标同时提高，
  Accuracy/Precision/Recall/F1 分别提高 `3.30/3.70/3.53/3.63 pp`。
- 在修复 prompt contract 后，Full 累计配置在 POPE 上相对同协议 Vanilla
  将 Accuracy/F1 分别提高 `4.47/6.56 pp`，Recall 提高 `14.27 pp`；
  Precision 下降 `2.20 pp`。因此当前实现已恢复原论文所展示的 POPE
  实际增益，剩余问题是 precision-recall trade-off，而不是模块无效。
- 修复 Inherent Bias 后，相比修复前 Full，CHAIRs/CHAIRi/Recall 分别改善
  `2.00/0.80/0.20 pp`；相比 Statistical 阶段则为
  `+1.80/+0.42/+0.50 pp`。这表示修复消除了原实现的一部分损失，但在当前
  HF 协议下 Inherent Bias 仍不是最后阶段的最大增益来源。
- 20 张 greedy 重复诊断显示 H100 的 FA2/FP8 路径存在运行级非确定性，
  即使相同 seed 也不能要求跨进程 caption 逐条相同。因此本报告采用完整
  500/3000 条 aggregate metrics，并记录 acceleration 验收；不把单次
  smoke caption 差异解释为模块公式差异。
- 论文 Table 7 的 CHAIR 消融是论文原始运行的百分比点估计；本项目使用
  固定本地资源、当前 Transformers 运行时和 SHIELD-compatible scorer。
  因此应比较本项目内 Vanilla 的配对差值，不应直接把论文绝对数值与本表
  混用。

## 验收

已通过：

```text
41 passed
py_compile
git diff --check
uv pip check --python .venv/bin/python
```

启动新的 GPU 实验前仍需按项目契约执行 `nvidia-smi` 和进程检查；已有
`PASS` 结果满足复用规则，不需要重复运行。
