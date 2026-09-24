# 三组件视觉幻觉抑制方法

## 摘要

本文档只记录当前仓库中的实际实现，不复述外部方法描述。当前方法是一个
推理阶段的可组合插件，由以下三个组件组成：

| 符号 | 组件 | 主要操作 |
|---|---|---|
| A | Adaptive Plausibility | 使用 clean branch 的置信度约束 candidate logits |
| V | Vulnerability Defense | 构造攻击视觉分支并执行 contrastive decoding |
| S | Statistical Bias | 使用 caption-token 相似度重加权 raw visual tokens |

默认组合为 `A + V + S`。代码中的 `Inherent Bias` 仍可作为独立研究插件使用，
但不属于本文档记录的默认三组件方法，也不纳入本文的三组件结果。

方法不更新视觉编码器、multimodal projector 或语言模型的参数。V 组件只在
推理阶段优化一个临时的图像扰动，用于构造 adversarial branch。

## 1. 问题定义

记号定义如下：

| 符号 | 含义 |
|---|---|
| `v` | 输入图像 |
| `t` | 文本 prompt |
| `y_<i>` | 第 `i` 个解码步骤之前已经生成的 token |
| `E` | LLaVA vision tower |
| `P` | LLaVA multimodal projector |
| `X` | raw visual-token features |
| `z_i` | 第 `i` 步的词表 logits |
| `Vocab` | 模型词表 |

视觉编码器首先产生 raw visual tokens：

```text
X = E(v)
```

视觉特征经过 multimodal projector 和语言模型后得到 logits：

```text
z_i = LLM(P(X), t, y_<i)
```

Vanilla sampling 使用：

```text
p_i(w) = softmax(z_i)[w]
y_i ~ p_i
```

三个插件分别作用于不同位置：

1. S 在 multimodal projector 之前修改视觉特征；
2. V 增加 adversarial visual branch，并组合两条分支的 logits；
3. A 在最终采样前过滤低可信 candidate logits。

## 2. 方法总览

完整实现的推理路径如下：

```text
raw visual features
    -> Statistical Bias feature transformation
    -> multimodal projector
    -> clean-branch logits

attacked image
    -> vision tower
    -> multimodal projector
    -> adversarial-branch logits

clean logits + adversarial logits
    -> Vulnerability Defense contrastive combination
    -> Adaptive Plausibility mask
    -> temperature / top-k / top-p sampling
```

三组件的统一表示为：

```text
X_S = S(X, caption_features)
z_clean = LLM(P(X_S), t, y_<i)
z_adv = LLM(P(E(attacked_image)), t, y_<i)
z_V = V(z_clean, z_adv)
z_final = A(z_clean, z_V)
```

未启用的插件退化为恒等映射：

```text
without S: X_S = X
without V: z_candidate = z_clean
without A: z_final = z_candidate
```

在完整 `A + V + S` 方法中，实际顺序为：

```text
raw visual features
    -> S
    -> multimodal projector
    -> clean logits
    -> V contrastive decoding
    -> A plausibility filtering
    -> sampling
```

核心实现文件为：

- `src/methods/shield_adaptive_plausibility.py`
- `src/methods/shield_vulnerability_defense.py`
- `src/methods/shield_statistical_bias.py`
- `src/methods/shield_pipeline.py`

## 3. Adaptive Plausibility

### 3.1 作用

Contrastive decoding 可能提高一个在 clean image 下本身并不可信的 token。
A 使用 clean branch 的分布构造动态候选集合，阻止这些 token 进入最终采样
空间。

A 不修改视觉特征，也不引入额外模型。

### 3.2 候选 token 约束

令 `z_clean` 表示 clean-branch logits，令 `z_candidate` 表示待采样 logits。
当前实现使用 `beta` 定义 clean branch 的可信度下界：

```text
Candidate_i =
{
    w :
    p_clean_i(w) >= beta * max_u p_clean_i(u)
}
```

由于 softmax 是单调函数，代码直接在 logits 空间判断等价条件：

```text
z_clean_i(w) >= max_u z_clean_i(u) + log(beta)
```

候选 logits 的 mask 为：

```text
z_masked_i(w) =
    z_candidate_i(w),  if w is in Candidate_i
    -inf,              otherwise
```

mask 完成后，再执行 temperature、top-k、top-p 和 sampling：

```text
p_final_i = softmax(z_masked_i / temperature)
```

当前默认参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `beta` | 0.35 | clean branch 最大概率的保留比例 |

实现细节：

- clean branch 始终提供 plausibility reference；
- 完整方法中，`z_candidate` 是 V 组合后的 logits；
- A-only 运行时，`z_candidate` 等于 `z_clean`；
- mask 在 top-k/top-p sampling 之前执行。

### 3.3 实际作用

A 主要负责置信度约束和 coverage 保持：

- CHAIR Recall 的 Shapley 贡献为 `+5.77 pp`；
- POPE Accuracy 的 Shapley 贡献为 `+5.01 pp`；
- POPE Precision 的 Shapley 贡献为 `+8.06 pp`；
- POPE F1 的 Shapley 贡献为 `+3.69 pp`；
- CHAIRs 的 Shapley 贡献为 `-1.77 pp`。

因此，A 不是单调降低 CHAIRs 的模块。它的主要价值是当其他模块抑制不可靠
视觉证据时，保留或恢复 grounded coverage。

## 4. Vulnerability Defense

### 4.1 作用

V 为每张图像构造一个 adversarial visual branch，用来暴露模型对视觉扰动
敏感的输出偏好。随后使用 clean branch 抵消 adversarial branch 中不稳定的
logit 偏好。

### 4.2 图像扰动优化

每张图像使用一个预生成的 naive caption `c`。CLIP image encoder 和 text
encoder 用于构造攻击目标：

```text
L_adv(delta) =
    cosine(
        CLIP_image(v + delta),
        CLIP_text(c)
    )
```

扰动使用 Adam 更新：

```text
delta_(k+1) =
    delta_k - attack_lr * grad_delta L_adv(delta_k)
```

每一步都对扰动幅度进行截断：

```text
delta_k = clip(delta_k, -epsilon, epsilon)
```

最终扰动经过 `attack_c` 缩放，并投影到 runtime 使用的合法归一化像素范围：

```text
v_adv = project_valid(v + attack_c * delta_star)
```

`v_adv` 使用与 clean image 相同的 LLaVA vision tower 处理。两条分支共享
模型权重。

### 4.3 Contrastive decoding

令 `z_clean_i` 和 `z_adv_i` 分别表示第 `i` 步的 clean logits 和
adversarial logits。V 的组合公式为：

```text
z_V_i = (1 + alpha) * z_clean_i - alpha * z_adv_i
```

对应分布为：

```text
p_V_i = softmax(z_V_i)
```

当前组合实现中，V 不直接执行 plausibility mask，mask 由 V 之后的 A 独立
执行。这样可以独立打开或关闭 A、V。

### 4.4 当前配置

| 参数 | 含义 | 当前值 |
|---|---|---:|
| `cd_alpha` | adversarial branch 的对比权重 | 2.0 |
| `epsilon` | 扰动截断幅度 | 0.14 |
| `attack_steps` | Adam 更新步数 | 30 |
| `attack_c` | 扰动缩放系数 | 12.0 |
| CHAIR `attack_lr` | CHAIR 攻击学习率 | 0.02 |
| POPE `attack_lr` | POPE 攻击学习率 | 0.14 |

经验证的 CHAIR 和 POPE 使用不同的 `attack_lr`，不能在两个数据集之间静默
复用同一配置。

### 4.5 实际作用

V 主要是高 Recall 模块：

- POPE Recall 的 Shapley 贡献为 `+13.72 pp`；
- POPE F1 的 Shapley 贡献为 `+2.27 pp`；
- POPE Precision 的 Shapley 贡献为 `-8.88 pp`；
- CHAIRs 的 Shapley 贡献为 `+7.73 pp`；
- CHAIRi 的 Shapley 贡献为 `+2.09 pp`；
- CHAIR Recall 的 Shapley 贡献为 `-2.54 pp`。

V 能减少 POPE 漏检，但也会使模型更倾向于预测 positive answer，因此会出现
Recall 上升、Precision 下降的 trade-off。

## 5. Statistical Bias

### 5.1 作用

S 作用于 multimodal projector 之前的 raw LLaVA visual features。它使用
图像对应的预生成 caption 和 CLIP caption-token features，识别与 caption
内容相关的视觉 patch，并增强这些 patch。

S 的输入包括：

- raw LLaVA visual features；
- 每张图像对应的一条 naive caption；
- CLIP text encoder 产生的 caption-token features。

### 5.2 Visual-caption similarity

令 `x_j` 表示 raw visual token，令 `c_k` 表示 CLIP caption token feature。
当前实现先将视觉 token 的特征维度变换为 CLIP 的 768 维：

```text
x_pool_j = adaptive_max_pool1d(x_j, output_size=768)
```

然后执行 L2 normalization：

```text
x_norm_j = x_pool_j / ||x_pool_j||_2
c_norm_k = c_k / ||c_k||_2
```

视觉 token 与 caption token 的相似度矩阵为：

```text
M[j, k] = dot(x_norm_j, c_norm_k)
```

其中 `M` 的形状为：

```text
[number_of_visual_tokens, number_of_caption_tokens]
```

### 5.3 Caption token 选择

对每个 caption token，先计算其与所有 visual tokens 的最大相似度：

```text
score_caption[k] = max_j M[j, k]
```

选择相似度超过阈值的 caption tokens：

```text
SelectedCaption =
{
    k : score_caption[k] > threshold
}
```

当前默认值为：

```text
threshold = 0.002
```

如果没有 caption token 通过阈值，S 直接返回原始视觉特征，不执行重加权。

### 5.4 Visual token 重加权

对每个 visual token，只保留它与选中 caption tokens 的最大相似度：

```text
score_visual[j] = max_k_in_SelectedCaption M[j, k]
```

执行 min-max normalization：

```text
score_norm[j] =
    (score_visual[j] - min(score_visual))
    / (max(score_visual) - min(score_visual))
```

再通过 `gamma_gain` 拉大权重差异：

```text
score_gamma[j] =
    minmax(score_norm[j] ** gamma_gain)
```

使用 `gain_per` 对低权重 visual tokens 进行门控：

```text
gain[j] =
    0,                if score_gamma[j] <= gain_per
    score_gamma[j],   otherwise
```

最终的增强视觉 token 为：

```text
x_enhanced[j] = x_j + x_j * gain[j]
```

当前默认参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `threshold` | 0.002 | caption-token 选择阈值 |
| `gamma_gain` | 3.0 | 权重差异增强系数 |
| `gain_per` | 0.55 | visual-token 增强门槛 |

增强后的视觉特征经过原有的 LLaVA multimodal projector，作为 clean branch
的视觉输入。

### 5.5 Caption token injection

在 cumulative runner 中启用 S 时，还会执行 caption token injection：

1. 将 CLIP token index 的连续片段映射到 LLaVA tokenizer 的 caption span；
2. 获取这些 caption tokens 的 LLaVA input embeddings；
3. 将选中的 caption embeddings 插入 projected visual embeddings 之后；
4. 重新构造 attention mask 和 position ids。

新的 input embedding 序列为：

```text
input_embeddings =
    [
        text_left,
        projected_visual_features,
        selected_caption_embeddings,
        text_right
    ]
```

caption token injection 是当前 S 实现的一部分，不作为独立的第四个组件。

### 5.6 实际作用

S 是 CHAIR 上最主要的幻觉抑制模块：

- CHAIRs 的 Shapley 贡献为 `+10.03 pp`；
- CHAIRi 的 Shapley 贡献为 `+2.67 pp`；
- CHAIR Recall 的 Shapley 贡献为 `-3.08 pp`；
- POPE Accuracy 的 Shapley 贡献为 `+0.19 pp`；
- POPE Recall 的 Shapley 贡献为 `+1.96 pp`；
- POPE F1 的 Shapley 贡献为 `+0.59 pp`；
- POPE Precision 的 Shapley 贡献为 `-0.84 pp`。

S 能降低幻觉对象提及，但可能同时减少模型主动提及 grounded objects 的
数量，因此主要收益体现在 CHAIR，而不是 POPE 分类准确率。

## 6. 三组件联合推理

### 6.1 Clean branch

启用 S 时，clean branch 的处理为：

```text
X_S = S(X, caption_features)
H_clean = multimodal_projector(X_S)
z_clean_i = LLM(H_clean, t, y_<i)
```

未启用 S 时，使用 `X_S = X`。

### 6.2 Adversarial branch

启用 V 时，adversarial branch 的处理为：

```text
v_adv = project_valid(v + attack_c * delta_star)
H_adv = multimodal_projector(vision_tower(v_adv))
z_adv_i = LLM(H_adv, t, y_<i)
```

当前实现将 S 应用于 clean branch。adversarial branch 使用攻击图像经过 vision
tower 得到的特征。

### 6.3 Logit processing

启用 V 时：

```text
z_candidate_i = (1 + alpha) * z_clean_i - alpha * z_adv_i
```

未启用 V 时：

```text
z_candidate_i = z_clean_i
```

启用 A 时，保留条件为：

```text
keep(w) =
    z_clean_i(w) >= max_u z_clean_i(u) + log(beta)
```

最终 logits 为：

```text
z_final_i(w) =
    z_candidate_i(w),  if keep(w)
    -inf,              otherwise
```

之后执行 temperature、top-k、top-p 和 sampling：

```text
next_token =
    sample(softmax(z_final_i / temperature))
```

### 6.4 推理伪代码

```text
X_clean = vision_tower(image)

if use_statistical_bias:
    X_clean = S(X_clean, clip_caption_token_features)

H_clean = multimodal_projector(X_clean)

if use_vulnerability_defense:
    delta = optimize_clip_attack(image, naive_caption)
    X_adv = vision_tower(attacked_image(image, delta))
    H_adv = multimodal_projector(X_adv)
    z_clean = forward(H_clean, prompt, prefix)
    z_adv = forward(H_adv, prompt, prefix)
    z_candidate = (1 + alpha) * z_clean - alpha * z_adv
else:
    z_clean = forward(H_clean, prompt, prefix)
    z_candidate = z_clean

if use_adaptive_plausibility:
    z_final = mask_by_clean_plausibility(
        z_clean,
        z_candidate,
        beta,
    )
else:
    z_final = z_candidate

next_token = sample_with_top_p_and_temperature(z_final)
```

## 7. 实验协议

### 7.1 CHAIR-500

| 项目 | 设置 |
|---|---|
| 模型 | LLaVA-1.5-7B |
| 样本 | 固定 500 张 COCO val2014 图像 |
| Prompt | `Please describe this image in detail.` |
| 解码 | sampling |
| Seed | 42 |
| Temperature | 1 |
| Top-p | 1 |
| Max new tokens | 512 |
| Batch size | 32 |
| Image batch size | 16 |

指标方向：

- `CHAIRs` 越低越好；
- `CHAIRi` 越低越好；
- `Recall` 越高越好。

本文只纳入详细描述 prompt 的运行结果。早期错误使用 POPE 一词 prompt 的
短输出不纳入结果矩阵。

### 7.2 POPE adversarial-3000

| 项目 | 设置 |
|---|---|
| 模型 | LLaVA-1.5-7B |
| 样本 | 3000 条 adversarial questions |
| Prompt | 官方一词回答 POPE prompt |
| 解码 | sampling |
| Seed | 42 |
| Temperature | 1 |
| Top-p | 1 |
| Max new tokens | 8 |
| Batch size | 8 |
| Image batch size | 16 |

POPE 的 Accuracy、Precision、Recall 和 F1 均越高越好。

## 8. 组合实验结果

### 8.1 CHAIR-500

| 组合 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.518000 | 0.163492 | 0.808083 |
| A | 0.496000 | 0.135772 | 0.840866 |
| V | 0.460000 | 0.143551 | 0.797748 |
| S | 0.454000 | 0.142385 | 0.795050 |
| A+V | 0.472000 | 0.139740 | 0.836637 |
| A+S | 0.432000 | 0.129335 | 0.828539 |
| V+S | 0.278000 | 0.109549 | 0.705422 |
| A+V+S | 0.358000 | 0.100977 | 0.809490 |

相对于 Vanilla，完整三组件方法：

- CHAIRs 降低 `16.00 pp`；
- CHAIRi 降低 `6.25 pp`；
- Recall 变化为 `+0.14 pp`。

`V+S` 的 CHAIRs 和 CHAIRi 最低，但 Recall 只有 `0.705422`。这是实际的
抑制与 coverage trade-off，不是此前的一词 prompt 协议错误。

### 8.2 POPE adversarial-3000

| 组合 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla | 79.0667% | 84.4392% | 71.2667% | 77.2957% |
| A | 82.3667% | 88.1383% | 74.8000% | 80.9232% |
| V | 77.4667% | 71.8452% | 90.3333% | 80.0354% |
| S | 79.6667% | 83.6103% | 73.8000% | 78.3994% |
| A+V | 83.2333% | 83.2112% | 83.2667% | 83.2389% |
| A+S | 82.7000% | 87.5287% | 76.2667% | 81.5105% |
| V+S | 76.3000% | 69.9545% | 92.2000% | 79.5513% |
| A+V+S | 83.6333% | 82.7810% | 84.9333% | 83.8434% |

相对于 Vanilla，完整三组件方法：

- Accuracy 提高 `4.57 pp`；
- Precision 变化为 `-1.66 pp`；
- Recall 提高 `13.67 pp`；
- F1 提高 `6.55 pp`。

## 9. 严格组件贡献

严格贡献分析使用完整的 `2^3` 组合矩阵：

```text
Vanilla
A
V
S
A+V
A+S
V+S
A+V+S
```

下表为 Shapley 平均边际贡献。正值表示改善；对于 CHAIRs 和 CHAIRi，
改善表示原始指标下降。

### 9.1 CHAIR

| 指标 | A | V | S |
|---|---:|---:|---:|
| CHAIRs | -1.77 pp | +7.73 pp | **+10.03 pp** |
| CHAIRi | +1.49 pp | +2.09 pp | **+2.67 pp** |
| Recall | **+5.77 pp** | -2.54 pp | -3.08 pp |

主要结论：

- S 是降低 CHAIR 幻觉率的主要来源；
- A 是提高 CHAIR coverage 的主要来源；
- V 提供次要的 CHAIR 抑制，但可能降低 Recall；
- A 能部分恢复 V+S 组合损失的 coverage。

### 9.2 POPE

| 指标 | A | V | S |
|---|---:|---:|---:|
| Accuracy | **+5.01 pp** | -0.64 pp | +0.19 pp |
| Precision | **+8.06 pp** | -8.88 pp | -0.84 pp |
| Recall | -2.01 pp | **+13.72 pp** | +1.96 pp |
| F1 | **+3.69 pp** | +2.27 pp | +0.59 pp |

主要结论：

- A 主要提升 Accuracy、Precision 和 F1；
- V 主要提升 Recall；
- S 对 POPE 的边际收益较小，主要收益集中在 CHAIR；
- 没有一个组件在所有指标上都占优。

### 9.3 交互作用

三个组件不是简单相加：

| 数据集和指标 | 交互项 | 解释 |
|---|---:|---|
| CHAIRs, V x S | +11.80 pp | 明显的幻觉抑制协同 |
| CHAIR Recall, V x S | -7.93 pp | 联合使用导致过度抑制 |
| CHAIR Recall, A x V x S | +6.45 pp | A 部分恢复 coverage |
| POPE Accuracy, A x V | +2.47 pp | 联合提升 Accuracy |
| POPE Precision, A x V | +7.67 pp | 联合提升 Precision |
| POPE Recall, A x V | -10.60 pp | Precision-Recall trade-off |

## 10. 组件作用总结

| 组件 | 主要作用位置 | 主要收益 | 主要代价 |
|---|---|---|---|
| A | 最终 logits 过滤 | CHAIR Recall、POPE Accuracy/Precision/F1 | 单独降低 CHAIRs 的作用不稳定 |
| V | adversarial branch 和 logits | POPE Recall、部分 CHAIR 抑制 | Precision 和 CHAIR Recall 可能下降 |
| S | projector 前的 raw visual features | 降低 CHAIRs 和 CHAIRi | 可能损失 grounded coverage，POPE 收益较小 |

三个组件的分工可以概括为：

```text
S: 调整不可靠的视觉 token 权重
V: 暴露并抵消对视觉扰动敏感的输出偏好
A: 阻止 clean branch 低置信度 token 被采样
```

完整贡献报告位于：

- `reports/shield_strict_contributions_20260922.md`
- `reports/shield_strict_contributions_20260922.json`
