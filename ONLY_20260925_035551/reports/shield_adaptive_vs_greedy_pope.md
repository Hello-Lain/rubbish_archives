# Adaptive Plausibility 与纯贪婪解码对照实验

## 1. 实验目的

本实验验证两个问题：

1. `Adaptive Plausibility`（模块 A）相对于 Vanilla sampling 的增益是否仍然存在；
2. A 是否本质上只是另一种贪婪解码策略。

为区分模块作用和解码随机性，使用同一模型、同一数据和同一 prompt，比较四组条件：

| 条件 | 模块 | 解码 |
|---|---|---|
| Vanilla-sample | 无 | sampling |
| A-sample | Adaptive Plausibility | sampling |
| Vanilla-greedy | 无 | greedy |
| A-greedy | Adaptive Plausibility | greedy |

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| 模型 | LLaVA-1.5-7B |
| 数据集 | POPE adversarial |
| 样本数 | 3000 |
| prompt | 官方一词回答 prompt |
| seed | 42 |
| temperature | 1.0 |
| top-p | 1.0 |
| max new tokens | 8 |
| dtype | bfloat16 |
| attention | FlashAttention-2 |
| FP8 | text MLP，已生效 |
| A 参数 | `beta=0.35` |

四个结果均满足：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

## 3. POPE 结果

### 3.1 四组绝对指标

| 条件 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla-sample | 79.0667% | 84.4392% | 71.2667% | 77.2957% |
| A-sample | 82.3667% | 88.1383% | 74.8000% | 80.9232% |
| Vanilla-greedy | 83.1667% | 90.0241% | 74.6000% | 81.5895% |
| A-greedy | 83.3333% | 90.3226% | 74.6667% | 81.7518% |

### 3.2 A 相对于相同解码方式的变化

| 对照 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| A-sample - Vanilla-sample | +3.3000 pp | +3.6991 pp | +3.5333 pp | +3.6275 pp |
| A-greedy - Vanilla-greedy | +0.1667 pp | +0.2984 pp | +0.0667 pp | +0.1623 pp |

逐条预测比较结果：

| 对照 | 预测不同的样本数 | 预测翻转方向 | 净正确数变化 |
|---|---:|---|---:|
| A-sample vs Vanilla-sample | 221/3000 | `No -> Yes`: 114；`Yes -> No`: 107 | +99 |
| A-greedy vs Vanilla-greedy | 19/3000 | `No -> Yes`: 8；`Yes -> No`: 11 | +5 |

在 sample 条件下，A 明显改变了输出分布；在 greedy 条件下，A 几乎不改变输出。

需要区分机制差异和 evaluator 路径差异：Vanilla-greedy 使用基础
`pope_llava_only_compare.py`，A-greedy 使用组合式
`shield_cumulative.py`。后者显式构造 `position_ids`，并通过同一 clean
输入的双分支 runtime 生成 logits。因此，19 条 residual difference 不能直接
归因于 A 的 mask；结合下面的 argmax 证明，更合理的解释是输入构造、缓存路径
或低精度执行顺序造成的数值差异。

### 3.3 纯 greedy 本身的影响

| 解码变化 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla-greedy - Vanilla-sample | +4.1000 pp | +5.5850 pp | +3.3333 pp | +4.2938 pp |
| A-greedy - A-sample | +0.9667 pp | +2.1843 pp | -0.1333 pp | +0.8286 pp |

在本次 POPE adversarial 实验中，单纯将 Vanilla 从 sampling 改为 greedy 带来的提升大于 A 在 sampling 下相对 Vanilla 的提升，尤其体现在 Accuracy、Precision 和 F1。

## 4. A 的解码机制

A-only 阶段没有 Vulnerability Defense，因此候选 logits 就是 clean logits。当前实现的过滤过程为：

```text
cutoff = log(beta) + max_j(z_j)

z'_j = z_j,       if z_j >= cutoff
       -infinity, otherwise
```

其中 `beta=0.35`，因此：

```text
log(beta) < 0
```

设 clean logits 的最大值为 `z_max`，则：

```text
cutoff = z_max + log(beta) < z_max
```

所以最大 logit 对应的 token 一定不会被过滤。对 greedy 解码而言：

```text
argmax_j(z'_j) = argmax_j(z_j)
```

也就是说，在同一组 clean logits 上，A 的 plausibility mask 不会改变 greedy 的 argmax。A 在 greedy 下不是一种新的决策规则，而是一个对最终结果不起作用的候选集约束。

sampling 则不同。A 会移除 clean probability 低于 `beta` 相对阈值的 token，然后在剩余 token 上重新归一化采样概率：

```text
p_A(token) = softmax(z')
```

因此，A 的有效作用是改变 sampling 的支持集，减少从低可信概率尾部抽到错误答案的机会。

## 5. 结论

1. **A 不是特殊的贪婪解码策略。** 在 A-only、candidate logits 等于 clean logits 的实现中，A 与 greedy 在 argmax 意义下理论等价。
2. **A 的主要作用发生在 sampling。** A-sample 相对于 Vanilla-sample 提升了 `3.30` 个百分点 Accuracy 和 `3.63` 个百分点 F1，并改变了 `221/3000` 条预测。
3. **A 在 greedy 下的边际贡献接近于零。** A-greedy 相对于 Vanilla-greedy 仅提升 `0.1667` 个百分点 Accuracy，只有 `19/3000` 条答案不同，净增加 `5` 条正确预测。
4. **此前 A-only 的部分 POPE 增益确实混入了解码策略收益。** Vanilla 从 sampling 改为 greedy 已带来 `+4.10` 个百分点 Accuracy 和 `+4.29` 个百分点 F1，超过 A-sample 相对于 Vanilla-sample 的增益。
5. **A 仍然具有独立的 sampling 作用，但不能把该作用表述为提升 greedy 决策能力。** 更准确的描述是：A 是一个基于 clean distribution 的自适应 plausibility truncation，在 sampling 时重新塑造候选分布。

## 6. 结果文件

| 条件 | Metrics | Predictions |
|---|---|---|
| Vanilla-sample | `outputs/pope_llava_7b_adversarial_promptfix_vanilla_20260922.metrics.json` | `outputs/pope_llava_7b_adversarial_promptfix_vanilla_20260922.jsonl` |
| A-sample | `outputs/pope_llava_7b_adversarial_cumulative_adaptive_official_20260922.metrics.json` | `outputs/pope_llava_7b_adversarial_cumulative_adaptive_official_20260922.jsonl` |
| Vanilla-greedy | `outputs/pope_llava_7b_adversarial_greedy_vanilla_20260923.metrics.json` | `outputs/pope_llava_7b_adversarial_greedy_vanilla_20260923.jsonl` |
| A-greedy | `outputs/pope_llava_7b_adversarial_greedy_adaptive_20260923.metrics.json` | `outputs/pope_llava_7b_adversarial_greedy_adaptive_20260923.jsonl` |
