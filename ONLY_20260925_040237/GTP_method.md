# GTP: Counterfactual Patch Consensus

## 1. 方法定位

GTP（Grounded Token Projection）当前采用 **Counterfactual Patch Consensus**
作为主方法。它将 GTP 的反事实思想从词表 logits 层前移到视觉 patch feature
层：

> 一个视觉 patch 只有同时满足“与当前视觉语义相关”和“在 shuffled-control
> 下不可复现”时，才获得增强权重。

该设计针对长序列 captioning 中的两个问题：

- 直接修改词表 logits 过于粗粒度，容易在自回归过程中累积错误；
- 自动生成 caption/evidence 可能携带语言先验，不能被当作视觉真值；
- 单纯增强高相关 patch 无法区分真实视觉证据与随机空间对应。

GTP patch consensus 的特点：

- 不修改 attention；
- 不实现 VGA 的 VSC、head balancing 或 PVG；
- 不注入预生成 caption token；
- 不对最终词表 logits 做全局 token 投影；
- 只在视觉 feature 层进行一次能量守恒重分配；
- 最终使用原始语言模型自然解码。

因此，方法差异化来自 **patch-level counterfactual agreement**，而不是 decoder
层面的强制纠偏。

## 2. 方法流程

```text
image
  -> LLaVA vision patch features x_i
  -> caption/image patch relevance s_i
  -> shuffled-control relevance c_i
  -> counterfactual patch consensus a_i
  -> energy-preserving feature probe x'_i
  -> multimodal projector
  -> original LM natural decoding
```

对每个样本只构造一次 patch probe。所有生成 token 使用同一组视觉 feature，
不会在每个 token 重新生成 caption，也不会把生成结果作为额外文本注入。

## 3. Patch-level 反事实共识

设视觉 patch feature 为：

```text
x_i in R^d, i = 1, ..., N
```

使用已有的 patch-caption relevance 计算原始视觉分数：

```text
s_i = max_j cosine(pool(x_i), c_j)
```

其中 `c_j` 是 caption token 的 CLIP feature。然后对 patch 分数做随机空间置换，
得到 shuffled-control 分数：

```text
c_i = shuffle(s)_i
```

`c_i` 不是“对象不存在”的绝对证明，而是检验空间对应关系是否必要的控制分支。
若某个 patch 只是由于全局语言相似性获得高分，则其 shuffled-control 分数也会
保持较高；只有真实空间相关 patch 才应满足 `s_i > c_i`。

### 3.1 相关性门

```text
r_i = sigmoid((s_i - mean(s)) / max(std(s), tau))
```

`r_i` 表示 patch 在当前图像中的相对视觉相关性。

### 3.2 反事实门

```text
d_i = sigmoid((s_i - c_i) / tau)
```

`d_i` 表示原始 patch 分数相对于 shuffled-control 的优势。

### 3.3 共识权重

```text
a_i = r_i * d_i
a_i = a_i / mean(a)
```

最终 patch consensus 同时要求：

1. patch 自身具有较高视觉相关性；
2. 该相关性不能由随机空间配对解释。

温度 `tau` 控制反事实判别的软硬程度。较小的 `tau` 更强调明确的真实/随机
差异，较大的 `tau` 则保留更多弱证据。

## 4. 能量守恒视觉增强

patch consensus 不直接替换视觉 feature，而是作为 feature probe 的方向信号。
令 patch energy 为：

```text
e_i = ||x_i||_2^2
```

先对共识权重做能量加权中心化：

```text
delta_i = a_i - sum_k a_k * e_k / sum_k e_k
```

再归一化并得到临时 feature：

```text
delta_i = delta_i / max_i |delta_i|
y_i = x_i * (1 + eta * delta_i)
```

最后进行全局范数校正：

```text
x'_i = y_i * sqrt(sum_i ||x_i||_2^2 / sum_i ||y_i||_2^2)
```

该操作满足：

```text
sum_i ||x'_i||_2^2 = sum_i ||x_i||_2^2
```

因此 GTP 不是无界放大视觉输入，而是在保持总视觉能量不变的前提下，把有限
的表达预算从低可信 patch 转移到高可信 patch。

当前验证配置：

```text
gtp_patch_gain        = 0.35
gtp_patch_temperature = 0.05
```

代码中的 `energy_preserving_probe` 实现该步骤。

## 5. 与旧版 token-level GTP 的关系

仓库仍保留旧版 GTP，用于复现与消融：

```text
gtp-mode=legacy
gtp-mode=selective
```

旧版流程是：

```text
clean logits + evidence logits + shuffled logits
  -> reliability/counterfactual token correction
  -> chi-square projection
```

旧版的问题是：

- 反事实信号在完整词表上操作，语义粒度过粗；
- 低概率但正确的名词可能被 hard active-set 截断；
- logits 的小幅偏移会在 CHAIR 长序列中逐步累积；
- caption injection 会增加语言先验污染和对象重复。

因此当前推荐部署与论文主实验使用：

```text
method=gtp_patch
gtp_patch_gain=0.35
gtp_patch_temperature=0.05
```

旧版 `gtp` 仅用于历史结果复现和严格消融，不应与 `gtp_patch` 混称。

## 6. 为什么该设计具有方法学价值

### 6.1 从“高相关”转向“不可由随机控制解释”

传统视觉增强只使用 `s_i`，隐含假设高相关即为可靠。GTP patch consensus
额外要求原始空间关系相对 shuffled-control 有增量信息：

```text
visual relevance + counterfactual specificity
```

这使得方法不必把自动 caption 当作真值，而是把它当作一个带噪 witness。

### 6.2 从词表纠偏转向证据源纠偏

CHAIR 的错误通常不是单个 token 独立错误，而是视觉证据在生成早期已经被错误
编码。直接在 logits 层纠正属于结果层干预；patch consensus 在进入语言模型前
修正证据分布，更适合处理长序列误差累积。

### 6.3 保留自然生成能力

GTP patch 不强制某些词出现，也不禁止某些词出现。它只改变视觉输入的空间
证据分配，最终 token 仍由原始语言模型决定。这避免了 token-level 方法常见的：

- 输出变长；
- 对象重复；
- 低概率正确名词被删除；
- 生成策略与方法规则耦合。

## 7. 正式实现

主要代码：

- `src/methods/gtp.py`
- `scripts/vrsc_probe.py`
- `configs/method/gtp.yaml`
- `tests/test_gtp.py`

核心入口：

```python
counterfactual_patch_scores(
    scores,
    control_scores,
    temperature=0.05,
)
```

研究 CLI：

```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate \
  --dataset chair \
  --limit 500 \
  --methods vanilla,gtp_patch \
  --output outputs/gtp_patch_chair500_<date> \
  --gtp-patch-gain 0.35 \
  --gtp-patch-temperature 0.05
```

POPE：

```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate \
  --dataset pope \
  --limit 3000 \
  --methods vanilla,gtp_patch \
  --output outputs/gtp_patch_pope3000_<date> \
  --gtp-patch-gain 0.35 \
  --gtp-patch-temperature 0.05
```

实验必须遵守项目统一协议：

```text
seed=42
sampling
temperature=1
top_p=1
attention=flash_attention_2
dtype=bfloat16
fp8=true
fp8_scope=text_mlp
```

## 8. 当前正式验证结果

详细报告：
`reports/gtp_patch_consensus_feasibility_20260924.md`

### 8.1 CHAIR-500

输出目录：
`outputs/gtp_patch_chair500_20260924`

| 方法 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.574000 | 0.167331 | 0.802614 |
| GTP patch consensus | **0.514000** | **0.138889** | **0.845245** |

相对 Vanilla：

```text
CHAIRs  -6.00 pp
CHAIRi  -2.84 pp
Recall  +4.26 pp
```

### 8.2 POPE adversarial-3000

输出目录：
`outputs/gtp_patch_pope3000_20260924`

| 方法 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla | 79.5000% | 84.1699% | 72.6667% | 77.9964% |
| GTP patch consensus | **82.5000%** | **89.6664%** | **73.4667%** | **80.7622%** |

相对 Vanilla：

```text
Accuracy  +3.00 pp
Precision +5.50 pp
Recall    +0.80 pp
F1        +2.77 pp
```

### 8.3 加速与工程验收

正式结果均满足：

```text
status=PASS
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
caption_token_injection=false
```

当前完整测试：

```text
107 passed
```

## 9. 顶刊级验证要求

当前结果证明 GTP patch consensus 已通过可行性门槛，并同时改善 CHAIR 与 POPE。
但单一模型、单个 gain/temperature 和单次 seed 仍不足以宣称顶刊级方法。

正式论文还需要：

1. 在独立验证集调节 `gtp_patch_gain` 与 `gtp_patch_temperature`，再在 holdout
   上报告，避免 CHAIR-64 过拟合；
2. 完成 `Vanilla / shuffled-control / feature-only / token-level GTP /
   GTP patch` 的严格消融；
3. 报告 patch consensus 与 hallucination reduction 的相关性；
4. 报告 per-image win/loss、失败案例和对象类别分布；
5. 使用多随机种子或 bootstrap 置信区间；
6. 在额外 MLLM 或额外幻觉数据集上验证迁移性；
7. 与 VGA、SHIELD 等方法使用完全一致的模型、prompt、seed、解码和加速协议
   进行比较；
8. 报告额外视觉预处理、feature probe 和 caption feature 计算成本。

在这些证据完成前，不使用以下表述：

- “SOTA”；
- “幻觉消除”；
- “对任意模型有效”；
- “无条件概率校准”；
- “distribution-free generation guarantee”。

当前最准确的结论是：

> GTP counterfactual patch consensus 已在 LLaVA-1.5-7B 的 CHAIR-500 和 POPE
> adversarial-3000 上通过同协议双指标验证，具备形成顶刊方法的核心机制与
> 实验基础，但仍需多模型、多 seed、严格消融和独立 holdout 验证后，才能
> 支撑顶刊级完整结论。
