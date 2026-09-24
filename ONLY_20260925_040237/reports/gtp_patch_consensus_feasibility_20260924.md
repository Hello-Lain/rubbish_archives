# Counterfactual Patch Consensus 可行性验证

## 结论

在 LLaVA-1.5-7B 上，反事实 patch consensus 具备明确的实验可行性：

- 不修改 attention；
- 不注入预生成 caption；
- 不在词表 logits 上做全局投影；
- 只使用原始 patch-caption relevance 与 shuffled-control 的 patch 差值；
- 通过能量守恒的视觉 feature probe 后，由原始语言模型自然解码。

在固定协议下，CHAIR 和 POPE 同时改善，说明该机制不是只针对单一评测
指标的后处理收益。

## 方法

对每个视觉 patch 计算原始分数 `s_i` 和 shuffled-control 分数 `c_i`。定义：

```text
relevance_i = sigmoid((s_i - mean(s)) / max(std(s), temperature))
counterfactual_i = sigmoid((s_i - c_i) / temperature)
consensus_i = relevance_i * counterfactual_i
```

`consensus` 只用于对视觉 patch feature 做能量守恒重分配，最终分支使用
base logits 自然生成。因此该方法与 VGA 的 attention 修改、head balancing
和 PVG 机制不同。

本次配置：

```text
gtp_patch_gain=0.35
gtp_patch_temperature=0.05
seed=42
decoding=sample
temperature=1
top_p=1
```

## CHAIR-500

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

## POPE adversarial-3000

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

## 协议与软件验证

- CHAIR count: 500，状态 `PASS`
- POPE count: 3000，状态 `PASS`
- effective attention: `flash_attention_2`
- dtype: `bfloat16`
- `fp8_effective=true`
- `fp8_native_fallback_calls=0`
- `fallback_events=[]`
- caption token injection: `false`
- 完整测试：`107 passed`
- `py_compile`、`git diff --check`、`uv pip check`：通过

## 顶刊价值判断

当前结果足以支持“机制可行且值得继续发展”的判断，尤其是：

1. 反事实信号从 token-level 移到 patch-level 后，CHAIR 长序列生成显著改善；
2. 同一机制在 POPE 上没有损失，反而提高；
3. 方法不依赖 attention 改写或 caption token 注入，具有清晰的差异化。

但这还不能单凭一次 LLaVA-1.5-7B、一次固定 gain/temperature 和两个数据集
宣称顶刊级方法。论文级证据仍需要：

- `gain` 和 `temperature` 的预注册式验证集调参，并在 holdout 上报告；
- vanilla、shuffled-control、feature-only、token-level GTP 的完整消融；
- 至少一个额外 MLLM 或额外幻觉评测；
- 多随机种子或 bootstrap 置信区间；
- patch consensus 与 hallucination reduction 的相关性和失败案例分析；
- 与 VGA/SHIELD 等方法的严格同协议比较。

因此当前结论是：**反事实 patch consensus 已通过可行性门槛，具备形成顶刊方法
叙事的核心实验依据，但尚未完成顶刊级完整验证。**
