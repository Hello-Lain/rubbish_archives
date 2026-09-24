# CAP 方法精炼与完整验证报告

## 结论

本轮将 CAP-C 精炼为 `CAP-R`，核心改造是把固定的整段 probe 干预改为
逐生成步的可靠性门控：

```text
J_t       = JSD(p_t || q_t)
DeltaH_t  = (H(p_t) - H(q_t)) / log(|V|)
r_t       = sigmoid((J_t - j0) / tau_J)
            * sigmoid(DeltaH_t / tau_H)
lambda_t  = lambda_max * r_t

g_t(w)    = sigmoid(
              (log p_t(w) - max(log p_t) - log(beta)) / tau_P
            )

log q_R,t(w) =
    log p_t(w)
    + lambda_t * g_t(w) * (log q_t(w) - log p_t(w))
```

这借鉴了 `VCD`、`HALC`、`OPERA`、`DoLa`、`SAVAA` 等方法的分支对比、逐
token/step 自适应和不确定性门控思想；文献、链接、公式映射见
[文献清单.md](/data/lcq/Downloads/ONLY/文献清单.md)。

**工程可行性通过，完整效果门槛未通过。** CAP-R 可以在固定协议下完成
CHAIR-500 和 POPE adversarial-3000，且加速验收通过；但它没有同时改善
CHAIR 的 Recall 和 POPE，不能作为当前默认部署方法。优化后的研究配置为：

```text
cap_attribution=contrastive_hidden
probe_gain=0.10
max_weight=0.75
jsd_threshold=0.005
jsd_temperature=0.01
entropy_temperature=0.01
plausibility_beta=0.10
plausibility_temperature=0.05
```

`configs/method/cap_refined.yaml` 已更新为该配置。历史 `cap` 路径和其默认
参数没有修改；直接使用 `scripts/vrsc_probe.py` 时应显式传
`--cap-probe-gain 0.10`。

## 方法精炼

### 失败模式

CAP-C 使用 contrastive-hidden feature probe：

```text
a_i = cos(p_i, h_visual) - cos(p_i, h_blank)
x'_i = EnergyPreservingProbe(x_i, a_i, probe_gain)
```

CAP-C 的信号方差已经足够区分，但在 CHAIR-64 的 gain sweep 中产生孤立低点：
`gain=0.15` 的 CHAIRs 为 `0.468750`，相邻 `0.10` 和 `0.20` 分别为
`0.625000` 和 `0.562500`。这说明固定 probe 强度会同时改写正确 token 和
幻觉 token，且不能识别当前生成步的可靠性。

CAP-R 保留该 feature probe，不引入新模型、检测器、检索库、caption 注入或
beam search，只在 logits 解码端增加：

- `JSD(base, probe)`：检测两条视觉路径是否存在需要判断的分歧；
- `DeltaH`：只有 probe 让分布更集中时才提高干预；
- soft plausibility gate：避免硬 APC 将 sampling 退化成近似 greedy；
- `lambda_t`：每个生成步独立控制 probe 影响，分歧不可靠时回退到 Vanilla。

### 候选参数筛选

固定 LLaVA-1.5-7B、seed=42、sampling、CHAIR-64、max_new_tokens=512，
只筛选 `probe_gain`。`max_weight`、JSD 和 plausibility 参数固定为上面的
配置。

| 方法/参数 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.609375 | 0.177632 | 0.814193 |
| CAP-R, gain=0.05 | 0.562500 | 0.172727 | 0.802995 |
| CAP-R, gain=0.10 | **0.562500** | **0.160271** | **0.819661** |
| CAP-R, gain=0.15 | 0.656250 | 0.191358 | 0.798698 |
| CAP-R, gain=0.20 | 0.625000 | 0.188720 | 0.781380 |

`gain=0.10` 是本轮唯一同时改善 CHAIRs、CHAIRi 和 Recall 的候选，因此
冻结为完整验证参数。该筛选仍是单 seed、小样本选择，不能视为独立 holdout
调参证明。

## 完整 CHAIR-500

### 结果

| 方法 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.574000 | 0.167331 | 0.802614 |
| CAP-R, gain=0.10 | **0.542000** | **0.166519** | 0.793425 |

相对 Vanilla：

```text
CHAIRs  -3.2000 pp
CHAIRi  -0.0811 pp
Recall  -0.9189 pp
```

CAP-R 减少了 `23` 个 hallucinated object mentions
（`588 -> 565`），同时总 object mentions 从 `3514` 降到 `3393`；
句子级和对象级幻觉率略有改善，但 grounded object coverage 下降。这是
“降低输出对象数量”而不是“全面提升视觉 grounding”的证据。

正式产物：

- `outputs/capr_chair500_gain0.10_20260925/vanilla.jsonl`
- `outputs/capr_chair500_gain0.10_20260925/cap_refined.jsonl`
- `outputs/capr_chair500_gain0.10_20260925/vanilla.metrics.json`
- `outputs/capr_chair500_gain0.10_20260925/cap_refined.metrics.json`
- `outputs/capr_chair500_gain0.10_20260925/manifest.json`

两种方法均为 `status=PASS`、`count=500`，预测文件各 500 行。

## 完整 POPE adversarial-3000

### 结果

| 方法 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla | 79.5000% | 84.1699% | 72.6667% | 77.9964% |
| CAP-R, gain=0.10 | 79.5000% | 84.1699% | 72.6667% | 77.9964% |

相对 Vanilla：

```text
Accuracy  +0.0000 pp
Precision +0.0000 pp
Recall    +0.0000 pp
F1        +0.0000 pp
```

两种方法的混淆矩阵完全相同：

```text
TP=1090, TN=1295, FP=205, FN=410
```

这说明 CAP-R 的软门控在当前参数下没有改变 POPE 的最终一词决策。它没有
复现旧 CAP-C 在 POPE 上的轻微下降，但也没有形成新的 adversarial 性能收益。

正式产物：

- `outputs/capr_pope3000_gain0.10_20260925/vanilla.jsonl`
- `outputs/capr_pope3000_gain0.10_20260925/cap_refined.jsonl`
- `outputs/capr_pope3000_gain0.10_20260925/vanilla.metrics.json`
- `outputs/capr_pope3000_gain0.10_20260925/cap_refined.metrics.json`
- `outputs/capr_pope3000_gain0.10_20260925/manifest.json`

两种方法均为 `status=PASS`、`count=3000`，预测文件各 3000 行。

## 与历史方法对照

固定协议下，历史 CAP-C 的完整结果为：

| 方法 | CHAIRs | CHAIRi | CHAIR Recall | POPE Accuracy | POPE F1 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.574000 | 0.167331 | 0.802614 | 79.5000% | 77.9964% |
| CAP-C | 0.520000 | 0.171576 | 0.797504 | 79.3000% | 77.7658% |
| CAP-R | 0.542000 | **0.166519** | 0.793425 | **79.5000%** | **77.9964%** |
| GTP-Patch | **0.514000** | **0.138889** | **0.845245** | **82.5000%** | **80.7622%** |

CAP-R 相比 CAP-C 的主要变化是：

- POPE 从轻微退化恢复到 Vanilla，但没有超过 Vanilla；
- CHAIRi 略好于 CAP-C；
- CHAIR Recall 仍然下降，且低于 CAP-C；
- GTP-Patch 在三项 CHAIR 指标和四项 POPE 指标上仍明显更强。

因此 CAP-R 是比 CAP-C 更安全、但仍未达到部署效果门槛的精炼版本。

## 固定协议与加速验收

模型和数据：

```text
model revision:
  b234b804b114d9e37bb655e11cbbb5f5e971b7a9
model config sha256:
  0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f
CHAIR questions sha256:
  d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e
POPE data sha256:
  185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac
```

CHAIR：

```text
prompt=Please describe this image in detail.
seed=42
decoding=sample
temperature=1
top_p=1
max_new_tokens=512
batch_size=32
image_batch_size=16
num_workers=4
prefetch_factor=4
cache=dynamic
pope_answer_instruction=false
```

POPE：

```text
official one-word answer instruction=true
seed=42
decoding=sample
temperature=1
top_p=1
max_new_tokens=8
batch_size=8
image_batch_size=16
num_workers=4
prefetch_factor=4
cache=dynamic
```

两套完整产物均通过：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
dtype=bfloat16
fp8_scope=text_mlp
```

模型、图像和数据均来自固定 Hugging Face 本地缓存。Vanilla 与 CAP-R 使用
相同模型、数据、prompt、seed、解码和 batch。

## 资源与异常记录

一次 `gain=0.15` 的 CHAIR-64 运行在共享 GPU 上因其他用户进程增长而
`CUDA out of memory`，当时目标 GPU 只剩约 376 MiB。未终止或干预其他用户
进程；随后调度器重新选择空闲 GPU0，在 canonical batch 和 token 配置下完成
重跑，结果记录为：

```text
outputs/capr_chair64_gain_0.15_20260925_retry/
CHAIRs=0.656250, CHAIRi=0.191358, Recall=0.798698
```

该失败运行不进入性能统计。原始资源证据和 rework 记录见：

- `.codex/work/20260925-cap-refinement/artifacts/rework-guidance-1.md`
- `.codex/work/20260925-cap-refinement/artifacts/taskC-execution.md`

## 验证结论

### 通过项

- CAP-R 纯张量逻辑和 method contract 测试通过。
- CAP-R 最新代码 POPE-32 smoke 可运行。
- CHAIR-64 四个候选 gain 均在完整运行后得到指标，失败的 OOM 运行已隔离。
- CHAIR-500 和 POPE adversarial-3000 均完整生成并通过 `status=PASS`。
- 两方法输入顺序哈希一致，行数和 metrics count 一致。
- 加速验收通过，无 native FP8 fallback 和 fallback events。
- `文献清单.md` 已记录顶会/近期论文、链接和 CAP-R 公式借鉴关系。

### 未通过项

- CAP-R 没有同时改善 CHAIRs、CHAIRi、Recall。
- CAP-R 没有改善 POPE adversarial-3000 的任何聚合指标。
- 只有单个模型、单个随机种子和一次小样本 gain 筛选，尚不能作为顶会级泛化
  证据。
- `gain=0.10` 是当前实验的最佳可行配置，不是经过独立 holdout 和多 seed
  统计验证的全局最优参数。

### 最终判断

```text
机制/工程可行性: PASS
完整 CHAIR 通过全面改善门槛: FAIL
完整 POPE 通过增益门槛: FAIL
当前部署建议: 不部署 CAP-R 作为主要幻觉抑制方法
```

CAP-R 可以保留为后续研究分支，用于研究 token-level reliability gating；
如果目标是当前仓库的综合 CHAIR/POPE 指标，应优先继续使用
`GTP-Patch`，而不是把 CAP-R 的 CHAIRs 单项下降解释为 grounding 已解决。

## 归档与上传

使用 `zip-directory` skill 生成并上传了源码快照：

```text
local archive:
  .codex/ONLY_20260925_035551.zip
sha256:
  f0e39f042a2e3cf75b8e97baf7d7be3eaa4fc60066ea5ed86a7683b40367a739
files:
  151
uncompressed bytes:
  2201104
```

上传回执：

```text
repository=Hello-Lain/rubbish_archives
branch=main
snapshot=ONLY_20260925_035551
archive=ONLY_20260925_035551/ONLY_20260925_035551.zip
```

ZIP CRC 检查通过。按 skill 规则，`outputs/`、隐藏协调目录、环境、缓存、
metrics 文件和超过 3,000,000 bytes 的历史归档未打包；完整实验原始结果仍保留
在本地 `outputs/`。
