# CAP-C 全量结果报告

## 结论摘要

本报告汇总最新修正版 CAP-C 在完整 `CHAIR-500` 和
`POPE adversarial-3000` 上的结果。最佳参数来自 CHAIR-64 sweep 的最低点：

```text
cap_attribution=contrastive_hidden
cap_probe_gain=0.15
```

全量结果是**用户授权的探索性验证**。原因是 CAP-C 在 CHAIR-64 上虽然
满足改善和 Recall 门控，但相邻 gain 的稳定性门控失败；原始计划按门控
本应停止。本轮全量结果因此不能表述为“通过全部预注册门控的正式主结果”。

核心判断：

- CAP-C 相对同一轮 Vanilla 的 `CHAIRs` 从 `0.574` 降到 `0.520`
  （改善 `5.4 pp`）。
- 但 CAP-C 的 `CHAIRi` 上升 `0.4245 pp`，Recall 下降 `0.5109 pp`。
- POPE 上 CAP-C 的 Accuracy、Precision、Recall、F1 均略低于 Vanilla。
- GTP-Patch 在 CHAIRi、Recall 以及 POPE 四项指标上均明显优于 CAP-C。
- 因此当前证据不支持把 CAP-C 作为部署方法；GTP-Patch 是本轮更强的
  对照方案。

## 全量结果

### CHAIR-500

指标定义：`CHAIRs` 和 `CHAIRi` 越低越好，Recall 越高越好。

| 方法 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.574000 | 0.167331 | 0.802614 |
| CAP-C (`g=0.15`) | **0.520000** | 0.171576 | 0.797504 |
| GTP-Patch | 0.514000 | **0.138889** | **0.845245** |

相对同一轮 Vanilla：

| 方法 | ΔCHAIRs | ΔCHAIRi | ΔRecall |
|---|---:|---:|---:|
| CAP-C | **-5.4000 pp** | +0.4245 pp | -0.5109 pp |
| GTP-Patch | -6.0000 pp | **-2.8442 pp** | **+4.2632 pp** |

CAP-C 的 `CHAIRs` 有改善，但它不是全面改善：CAP-C 的 hallucinated
object mentions 为 `600`，高于 Vanilla 的 `588`；总 mentioned objects
为 `3497`，低于 Vanilla 的 `3514`，这解释了为什么 `CHAIRs` 改善没有
转化为 `CHAIRi` 和 Recall 的同步改善。

### POPE adversarial-3000

指标越高越好。

| 方法 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla | 79.5000% | 84.1699% | 72.6667% | 77.9964% |
| CAP-C (`g=0.15`) | 79.3000% | 83.9907% | 72.4000% | 77.7658% |
| GTP-Patch | **82.5000%** | **89.6664%** | **73.4667%** | **80.7622%** |

相对同一轮 Vanilla：

| 方法 | ΔAccuracy | ΔPrecision | ΔRecall | ΔF1 |
|---|---:|---:|---:|---:|
| CAP-C | -0.2000 pp | -0.1792 pp | -0.2667 pp | -0.2306 pp |
| GTP-Patch | **+3.0000 pp** | **+5.4965 pp** | **+0.8000 pp** | **+2.7658 pp** |

CAP-C 的 POPE 混淆矩阵为 `TP=1086, TN=1293, FP=207, FN=414`；
Vanilla 为 `TP=1090, TN=1295, FP=205, FN=410`。CAP-C 同时少判对
4 个正例和 2 个负例，多出 2 个 FP 和 4 个 FN，因此四项聚合指标均略降。

## 相对 GTP-Patch

CAP-C 与 GTP-Patch 的差值如下，正数不代表更好，只表示 CAP-C 数值更高：

### CHAIR-500

```text
CAP-C - GTP-Patch:
  CHAIRs  +0.6000 pp
  CHAIRi  +3.2687 pp
  Recall  -4.7741 pp
```

对于 CHAIR，CAP-C 在三项指标上都不如 GTP-Patch。

### POPE adversarial-3000

```text
CAP-C - GTP-Patch:
  Accuracy  -3.2000 pp
  Precision -5.6757 pp
  Recall    -1.0667 pp
  F1        -2.9963 pp
```

## 固定协议

### 模型与数据

```text
model:
  /data/lcq/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/
  snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9
model_revision:
  b234b804b114d9e37bb655e11cbbb5f5e971b7a9
model_config_sha256:
  0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f
```

```text
CHAIR questions:
  /data/lcq/.cache/huggingface/hub/chair/shield/questions.jsonl
SHA-256:
  d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e

POPE adversarial:
  /data/lcq/.cache/huggingface/hub/pope/coco/coco_pope_adversarial.json
SHA-256:
  185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac
```

所有模型、数据和图像均从 Hugging Face 本地缓存读取，未从仓库或临时
路径加载。

### 解码与批处理

```text
seed=42
decoding=sample
temperature=1
top_p=1
cache=dynamic
```

CHAIR：

```text
batch_size=32
image_batch_size=16
num_workers=4
prefetch_factor=4
max_new_tokens=512
prompt=Please describe this image in detail.
```

POPE：

```text
batch_size=8
image_batch_size=16
num_workers=4
prefetch_factor=4
max_new_tokens=8
pope_answer_instruction=true
```

三种方法使用相同模型、数据、prompt、seed、解码和批处理设置。

### 加速验收

两套全量目录的每个方法均满足：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
dtype=bfloat16
fp8_scope=text_mlp
```

## 完整性审计

正式输出：

```text
outputs/capc_chair500_20260924/
outputs/capc_pope3000_20260924/
```

每个目录均包含 `vanilla`、`cap`、`gtp_patch` 三种方法：

```text
CHAIR: 500 + 500 + 500 rows
POPE:  3000 + 3000 + 3000 rows
```

审计通过项：

- 两个 manifest 的 `status=PASS`、`count`、数据哈希和模型哈希正确。
- 三种方法的 JSONL 行数完整且无坏行。
- 三种方法的 question/image 顺序完全一致。
- POPE 的 TP/TN/FP/FN 与 metrics 重算一致。
- CHAIR 的逐图 CHAIRs/Recall 与 metrics 重算一致。
- manifest 中记录的源代码哈希与当前文件一致。
- 没有覆盖此前的部分或历史输出；全量结果保存在独立的 `capc_*`
  目录中。

本轮续跑发现目标目录已经包含完整输出，因此依据项目的“大型实验复用”
规则直接做协议与结果审计，没有再次覆盖或重复生成同一目录。

## 门控状态与结果定位

CHAIR-64 sweep 的门控结果：

```text
CHAIRs improvement: PASS
Recall floor: PASS
curve stability: FAIL
max adjacent fluctuation: 0.156250
threshold: < 0.05
```

原计划的正式决策仍是停止并保留：

```text
reports/cap_negative_20260924.md
```

本报告记录的是用户授权后的探索性全量验证。它回答“如果继续跑，
完整 CHAIR/POPE 的数值到底是什么”，但不改变稳定性门控失败这一事实。

## 产物索引

- 全量结果报告：
  `reports/cap_v3_full_results_20260924.md`
- CHAIR-500 原始输出：
  `outputs/capc_chair500_20260924/`
- POPE-3000 原始输出：
  `outputs/capc_pope3000_20260924/`
- 全量证据 ZIP 归档：
  `reports/CAP-C_full_results_20260924.zip`
- CHAIR-64 sweep 报告：
  `reports/capc_chair64_sweep_20260924.md`
- 稳定性门控负结果：
  `reports/cap_negative_20260924.md`
- 计划执行最终记录：
  `.codex/work/20260924-capc/artifacts/final-report.md`

## 仓库上传状态

本地提交已经创建：

```text
ef3ccbb CAP-C: add exploratory full CHAIR and POPE evaluation
```

归档文件已进入当前工作区的 `reports/` 路径。向原始 GitHub 仓库
推送时，远端返回：

```text
403 Permission to zifuwan/ONLY.git denied to Hello-Lain
```

因此本报告不宣称远端分支已经上传成功；需要具有
`zifuwan/ONLY` 写权限的 GitHub 凭据后再推送当前分支。
