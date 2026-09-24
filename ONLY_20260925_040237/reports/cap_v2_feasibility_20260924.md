# CAP v2 可行性验证报告

## 结论

CAP v2 的主方案 ACP（Attention-Calibrated Probe）未通过 Phase 2 信号门控，
因此没有启动 Phase 3-7 的 CHAIR/POPE 全量实验。按上传计划的 Fallback B
分支实现的 contrastive-hidden attribution 通过了同样的 POPE-32 离散度
门控，可以进入上传计划规定的 CHAIR-64 gain sweep。

当前结论是：

- ACP attention attribution 在 LLaVA-1.5-7B、FA2/BF16/FP8 固定协议下不可行；
- Fallback B contrastive hidden 具备可辨识的视觉-语言分离信号；
- 该结果仍只是方法可行性证据，不是 CHAIR 或 POPE 性能收益结论；
- CHAIR-436/500 和 POPE-3000 必须在 CHAIR-64 gain sweep 及随机性门控
  通过后才能运行。

## 执行范围

本轮只执行到上传计划的最早决策点，没有跳过门控：

1. 实现并验证 ACP attention attribution；
2. 运行 ACP POPE-32 信号体检；
3. 按上传计划的决策树停止 ACP；
4. 实现并验证 Fallback B contrastive hidden；
5. 运行 Fallback B POPE-32 信号体检；
6. 比较两条分支并给出条件性可行性结论。

## 实验协议

两次冒烟均使用相同的模型、数据、prompt、seed、解码和批处理协议：

```text
model: LLaVA-1.5-7B
model revision: b234b804b114d9e37bb655e11cbbb5f5e971b7a9
model config SHA-256:
  0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f
dataset: POPE adversarial
dataset SHA-256:
  185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac
seed: 42
decoding: sample
temperature: 1.0
top_p: 1.0
batch_size: 8
image_batch_size: 16
num_workers: 4
prefetch_factor: 4
max_new_tokens: 8
pope_answer_instruction: true
attention: flash_attention_2
dtype: bfloat16
fp8_scope: text_mlp
```

两次 manifest 均记录：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

## ACP 主方案

实现上保留原有 cosine attribution，并新增：

- `attention_attribution_weights()`：从生成位置取最后一个语言层的
  mean-head attention，对 576 个图像 patch 返回 `[576]` 分数；
- `build_vulnerability_prompt_batch(..., return_image_positions=True)`：
  返回经过左 padding 修正后的精确图像 token 位置；
- `--cap-attribution attention`：从 CAP CLI 选择 ACP；
- 临时 eager last-layer attention capture：绕过当前
  `transformers==4.57.6` 与 FA2 组合无法通过
  `output_attentions=True` 返回权重的问题；归因前向前完成后恢复 FA2，
  生成阶段仍保持 canonical 配置。

运行结果：

```text
output: outputs/acp_pope32_smoke_20260924/
rows: vanilla=32, cap=32
image position endpoints: 35..38 through 610..613
generation positions: 632..635
score length per row: 576
```

Phase 2 门控要求：

```text
std(scores) > 0.005
max-min > 0.02
```

实际结果：

| 指标 | 数值 | 结果 |
|---|---:|---|
| Pooled score std | 0.0004818293 | FAIL |
| Pooled score range | 0.0094588375 | FAIL |
| Mean per-row std | 0.0004745604 | FAIL |
| Mean per-row range | 0.007359 | FAIL |

因此 ACP 被判定为主方案不可行，并停止在 Phase 2。没有创建
`outputs/acp_chair*` 或 `outputs/acp_pope3000*`。

## Fallback B

Fallback B 使用同一 prompt 的第二个语言模型前向，将图像 token 位置的
embedding 置零，并计算：

```text
a_i = cosine(p_i, h_visual) - cosine(p_i, h_blank)
```

其中：

```text
p_i: 第 i 个图像 patch 的 projected feature
h_visual: 原始 prompt 在生成位置的 hidden state
h_blank: 图像 embedding 置零后的同位置 hidden state
```

实现同时记录：

```text
cap_visual_hidden_norm
cap_blank_hidden_norm
cap_blank_forward_mode=zero_image_embeddings
cap_image_positions
cap_attribution_scores
```

运行结果：

```text
output: outputs/acp_fallbackb_pope32_smoke_20260924/
rows: vanilla=32, cap=32
score length per row: 576
```

| 指标 | 数值 | 结果 |
|---|---:|---|
| Pooled score std | 0.0198397019 | PASS |
| Pooled score range | 0.1485607401 | PASS |
| Negative scores | 72.19% | signed separation |
| Positive scores | 27.81% | signed separation |
| Zero scores | 0 | non-degenerate |
| Mean per-row std | 0.0144149076 | PASS |
| Mean per-row range | 0.0831313882 | PASS |
| Probe norm ratio | 0.99969..1.00032 | energy preservation PASS |

Fallback B 通过门控，说明单一 hidden-state cosine 的弱信号在加入
visual-minus-blank 差分后，至少在当前 POPE-32 诊断上具有可测量的离散度。

## 验证结果

代码与运行验证均通过：

```text
.venv/bin/python -m pytest tests/test_cap.py tests/test_vrsc_probe.py -q
54 passed

.venv/bin/python -m pytest -q
150 passed

.venv/bin/python -m py_compile \
  src/methods/cap.py scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py
passed

.venv/bin/ruff check \
  src/methods/cap.py scripts/vrsc_probe.py \
  scripts/shield_vulnerability_runtime.py \
  tests/test_cap.py tests/test_vrsc_probe.py
All checks passed!

uv pip check --python .venv/bin/python
Checked 121 packages; all compatible

python /data/lcq/.codex/skills/plan2do/scripts/validate_execution.py \
  .codex/work/20260924-cap-v2
VALID
```

## 决策与下一步

推荐的下一步是只运行上传计划中的 CHAIR-64 gain sweep，使用：

```text
--cap-attribution contrastive_hidden
--cap-probe-gain {0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60}
```

CHAIR-64 必须继续满足上传计划的门控：

```text
存在 gain 使 CHAIRs < Vanilla
gain-CHAIRs 曲线非随机
```

在这些条件通过之前，不应运行 CHAIR-436 holdout、CHAIR-500 或
POPE-3000。如果 CHAIR-64 sweep 失败，应把当前结果记录为 negative
result，并转向 GTP-Patch 的多 token caption grounding 路线。

## 产物

代码与运行产物：

- `src/methods/cap.py`
- `scripts/vrsc_probe.py`
- `scripts/shield_vulnerability_runtime.py`
- `tests/test_cap.py`
- `tests/test_vrsc_probe.py`
- `outputs/acp_pope32_smoke_20260924/`
- `outputs/acp_fallbackb_pope32_smoke_20260924/`

协调与验收记录：

- `.codex/work/20260924-cap-v2/artifacts/task1-verification.md`
- `.codex/work/20260924-cap-v2/artifacts/task2-verification.md`
- `.codex/work/20260924-cap-v2/artifacts/task3-decision.md`
- `.codex/work/20260924-cap-v2/artifacts/task4-verification.md`
- `.codex/work/20260924-cap-v2/artifacts/task5-verification.md`
- `.codex/work/20260924-cap-v2/artifacts/task6-decision.md`
- `.codex/work/20260924-cap-v2/artifacts/final-report.md`

本报告只总结已完成的可复现证据，不把 POPE-32 可行性门控表述为
CHAIR/POPE 性能结论。
