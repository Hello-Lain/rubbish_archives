# ONLY 项目关键执行契约

## 统一推理协议

- 环境：项目根目录 `.venv/bin/python`，由 `uv` 管理。
- 模型、数据和图像只从 `/data/lcq/.cache/huggingface/hub` 读取。
- CUDA：BF16、FlashAttention-2、Transformer Engine FP8 `text_mlp`、TF32、
  dynamic KV cache。
- 批处理：`batch_size=8`、`image_batch_size=16`、`num_workers=4`、
  `prefetch_factor=4`。
- 解码：`seed=42`、sampling、`temperature=1`、`top_p=1`、
  `max_new_tokens=8`。
- Vanilla 与所有方法必须使用相同的模型、数据、prompt、seed、解码和批处理。
- 启动 GPU 实验前必须使用 `gpu-scheduler` 检查显存和进程，不得干预其他用户
  的进程。

加速验收必须同时满足：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

ONLY 的 enhance layer 和 final layer 保留手写 attention；其余层使用统一
attention 后端。`torch.compile`、FlashAttention-3 尚未纳入 canonical 协议。

## 固定资源

- POPE adversarial：
  `/data/lcq/.cache/huggingface/hub/pope/coco/coco_pope_adversarial.json`
- COCO 图像：
  `/data/lcq/.cache/huggingface/hub/coco2014/val2014`
- POPE SHA-256：
  `185d7eb49958c882f15d46c51d5c0456fbd5efbf24c57ff8b7adc6648870c6ac`
- SHIELD-compatible CHAIR questions：
  `/data/lcq/.cache/huggingface/hub/chair/shield/questions.jsonl`
- CHAIR questions SHA-256：
  `d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e`
- CHAIR ground-truth annotations：
  `/data/lcq/.cache/huggingface/hub/coco2014/annotations/`
- CHAIR ground-truth cache：
  `outputs/chair_shield_ground_truth_v2.json`
- LLaVA-1.5-7B：
  `/data/lcq/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9`
- LLaVA revision/config SHA-256：
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` /
  `0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f`
- Qwen2.5-VL-3B：
  `/data/lcq/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`
- Qwen revision/config SHA-256：
  `66285546d2b821cf421d4f5eb2576359d3770cd3` /
  `7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5`

## 最新 POPE 结果

POPE adversarial 固定 3000 条。LLaVA 的 SHIELD 对比使用官方一词回答
指令：

```text
{system_prompt} USER: <image>
{question} Please answer this question with one word. ASSISTANT:
```

同一 prompt、模型、数据、seed、解码、batch 和加速协议下：

| 模型 | 方法 | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | Vanilla（官方 POPE prompt） | 79.0667% | 84.4392% | 71.2667% | 77.2957% |
| LLaVA-1.5-7B | SHIELD 推荐三组件（官方 POPE prompt） | **83.6333%** | 82.7810% | **84.9333%** | **83.8434%** |
| LLaVA-1.5-7B | SHIELD Full 四组件历史消融 | 83.5333% | 82.2436% | 85.5333% | **83.8562%** |
| Qwen2.5-VL-3B | Vanilla | 83.0000% | 83.2661% | 82.6000% | 82.9317% |
| Qwen2.5-VL-3B | 默认 ONLY | 83.6000% | 94.5230% | 71.3333% | 81.3070% |
| Qwen2.5-VL-3B | AHT tuned ONLY | **85.0000%** | 86.1570% | **83.4000%** | **84.7561%** |

LLaVA 官方 prompt Vanilla：
`outputs/pope_llava_7b_adversarial_promptfix_vanilla_20260922.metrics.json`

LLaVA 官方 prompt SHIELD Full（四组件历史消融）：
`outputs/pope_llava_7b_adversarial_promptfix_full_20260922.metrics.json`

LLaVA 官方 prompt SHIELD 推荐三组件配置
（Adaptive Plausibility + Vulnerability Defense + Statistical Bias）：
`outputs/pope_llava_7b_adversarial_cumulative_statistical_official_20260922.metrics.json`

推荐三组件 POPE 指标：

```text
Accuracy 83.6333%
Precision 82.7810%
Recall 84.9333%
F1 83.8434%
```

相对同协议 Vanilla，SHIELD 推荐三组件的变化为：

```text
Accuracy  +4.5667 pp
Precision -1.6582 pp
Recall   +13.6667 pp
F1       +6.5476 pp
```

旧的 LLaVA ONLY 结果目录
`outputs/pope_llava_7b_adversarial_accel_full_20260912/` 使用无一词回答后缀
的 legacy prompt，只能作为历史诊断，不能与上述官方 prompt 结果直接比较。

Qwen baseline 结果目录：
`outputs/pope_qwen25_3b_adversarial_accel_full_20260912/`

Qwen 最新 Vanilla/调参 ONLY 重跑目录：
`outputs/pope_qwen25_3b_adversarial_rerun_20260912/`

Qwen AHT tuned ONLY 参数：

```text
only_tvd_gamma=1.99856
only_alpha_neg=0.0
only_alpha_pos=2.0
only_beta=0.135
```

Qwen tuned 完整结果：
`.codex/aht/sessions/20260912-192629/full/tuned_trial_0007/`

默认 SHIELD 方法只启用上述三个组件；Inherent Bias 和四组件 Full
保留为可选研究消融，不属于默认部署组合。

## 最新 CHAIR 结果

SHIELD-compatible CHAIR-500 协议，固定 500 张 COCO val2014 图像：

```text
prompt=Please describe this image in detail.
decoding=sample
seed=42
temperature=1
top_p=1
max_new_tokens=512
batch_size=32
image_batch_size=16
num_workers=4
prefetch_factor=4
cache=dynamic
```

Vanilla 与 ONLY 使用相同的模型、图像、prompt、seed、解码、批处理和加速协议：

| 模型 | 方法 | CHAIRs | CHAIRi | Recall |
|---|---|---:|---:|---:|
| LLaVA-1.5-7B | Vanilla | 0.518000 | 0.163492 | 0.808083 |
| LLaVA-1.5-7B | ONLY | **0.506000** | **0.145494** | **0.833035** |

CHAIR 定义：

- `CHAIRs=hallucinated_captions/captions`
- `CHAIRi=hallucinated_object_mentions/object_mentions`
- `Recall` 为每张图 grounded COCO object coverage 的 macro mean
- ground truth 使用 COCO train2014/val2014 instances 与 captions，并采用 SHIELD `chair_eval.py` 兼容的 synonym/singularization 规则

正式结果文件：

- Vanilla metrics：
  `outputs/chair_llava_7b_shield_vanilla_b32_20260921.metrics.json`
- ONLY metrics：
  `outputs/chair_llava_7b_shield_only_b32_capture_topk20_v2_20260921.metrics.json`
- ONLY predictions：
  `outputs/chair_llava_7b_shield_only_b32_capture_topk20_v2_20260921.jsonl`
- ONLY branch-logit manifest：
  `outputs/chair_llava_7b_shield_only_b32_capture_topk20_v2_20260921.logits/manifest.json`

SHIELD 推荐三组件 CHAIR 结果复用：

`outputs/chair_llava_7b_shield_cumulative_statistical_b32_20260922.metrics.json`

```text
CHAIRs 0.358000
CHAIRi 0.100977
Recall 0.809490
```

ONLY branch-logit capture：

- `original_*` 对应原分支 `base_logits`
- `language_prior_*` 对应语言先验分支 `auxiliary_logits`
- 当前正式 sidecar 为 `topk=20`，`row_count=500`、`max_new_tokens=512`、`vocab_size=32064`
- `valid_steps=59036`，与 runtime `generated_tokens=59036` 一致
- `branch.npy` 使用 `0=negative branch`、`1=positive branch`
- `full` 模式会额外写入两组完整词表 float32 logits，500 图约需 `61.16 GiB`，运行前必须检查磁盘空间

CHAIR 加速验收同样必须满足：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

## 严格组件贡献结果

SHIELD 三组件的严格贡献必须使用完整 `2^3` 组合矩阵：

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

其中 `A=Adaptive Plausibility`、`V=Vulnerability Defense`、
`S=Statistical Bias`。不能用
`Vanilla -> A -> A+V -> A+V+S` 的累计差值替代独立贡献。

固定协议下的 full leave-one-out 结果（单位：pp，正值表示改善）：

| 数据集/指标 | A | V | S |
|---|---:|---:|---:|
| CHAIRs | -8.00 | +7.40 | +11.40 |
| CHAIRi | +0.86 | +2.84 | +3.88 |
| CHAIR Recall | +10.41 | -1.90 | -2.71 |
| POPE Accuracy | +7.33 | +0.93 | +0.40 |
| POPE Recall | -7.27 | +8.67 | +1.67 |
| POPE F1 | +4.29 | +2.33 | +0.60 |

Shapley 平均贡献：

| 数据集/指标 | A | V | S |
|---|---:|---:|---:|
| CHAIRs | -1.77 | +7.73 | +10.03 |
| CHAIRi | +1.49 | +2.09 | +2.67 |
| CHAIR Recall | +5.77 | -2.54 | -3.08 |
| POPE Accuracy | +5.01 | -0.64 | +0.19 |
| POPE Recall | -2.01 | +13.72 | +1.96 |
| POPE F1 | +3.69 | +2.27 | +0.59 |

完整组合矩阵、pairwise/third-order interaction 和 artifact 路径：
`reports/shield_strict_contributions_20260922.md`。

CHAIR prompt 审计：早期四个显式组件补跑节点错误继承了 POPE 的一词回答
prompt，产生 `Camera`/`Food`/`Yes` 等短答案，不能使用。正式严格矩阵改用
`protocolfix` artifacts；当前入口默认 `POPE=true`、`CHAIR=false`，并在
metrics 中记录实际 `pope_answer_instruction`。

## 复用规则

- 大型实验（如 CHAIR、POPE）已有结果通过协议与完整性校验且满足当前需求时，直接复用，不重复运行。
- 运行前检查目标目录的 metrics、预测条数、模型/数据哈希和协议字段。
- POPE 满足 `status=PASS`、`count=3000` 及加速验收条件时直接复用，不重复运行。
- CHAIR 满足 `status=PASS`、`count=500`、CHAIR 协议字段与固定资源哈希一致，且加速验收条件通过时直接复用，不重复运行。
- 若只需要分析 ONLY 两个分支 logits，额外检查 logits `manifest.json` 的 `complete=true`、`mode`、`topk`/词表大小、`valid_steps` 与 metrics 的 `runtime.generated_tokens` 一致。
- 修改模型、数据、prompt、解码、加速协议或方法参数时，使用新的带日期输出目录。

## 最小验收

```bash
.venv/bin/python -m py_compile <changed_script.py>
git diff --check
uv pip check --python .venv/bin/python
```
