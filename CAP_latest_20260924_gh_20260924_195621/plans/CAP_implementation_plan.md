# CAP 方法改造与可行性验证 — 完整执行计划

> 本计划面向执行 agent，提供从代码集成到论文验证的完整路径。
> 每个 Task 可独立分配，含明确输入、输出、验收标准和依赖关系。

---

## 总体架构

```
Phase 0: 环境与基线复现          [Day 1]
    ↓
Phase 1: CAP 代码集成与单元测试    [Day 1-2]
    ↓
Phase 2: CHAIR-500 主实验          [Day 2-3]
    ↓
Phase 3: POPE-3000 主实验          [Day 3]
    ↓
Phase 4: 超参数扫描与消融          [Day 3-4]
    ↓
Phase 5: 失败案例分析与可视化      [Day 4-5]
    ↓
Phase 6: 交叉验证与鲁棒性检查      [Day 5-6]
    ↓
Phase 7: 论文实验表与图生成        [Day 6-7]
```

---

## Phase 0: 环境与基线复现

### Task 0.1 — 验证现有环境可运行

**目标**：确认现有代码库可正常执行 vanilla 和 gtp_patch 方法。

**执行步骤**：
1. 进入项目根目录，确认 `.venv` 存在且可用
2. 运行 `107 passed` 的测试套件确认无回归：
   ```bash
   cd <project_root>
   .venv/bin/python -m pytest tests/ -x -q
   ```
3. 确认模型权重、数据集、CLIP 模型路径正确（参照 `HPARAM.md`）

**验收标准**：
- [ ] `pytest` 全部通过（107 passed）
- [ ] `/data/lcq/.cache/huggingface/hub` 下的模型和数据文件可访问

**产出**：环境状态报告（`reports/env_status.md`）

---

### Task 0.2 — 复现 Vanilla 基线

**目标**：获取 Vanilla 在 CHAIR-500 和 POPE-3000 上的基线分数。

**执行步骤**：
```bash
# CHAIR-500
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 500 \
  --methods vanilla \
  --output outputs/vanilla_chair500_baseline_$(date +%Y%m%d)

# POPE adversarial-3000
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset pope --limit 3000 \
  --methods vanilla \
  --output outputs/vanilla_pope3000_baseline_$(date +%Y%m%d)
```

**验收标准**：
- [ ] CHAIR-500 Vanilla: CHAIRs ≈ 0.574, CHAIRi ≈ 0.167, Recall ≈ 0.803
- [ ] POPE-3000 Vanilla: Accuracy ≈ 79.5%, Precision ≈ 84.2%, F1 ≈ 78.0%
- [ ] 状态标记为 `PASS`（full count run）
- [ ] 加速协议正确：`flash_attention_2`, `fp8_effective=true`, `fp8_native_fallback_calls=0`

**产出**：`outputs/vanilla_*_baseline_*/vanilla.metrics.json`

**依赖**：Task 0.1

---

### Task 0.3 — 复现 GTP Patch Consensus 基线

**目标**：获取 GTP Patch 的现有结果，作为 CAP 的直接对比基线。

**执行步骤**：
```bash
# CHAIR-500
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 500 \
  --methods vanilla,gtp_patch \
  --output outputs/gtp_patch_chair500_baseline_$(date +%Y%m%d) \
  --gtp-patch-gain 0.35 --gtp-patch-temperature 0.05

# POPE-3000
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset pope --limit 3000 \
  --methods vanilla,gtp_patch \
  --output outputs/gtp_patch_pope3000_baseline_$(date +%Y%m%d) \
  --gtp-patch-gain 0.35 --gtp-patch-temperature 0.05
```

**验收标准**：
- [ ] CHAIR-500 GTP Patch: CHAIRs ≈ 0.514, CHAIRi ≈ 0.139, Recall ≈ 0.845
- [ ] POPE-3000 GTP Patch: Accuracy ≈ 82.5%, Precision ≈ 89.7%, F1 ≈ 80.8
- [ ] 状态标记为 `PASS`

**产出**：`outputs/gtp_patch_*_baseline_*/gtp_patch.metrics.json`

**依赖**：Task 0.1

---

## Phase 1: CAP 代码集成与单元测试

### Task 1.1 — 将 CAP 核心代码集成到项目中

**目标**：将 `cap.py` 复制到项目 `src/methods/` 并通过 import 测试。

**执行步骤**：

1. 将 `GTP_restructured_cap/src/methods/cap.py` 复制到 `<project_root>/src/methods/cap.py`
2. 在 `src/methods/__init__.py` 中添加 CAP 导出：
   ```python
   from methods.cap import CAPConfig, CAPMethod, causal_attribution_weights, energy_preserving_probe
   ```
3. 验证 import 成功：
   ```bash
   .venv/bin/python -c "from methods.cap import CAPConfig, causal_attribution_weights; print('OK')"
   ```

**验收标准**：
- [ ] `from methods.cap import ...` 无报错
- [ ] `CAPConfig(probe_gain=0.20)` 构造成功
- [ ] `CAPConfig(probe_gain=1.0)` 抛出 `ValueError`

**产出**：修改后的 `src/methods/cap.py` 和 `src/methods/__init__.py`

**依赖**：Task 0.1

---

### Task 1.2 — 集成 CAP 到 vrsc_probe.py

**目标**：让 `scripts/vrsc_probe.py` 支持 `--methods cap`。

**执行步骤**（参照 `integration/vrsc_probe_cap_patch.py` 中的 7 步指令）：

**步骤 A — 添加 import**（vrsc_probe.py 顶部）：
```python
from methods.cap import (
    CAPConfig,
    causal_attribution_weights,
    energy_preserving_probe as cap_energy_probe,
)
```

**步骤 B — 添加 branch routing**（`branch_names` 函数中）：
```python
"cap": ("base", "cap", None),
```

**步骤 C — 添加 CLI 参数**（`parse_args` 函数中）：
```python
parser.add_argument("--cap-probe-gain", type=float, default=0.20)
```

**步骤 D — 添加 config 验证**（`parse_args` 末尾）：
```python
CAPConfig(probe_gain=args.cap_probe_gain)
```

**步骤 E — 添加 `cap` 到 EXTRA_METHODS**：
```python
EXTRA_METHODS = (
    "evidence_consensus",
    "evidence_shrink",
    "evidence_visual",
    "gtp",
    "gtp_patch",
    "cap",       # <-- ADD
)
```

**步骤 F — 添加 `build_cap_feature_cache` 函数**：

这是最关键的集成步骤。CAP 需要在 `main()` 的 feature preparation 阶段提取 `h_gen`。

核心逻辑（伪代码）：
```python
def build_cap_feature_cache(
    model, processor, image_paths, rows,
    projected_base_cache,   # base projected features
    raw_features_cache,     # raw vision-tower features
    args, device, dtype, state, recipe,
) -> dict[str, torch.Tensor]:
    """For each image, compute causal attribution and apply probe."""
    cache = {}
    for path in tqdm(image_paths, desc="cap-attribution"):
        image_name = path.name
        row = next(r for r in rows if str(r["image"]) == image_name)

        # 1. Build prompt embeddings for this image
        prompt = format_pope_prompt(str(row["text"]), one_word=args.pope_answer_instruction)
        count = image_token_count(processor)
        input_ids = tokenize_legacy_image_prompt(
            processor.tokenizer, prompt,
            int(model.config.image_token_index), count,
        )
        tokenized = torch.tensor([input_ids], dtype=torch.long, device=device)
        text_embeds = model.get_input_embeddings()(tokenized)

        # 2. Get projected features
        p_i = projected_base_cache[str(path)].to(device=device, dtype=dtype)

        # 3. Insert into prompt
        image_positions = torch.where(
            tokenized[0] == int(model.config.image_token_index)
        )[0]
        image_start, image_end = int(image_positions[0]), int(image_positions[-1])
        fused = torch.cat([
            text_embeds[0, :image_start],
            p_i,
            text_embeds[0, image_end + 1:],
        ], dim=0).unsqueeze(0)
        attention_mask = torch.ones(1, fused.shape[1], dtype=torch.long, device=device)

        # 4. Single forward step → h_gen
        with torch.inference_mode(), fp8_context(state, recipe):
            outputs = model.language_model(
                inputs_embeds=fused, input_ids=None,
                attention_mask=attention_mask,
                position_ids=torch.arange(fused.shape[1], device=device).unsqueeze(0),
                past_key_values=None,
                cache_position=torch.arange(fused.shape[1], device=device),
            )
        h_gen = outputs.last_hidden_state[0, -1, :].detach()

        # 5. Causal attribution
        scores = causal_attribution_weights(p_i, h_gen)

        # 6. Energy-preserving probe on RAW features
        raw_x = raw_features_cache[str(path)].to(device=device, dtype=dtype)
        probed = cap_energy_probe(raw_x, scores, args.cap_probe_gain)
        cache[str(path)] = probed.detach()

    return cache
```

**步骤 G — 在 `main()` 中调用**：

在 caches 构建部分之后，添加：
```python
if "cap" in args.methods:
    required_caches.add("cap")
    cap_raw = build_cap_feature_cache(
        model, processor, paths, rows,
        caches["base"],  # projected base features (already built)
        raw,             # raw vision features (already built)
        args, device, dtype, state, recipe,
    )
    caches["cap"] = build_projected_feature_cache(
        model, paths, cap_raw, device, dtype,
        args.image_batch_size, args.progress, "cap-project",
    )
```

**验收标准**：
- [ ] `--methods cap` 不报错
- [ ] `--methods vanilla,cap` 可同时运行
- [ ] manifest.json 中记录 `cap` 的 branch routing 为 `("base", "cap", None)`
- [ ] `probe_diagnostics.json` 包含每张图的 cap 诊断信息

**产出**：修改后的 `scripts/vrsc_probe.py`

**依赖**：Task 1.1

---

### Task 1.3 — 编写并运行 CAP 单元测试

**目标**：验证 CAP 核心函数的数学正确性。

**执行步骤**：
1. 将 `GTP_restructured_cap/tests/test_cap.py` 复制到 `<project_root>/tests/test_cap.py`
2. 运行测试：
   ```bash
   .venv/bin/python -m pytest tests/test_cap.py -v -x
   ```
3. 如有失败，修复 `src/methods/cap.py` 后重新运行

**验收标准**（全部 17 个测试）：
- [ ] `test_causal_attribution_basic_properties` — 输出在 [-1, 1] 且 finite
- [ ] `test_causal_attribution_aligned_patch_is_highest` — 对齐 patch 得分最高
- [ ] `test_causal_attribution_anti_aligned_patch_is_lowest` — 反对齐 patch 得分最低
- [ ] `test_causal_attribution_rejects_mismatched_dimensions` — 维度不匹配报错
- [ ] `test_causal_attribution_rejects_bad_shapes` — 形状错误报错
- [ ] `test_causal_attribution_rejects_nonfinite` — 非有限值报错
- [ ] `test_probe_zero_gain_is_identity` — η=0 返回原始特征
- [ ] `test_probe_preserves_total_norm` — 能量守恒（误差 < 0.1%）
- [ ] `test_probe_high_scores_get_enhanced` — 高分 patch 增强
- [ ] `test_probe_rejects_bad_eta` — 非法 η 报错
- [ ] `test_probe_rejects_mismatched_shapes` — 形状不匹配报错
- [ ] `test_config_default_is_valid` — 默认配置合法
- [ ] `test_config_rejects_invalid_gain` × 3 — 非法 gain 报错
- [ ] `test_method_contract_dispatches_cap` — Method adapter 符合项目合约

**产出**：所有测试通过的报告

**依赖**：Task 1.1

---

### Task 1.4 — 运行完整测试套件确认无回归

**目标**：CAP 集成不破坏现有方法。

**执行步骤**：
```bash
.venv/bin/python -m pytest tests/ -x -q
```

**验收标准**：
- [ ] 原有 107 个测试全部通过
- [ ] 新增 17 个 CAP 测试全部通过
- [ ] 总计 124 passed, 0 failed

**依赖**：Task 1.2, Task 1.3

---

## Phase 2: CHAIR-500 主实验

### Task 2.1 — CHAIR-500 CAP 实验（首次运行）

**目标**：在 CHAIR-500 上首次运行 CAP，获取初步结果。

**执行步骤**：
```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 500 \
  --methods vanilla,cap \
  --output outputs/cap_chair500_$(date +%Y%m%d) \
  --cap-probe-gain 0.20
```

**验收标准**：
- [ ] 状态标记为 `PASS`（full 500 images）
- [ ] 加速协议正确（flash_attention_2, fp8, bf16）
- [ ] `caption_token_injection: false`
- [ ] 输出包含 `vanilla.metrics.json` 和 `cap.metrics.json`

**关键指标预期**（目标，非硬性要求）：
- CHAIRs < 0.520（vs GTP Patch 0.514，目标持平或更优）
- CHAIRi < 0.140（vs GTP Patch 0.139）
- Recall > 0.840（vs GTP Patch 0.845）

**产出**：
- `outputs/cap_chair500_*/cap.metrics.json`
- `outputs/cap_chair500_*/cap.jsonl`（逐句结果）
- `outputs/cap_chair500_*/probe_diagnostics.json`

**依赖**：Task 1.4

---

### Task 2.2 — CHAIR 结果初步分析

**目标**：分析 CAP 在 CHAIR 上的表现，识别成功和失败模式。

**执行步骤**：
1. 比较 vanilla vs cap 的 CHAIR_s, CHAIR_i, Recall
2. 逐句比较，找出：
   - CAP 修复了哪些 vanilla 的幻觉对象（win cases）
   - CAP 引入了哪些新幻觉对象（loss cases）
   - 对象类别分布（哪些类别的幻觉被消除/引入）
3. 统计 win/loss ratio

**产出**：`reports/cap_chair500_analysis.md`，包含：
- 指标对比表
- Win/loss 统计
- Top-10 修复的幻觉对象类别
- Top-10 新引入的幻觉对象类别（如有）
- 3-5 个典型 win case 的详细分析
- 3-5 个典型 loss case 的详细分析

**依赖**：Task 2.1

---

## Phase 3: POPE-3000 主实验

### Task 3.1 — POPE adversarial-3000 CAP 实验

**目标**：验证 CAP 在短序列问答任务上同样有效。

**执行步骤**：
```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset pope --limit 3000 \
  --methods vanilla,cap \
  --output outputs/cap_pope3000_$(date +%Y%m%d) \
  --cap-probe-gain 0.20
```

**验收标准**：
- [ ] 状态标记为 `PASS`（full 3000 questions）
- [ ] 加速协议正确

**关键指标预期**：
- Accuracy ≥ 82%（vs GTP Patch 82.5%，目标持平或更优）
- Precision ≥ 89%（vs GTP Patch 89.7%）
- F1 ≥ 80%（vs GTP Patch 80.8）

**产出**：`outputs/cap_pope3000_*/cap.metrics.json`

**依赖**：Task 1.4

---

### Task 3.2 — POPE 结果分析

**执行步骤**：
1. 比较 vanilla vs cap 的 Accuracy, Precision, Recall, F1
2. 分析 yes_ratio 变化（CAP 是否改变了模型的 yes/no 倾向）
3. 按 adversarial 类型分组分析（如果数据中有此字段）

**产出**：`reports/cap_pope3000_analysis.md`

**依赖**：Task 3.1

---

## Phase 4: 超参数扫描与消融

### Task 4.1 — Probe Gain 扫描

**目标**：找到 CAP 在 CHAIR 和 POPE 上的最优 gain，并绘制 sensitivity 曲线。

**执行步骤**：

在 CHAIR-64（快速探索集）上扫描：
```bash
for gain in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60; do
  .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset chair --limit 64 --start 0 \
    --methods vanilla,cap \
    --output outputs/cap_chair64_gain_${gain} \
    --cap-probe-gain ${gain}
done
```

在 POPE-256 上扫描：
```bash
for gain in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60; do
  .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset pope --limit 256 --start 0 \
    --methods vanilla,cap \
    --output outputs/cap_pope256_gain_${gain} \
    --cap-probe-gain ${gain}
done
```

**产出**：
- 各 gain 的 metrics JSON
- sensitivity 曲线（CHAIRs / POPE Accuracy vs gain）
- 最优 gain 推荐

**关键分析**：
- CHAIR 和 POPE 的最优 gain 是否一致？
- 曲线是否平滑（robust）还是尖锐（fragile）？
- 是否存在一个 gain 在两个指标上都表现良好？

**验收标准**：
- [ ] 至少完成 10 个 gain 值的扫描
- [ ] 绘制 sensitivity 曲线图
- [ ] 推荐在 CHAIR-500 和 POPE-3000 上验证的 1-2 个候选 gain

**依赖**：Task 1.4

---

### Task 4.2 — 独立验证集确认最优 gain

**目标**：用 CHAIR-500 的 holdout 部分和 POPE-3000 确认 Task 4.1 推荐的 gain。

**执行步骤**：

**注意**：CHAIR-64 用于探索，CHAIR-500 用于确认。
- Task 4.1 中如果 CHAIR 探索集使用 start=0, limit=64
- 则确认集应使用 start=64, limit=436（即 CHAIR 64-499）

```bash
# 在 CHAIR holdout 上确认
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 436 --start 64 \
  --methods vanilla,cap \
  --output outputs/cap_chair436_confirm_$(date +%Y%m%d) \
  --cap-probe-gain <best_gain>

# 在 POPE-3000 上确认
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset pope --limit 3000 \
  --methods vanilla,cap \
  --output outputs/cap_pope3000_confirm_$(date +%Y%m%d) \
  --cap-probe-gain <best_gain>
```

**验收标准**：
- [ ] holdout CHAIR 上 CAP 优于 vanilla
- [ ] POPE-3000 上 CAP 优于 vanilla
- [ ] gain 的表现在 holdout 上与探索集一致（无严重过拟合）

**产出**：`reports/gain_confirmation.md`

**依赖**：Task 4.1

---

### Task 4.3 — 消融实验（Ablation Study）

**目标**：验证 CAP 每个组件的贡献。

**消融项**（需要在代码中添加相应的消融分支）：

| 消融 | 说明 | 方法名 |
|------|------|--------|
| A1: 无干预 | Vanilla baseline | `vanilla` |
| A2: 随机 attribution | a_i ~ Uniform, 不用 h_gen | `cap_random` |
| A3: CLIP attribution | 用 CLIP caption similarity 代替 h_gen | `cap_clip` |
| A4: 无能量守恒 | 直接 scale 不做 norm 校正 | `cap_no_conservation` |
| A5: 无 energy-centering | 用 raw attribution 不做 energy-weighted centering | `cap_no_centering` |
| A6: CAP（完整） | 完整方法 | `cap` |

**执行步骤**：

需要在 `vrsc_probe.py` 中添加 A2-A5 的变体分支。每个变体只需修改 `build_cap_feature_cache` 中的一个步骤。

```bash
# CHAIR-500 消融
for method in vanilla cap_random cap_clip cap_no_conservation cap_no_centering cap; do
  .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset chair --limit 500 \
    --methods vanilla,${method} \
    --output outputs/ablation_chair500_${method}_$(date +%Y%m%d) \
    --cap-probe-gain <best_gain>
done

# POPE-3000 消融（如有时间）
for method in vanilla cap_random cap_clip cap_no_conservation cap_no_centering cap; do
  .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset pope --limit 3000 \
    --methods vanilla,${method} \
    --output outputs/ablation_pope3000_${method}_$(date +%Y%m%d) \
    --cap-probe-gain <best_gain>
done
```

**验收标准**：
- [ ] A6（完整 CAP）优于 A1（vanilla）
- [ ] A6 优于 A2（random），证明因果信号有效
- [ ] A6 优于 A3（CLIP），证明模型自身信号优于外部 CLIP
- [ ] A6 优于 A4（no conservation），证明能量守恒有贡献
- [ ] A6 优于 A5（no centering），证明 energy-centering 有贡献

**产出**：
- 消融结果表（Table 2 in paper）
- `reports/ablation_study.md`

**依赖**：Task 4.2（需要确认的 gain）

---

### Task 4.4 — 与 GTP Patch 的 head-to-head 对比

**目标**：在完全相同的条件下比较 CAP 和 GTP Patch。

**执行步骤**：
```bash
# CHAIR-500
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 500 \
  --methods vanilla,gtp_patch,cap \
  --output outputs/head2head_chair500_$(date +%Y%m%d) \
  --gtp-patch-gain 0.35 --gtp-patch-temperature 0.05 \
  --cap-probe-gain <best_gain>

# POPE-3000
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset pope --limit 3000 \
  --methods vanilla,gtp_patch,cap \
  --output outputs/head2head_pope3000_$(date +%Y%m%d) \
  --gtp-patch-gain 0.35 --gtp-patch-temperature 0.05 \
  --cap-probe-gain <best_gain>
```

**验收标准**：
- [ ] 三个方法使用完全相同的 seed、model、prompt、decoding、acceleration
- [ ] 输出包含三个独立的 metrics JSON

**产出**：head-to-head 对比表（Table 1 in paper）

**依赖**：Task 4.2

---

## Phase 5: 失败案例分析与可视化

### Task 5.1 — CHAIR 失败案例深度分析

**目标**：理解 CAP 何时失败，以及失败原因。

**执行步骤**：
1. 从 Task 4.4 的 CHAIR-500 结果中，找出：
   - vanilla 正确但 CAP 幻觉的样本（CAP regression）
   - vanilla 幻觉且 CAP 仍然幻觉的样本（CAP 无效）
   - vanilla 幻觉但 CAP 正确的样本（CAP improvement）
2. 对 regression cases 分析：
   - 是哪些对象类别？
   - attribution map 显示什么模式？
   - 能量重分配是否过于激进？
3. 对无效 cases 分析：
   - 是视觉特征本身的问题（图像质量）？
   - 还是 attribution 信号不够强？

**产出**：`reports/cap_failure_analysis.md`，包含：
- Regression/Improvement/No-change 统计
- 每类 5 个详细案例（含图片名、对象列表、attribution 分数）
- 失败模式分类和改进建议

**依赖**：Task 4.4

---

### Task 5.2 — Causal Attribution 可视化

**目标**：在典型图像上展示 CAP 的 causal attribution map。

**执行步骤**：
1. 从 CHAIR-500 中选择 10-20 张代表性图像：
   - 5 张 CAP 成功消除幻觉的
   - 5 张 CAP 引入幻觉的（regression）
   - 5 张包含多对象复杂场景的
   - 5 张简单场景
2. 对每张图，提取 `a_i` scores（shape: [576]）
3. 将 scores reshape 为 24×24 空间网格
4. 与原图叠加生成 heatmap
5. 同时对比 GTP Patch 的 `counterfactual_patch_scores` 的空间分布

**实现参考**：
```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_attribution(image_path, scores, save_path):
    """Overlay causal attribution heatmap on image."""
    scores_np = scores.cpu().numpy().reshape(24, 24)
    scores_np = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min())

    image = Image.open(image_path).resize((384, 384))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(scores_np, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title("Causal Attribution (CAP)")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(scores_np, cmap="RdYlGn", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
```

**产出**：
- `figures/attribution_maps/` — 20 张可视化图
- `reports/attribution_visualization.md` — 视觉分析报告

**依赖**：Task 4.4（需要 cap probe_diagnostics 中的 scores）

---

## Phase 6: 交叉验证与鲁棒性检查

### Task 6.1 — 多种子验证

**目标**：验证 CAP 结果对随机种子的鲁棒性。

**执行步骤**：
```bash
for seed in 42 123 456 789 2026; do
  .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset chair --limit 500 \
    --methods vanilla,cap \
    --output outputs/cap_chair500_seed${seed} \
    --cap-probe-gain <best_gain> \
    --seed ${seed}
done
```

**注意**：CAP 本身的 attribution 计算不涉及随机性（它是确定性的），但 LLM 的 sampling decoding 有随机性。因此多种子验证主要测试解码阶段的鲁棒性。

**验收标准**：
- [ ] 5 个 seed 的 CHAIRs 标准差 < 0.03
- [ ] 所有 seed 上 CAP 优于 vanilla
- [ ] 报告 mean ± std

**产出**：
- `reports/multiseed_chair.md`
- bootstrap 95% 置信区间

**依赖**：Task 4.2

---

### Task 6.2 — CHAIR 子集稳定性

**目标**：验证 CAP 在不同 CHAIR 子集上的一致性。

**执行步骤**：
```bash
# 5 折交叉验证
for fold in 0 1 2 3 4; do
  start=$((fold * 100))
  .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset chair --limit 100 --start ${start} \
    --methods vanilla,cap \
    --output outputs/cap_chair_fold${fold} \
    --cap-probe-gain <best_gain>
done
```

**验收标准**：
- [ ] 5 折中 CAP 均优于 vanilla
- [ ] fold 间方差合理

**产出**：`reports/cross_fold_stability.md`

**依赖**：Task 4.2

---

### Task 6.3 — 计算开销分析

**目标**：量化 CAP 的额外计算成本。

**执行步骤**：
1. 从 manifest.json 中提取 `feature_and_load_seconds` 和 `generation_seconds`
2. 比较 vanilla vs cap 的总推理时间
3. 分解 CAP 额外开销：
   - h_gen 提取（1 次额外 forward step）
   - cosine similarity 计算
   - energy-preserving probe
   - 重新 projection

**验收标准**：
- [ ] CAP 额外时间开销 < 10%（相对于 vanilla）
- [ ] 无额外 GPU 显存开销（或 < 5%）

**产出**：`reports/computational_cost.md`，包含：
- 方法对比的 wall-clock 时间表
- 开销分解
- 与 GTP Patch 的计算成本对比（GTP 需要 CLIP 编码）

**依赖**：Task 4.4

---

## Phase 7: 论文实验表与图生成

### Task 7.1 — Table 1: Main Results

**目标**：生成论文主实验结果表。

**格式**：
```
| Method       | CHAIR_s ↓ | CHAIR_i ↓ | Recall ↑ | POPE Acc ↑ | POPE Prec ↑ | POPE F1 ↑ |
|-------------|-----------|-----------|----------|------------|-------------|-----------|
| Vanilla     | 0.574     | 0.167     | 0.803    | 79.5       | 84.2        | 78.0      |
| SHIELD (AS) | ...       | ...       | ...      | ...        | ...         | ...       |
| GTP Patch   | 0.514     | 0.139     | 0.845    | 82.5       | 89.7        | 80.8      |
| CAP (Ours)  | ???       | ???       | ???      | ???        | ???         | ???       |
```

**数据来源**：Task 0.2, 0.3, 4.4

**产出**：`tables/main_results.tex`

**依赖**：Task 4.4

---

### Task 7.2 — Table 2: Ablation Study

**格式**：
```
| Variant             | CHAIR_s ↓ | CHAIR_i ↓ | POPE Acc ↑ |
|--------------------|-----------|-----------|------------|
| Vanilla            |           |           |            |
| CAP (random a_i)   |           |           |            |
| CAP (CLIP a_i)     |           |           |            |
| CAP (no conserve)  |           |           |            |
| CAP (no center)    |           |           |            |
| CAP (full)         |           |           |            |
```

**数据来源**：Task 4.3

**产出**：`tables/ablation.tex`

**依赖**：Task 4.3

---

### Task 7.3 — Figure 1: 方法概览图

**描述**：展示 CAP 的完整流程。

**元素**：
1. 输入图像
2. Vision Tower → raw features x_i
3. Multi-modal Projector → p_i
4. LLM Forward → h_gen（高亮生成位置）
5. Causal Attribution: cosine_sim(p_i, h_gen)（带公式）
6. Energy-Preserving Probe（带公式和示意图）
7. Re-project → LLM Decoding

**实现**：使用 matplotlib 或 draw.io 生成矢量图。

**产出**：`figures/cap_overview.pdf`

**依赖**：无（可并行）

---

### Task 7.4 — Figure 2: Causal Attribution 可视化

**数据来源**：Task 5.2

**产出**：`figures/attribution_maps.pdf`

**依赖**：Task 5.2

---

### Task 7.5 — Figure 3: Gain Sensitivity 曲线

**数据来源**：Task 4.1

**格式**：
- X 轴: probe_gain (0.05 ~ 0.60)
- Y 轴 (左): CHAIR_s (↓)
- Y 轴 (右): POPE Accuracy (↑)
- 两条曲线 + 最优点标注

**产出**：`figures/gain_sensitivity.pdf`

**依赖**：Task 4.1

---

### Task 7.6 — Figure 4: Failure Case 分析

**描述**：展示 CAP regression 和 improvement 的典型案例。

**格式**：
- 每行一个案例：原图 | Vanilla caption | CAP caption | Attribution heatmap
- 红色标注幻觉对象，绿色标注正确对象

**数据来源**：Task 5.1, 5.2

**产出**：`figures/failure_cases.pdf`

**依赖**：Task 5.1, 5.2

---

## 附录：Agent 分配建议

| Agent | 负责 Tasks | 核心技能要求 |
|-------|-----------|-------------|
| **Agent A: 集成工程师** | 0.1, 1.1, 1.2, 1.3, 1.4 | Python, PyTorch, 项目结构理解 |
| **Agent B: 实验执行者** | 0.2, 0.3, 2.1, 3.1, 4.1, 4.2, 4.4, 6.1, 6.2 | GPU 资源, 脚本执行, 日志分析 |
| **Agent C: 分析师** | 2.2, 3.2, 4.3, 5.1, 6.3 | 数据分析, 统计, 报告撰写 |
| **Agent D: 可视化/论文** | 5.2, 7.1-7.6 | matplotlib, LaTeX, 论文写作 |

**关键路径**：Task 0.1 → 1.1 → 1.2 → 1.4 → 2.1 → 4.1 → 4.2 → 4.4 → 7.1

**预计总工时**：5-7 天（单 GPU）

---

## 附录：风险与降级方案

| 风险 | 概率 | 影响 | 降级方案 |
|------|------|------|---------|
| CHAIR 改善不显著 | 中 | 高 | 调整 gain 范围；尝试在 projector 后操作而非 vision tower |
| POPE 退化 | 低 | 高 | 分析 attribution 方向；尝试反转信号（抑制高分 patch） |
| 计算开销过大 | 低 | 中 | 缓存 h_gen；批量计算；只对 top-k patch 做 probe |
| h_gen 信号噪声大 | 中 | 中 | 尝试多层 hidden state 平均；或使用 attention weights 直接 |
| gain 过拟合 CHAIR-64 | 中 | 中 | 使用 CHAIR holdout 验证；报告探索/验证分离 |
| 能量守恒约束过紧 | 低 | 中 | 尝试软约束（λ·KL penalty）代替硬约束 |

---

## 附录：代码变更清单

| 文件 | 变更类型 | 变更量 |
|------|---------|--------|
| `src/methods/cap.py` | **新增** | ~150 行 |
| `src/methods/__init__.py` | 修改 | +1 行 import |
| `scripts/vrsc_probe.py` | 修改 | ~80 行（import + branch + CLI + cache builder） |
| `tests/test_cap.py` | **新增** | ~120 行 |
| `configs/method/cap.yaml` | **新增** | 2 行 |
| `configs/method/cap_random.yaml` | **新增**（消融） | 2 行 |
| `configs/method/cap_clip.yaml` | **新增**（消融） | 2 行 |

**总代码变更**：~350 行新增 + ~80 行修改
