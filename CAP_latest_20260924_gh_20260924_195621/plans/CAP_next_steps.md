# CAP 下一步建议（基于当前验证状态）

---

## 当前状态总结

| 项目 | 状态 |
|------|------|
| 代码集成 | ✅ 完成，131 passed |
| 静态检查 | ✅ Ruff / py_compile / git diff / uv pip check 全通过 |
| GPU Smoke (POPE 8条) | ✅ 链路可行，加速协议合规 |
| Attribution 诊断 | ✅ 576 个 score 正常，probe norm ratio ≈ 1.00015 |
| 核心未验证项 | CHAIR-500 全量 / POPE-3000 全量 / gain sweep / 消融 |

---

## 建议执行顺序（按优先级）

### 第 1 步：修复 per-question attribution（阻塞项，~1 小时）

**问题**：同图多问时，当前实现只缓存一条 attribution probe，对后续问题使用同一组 `a_i`。
这会导致 POPE-3000 中同一张 COCO 图的 6 个问题共享 attribution，但每个问题的"关注重点"不同（如"有人吗？"关注人，"有狗吗？"关注动物）。

**两种修复策略（二选一）**：

| 策略 | 做法 | 代价 | 推荐度 |
|------|------|------|--------|
| **A: per-question attribution** | 每个问题独立 build prompt → 提取 h_gen → 计算 a_i | 推理时间 × 问题数/图数（POPE 约 ×4-6） | ⭐⭐⭐ 论文可辩护 |
| **B: fixed representative prompt** | 统一用 `"Please describe this image in detail."` 作为 prompt 计算 attribution | 与 CHAIR 描述任务一致，POPE 上可能略差但公平 | ⭐⭐ 简洁，但需要论证 |

**推荐策略 A**，理由：
- POPE 的问题是特定对象查询（"Is there a dog?"），用对应问题做 prompt 让 h_gen 真正编码"该问题需要关注什么"
- CHAIR 本身每图只有一个描述 prompt，不受影响
- 论文审稿人不会质疑"per-question attribution"的合理性

**实现要点**：
```python
# 在 build_cap_feature_cache 中：
# 不再按 image 缓存，改为按 (image_path, question_text) 缓存
# key 从 str(path) 改为 (str(path), row["text"])
# CHAIR 场景：每图唯一 prompt，退化为 per-image
# POPE 场景：每问题独立 prompt，per-question
```

**验收**：运行 32 条 POPE smoke，检查 `probe_diagnostics.json` 中是否每个 question_id 都有独立的 attribution score。

---

### 第 2 步：CHAIR-500 全量实验（核心证据，~3-4 小时 GPU）

**为什么先跑 CHAIR 而不是 POPE**：
- CHAIR 是 CAP 设计的首要目标（解决 GTP Patch 在长序列上表现差的问题）
- CHAIR 结果决定论文叙事的核心 claim
- 如果 CHAIR 改善不显著，需要重新审视方法设计

**命令**：
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/vrsc_probe.py \
  --mode generate \
  --dataset chair \
  --limit 500 \
  --methods vanilla,cap \
  --output outputs/cap_chair500_$(date +%Y%m%d) \
  --cap-probe-gain 0.20
```

**硬性验收**：
- [ ] status = PASS（500 images, full count）
- [ ] acceleration 合规（flash_attention_2, fp8, fallback=0）
- [ ] caption_token_injection = false
- [ ] 输出包含 vanilla.metrics.json 和 cap.metrics.json

**结果判定**：

| CHAIR_s 变化 | CHAIR_i 变化 | 结论 | 后续动作 |
|-------------|-------------|------|---------|
| < -3pp | < -1.5pp | **强信号**，方法有效 | 直接进入 POPE + gain sweep |
| -1 ~ -3pp | -0.5 ~ -1.5pp | **中等信号**，需 gain 调优 | 先做 gain sweep 再确认 |
| > -1pp 或恶化 | 无改善 | **弱信号** | 见下文"如果 CHAIR 无改善" |

---

### 第 3 步：POPE-3000 全量实验（并行或紧随 CHAIR，~2 小时 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/vrsc_probe.py \
  --mode generate \
  --dataset pope \
  --limit 3000 \
  --methods vanilla,cap \
  --output outputs/cap_pope3000_$(date +%Y%m%d) \
  --cap-probe-gain 0.20
```

**验收**：
- [ ] status = PASS（3000 questions）
- [ ] Accuracy ≥ 80%（不应比 vanilla 差太多）

---

### 第 4 步：Gain Sweep（CHAIR-64 探索，~30 分钟 GPU）

**在 CHAIR-64 上快速扫描，找到最优 gain 区间**：

```bash
for gain in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/vrsc_probe.py \
    --mode generate --dataset chair --limit 64 --start 0 \
    --methods vanilla,cap \
    --output outputs/cap_chair64_gain_${gain} \
    --cap-probe-gain ${gain}
done
```

**分析**：绘制 CHAIR_s vs gain 曲线，找最优区间。

**重要**：CHAIR-64 是探索集，最优 gain 需在 CHAIR 64-499（holdout）上确认，避免过拟合。

---

### 第 5 步：Holdout 确认 + Head-to-head（~4 小时 GPU）

```bash
# CHAIR holdout（start=64, limit=436）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 436 --start 64 \
  --methods vanilla,gtp_patch,cap \
  --output outputs/h2h_chair436_confirm_$(date +%Y%m%d) \
  --gtp-patch-gain 0.35 --gtp-patch-temperature 0.05 \
  --cap-probe-gain <best_gain>
```

**验收**：
- [ ] holdout 上 CAP ≥ GTP Patch（CHAIR_s 和 CHAIR_i）
- [ ] 三方法使用完全相同的协议

---

## 如果 CHAIR 无改善：诊断与修复路径

**不建议直接放弃**，应按以下顺序诊断：

### 诊断 1：Attribution 信号质量

```python
# 从 probe_diagnostics.json 分析：
# 1. a_i 的分布：是否集中在少数 patch？还是均匀分布？
# 2. a_i 的 max-min range：信号是否有区分度？
# 3. a_i 最高值对应的 patch 位置：是否在图中有意义的区域？
```

**如果 a_i 分布过于均匀** → 信号弱 → h_gen 可能没有足够的区分度
- 修复：尝试用注意力权重替代 cosine similarity
- 修复：尝试用多层 hidden state 的平均
- 修复：在 prompt 中添加更强的引导（如 "List all objects you see."）

### 诊断 2：Probe gain 是否合适

- 如果 gain 太小（< 0.05），特征变化不足以影响生成
- 如果 gain 太大（> 0.50），能量重分配过于激进，可能引入新误差
- 解决：Task 4 的 gain sweep 会揭示这一点

### 诊断 3：Probe 操作空间是否正确

当前 CAP 在 vision tower 特征空间操作（multi-modal projector 之前）。

**替代方案**：在 projector 之后操作

```text
# 当前（vision tower space）：
raw_features → CAP probe → multi_modal_projector → LLM

# 替代（projector space）：
raw_features → multi_modal_projector → CAP probe → LLM
```

**区别**：multi_modal_projector 是一个 MLP，非线性变换后特征空间不同。在 projector 后操作的 probe 直接作用于 LLM 的输入空间，可能更有效。

**实现方式**：修改 `build_cap_feature_cache`，将 probe 从 raw features 移到 projected features。

### 诊断 4：是否需要保留 caption 信号作为辅助

如果纯 causal signal 不够强，可以考虑 **hybrid 方案**：

```text
a_i = α · cosine_sim(p_i, h_gen) + (1-α) · max_j cosine(pool(x_i), c_j)
```

其中 α ∈ [0, 1]。α=1 是纯 CAP，α=0 退化为 GTP Patch 的 attribution。

这虽然增加了复杂度，但可以作为消融实验的一个有意义的数据点，证明"纯 causal 信号足够好"或"hybrid 更好"。

---

## 附加建议

### 多种子验证（在第 2-3 步之后）

CHAIR 和 POPE 各取 1-2 个 seed 做验证：
```bash
--seed 123  # 除默认 42 外的一个
```
不需做 5 个 seed，2-3 个足够证明非偶然。

### Attribution 可视化（在第 2 步之后）

从 CHAIR-500 中选 10 张典型图，保存 `a_i` 并 reshape 为 24×24 heatmap 叠加到原图。
这将为论文 Figure 2 提供素材，也能帮助诊断 attribution 质量。

### 计算开销记录

从 CHAIR-500 的 manifest.json 提取 `feature_and_load_seconds` 和 `generation_seconds`，与 vanilla 对比。
CAP 的额外开销应 < 10%（只有 1 次额外 forward step + cosine sim）。

---

## 时间线总结

```
Day 1  上午  修复 per-question attribution           [~1h]
       下午  CHAIR-500 全量 + POPE-3000 全量          [~4-6h GPU]
Day 2  上午  Gain sweep (CHAIR-64)                    [~30min GPU]
       下午  Holdout 确认 + H2H 对比                   [~4h GPU]
       晚上  结果分析 + Attribution 可视化              [~2h]
Day 3  消融实验（如主实验结果好）                        [~4h GPU]
       论文表格/图片生成                                [~2h]
```

**关键决策点**：CHAIR-500 结果出来后（Day 1 晚），根据改善幅度决定后续投入程度。
