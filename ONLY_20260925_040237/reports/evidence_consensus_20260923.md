# A+S 替代研究：修正与 ECP

日期：2026-09-23

## 目标与状态

不是“代码可运行”就算完成。目标是在同一 LLaVA-1.5-7B、POPE-3000 /
CHAIR-500 canonical 协议下，新模块接近 A+S，且 Recall 损失不超过声明门槛。
原始 SHIELD 默认配置不改；本轮没有重跑已有 A+S、Vanilla 或 direct VRSC。

**当前判断：ECP 的完整样本及排除本轮 pilot 图像后的六项点估计均达标，
已有证据支持它作为接近 A+S 的研究候选。未证明统计等效或显著超越。**
参数在完整运行前冻结，没有依据完整结果继续调整。

上一轮 residual 生成有分支路由错误，四组旧 smoke 不能用于机制结论。
修复与失效目录见 `reports/vrsc_residual_routing_correction_20260923.md`。

## 修正后的 residual

固定 `eta=0.25, weight=2, trust=1, clip=1`，实际路由
`base/semantic/shuffled`，先做原始正交残差再裁剪：

| 方法 | POPE Acc | POPE F1 | POPE Recall | CHAIRs | CHAIRi | CHAIR Recall |
|---|---:|---:|---:|---:|---:|---:|
| A+S，复用 | 82.700% | 81.511% | 76.267% | 0.432 | 0.129335 | 0.828539 |
| Direct VRSC，复用 | 82.700% | 81.107% | 74.267% | 0.504 | 0.144168 | 0.830550 |
| Residual，修正后 | 82.700% | 81.134% | 74.400% | 0.494 | 0.136131 | 0.839315 |

修正有实际改善，但仍未达到 A+S。不能把这个实现错误当作唯一失败原因。
CHAIRs 相对 A+S 的差距为 +6.2 pp，图像配对 bootstrap 95% 区间
[+1.40, +10.405] pp。

新结果：

- `outputs/vrsc_20260923/pope3000_residual_routefix_v2/`
- `outputs/vrsc_20260923/chair500_residual_routefix_v2/`
- `outputs/vrsc_20260923/residual_pope_as_audit.json`
- `outputs/vrsc_20260923/residual_chair_as_audit.json`

## 新机制

ECP 保留“合理性约束”和“选择性重读视觉证据”，但不沿用 A 的固定概率
阈值或 S 的 min-max/power/gain 双门控。它包含两个数学步骤：

1. 在 patch-caption 对上求一个稀疏联合证据分布；其 patch 边缘分布控制
   视觉增强，caption 边缘分布控制记忆保留。
2. 对 clean 与 evidence 两个 next-token 分布进行双参考卡方投影，输出
   支持集与重排序同时由一个拉格朗日乘子求出。

```text
u = (1-a) log p + a log g
q = argmax <q,u> - tau/2 * [(1-a) chi_square(q,p) + a chi_square(q,g)]
h = 1 / [(1-a)/p + a/g]
q = h * [1 + (u-nu)/tau]_+
```

这改变的是信赖代价本身，不是把两条 logits 作加减后再套 A。
联合证据分布没有固定行列边缘约束，不将它包装成标准最优输运；
CLIP affinity 也不是已验证的视觉真实性概率。

首版固定 `a=0.75, tau=1, transport_trust=1, feature_gain=1`，POPE 与
CHAIR 共用参数。简化消融 `evidence_shrink` 将 clean 和 evidence 都设为
evidence 分布，检验额外的 clean 约束是否必要。`evidence_visual` 去掉
caption 记忆，保留相同视觉增强与双参考投影。

## 初筛

使用 POPE-256 / CHAIR-64 新运行，与已有 A+S 相同题目/图像的预测对齐，
不是拿子集数字直接对比 A+S 全集数字。

CHAIR-64：

| 方法 | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| A+S，同图像复用 | 0.437500 | 0.131004 | 0.840234 |
| ECP | 0.453125 | 0.130612 | 0.843490 |
| evidence_shrink | 0.406250 | 0.106250 | 0.839193 |
| evidence_visual | 0.578125 | 0.141104 | 0.846745 |

ECP POPE-256 的 Accuracy=78.906%，相对复用 A+S 同题目的差距 -0.391 pp，
F1 差距 -0.324 pp。64 图的区间很宽，只支持扩展验证，不能宣称成功。
简化版本更好的点估计要求保留消融，不应为了叙事强行增加双参考复杂性。

## 完整验证约定

2026-09-23 14:08（Asia/Shanghai）冻结首版参数，只运行上述 ECP 和
evidence_shrink 的完整数量级，不继续扫描系数。

用于描述“接近”的工程门槛（并非统计等效证明）：

- POPE Accuracy、F1 相对 A+S 最多降低 0.5 pp；Recall 最多降低 2 pp。
- CHAIRs、CHAIRi 相对 A+S 最多上升 2 pp、1 pp；Recall 最多降低 1 pp。
- 同时报告原始指标、图像配对区间，以及排除初筛图像后的结果。
- 任何更好结果都只代表此模型、此协议、此 seed；不给出跨模型或因果证明。

以上约定保留，后续只按此门槛评价，不修改门槛迁就结果。

## 完整结果

ECP 与 evidence_shrink 已于 2026-09-23 14:25 前全部完成。两任务各自
`status=PASS`、完整数量、资源/协议对齐、逐条预测重算、加速验收均通过。

| 方法 | POPE Acc | POPE Precision | POPE Recall | POPE F1 | CHAIRs | CHAIRi | CHAIR Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A+S，复用 | 82.700% | 87.529% | 76.267% | 81.511% | 0.432 | 0.129335 | 0.828539 |
| ECP | 83.067% | 88.509% | 76.000% | 81.779% | 0.422 | 0.124098 | 0.829856 |
| evidence_shrink | 82.467% | 87.577% | 75.667% | 81.187% | 0.408 | 0.119522 | 0.819597 |
| evidence_visual，无记忆 | 82.833% | 89.243% | 74.667% | 81.307% | 0.490 | 0.137931 | 0.844041 |

ECP POPE confusion counts 为 `TP=1140,TN=1352,FP=148,FN=360`，
A+S 为 `1144,1337,163,356`；差异并非指标四舍五入造成。
CHAIR 为 `211/500` 个含幻觉描述、`447/3602` 个幻觉对象提及；
A+S 对应 `216/500`、`455/3518`。

ECP 平均描述词数 89.880，A+S 为 86.168；对象提及数也更多，不能把此次
CHAIR 改善简单解释为“更短、少说话”。这些长度统计不构成其他质量保证。

### 配对不确定性

下表均为 ECP 减 A+S，单位 pp，2000 次图像聚类配对 bootstrap；
POPE 以 500 张图而非 3000 个独立问题为抽样单位。
CHAIRi 每次由幻觉/全部对象总数计算，不平均逐图 CHAIRi。

| 指标 | 差值 | 95% 区间 | 点估计接近门槛 |
|---|---:|---:|---|
| POPE Accuracy | +0.367 | [-0.600, +1.300] | 通过 |
| POPE F1 | +0.269 | [-0.866, +1.331] | 通过 |
| POPE Recall | -0.267 | [-1.867, +1.135] | 通过 |
| CHAIRs | -1.000 | [-5.400, +3.000] | 通过 |
| CHAIRi | -0.524 | [-2.018, +1.017] | 通过 |
| CHAIR Recall | +0.132 | [-1.166, +1.405] | 通过 |

多个区间超出预设接近边界，且所有区间跨零。因此通过的是描述性工程门槛，
不是非劣效检验、等效检验或显著优越性检验。

### 排除初筛图像

POPE 的前 256 问题涉及 43 张图，排除前 258 条以完整排除这些图，
剩余 2742 问题/457 图。CHAIR 排除前 64 图，剩余 436 图。
审计脚本会拒绝切开同一图像的 `--start`。

| 指标 | ECP | 同图 A+S | 差值 pp |
|---|---:|---:|---:|
| POPE Accuracy | 83.443% | 83.005% | +0.438 |
| POPE F1 | 82.168% | 81.839% | +0.329 |
| POPE Recall | 76.295% | 76.586% | -0.292 |
| CHAIRs | 0.417431 | 0.431193 | -1.376 |
| CHAIRi | 0.123072 | 0.129085 | -0.601 |
| CHAIR Recall | 0.827855 | 0.826822 | +0.103 |

六项仍通过。这里只排除了本轮 pilot 图像，过去的原型也观察过完整任务；
不得称为从未接触过的独立测试集，更不能消除模型选择偏差。

## 消融与叙事纠正

`evidence_shrink` 在 CHAIR 幻觉率上更好，但 CHAIR Recall 低于 ECP
1.026 pp，POPE Accuracy/F1 低 0.600/0.592 pp。ECP 减 shrink 的
POPE Accuracy 配对区间为 [+0.133,+1.067] pp，F1 为 [+0.050,+1.126] pp。
这支持 clean 约束在该 POPE 运行中的价值，不支持它对所有任务占优。
CHAIR Recall 的差异区间 [-0.151,+2.182] pp 仍跨零。

简化版也通过全量的点估计门槛；在排除 pilot 的 CHAIR Recall 上差
-1.010 pp，略超 -1 pp 边界，不能把这个微小差距当统计拒绝。
它保留为更简单的 Pareto 选项，不宣称复杂版是唯一有效方案。

额外 CPU 检查对齐同一 CLIP tokenizer 的 caption 长度与
`probe_diagnostics.json` 的 `selected_caption_tokens`：

| 数据集 | 图数 | 每图 caption token 均值 | 实际保留率 | 联合分布非零比例均值 |
|---|---:|---:|---:|---:|
| POPE | 500 | 51.964 | 100% | 0.814143 |
| CHAIR | 500 | 51.154 | 100% | 0.813272 |

**联合稀疏不等于 caption 边缘稀疏。** 此配置保留了全部 caption tokens，
不能把改进归因为“精准 token 筛选”。当前实测支持的叙事是：caption
作为自生成记忆提议，原始分布通过双参考投影约束其概率跃迁。
视觉增益和记忆筛选的各自因果作用仍需进一步消融。

2026-09-23 14:33 开始补充完整 `evidence_visual` 消融，只去掉 caption
注入、保留视觉增强和双参考公式，不调整其他参数。GPU0 的其他用户进程
保持不动；两张卡检查进程归属与峰值显存后各启动一个本用户进程。

消融已完成，POPE-3000 与 CHAIR-500 均通过运行/加速/协议审计。
去掉记忆后 CHAIRs 相对 A+S 增加 5.800 pp，配对区间
[+0.995,+10.600] pp；不满足接近门槛。ECP 相对无记忆版本的 CHAIRs
降低 6.800 pp，配对区间 [-11.800,-2.000] pp。

与此同时无记忆版本 CHAIR Recall 较 ECP 高 1.418 pp。这支持 caption
路径对当前幻觉/覆盖率权衡的作用，不等于证明记忆无偏，也不把跨 token
采样轨迹变化包装成确定性的逐词因果证明。视觉增强本身是否必需、是否
优于简单均匀增强尚未独立验证。

## 产物与复用

正式 A+S 原文件仅被读取：

- `outputs/pope_llava_7b_adversarial_subset_adaptive_statistical_official_20260922.metrics.json`
- `outputs/chair_llava_7b_shield_subset_adaptive_statistical_protocolfix_20260922.metrics.json`

完整 ECP/shrink 预测、配置、源码哈希和指标：

- `outputs/vrsc_20260923/pope3000_ecp_frozen_v1/`
- `outputs/vrsc_20260923/chair500_ecp_frozen_v1/`
- `outputs/vrsc_20260923/pope3000_ecp_visual_ablation_v1/`
- `outputs/vrsc_20260923/chair500_ecp_visual_ablation_v1/`

CPU 审计文件位于 `outputs/vrsc_20260923/`：

- `ecp_{pope,chair}_as_full_audit.json`
- `ecp_{pope,chair}_as_excluding_pilot_audit.json`
- `ecp_shrink_{pope,chair}_as_full_audit.json`
- `ecp_shrink_{pope,chair}_as_excluding_pilot_audit.json`
- `ecp_vs_shrink_{pope,chair}_full_audit.json`
- `ecp_visual_{pope,chair}_as_full_audit.json`
- `ecp_vs_visual_{pope,chair}_full_audit.json`

审计会拒绝未完成状态、错误数量、重复 ID、错误实际分支路由、
预测与 metrics 不一致、资源哈希不一致、协议不一致或加速失败。
它验证 artifact 内记录的资源哈希，而非逐次重新读取全部权重。

复算示例，`--output` 必须为不存在的新文件，不覆盖历史证据：

```bash
.venv/bin/python scripts/compare_evidence_results.py \
  --candidate outputs/vrsc_20260923/pope3000_ecp_frozen_v1/evidence_consensus.metrics.json \
  --reference outputs/pope_llava_7b_adversarial_subset_adaptive_statistical_official_20260922.metrics.json \
  --dataset pope --start 258 \
  --output outputs/vrsc_20260923/ecp_pope_audit_new.json
```

## 数值修复与成本

新增的极端反向 logits 测试暴露 `log_h-logsumexp(log_h)` 在约 -2000
处的 float32 相消误差。已修复：仅在 prior 未通过原投影器的归一化检查时
改用稳定 softmax；原成功路径不变，四个权重端点/内点逐位回归通过。
冻结 ECP 实验均成功结束，因此不可能经过原来会抛异常的失败路径。
不将修复后的源码哈希冒充冻结结果的源码哈希。

ECP 需要 clean/evidence 两条 KV cache；当前简化消融沿用双分支 runtime，
没有专门优化为单次前向。记录的 ECP generation time 为 POPE 379.16 秒、
CHAIR 443.93 秒，峰值 allocated VRAM 为 23.70/45.98 GiB。
这些时间受共享 GPU 负载影响，且不包含复用 naive caption 的生成开销，
不用于声称加速或严格计算量优势。

## 当前结论

旧 VRSC 的 A+S 替代结论应是否定。修正后的 ECP 已补上完整数量级的
接近效果证据；新叙事不再依赖未证实的扰动真实性或 caption 精准筛选。
保持研究候选状态，下一阶段若主张等效或优越，需要独立 seed/数据/模型的
验证，不能用本次点估计替代。

工程验证：全仓 `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q`
通过 92 项；相关文件 Ruff、py_compile、`uv pip check`、`git diff --check`
通过。极端输入、KKT、冻结路径一致性、双/三分支 prefix 与 KV 路由、结果
拒绝规则和 CHAIR micro 聚合均有测试。所有本轮 GPU session 正常退出，
未干预其他用户进程。
