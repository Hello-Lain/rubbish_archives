# ECP：证据共识投影

日期：2026-09-23  
状态：固定参数的 POPE-3000 / CHAIR-500 已达到预先记录的 A+S 接近门槛。
这是单模型、单 seed 的实测结论，不是统计等效、显著超越或学术新颖性证明。
不替换默认 SHIELD A+V+S。

## 1. 问题应重新表述

旧 VRSC 把“视觉扰动导致概率变化”当作可信证据，但敏感性并不等于真实性；
它也丢掉了本地 S 实现的 caption 记忆路径。即使修复 residual 路由，
完整 CHAIRs 仍为 0.494，落后 A+S 的 0.432。因此当前方案不继续放大响应。

**新叙事：合理性不是先裁掉一批词，视觉证据也不是无条件放大另一批词。
一次概率跃迁必须同时承担原始读图分布与证据重读分布的代价。**

- A 的核心被保留为低支持候选的高代价，而非固定 `beta*p_max` 门槛。
- S 的核心被保留为基于图文关系重读视觉与自生成记忆，但替换其
  min-max、幂次、增益门槛与固定 caption 相似度阈值。
- 两者在同一个 token 决策目标中耦合；不是两个 logits 任意相加后再套 A。

名称：**Evidence Consensus Projection (ECP)**。caption 仍来自模型，
两分支并非独立证人，不能把“共识”解释为已校准的真实概率。

## 2. 同一证据测度产生视觉重点与记忆

沿用现有本地 CLIP 特征接口，得到 patch-caption affinity `C[i,j]`。
通道池化只是与现有 runtime 对接的 proposal，不声称是已验证的空间对齐。
令 `Z=(C-mean(C))/std(C)`，均匀参考分布 `rho[i,j]=1/(N*M)`：

```text
Pi* = argmax_{Pi >= 0, sum(Pi)=1}
      <Pi,Z> - tau_e/2 * sum_ij (Pi[i,j]-rho[i,j])^2 / rho[i,j]

Pi*[i,j] = rho[i,j] * [1 + (Z[i,j]-nu_e)/tau_e]_+
v_i = sum_j Pi*[i,j]
m_j = sum_i Pi*[i,j]
```

这是自由边缘的稀疏联合分布，不是固定行列边缘的最优输运。
两个边缘使用同一个对偶解，避免分别设计互不一致的视觉/记忆阈值：

```text
d_i  = N*v_i
x'_i = x_i * (1 + gain * [d_i-1]_+ / (d_i+1))
memory = caption spans with m_j > 0
```

`[d-1]_+/(d+1)` 是相对于均匀密度的有界正增益；不会负向抹去原图信息。
空或常数 affinity 不增强、不注入记忆。稀疏联合支持不保证 caption 边缘
也高度稀疏，实际保留比例需报告，不能仅凭公式声称实现精准 token grounding。

## 3. 核心公式：同时约束概率跃迁

在相同生成前缀 `y_<t` 下，分别得到：

```text
p(w) = P(w | image, prompt, y_<t)
g(w) = P(w | enhanced_image, selected_memory, prompt, y_<t)
u(w) = (1-a) log p(w) + a log g(w)
```

只用一个严格凹目标同时决定支持集和概率：

```text
q* = argmax_{q >= 0, sum(q)=1}
     <q,u> - tau/2 * [(1-a) chi_square(q,p) + a chi_square(q,g)]

chi_square(q,p) = sum_w (q(w)-p(w))^2 / p(w)
```

定义加权调和支撑：

```text
h(w) = 1 / [(1-a)/p(w) + a/g(w)]
```

KKT 条件给出精确解：

```text
q*(w) = h(w) * [1 + (u(w)-nu)/tau]_+
sum_w q*(w) = 1
```

关键不是平均分数，而是二次代价的曲率变成
`(1-a)/p + a/g`：只要某一分支强烈反对一个词，把它推到大概率就很昂贵。
所有候选共用一个由归一化求出的 `nu`，不再引入额外概率截断阈值。

进一步令 `H=sum(h)<=1`、`b=h/H`。将目标乘以正数 `H`、去掉常数后：

```text
q* = argmax_q <q,H*u> - tau/2 * chi_square(q,b)
```

因此可直接使用排序 active-set 求解器，时间复杂度 `O(V log V)`。
`H` 小意味着两分支整体分歧大，此时相对调和参考 `b` 的效用推动也变小，
而不是越分歧越激进地做差分外推。

可检验退化性质：

- `p=g` 时精确退化到单分布卡方收缩，不伪称 Vanilla。
- `a=0/1` 分别退化到原图/证据分布收缩。
- 交换 `p,g` 并以 `1-a` 替换 `a`，解不变。
- 概率非负、归一化；有效支持上的目标梯度等于同一对偶常数。

## 4. 算法与实现

```text
per image:
    construct affinity C from existing local features
    solve Pi once; derive enhanced patches and caption spans
    prepare clean and evidence prompt embeddings

per decoding step:
    forward clean and evidence branches with independent KV caches
    log_p, log_g <- float32 log_softmax
    log_h <- -logaddexp(log(1-a)-log_p, log(a)-log_g)
    H <- exp(logsumexp(log_h)); b <- normalize(exp(log_h))
    q <- chi_square_projection(b, H*((1-a)*log_p+a*log_g), tau)
    sample one token from q; append the same token to both branches
```

两任务固定 `a=0.75, tau=1, tau_e=1, gain=1`，不是按任务分别调系数。
完整参数在结果生成前冻结，之后没有根据完整指标搜参。
此配置下 POPE/CHAIR 各 500 图都保留了全部 caption tokens；联合分布的
非零项约 81.4%/81.3%，但 caption 边缘并未变稀疏。当前不能以
“筛除了不可信记忆”解释效果，记忆可信性主要由下一步的双参考代价约束。

代码：`src/methods/evidence_consensus.py`。  
配置：`configs/method/evidence_consensus.yaml`。  
实际 LLaVA 实验入口：`scripts/vrsc_probe.py --methods evidence_consensus`。  
CPU 对比入口：`scripts/compare_evidence_results.py`。

方法 adapter 遵循本仓库 backend contract；现有通用 wrapper 并未因此自动
获得 ECP，真实运行使用上述研究 CLI。不宣称 Qwen 或任意 backend 已适配。

## 5. 完整实测

同模型、数据、prompt、seed、解码、batch 与加速协议；A+S 直接复用已有结果。

| 方法 | POPE Acc | POPE F1 | POPE Recall | CHAIRs | CHAIRi | CHAIR Recall |
|---|---:|---:|---:|---:|---:|---:|
| A+S | 82.700% | 81.511% | 76.267% | 0.432 | 0.129335 | 0.828539 |
| 修正 residual | 82.700% | 81.134% | 74.400% | 0.494 | 0.136131 | 0.839315 |
| **ECP** | **83.067%** | **81.779%** | 76.000% | **0.422** | **0.124098** | **0.829856** |
| evidence_shrink | 82.467% | 81.187% | 75.667% | 0.408 | 0.119522 | 0.819597 |
| evidence_visual，无记忆 | 82.833% | 81.307% | 74.667% | 0.490 | 0.137931 | 0.844041 |

ECP 在 POPE Acc/F1、CHAIRs/CHAIRi/Recall 上的点估计不差于 A+S；
POPE Recall 低 0.267 pp。排除本轮 pilot 图像后六项接近门槛仍全部通过。
这些区间并未证明等效或优越，当前结论是**有全量证据支持的替代候选**。

简化版在 CHAIR 幻觉率上更好，但降低 Recall，并在 POPE 上落后 ECP。
保留这个更简单的 Pareto 选项，不为叙事掩盖消融结果。ECP 相对简化版的
POPE Accuracy 差为 +0.600 pp，图像配对 95% 区间 [+0.133, +1.067] pp；
这是探索性单 seed 证据，不是普遍优势证明。

无记忆版本则未接近 A+S 的 CHAIRs。ECP 相比该版本降低 CHAIRs 6.800 pp，
图像配对区间 [-11.800,-2.000] pp，支持保留记忆路径；但 CHAIR Recall
也降低 1.418 pp，仍是权衡而非所有指标都改善。

完整配对区间、排除规则、协议、成本和消融见
`reports/evidence_consensus_20260923.md`。

## 6. 结论边界与下一步

已解决的是原先明显落后 A+S 的完整 benchmark 表现，不是所有机制归因。
不把多个任务的指标加总成一个胜率，也不把实现 PASS 当作效果 PASS。

- 尚无跨 seed、跨模型、独立数据集证据；排除 pilot 不抹去过去对这批数据的观察。
- caption token span 映射沿用已有近似接口，未证明语义选择的准确性。
- 无记忆完整消融支持记忆路径的作用；视觉增强的独立作用尚未隔离，
  现有结果不证明稀疏 affinity 优于均匀或随机 affinity。
- clean 约束并非所有指标都更好，不能为了“巧妙”强行删除简化对照。
- 保留每次实现的源码哈希，不覆盖旧实验；新代码只修复极端归一化报错
  路径，冻结成功路径逐位一致。

旧 VRSC 推导及无效实验历史保存在
`reports/vrsc_design_history_20260923.md`；路由错误的正式更正见
`reports/vrsc_residual_routing_correction_20260923.md`。
