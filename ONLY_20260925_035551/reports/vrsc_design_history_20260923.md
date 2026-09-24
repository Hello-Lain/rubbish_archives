# VRSC：视觉响应驱动的稀疏校准

本文件保存被替换的方案与实验历史，不代表当前推荐方向。
当前方案、公式与完整结果见根目录 `idea.md` 以及
`reports/evidence_consensus_20260923.md`。第 7 节已明确标为无效历史。

日期：2026-09-23  
状态：目标未达成，正在修正实现与机制；不进入默认 SHIELD 部署。

**更正**：第 7 节的 residual 自回归结果因分支路由错误而无效，相关停止
结论撤回，见 `reports/vrsc_residual_routing_correction_20260923.md`。

## 1. 新叙事

A 的有效部分是限制候选集合，避免低概率词因为采样噪声进入输出；S 的有效
部分是重新分配视觉 patch 的注意重点。两者不再串联成“先增强视觉特征、
再做候选硬阈值”，而是合并为一个分布级决策模块：

> 先构造一个不改变视觉总能量的“重新看图”探针，再让同一生成前缀下的
> clean/probe 输出差异成为词表级视觉响应，最后在原分布附近求一个带信赖
> 区域的稀疏最优分布。候选截断和视觉重排序由同一个优化问题决定。

模块名为 **VRSC (Visual-Response Sparse Calibration)**。

## 2. 能量守恒视觉探针

给定视觉 patch 特征 `x_i` 和由 S 的 CLIP-caption 相似度产生的 patch score
`s_i`，定义：

```text
w_i = ||x_i||^2 / sum_j ||x_j||^2
d_i = s_i - sum_j w_j*s_j
g_i = d_i / max_j |d_j|
```

先沿切空间重新分配 patch 能量，再恢复 Frobenius 范数：

```text
x'_i = (1 + eta*g_i) x_i
x*   = x' * ||x||_F / ||x'||_F
```

其中 `0 <= eta < 1`。常数 score、零特征或空 caption 直接返回原特征。
因为 `sum_i w_i*d_i = 0`，方向对总能量是一阶正交的，最后归一化保证
`||x*||_F = ||x||_F`。它表达的是视觉证据重新分配，而不是整体放大。

S 在这里只提供方向 proposal，不执行原始增强公式、不注入 caption token，
也不把 CLIP cosine 分数解释为 grounding probability。

## 3. 视觉响应与统一决策

在完全相同的 prompt/prefix 下，得到：

```text
p(w)       = softmax(z(w | x, prefix))
p_probe(w) = softmax(z(w | x*, prefix))
r(w)       = clip(log p_probe(w) - log p(w), -c, c)
```

令 `p` 为 clean 分布，构造 utility：

```text
u(w) = log p(w) - max_v log p(v) + lambda*r(w)
```

最终分布是 simplex 上的 chi-square trust-region 优化：

```text
q = argmax_q <q,u>
    - tau/2 * sum_w (q(w)-p(w))^2 / p(w)

subject to q(w) >= 0, sum_w q(w)=1
```

KKT 条件给出：

```text
q(w) = p(w) * [1 + (u(w)-nu)/tau]_+
sum_w q(w) = 1
```

`nu` 是所有 token 共用的对偶阈值，通过排序 active-set 精确求解。该公式
同时完成：

1. `log p` 保留 A 的候选合理性；
2. `r` 只奖励模型自身测得的视觉响应；
3. `1/p` 惩罚阻止极低概率 token 因单次响应大幅跃迁。

因此它不是 A 和 S 的顺序叠加，也不是简单的 contrastive logits 加权。
`lambda=0` 明确是纯收缩 control，不称为 Vanilla。

## 4. 固定原型与证伪条件

```text
eta=0.25
lambda=evidence_weight=2.0
tau=trust=1.0
c=response_clip=1.0
seed=42
```

必须保留 Vanilla、Adaptive、`lambda=0` 的 Shrink、语义 probe、固定随机
置换的 Shuffled probe、同熵 temperature control、Greedy 诊断和 A+S runtime
branch。首步 replay 不能代替自回归 POPE。

核心停止条件是：

```text
semantic response - shuffled response
```

不能稳定为正，或 VRSC 与 Shrink/Matched-Entropy 的收益相同，则不能把收益
归因于视觉证据。

## 5. 当前验证结论

POPE-256 首步配对 replay：

```text
VRSC expected correct: 81.708%
Shrink:                81.750%
Shuffled:              81.564%
semantic - shuffled correct-answer log-odds:
  -0.0132, image-cluster 95% interval [-0.0364, +0.0102]
```

POPE-64 自回归 smoke 中，VRSC、Shrink、Shuffled 均为 `84.375%` accuracy。
CHAIR-32 中 VRSC 为 `CHAIRs=0.5625`、`CHAIRi=0.153543`、
`Recall=0.825521`，没有超过主要对照。

结论：公式、数值实现和运行链路可行；“语义视觉响应优于随机探针”的机制
尚未得到支持。下一轮应重新设计视觉证据构造，而不是继续在当前四个系数上
调参。完整证据见 `reports/vrsc_feasibility_20260923.md`。

## 6. 针对性更新：随机扰动正交残差

完整 POPE-3000/CHAIR-500 复用验证显示，direct VRSC 没有超过
distribution-only controls，尤其 CHAIR hallucination 指标明显落后 A+S。
问题更像是 probe response 混入了“任意视觉扰动都会改变语言分布”的通用
nuisance，而不是单纯系数不足。

因此新增可选响应模式。令：

```text
delta_s = log p_semantic - log p
delta_n = log p_shuffled  - log p
a       = <delta_s, delta_n> / (||delta_n||^2 + epsilon)
r_res   = clip(delta_s - a*delta_n, -c, c)
```

这一步先从 semantic response 中正交去除与 shuffled response 同方向的
通用扰动，再送入原有 chi-square simplex 投影。它不使用标签，不改变
`q = argmax <q,u> - tau/2 * chi-square(q,p)` 的核心公式。

实现：

```text
configs/method/vrsc_residual.yaml
VRSCConfig(response_mode="orthogonal_residual")
```

复用 POPE-256 已捕获 logits 的首步验证：

```text
direct VRSC:       81.708%
orthogonal residual:81.938%
residual - direct: +0.230 pp
95% image-cluster interval: [-0.198, +0.696] pp
```

点估计改善，但区间跨 0，且没有新的自回归结果。因此 residual 是下一轮
端到端实验的候选，不是当前默认部署方法。

## 7. 端到端验证与停止决策

本节旧实验因路由错误无效，仅保留审计历史，不作为停止研究的依据。

为避免 residual 只在首 token replay 上看起来有效，已把解码器扩展为
clean/semantic/shuffled 三个独立 KV cache，并在相同前缀下计算三分支
响应。已有 direct、Shrink、Shuffled 与 A+S 结果均复用；只新增运行
residual 候选。

默认 residual（`trust=1.0`、`evidence_weight=2.0`）：

```text
POPE-64:  Accuracy 84.375%, F1 83.871%
CHAIR-32: CHAIRs 0.593750, CHAIRi 0.169231, Recall 0.847396
```

它与已有 Shrink 完全一致，说明正交残差没有改变实际采样轨迹；没有把
首 token 的 `+0.230 pp` 当成端到端收益。

对首 token 离线筛选出的高穿透候选（`trust=0.25`、`evidence_weight=3.0`）
也进行了 smoke：

```text
POPE-64:  Accuracy 81.250%, F1 80.645%
CHAIR-32: CHAIRs 0.531250, CHAIRi 0.153846, Recall 0.907813
```

该候选只改善了小规模 CHAIR，牺牲了 POPE-64 的 `3.125 pp` Accuracy，
不能作为跨任务默认机制，因此不更新 `vrsc_residual.yaml` 的默认参数。
新增三分支运行链路和结果文件保留为可复现实验，但 VRSC 全部变体目前
仍是 research-only，继续扩大 benchmark 没有证据基础。

最终判断：

```text
Direct VRSC:             deployment NO-GO
Default residual VRSC:   replacement NO-GO
Aggressive residual:     task-tradeoff only, deployment NO-GO
A+S reuse:               complete, no rerun required
```

## 8. 新候选：Evidence Consensus Projection（ECP）

原先的设计把“受扰动影响”误作“更可信”，且丢掉了 S 中的 caption 记忆
路径。新假设改为：**视觉与语言自生成记忆提出证据，原始读图分布负责
约束证据造成的概率跃迁；两者共同支持的词才值得放大。**
这仍是假设，caption 来自同一模型，并不是独立真值。

首先保留现有 CLIP 特征接口作为 affinity proposal，而不是 grounding
probability。令 `C[i,j]` 为 patch-caption 相似度，标准化后的 `Z` 输入
一个联合稀疏优化问题（`rho` 为均匀分布）：

```text
Pi = argmax_Pi <Pi,Z> - tau_e/2 * chi_square(Pi,rho)
Pi >= 0, sum_ij Pi[i,j] = 1

v_i = sum_j Pi[i,j]
m_j = sum_i Pi[i,j]
x'_i = x_i * (1 + gain * [N*v_i - 1]_+ / (N*v_i + 1))
memory = caption spans with m_j > 0
```

patch gain 与 caption memory 不再分别用 `gamma_gain/gain_per/threshold`
决策，而是从同一证据测度取边缘分布。常数 affinity 不增强也不注入记忆。
tokenizer span 映射复用本地已有接口，不将此实现细节算作新算法。

然后得到同一生成前缀下的 clean 分布 `p` 和 evidence 分布 `g`。令
`u = (1-a) log p + a log g`，统一解码目标为：

```text
q = argmax_q <q,u>
    - tau/2 * [(1-a) chi_square(q,p) + a chi_square(q,g)]
q >= 0, sum_w q(w)=1
```

其精确解为：

```text
h(w) = 1 / [(1-a)/p(w) + a/g(w)]
q(w) = h(w) * [1 + (u(w)-nu)/tau]_+
```

`nu` 由全词表归一化决定。与原 VRSC 的不同不只是换一个 response：
约束几何已经从 `1/p(w)` 改为 `(1-a)/p(w)+a/g(w)`。一边不支持的词，其
概率跃迁成本会变大；两个分支一致时退化到已知 Shrink，对照可被精确检验。
原 A 的固定 `beta*p_max` 硬阈值不再出现，支持集由对偶解决定。

计算时 `H=sum_w h(w)<=1`、`b=h/H`，可等价调用现有精确求解器：

```text
q = chi_square_projection(b, H*u, tau)
```

固定首版 `a=0.75, tau=1, tau_e=1, gain=1`，不依赖标签构造证据。
必须同时比较：ECP、`evidence_shrink`（仅 evidence 分布收缩）、
`evidence_visual`（去掉记忆）以及复用的 A+S。
先运行 POPE-256/CHAIR-64，选择后再冻结参数验证完整数量级；
不在同一评估集上不断搜索直到达到目标。此候选当前尚无效果结论。
