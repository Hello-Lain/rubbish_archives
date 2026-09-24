# ECP：证据共识投影

本文记录 Evidence Consensus Projection（ECP）的方法定义、数学目标与推理流程。完整 benchmark 指标和统计分析见 `reports/evidence_consensus_20260923.md`；实现见 `src/methods/evidence_consensus.py`。

## 1. 方法目标

ECP 面向视觉语言模型的幻觉抑制。其基本判断是：视觉增强分支提供的是证据提议，不应自动被视作事实；原始图像分支则提供未经该证据变换的参考。最终 token 分布应同时考虑两者的支持程度，并限制只被一个分支支持的 token 发生过大的概率跃迁。

ECP 将过程分成两个相连的步骤：

1. 从 patch-caption 关联构造一个联合证据分布，并由其边缘得到视觉 patch 增益和 caption 记忆候选。
2. 在相同解码前缀下，联合原始分布与证据分布，求得最终 next-token 分布。

这里的“共识”指双参考约束，不表示两个分支是统计独立的证人，也不表示输出概率已经校准为视觉真实性概率。

## 2. 视觉证据测度

设图像有 $N$ 个视觉 patch，caption 有 $M$ 个 CLIP token。使用现有特征接口计算 patch 与 caption token 的余弦相似度：

$$
C_{ij} =
\operatorname{cos}\bigl(\operatorname{pool}(x_i), c_j\bigr)
$$

其中 $x_i$ 是图像 patch 特征，$c_j$ 是 caption token 特征；$\operatorname{pool}$ 将视觉特征通道映射到 CLIP 特征维度。对相似度矩阵作全局标准化：

$$
Z_{ij}=\frac{C_{ij}-\operatorname{mean}(C)}
{\max(\operatorname{std}(C),\epsilon)}
$$

以 patch-caption 对上的均匀分布 $\rho_{ij}=1/(NM)$ 为参考，求解单纯形上的稀疏联合分布：

$$
\Pi^*=\arg\max_{\Pi\ge 0,\ \sum_{ij}\Pi_{ij}=1}
\left[
\sum_{ij}\Pi_{ij}Z_{ij}
-\frac{\tau_e}{2}
\sum_{ij}\frac{(\Pi_{ij}-\rho_{ij})^2}{\rho_{ij}}
\right]
$$

这是自由边缘的联合优化，没有指定行、列边缘约束，因此不将其称为具有固定边缘的最优输运。对该目标使用 KKT 条件，可写成：

$$
\Pi^*_{ij}=\rho_{ij}
\left[1+\frac{Z_{ij}-\nu_e}{\tau_e}\right]_+,
\qquad \sum_{ij}\Pi^*_{ij}=1
$$

$\nu_e$ 是使联合分布归一化的公共对偶阈值，$[z]_+=\max(z,0)$。

从联合分布取两个边缘：

$$
v_i=\sum_j\Pi^*_{ij},\qquad
m_j=\sum_i\Pi^*_{ij}
$$

其中 $v_i$ 表示 patch 的相对证据密度，$m_j$ 表示 caption token 的联合支持质量。定义 patch 密度 $d_i=Nv_i$，并使用有界的正向增益：

$$
\gamma_i=\frac{[d_i-1]_+}{d_i+1},\qquad
x'_i=x_i(1+\kappa\gamma_i)
$$

$\kappa\ge0$ 是视觉增益系数。均匀密度 $d_i=1$ 不产生增益；高于均匀密度的 patch 得到正向增强；由于 $0\le\gamma_i<1$，增强倍数有界于 $1$ 与 $1+\kappa$ 之间。此公式不以负增益抹去原图特征。

caption 记忆由 $m_j>0$ 的 token 位置生成，并按连续 span 映射到语言模型的 caption token embedding。当前实现沿用已有 CLIP 到语言模型 token span 的近似映射；它不是精确词级对齐。

## 3. 双参考 token 投影

对同一 prompt 与生成前缀 $y_{<t}$，分别计算原始分支与证据分支：

$$
p(w)=P(w\mid x,\text{prompt},y_{<t})
$$

$$
g(w)=P(w\mid x',\text{selected memory},\text{prompt},y_{<t})
$$

其中 $p$ 来自原始图像，$g$ 来自 patch 增强图像及 caption 记忆。按权重 $a\in[0,1]$ 构造效用：

$$
u(w)=(1-a)\log p(w)+a\log g(w)
$$

最终分布 $q$ 由以下严格凹优化问题确定：

$$
q^*=\arg\max_{\substack{q(w)\ge0\\\sum_wq(w)=1}}
\left[
\sum_wq(w)u(w)
-\frac{\tau}{2}
\left((1-a)\chi^2(q,p)+a\chi^2(q,g)\right)
\right]
$$

其中

$$
\chi^2(q,p)=\sum_w\frac{(q(w)-p(w))^2}{p(w)}
$$

是以参考分布 $p$ 为基准的卡方距离，$\tau>0$ 控制分布变化的代价。

### 闭式 active-set 形式

定义加权调和支撑：

$$
h(w)=\frac{1}{(1-a)/p(w)+a/g(w)}
$$

KKT 条件表明，最优分布可写为：

$$
q^*(w)=h(w)
\left[1+\frac{u(w)-\nu}{\tau}\right]_+,
\qquad \sum_wq^*(w)=1
$$

$\nu$ 是由归一化确定的公共阈值。阈值以上的 token 构成输出支持集，支持集内的概率则按同一目标重分配。实现通过排序 active-set 求解该阈值。

等价地，令

$$
H=\sum_wh(w),\qquad b(w)=\frac{h(w)}{H}
$$

可将目标变换成以 $b$ 为单一参考分布的卡方投影：

$$
q^*=\arg\max_{q\in\Delta}
\left[
\langle q,Hu\rangle-\frac{\tau}{2}\chi^2(q,b)
\right]
$$

其中 $\Delta$ 是词表概率单纯形。该形式可复用排序求解器，词表维度为 $V$ 时复杂度为 $O(V\log V)$。

### 直观解释

- 若某 token 同时获得原始分支和证据分支支持，其效用与调和支撑均允许它参与概率竞争。
- 若某 token 只在一个分支中概率很高，另一个分支的低概率会压低其调和支撑，使其大幅跃迁变得昂贵。
- $\nu$ 对整个词表共享，候选筛选与概率校准来自同一个优化问题，不需要再叠加固定的概率截断门槛。

## 4. 自回归推理流程

```text
每张图像：
    提取视觉 patch 特征与 caption token 特征
    计算 patch-caption 相似度并求解联合证据分布 Pi
    从 Pi 的 patch 边缘构造增强特征 x'
    从 Pi 的 caption 边缘构造 caption memory
    分别准备 clean 与 evidence 两个分支的输入和 KV cache

每个解码步：
    用相同的已生成前缀分别计算 clean logits 与 evidence logits
    将两组 logits 转为 float32 概率 p、g
    计算加权效用 u、调和支撑 h 及其总量 H
    用排序 active-set 求解 q
    从 q 采样下一个 token，并将同一个 token 追加到两个分支
```

两个分支共享采样出的 token 序列，但维护各自的 KV cache。该设计保证每一步的概率比较对应相同生成前缀。

## 5. 参数

当前研究配置为：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `evidence_weight`（$a$） | 0.75 | token 效用与双参考代价中证据分支的权重 |
| `trust`（$\tau$） | 1.0 | token 分布相对双参考几何变化的代价尺度 |
| `transport_trust`（$\tau_e$） | 1.0 | patch-caption 联合测度偏离均匀分布的代价尺度 |
| `feature_gain`（$\kappa$） | 1.0 | patch 正向增益系数 |

POPE 与 CHAIR 使用相同参数。上述值是当前固定研究设置，不表示已证明其对其他模型或任务最优。

## 6. 与 A+S 的方法差异

A+S 中，S 主要在视觉侧工作：根据 patch-caption 相似度增强视觉 patch，并选择 caption span 注入记忆；随后 A 根据原始分布的相对最大概率设置硬候选门槛。A 的筛选条件可写成：

$$
p(w)\ge\beta p_{\max}
$$

ECP 保留“低视觉支持不应轻易主导输出”及“利用图文关系重新组织视觉证据”的目标，但改动了决策结构：

| 维度 | A+S | ECP |
|---|---|---|
| 候选控制 | A 用固定相对概率门槛屏蔽候选 | 由双参考目标的公共对偶阈值形成支持集 |
| 视觉处理 | S 使用相似度阈值、min-max、幂次和增益门槛 | 从同一联合证据分布的 patch 边缘导出有界正增益 |
| Caption 记忆 | S 按选中相似度 span 注入 | 从联合分布的 caption 边缘产生记忆候选 |
| Token 决策 | 视觉侧 S 与 token 侧 A 串联 | clean/evidence 两分布共同进入一个优化问题 |
| 分歧处理 | A 主要看原始分布，不能表达双分布不一致代价 | 加权调和支撑 $h(w)$ 提高单边支持 token 的跃迁代价 |

因此 ECP 不是把 A 和 S 的输出相加，也不是把视觉增强 logits 直接替换原 logits。其核心差异在于概率跃迁的代价几何由两分支共同决定。

## 7. 数学性质与边界

- 当 $a=0$ 或 $a=1$ 时，目标分别退化为只以 $p$ 或只以 $g$ 为参考的单分布卡方投影。
- 当 $p=g$ 时，双参考项退化为同一分布上的卡方收缩；这不等同于 Vanilla。
- 同时交换 $p,g$ 并将 $a$ 替换为 $1-a$，目标保持不变。
- 对有限 logits，数值实现使用 float32 log-softmax 与 log-domain 调和支撑计算；极端分支分歧时以稳定归一化处理浮点误差。
- caption 是同一模型生成的记忆提议，不是独立真值；CLIP affinity 也未被校准成 grounding probability。
- 联合分布稀疏不保证 caption 边缘稀疏。当前配置的完整 POPE/CHAIR 运行实际保留了所有 caption token；不能宣称已验证精准 token 过滤。
- 现有消融支持保留 caption 记忆路径在当前 CHAIR 幻觉/覆盖率权衡中的作用，但视觉增益的独立贡献、该公式跨模型的效果仍未充分识别。

## 8. 代码对应

- 核心数学与视觉测度：`src/methods/evidence_consensus.py`
- 默认研究参数：`configs/method/evidence_consensus.yaml`
- LLaVA 自回归验证入口：`scripts/vrsc_probe.py --methods evidence_consensus`
- 结果与方法边界：`reports/evidence_consensus_20260923.md`

`EvidenceConsensusMethod` 提供项目的 method adapter 接口；具体模型是否支持 ECP，仍取决于其 backend 是否实现相应分支和特征构造，不能仅凭 adapter 存在推断所有模型均已接入。
