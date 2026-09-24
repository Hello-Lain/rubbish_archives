# VRSC residual 实验更正

日期：2026-09-23

## 已确认的错误

旧 `scripts/vrsc_probe.py` 只把 `vrsc` 映射到 `semantic`，没有包含
`vrsc_residual`。因此该方法实际使用 `base/base/shuffled` 三个输入，
`delta_s=0`，残差恒为零，退化成 Shrink。测试原先只覆盖公式，没有覆盖
运行入口的分支映射。

以下目录保留原始内容，但不能用于评价 residual 机制：

- `outputs/vrsc_20260923/pope64_residual_v1/`
- `outputs/vrsc_20260923/chair32_residual_v1/`
- `outputs/vrsc_20260923/pope64_residual_aggressive_v1/`
- `outputs/vrsc_20260923/chair32_residual_aggressive_v1/`

其中激进候选只是更强的分布收缩，不是更强的视觉响应。撤回以前据此得出
的“residual 没有传递到自回归”和“residual 导致跨任务 trade-off”结论。
direct VRSC 的完整实验不受此路由错误影响，未达到 A+S 的结论仍成立。

另一个公式偏差是对 semantic response 先裁剪再投影；现已改为先计算原始
正交残差再裁剪，与原设计（现存于 `reports/vrsc_design_history_20260923.md`）
的公式一致。旧 replay 数字仅代表旧实现。

## 修正验收

显式分支表、实际方法配置、源码哈希和响应诊断写入新 artifact；
新增测试要求 residual 走 `base/semantic/shuffled` 并能改变非退化输入的
输出分布。新实验使用新目录，不覆盖旧结果，也不重跑 A+S。

修正后的 residual 完整结果为 POPE Accuracy=82.700%、F1=81.134%、
Recall=74.400%，CHAIRs=0.494、CHAIRi=0.136131、Recall=0.839315。
仍未接近 A+S 的 CHAIRs=0.432，路由错误并非唯一原因。

后续改为 ECP 的证据共识目标。其完整结果与结论在
`reports/evidence_consensus_20260923.md`，不以新候选的成功追认旧无效结果。
