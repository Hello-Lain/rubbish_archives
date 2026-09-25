# GTP 优化候选确认实验（2026-09-23）

## 结论

冻结候选 `GTP-CaptionTransport+Memory` 在固定 SHIELD-compatible CHAIR-500 上通过运行与加速协议验收，指标为 `CHAIRs=0.480`、`CHAIRi=0.131422`、`Recall=0.842098`。

该候选较 Vanilla 的 CHAIRi 和 Recall 有明确改善，CHAIRs 点估计也更低；较原 GTP 的三项点估计均改善，但配对区间都跨 0，尚不能确认提升超出样本波动。**候选未达到 ECP 水平**：相对 ECP，CHAIRs 高 `5.8 pp`，其配对 95% bootstrap 区间为 `[+1.4,+10.4] pp`；CHAIRi 和 Recall 与 ECP 的差异区间跨 0。因此当前结果支持“可运行且有一定效果”，不支持“追平或超过 ECP”。

## 冻结候选

候选只在与固定 CHAIR-500 不重叠的 128 张开发集上选择，随后锁定参数，在固定集上仅做一次确认评估。

| 参数 | 值 |
|---|---|
| 方法 | GTP-CaptionTransport+Memory |
| `a_u` / `a_h` | `0.75` / `0.75` |
| `tau` / `tau_e` / `kappa` | `1.0` / `1.0` / `1.0` |
| grounding | 逐 caption token，`caption_tokens` |
| caption memory | 开启，仅注入 grounded branch，32 slots |
| claim draft | clean caption prefix，最多 32 tokens |

memory 通过 caption-token grounding 形成视觉支持的 token 表示，并只注入 grounded branch；clean branch 不变，两个分支使用各自独立长度的 dynamic KV cache。

开发集候选指标为 CHAIRs `0.468750`、CHAIRi `0.136842`、Recall `0.794978`。相对同一 128 张图的 Vanilla，配对变化依次为 `-8.59 pp`、`-3.09 pp`、`+0.65 pp`。以 20,000 次配对 percentile bootstrap 复算，CHAIRs 95% CI 为 `[-17.19,0.00] pp`，CHAIRi 为 `[-6.07,-0.13] pp`，Recall 为 `[-2.70,+4.04] pp`。开发集用于候选冻结，固定 CHAIR-500 结果用于确认。

## 固定 CHAIR-500

### 协议

- 模型：LLaVA-1.5-7B，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，`config.json` SHA-256 `0bde54495c54bcc346064a0d314b010d1c4d3ca7f7e583b6f711949a35352c0f`
- 数据：SHIELD-compatible CHAIR questions，500 张 COCO val2014 图；questions SHA-256 `d94e5b7f28bfc7ce74f8bf29c013436da52e5ab9697a7332aaaadbf7628d7a1e`
- prompt：`Please describe this image in detail.`
- 解码：sampling，seed `42`，temperature `1`，top-p `1`，max-new-tokens `512`
- 批处理：batch `32`，image batch `16`，workers `4`，prefetch `4`
- cache：dynamic
- 数值与加速：BF16、FlashAttention-2、TF32、Transformer Engine FP8 `text_mlp`
- 设备：物理 GPU 1（H100 PCIe）；进程内 `cuda:0` 是由 `CUDA_VISIBLE_DEVICES=1` 映射所得

所有对照均为相同 500 个唯一 image ID，且 prompt、基础解码参数及 LLaVA revision/config 一致。CHAIRs 越低越好，CHAIRi 越低越好，Recall 越高越好。

### 指标

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|---|---:|---:|---:|
| Vanilla | 0.518000 | 0.163492 | 0.808083 |
| 原始 GTP | 0.508000 | 0.144132 | 0.829442 |
| ONLY | 0.506000 | 0.145494 | 0.833035 |
| GTP-CaptionTransport+Memory（冻结候选） | **0.480000** | **0.131422** | **0.842098** |
| ECP | 0.422000 | 0.124098 | 0.829856 |
| SHIELD 推荐三组件 | 0.358000 | 0.100977 | 0.809490 |

候选总体计数为 `486/3698` 个幻觉对象提及，句子级幻觉为 `240/500`。相对原 GTP 的点变化为 CHAIRs `-2.80 pp`、CHAIRi `-1.27 pp`、Recall `+1.27 pp`；相对 Vanilla 为 `-3.80 pp`、`-3.21 pp`、`+3.40 pp`。

### 配对不确定区间

对同一 500 个 image ID 进行 20,000 次配对 percentile bootstrap，随机种子为 `20260923`。CHAIRs 和 Recall 按图像重采样后求宏平均；CHAIRi 在每次重采样中重新计算幻觉对象提及总数与对象提及总数之比。下表为“候选减对照”，单位为百分点。

| 对照 | CHAIRs Δ（95% CI） | CHAIRi Δ（95% CI） | Recall Δ（95% CI） |
|---|---:|---:|---:|
| 原始 GTP | -2.80 `[-7.40,+2.00]` | -1.27 `[-2.65,+0.13]` | +1.27 `[-0.34,+2.86]` |
| Vanilla | -3.80 `[-8.80,+1.40]` | -3.21 `[-4.90,-1.49]` | +3.40 `[+1.54,+5.27]` |
| ECP | +5.80 `[+1.40,+10.40]` | +0.73 `[-0.78,+2.23]` | +1.22 `[-0.17,+2.63]` |

因此，候选相对 Vanilla 在 CHAIRi 与 Recall 上的区间不跨 0；CHAIRs 区间跨 0。相对原始 GTP，三项区间均跨 0。相对 ECP，CHAIRs 明确更差；CHAIRi 与 Recall 的区间跨 0，不能据此宣称这两项已追平或显著落后。

## 运行验收

正式候选结果为 `status=PASS`、`count=500`，且满足项目加速契约：

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

grounding 无效样本数为 `0`。运行耗时 `368.10 s`，生成 `58,007` tokens。加载时 Transformer Engine 对本机 flash-attn `2.8.3` 高于其声明上限 `2.8.1` 输出兼容性 warning；本次 runtime metrics 仍报告 FlashAttention-2/FP8 有效且 fallback 为零。后续环境升级或复现实验仍应重新核验实际 runtime 字段，不能仅凭该 warning 或安装版本推断加速状态。

## 产物

- 固定 CHAIR-500 metrics：`outputs/gtp_evidence_chair500_frozen075_memory_20260923.metrics.json`
- 固定 CHAIR-500 predictions：`outputs/gtp_evidence_chair500_frozen075_memory_20260923.jsonl`
- 开发集 metrics：`outputs/gtp_evidence_dev128_ecp075_memory_20260923.metrics.json`
- 开发集 predictions：`outputs/gtp_evidence_dev128_ecp075_memory_20260923.jsonl`
- 原始 GTP 对照：`outputs/gtp_llava_chair_20260923.metrics.json`
- ECP 对照：`/data/lcq/Downloads/ONLY/outputs/vrsc_20260923/chair500_ecp_frozen_v1/evidence_consensus.metrics.json`
- SHIELD 对照：`/data/lcq/Downloads/ONLY/outputs/chair_llava_7b_shield_cumulative_statistical_b32_20260922.metrics.json`

## 判断与后续

这次确认说明 caption-token transport 加 grounded-only memory 能在固定协议下运行，并相对 Vanilla 改善 CHAIRi 与 Recall；但相对原 GTP 的收益尚不确定，且 CHAIRs 未达到 ECP，不能作为“达到 ECP 实验水准”的证据。

后续若继续优化，应另建与上述开发集和固定 CHAIR-500 均不重叠的开发 split，优先针对句子级幻觉率做机制迭代与消融；保持本次 CHAIR-500 作为已使用过的确认集，不再据此选择新参数。每个新候选仍须在 POPE adversarial 上检查误报/召回权衡，并在新的固定确认集上复验。
