# CAP-C CHAIR-64 Gain Sweep

## Scope

This report executes Phase 3 of
`CAP_corrected_method_and_plan.md`. The experiment uses the already verified
CAP-C attribution:

```text
a_i = cosine(p_i, h_visual) - cosine(p_i, h_blank)
```

where the blank forward uses the same prompt with image-token embeddings
zeroed. No source code, model cache, data cache, CLIP model, or caption
injection was added for this sweep.

## Protocol

All eight gains use the same LLaVA-1.5-7B snapshot, 64 SHIELD-compatible CHAIR
questions, seed `42`, sampling, temperature `1`, top-p `1`, max-new-tokens
`512`, batch size `32`, image batch size `16`, four workers, prefetch factor
`4`, BF16, FlashAttention-2, dynamic cache, and Transformer Engine FP8
`text_mlp`.

Each output manifest records:

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
cap_attribution=contrastive_hidden
```

Every gain has 64 vanilla rows, 64 CAP-C rows, 64 diagnostics, and 576 finite
attribution scores per diagnostic row.

The repository does not contain the plan-referenced standalone
`scripts/score_chair.py`. The runtime's canonical `ChairScorer` generated
`vanilla.metrics.json` and `cap.metrics.json` during each run; these files are
the scored inputs used below.

## Results

The vanilla result is identical across the sweep:
`CHAIRs=0.609375`, `CHAIRi=0.1776315789`, `Recall=0.8141927083`.

The plotted sweep is available at
`reports/capc_chair64_sweep_20260924.svg`.

| Gain | CHAIRs | CHAIRi | Recall | CHAIRs delta vs vanilla |
|---:|---:|---:|---:|---:|
| 0.05 | 0.546875 | 0.148315 | 0.806510 | -0.062500 |
| 0.10 | 0.625000 | 0.167715 | 0.807422 | +0.015625 |
| 0.15 | **0.468750** | 0.148472 | 0.823828 | **-0.140625** |
| 0.20 | 0.562500 | 0.152941 | 0.821615 | -0.046875 |
| 0.30 | 0.609375 | 0.171492 | 0.792448 | 0.000000 |
| 0.40 | 0.609375 | 0.186047 | 0.824349 | 0.000000 |
| 0.50 | 0.625000 | 0.171548 | 0.813542 | +0.015625 |
| 0.60 | 0.531250 | 0.146572 | 0.816276 | -0.078125 |

The deterministic selection rule gives:

```text
g* = 0.15
```

There is no tie on CHAIRs. The CAP-C diagnostics for `g*` retain the same
contrastive-hidden protocol and energy-preserving probe as the POPE-32
feasibility smoke.

## Phase 3 Gates

The corrected plan requires all three gates to pass.

### Gate 1: CHAIRs improvement

Required:

```text
exists gain: CHAIRs(gain) < 0.518 - 0.01 = 0.508
```

Observed:

```text
CHAIRs(0.15) = 0.468750
```

Result: **PASS**.

### Gate 2: Non-random curve

Required:

```text
absolute adjacent-gain CHAIRs fluctuation < 0.05
```

Observed adjacent absolute differences in gain order
`0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60`:

```text
[0.078125, 0.156250, 0.093750, 0.046875,
 0.000000, 0.015625, 0.093750]
```

The maximum adjacent fluctuation is `0.156250`, which exceeds `0.05`.

The local-neighbor subcondition does pass:

```text
g*=0.15 neighbors: CHAIRs(0.10)=0.625000,
                   CHAIRs(0.20)=0.562500
vanilla + 0.02 = 0.629375
```

Both neighbors are at or below `0.629375`, but this does not repair the
failed adjacent-fluctuation condition.

Result: **FAIL**.

### Gate 3: Recall floor

Required:

```text
Recall(g*) >= 0.808 - 0.02 = 0.788
```

Observed:

```text
Recall(0.15) = 0.823828125
```

Result: **PASS**.

## Decision

Phase 3 **FAILS** because the gain-response curve is not sufficiently smooth
under the corrected plan's explicit non-randomness threshold. The low
`CHAIRs=0.468750` at `g=0.15` is an encouraging isolated point, but the
adjacent gain instability prevents treating it as a validated causal
intervention.

Consequently:

- `g*=0.15` is recorded as the diagnostic minimum, not as a confirmed
  deployment gain;
- CHAIR-436 holdout is not run;
- CHAIR-500 and POPE-3000 are not run;
- no full-count effectiveness claim is made;
- the prescribed negative-result report is emitted separately.

## Historical CAP-v1 Context

The older `outputs/cap_chair64_gain_*_20260924/` directories are not included
as CAP-C evidence. Their manifests have no `cap_attribution=contrastive_hidden`
field and belong to the earlier cosine implementation. They may be used only
as historical context, not as a same-protocol control for the corrected claim.

## Reproduction

Representative command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 64 --start 0 \
  --methods vanilla,cap \
  --output outputs/capc_chair64_gain_0.15_20260924 \
  --cap-attribution contrastive_hidden --cap-probe-gain 0.15
```

The complete raw outputs are in:

```text
outputs/capc_chair64_gain_0.05_20260924/
outputs/capc_chair64_gain_0.10_20260924/
outputs/capc_chair64_gain_0.15_20260924/
outputs/capc_chair64_gain_0.20_20260924/
outputs/capc_chair64_gain_0.30_20260924/
outputs/capc_chair64_gain_0.40_20260924/
outputs/capc_chair64_gain_0.50_20260924/
outputs/capc_chair64_gain_0.60_20260924/
```
