# VRSC Feasibility Report

Date: 2026-09-23

**Correction:** the four residual generation runs reported below used an
incorrect `base/base/shuffled` routing and are invalid as residual evidence.
Their mechanism no-go and task-tradeoff conclusions are withdrawn. See
`reports/vrsc_residual_routing_correction_20260923.md`. Direct VRSC full-count
results remain valid. The replacement objective is not complete.

## Decision

VRSC is a runnable research prototype, but the current evidence does not
support the intended mechanism claim that a semantic visual probe provides a
better visual-truth signal than a shuffled probe. The prototype should not
replace the default SHIELD composition or proceed directly to full-scale
hyperparameter tuning. The next iteration should redesign visual evidence
construction instead of only tuning `eta`, `evidence_weight`, or `trust`.

## Method

VRSC uses the S-side patch-caption score only to choose a probe direction. It
does not inject caption tokens or apply the S enhancement directly.

For patch features `x_i` and scores `s_i`, define energy weights
`w_i = ||x_i||^2 / sum_j ||x_j||^2` and the weighted-centered direction:

```text
d_i = s_i - sum_j w_j*s_j
```

The probe rescales each patch and restores the total feature norm:

```text
x'_i = x_i * (1 + eta*d_i/max_j|d_j|)
x*   = x' * ||x||_F / ||x'||_F
```

With the same prefix, let `p` be the clean next-token distribution and
`p_probe` the probed distribution:

```text
r_i = clip(log p_probe(i) - log p(i), -c, c)
u_i = log p(i) - max_j log p(j) + beta*r_i
```

The final distribution is the chi-square trust-region simplex solution:

```text
q_i = p_i * [1 + (u_i-nu)/tau]_+
sum_i q_i = 1
```

The shared dual threshold is solved by an active-set scan. This is one
distribution-level optimization, rather than an independent A cutoff followed
by an S feature amplification.

Fixed prototype parameters:

```text
eta=0.25
evidence_weight=2.0
trust=1.0
response_clip=1.0
```

## Implementation And Checks

Changed research files:

- `idea.md`
- `src/methods/vrsc.py`
- `configs/method/vrsc.yaml`
- `scripts/vrsc_probe.py`
- `scripts/analyze_vrsc_probe.py`
- `tests/test_vrsc.py`
- `tests/test_vrsc_probe.py`
- `README.md`

The implementation is an adapter/plugin and does not modify the backbone,
default SHIELD YAML, fixed datasets, or existing result artifacts.

The capture and generation runs used the local canonical protocol:

```text
model: LLaVA-1.5-7B local snapshot
seed: 42
dtype: bfloat16
attention: flash_attention_2
fp8: text_mlp
tf32: true
cache: dynamic
decoding: sample
temperature: 1
top_p: 1
```

All completed runs reported:

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

The runtime log contains a flash-attn compatibility warning because the local
installed version is `2.8.3` while the helper's advisory range ends at
`2.8.1`; the effective backend remained `flash_attention_2` and no fallback was
recorded.

## Paired First-Token Replay

Source artifact: `outputs/vrsc_20260923/pope256_capture/`.
Recomputed analysis: `outputs/vrsc_20260923/pope256_analysis_v2/`.
This is a first-token replay diagnostic, not generated POPE accuracy.

| Method | Expected correct probability | Exact Yes/No argmax |
|---|---:|---:|
| Vanilla | 78.645% | 82.031% |
| Adaptive | 81.819% | 82.031% |
| Shrink, no visual response | 81.750% | 82.031% |
| VRSC | 81.708% | 82.422% |
| Shuffled probe | 81.564% | 82.031% |
| Entropy-matched control | 81.744% | 82.031% |
| Greedy | 82.031% | 82.031% |
| A+S runtime branch | 79.459% | 80.078% |

Paired bootstrap deltas for VRSC expected correct probability:

| Comparison | Delta | 95% image-cluster interval |
|---|---:|---:|
| Vanilla | +3.063 pp | [+1.966, +4.084] |
| Adaptive | -0.111 pp | [-0.895, +0.642] |
| Shrink | -0.042 pp | [-0.427, +0.392] |
| Shuffled | +0.144 pp | [-0.257, +0.590] |
| Entropy-matched | -0.036 pp | [-0.143, +0.056] |
| A+S | +2.249 pp | [+0.357, +4.306] |
| Entropy-matched, reachable rows only | -0.061 pp | [-0.158, +0.003] |

The direct mechanism diagnostic is unfavorable:

```text
semantic probe correct-answer log-odds change: -0.0278
shuffled probe correct-answer log-odds change: -0.0146
semantic minus shuffled: -0.0132
95% image-cluster interval: [-0.0364, +0.0102]
```

The semantic probe did not show a reliable visual-truth response advantage over
random score permutation. VRSC was also indistinguishable from the no-response
shrink control and the entropy control on this subset.

The entropy control now records unattainable targets explicitly. One row had a
target entropy below the clean distribution's minimum because its largest
logits were tied. That row uses the nearest temperature boundary. The other
255 rows match to numerical precision; the maximum reachable-row error is
`1.27e-7`.

## Small Autoregressive POPE

Source artifact: `outputs/vrsc_20260923/pope64_generate/`.
This is a 64-question smoke, not the fixed 3000-question benchmark.

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla | 78.125% | 80.000% | 75.000% | 77.419% |
| Adaptive | 81.250% | 81.250% | 81.250% | 81.250% |
| Shrink | 84.375% | 86.667% | 81.250% | 83.871% |
| VRSC | 84.375% | 86.667% | 81.250% | 83.871% |
| Shuffled | 84.375% | 86.667% | 81.250% | 83.871% |
| A+S | 75.000% | 75.000% | 75.000% | 75.000% |

VRSC exactly matched both the no-response shrink control and the shuffled
probe on this smoke. This is consistent with the paired replay diagnosis that
the observed gain is distribution concentration/reordering rather than
validated semantic visual evidence.

## Small Autoregressive CHAIR

Source artifact: `outputs/vrsc_20260923/chair32_generate/`.
This is a 32-image smoke, not the fixed 500-image CHAIR evaluation.

| Method | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.5625 | 0.156118 | 0.865625 |
| Adaptive | 0.46875 | 0.103175 | 0.895313 |
| Shrink | 0.59375 | 0.169231 | 0.847396 |
| VRSC | 0.5625 | 0.153543 | 0.825521 |
| Shuffled | 0.59375 | 0.147287 | 0.871875 |
| A+S | 0.5000 | 0.191964 | 0.808854 |

VRSC did not improve CHAIRs over Vanilla, had lower Recall than every control
except A+S, and did not show a consistent advantage over Shuffled.

## Full-Count Reuse Validation

To answer whether VRSC is close to the original A+S combination, the existing
verified A+S artifacts were reused instead of rerunning them. A new runner
process was used only to obtain full-count Vanilla, A, Shrink, Shuffled, and
VRSC metrics; it was stopped after VRSC because the remaining A+S phase was
redundant. The per-method VRSC files contain all 3000 POPE or 500 CHAIR rows and
pass the acceleration checks; their parent manifest is intentionally marked
`RUNNING` because the comparison process was stopped before the redundant
phase.

### POPE-3000

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vanilla, reused verified | 79.067% | 84.439% | 71.267% | 77.296% |
| A, reused verified | 82.367% | 88.138% | 74.800% | 80.923% |
| S, reused verified | 79.667% | 83.610% | 73.800% | 78.399% |
| A+S, reused verified | 82.700% | 87.529% | 76.267% | 81.511% |
| VRSC, full-count run | 82.700% | 89.334% | 74.267% | 81.107% |
| Shrink, same run | 83.033% | 89.735% | 74.600% | 81.471% |
| Shuffled, same run | 83.100% | 89.688% | 74.800% | 81.570% |

Against A+S, direct VRSC is equal on Accuracy, `+1.806 pp` on Precision,
`-2.000 pp` on Recall, and `-0.404 pp` on F1. It is therefore close on
classification accuracy but not better overall. More importantly, Shuffled is
`+0.400 pp` above VRSC on Accuracy and `+0.463 pp` above it on F1 in the same
full-count run. This rejects the claim that the semantic probe is the source
of the observed gain.

### CHAIR-500

| Method | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla, reused verified | 0.518000 | 0.163492 | 0.808083 |
| A, reused verified | 0.496000 | 0.135772 | 0.840866 |
| S, reused verified | 0.454000 | 0.142385 | 0.795050 |
| A+S, reused verified | 0.432000 | 0.129335 | 0.828539 |
| VRSC, full-count run | 0.504000 | 0.144168 | 0.830550 |
| Shrink, same run | 0.500000 | 0.142589 | 0.833862 |
| Shuffled, same run | 0.510000 | 0.140199 | 0.847146 |

Against A+S, direct VRSC is worse by `+7.200 pp` CHAIRs and `+1.483 pp`
CHAIRi, while Recall is only `+0.201 pp` higher. Shuffled has lower CHAIRi and
higher Recall than VRSC. Direct VRSC is not close enough to A+S on hallucination
quality to justify deployment.

The reused A+S and newly obtained VRSC runs have matching model revision,
dataset hashes, prompt mode, decoding, batch settings, and acceleration fields.
They are protocol-matched independent runs rather than paired identical random
draw streams; the same-run Shrink/Shuffled controls are the stronger mechanism
comparison.

## Targeted Mechanism Update

The full-count result identifies a specific failure: semantic probe response is
not distinguishable from generic response to a norm-preserving perturbation.
The optional residual mode removes the component aligned with the shuffled
response:

```text
delta_s = log p_semantic - log p
delta_n = log p_shuffled  - log p
a       = <delta_s, delta_n> / (||delta_n||^2 + epsilon)
r_res   = clip(delta_s - a*delta_n, -c, c)
```

The same chi-square projection then consumes `r_res`. This is implemented in
`VRSCConfig(response_mode="orthogonal_residual")` and
`configs/method/vrsc_residual.yaml`; the direct default remains unchanged.

Using only the already captured POPE-256 logits:

| Method | Expected correct first-token probability |
|---|---:|
| Direct VRSC | 81.708% |
| Orthogonal residual VRSC | 81.938% |
| Shuffled | 81.564% |

The residual point estimate is `+0.230 pp` over direct VRSC, but its paired
image-cluster interval is `[-0.198, +0.696] pp`, which crosses zero. This is
not an end-to-end claim and does not overwrite the direct VRSC result.

## Autoregressive Residual Validation

The research runner now supports a third, independent reference KV cache, so
residual decoding uses clean, semantic-probe, and shuffled-probe branches under
the same generated prefix. Existing direct, Shrink, Shuffled, and A+S results
were reused; only residual candidates were newly run. Both new runs passed the
canonical acceleration checks:

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

### Default Residual Smoke

Artifacts:

- `outputs/vrsc_20260923/pope64_residual_v1/`
- `outputs/vrsc_20260923/chair32_residual_v1/`

| Method | POPE Accuracy | POPE Precision | POPE Recall | POPE F1 |
|---|---:|---:|---:|---:|
| Vanilla, reused | 78.125% | 80.000% | 75.000% | 77.419% |
| Adaptive, reused | 81.250% | 81.250% | 81.250% | 81.250% |
| Shrink, reused | 84.375% | 86.667% | 81.250% | 83.871% |
| VRSC direct, reused | 84.375% | 86.667% | 81.250% | 83.871% |
| Shuffled, reused | 84.375% | 86.667% | 81.250% | 83.871% |
| VRSC residual, new | 84.375% | 86.667% | 81.250% | 83.871% |

| Method | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla, reused | 0.562500 | 0.156118 | 0.865625 |
| Adaptive, reused | 0.468750 | 0.103175 | 0.895313 |
| Shrink, reused | 0.593750 | 0.169231 | 0.847396 |
| VRSC direct, reused | 0.562500 | 0.153543 | 0.825521 |
| Shuffled, reused | 0.593750 | 0.147287 | 0.871875 |
| VRSC residual, new | 0.593750 | 0.169231 | 0.847396 |

The default residual exactly matched the distribution-only Shrink control on
both smokes. The first-token replay improvement therefore did not survive
autoregressive sampling.

### High-Penetration Candidate

An offline replay suggested `trust=0.25` and `evidence_weight=3.0`. The
candidate was tested without changing the default YAML:

- `outputs/vrsc_20260923/pope64_residual_aggressive_v1/`
- `outputs/vrsc_20260923/chair32_residual_aggressive_v1/`

It reached POPE Accuracy/F1 of `81.250%/80.645%`, below the reused
Shrink/residual control at `84.375%/83.871%`. It improved CHAIR-32 to
`CHAIRs=0.531250`, `CHAIRi=0.153846`, `Recall=0.907813`, but this is a
cross-task tradeoff on a 32-image smoke and not a deployment improvement.
The default residual configuration is therefore unchanged.

## Limitations

- POPE and CHAIR generation results are smoke subsets and have low statistical
  power.
- The semantic score uses reused caption features; caption construction cost
  is excluded from generation timing.
- The first-token replay evaluates a local distribution and cannot establish
  long-horizon autoregressive behavior.
- The energy probe is constrained to the current CLIP/vision feature interface;
  its score alignment with the language model visual pathway remains unverified.
- No claim is made that VRSC exceeds the official full SHIELD or ONLY results.
- Residual autoregressive validation is limited to POPE-64 and CHAIR-32 smokes;
  it is not evidence for a full benchmark replacement.

## Go / No-Go

```text
Engineering prototype: GO
Direct VRSC deployment: NO-GO
Default residual replacement: NO-GO
Aggressive residual deployment: NO-GO
Full residual benchmark escalation: NO-GO
```

The direct module is a reproducible falsifiable baseline, not a successful
replacement for A+S. The residual mode was the correct targeted test of the
measured shuffled-response nuisance, but its smoke results show that the
current probe is not a reliable causal visual-evidence signal.
