# Task 4 Gain Sweep And Holdout Decision

- Status: complete for the planned experimental scope; method effectiveness is negative.
- Constraint: no Vanilla, GTP Patch, or CHAIR-500 model rerun was launched. CAP-only runs were used where new generation was required; existing `PASS` predictions were reused and rescored for comparison.

## Gain Sweep

CHAIR-64, `start=0`, CAP-only, seed 42, BF16, FA2, FP8, zero fallback:

| gain | CHAIRs | CHAIRi | Recall |
|---:|---:|---:|---:|
| 0.05 | 0.500000 | 0.167816 | 0.801432 |
| 0.10 | 0.609375 | 0.183761 | 0.813542 |
| 0.15 | 0.531250 | 0.180401 | 0.808073 |
| 0.20 | 0.578125 | 0.182819 | 0.806641 |
| 0.25 | 0.609375 | 0.168182 | 0.826302 |
| 0.30 | 0.578125 | 0.159915 | 0.827083 |
| 0.35 | 0.593750 | 0.195853 | 0.810937 |
| 0.40 | 0.593750 | 0.173719 | 0.798828 |
| 0.50 | 0.609375 | 0.182432 | 0.816406 |

Selection rule: minimize CHAIRs, then CHAIRi, using CHAIR-64 only. Selected `gain=0.05`.

CHAIR-64 same-subset controls:

| method | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.578125 | 0.176606 | 0.796094 |
| GTP Patch | 0.593750 | 0.168932 | 0.861719 |
| CAP gain 0.05 | **0.500000** | **0.167816** | 0.801432 |

This 64-image exploration result did not transfer to the holdout.

## Holdout 64:500

CAP output: `outputs/cap_chair436_holdout_gain0.05_20260924/`

The output count is 436 and all canonical acceleration gates pass, but the runner marks any non-full subset as `SMOKE`; this does not make an effectiveness claim.

| method | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Vanilla | 0.509174 | 0.161636 | 0.809843 |
| GTP Patch | **0.502294** | **0.134150** | **0.842827** |
| CAP gain 0.05 | 0.538991 | 0.171333 | 0.805957 |

CAP minus Vanilla: `CHAIRs +2.982 pp`, `CHAIRi +0.970 pp`, `Recall -0.389 pp`.

CAP minus GTP Patch: `CHAIRs +3.670 pp`, `CHAIRi +3.718 pp`, `Recall -3.687 pp`.

## Full-Count Context

| dataset | method | primary metrics |
|---|---|---|
| CHAIR-500 | Vanilla | CHAIRs 0.518000, CHAIRi 0.163492, Recall 0.808083 |
| CHAIR-500 | CAP gain 0.20 | CHAIRs 0.542000, CHAIRi 0.171534, Recall 0.796584 |
| CHAIR-500 | GTP Patch | CHAIRs 0.514000, CHAIRi 0.138889, Recall 0.845245 |
| POPE-3000 | Vanilla | Accuracy 0.790667, Precision 0.844392, Recall 0.712667, F1 0.772957 |
| POPE-3000 | CAP gain 0.20 | Accuracy 0.794000, Precision 0.841860, Recall 0.724000, F1 0.778495 |
| POPE-3000 | GTP Patch | Accuracy 0.825000, Precision 0.896664, Recall 0.734667, F1 0.807622 |

CAP provides a small POPE F1 gain over Vanilla but does not approach GTP Patch. CHAIR is the method's primary target and is negative.

## Attribution Diagnosis

CHAIR-500 CAP attribution is not numerically degenerate:

- mean within-image score standard deviation: approximately `0.01545`;
- attribution range spans negative and positive values.

However, the within-image attribution mean is systematically negative, on average
approximately `-0.01271`. Under the current energy-preserving probe this means
most patches are suppressed while a smaller subset is relatively enhanced. The
intervention changes image feature norm by only about `0.064` relative L2, but
the sign and scaling of the causal direction do not produce hallucination
reduction. Per-image CHAIRs improves on 91 images, worsens on 103, and ties on 306.

## Decision

`NO-GO` for the current CAP formulation as a publishable improvement.

The implementation and per-question protocol are feasible and verified, but the
method does not beat Vanilla on the CHAIR holdout and is clearly behind GTP Patch.
The next bounded method iteration should change attribution construction, not
only probe gain:

1. center attribution within each image before energy redistribution;
2. test projector-space intervention or a signed/competing attribution rule;
3. re-run CHAIR-64 and the `start=64` holdout before another full CHAIR run;
4. keep POPE as a secondary check only after CHAIR improves.

## Raw Data Omitted

Per-image captions, full attribution vectors, and command logs remain in the cited
output and diagnostics directories.
