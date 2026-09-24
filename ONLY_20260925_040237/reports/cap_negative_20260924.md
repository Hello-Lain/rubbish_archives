# CAP-C Negative Result

## Status

**Phase 3 gate failed; later phases stopped.**

The corrected CAP plan required the CHAIR-64 gain sweep to satisfy all three
conditions before any holdout or full-count experiment:

1. at least one gain with `CHAIRs < 0.508`;
2. adjacent gain CHAIRs fluctuation below `0.05`, plus the local-neighbor
   condition;
3. selected-gain Recall at least `0.788`.

Observed outcomes:

```text
Gate 1: PASS
Gate 2: FAIL
Gate 3: PASS
```

The failure class is **method-effectiveness gate failure**, not a code or
runtime failure. All eight runs completed under the fixed protocol and passed
their machine-readable acceleration checks.

## Three-Signal Diagnosis

The fixed POPE-32 diagnostic established:

| Attribution | Std | Range | Interpretation |
|---|---:|---:|---|
| CAP v1 cosine | historical weak proxy | historical only | language-prior contaminated |
| ACP attention | 0.0004818293 | 0.0094588375 | too flat; primary gate failed |
| CAP-C contrastive hidden | 0.0198397019 | 0.1485607401 | non-degenerate signal |

CAP-C therefore fixes the signal-variance problem, but the CHAIR-64 sweep
demonstrates that variance alone does not establish a stable intervention
direction. The minimum at `g=0.15` is isolated by large adjacent changes:

```text
CHAIRs:
g=0.10 -> 0.625000
g=0.15 -> 0.468750
g=0.20 -> 0.562500
```

The maximum adjacent difference across all gains is `0.156250`, versus the
required maximum below `0.05`.

## Completed Evidence

- Eight CAP-C gain directories, each with 64 vanilla rows and 64 CAP-C rows.
- Each CAP-C diagnostic row contains 576 finite contrastive-hidden scores.
- All manifests use `contrastive_hidden`, seed `42`, sampling, CHAIR
  max-new-tokens `512`, BF16, effective FlashAttention-2, effective FP8
  `text_mlp`, zero native fallback calls, and no fallback events.
- Full repository tests: `150 passed`.
- Focused CAP/runtime tests: `54 passed`.
- `py_compile`, Ruff, `uv pip check`, and `git diff --check`: passed.

## Decision

CAP-C is **not validated as a stable CHAIR intervention** under the corrected
plan's Phase 3 gate. This result does not erase the POPE-32 signal-feasibility
evidence, but it prevents a full-count performance claim.

No CHAIR-436, CHAIR-500, or POPE-3000 CAP-C output was created by this plan
after the failed gate.

## Next Action

The corrected plan's own negative-result path applies: preserve this
reproducible diagnosis and return to a method with stronger grounding
structure, such as multi-token caption grounding / ECP / GTP, or design a
newly pre-registered CAP intervention before any further CAP-C gain tuning.
Do not select `g=0.15` for deployment solely because it is the best point in
this unstable 64-example sweep.

## Reproduction Artifacts

- Sweep report: `reports/capc_chair64_sweep_20260924.md`
- CAP-C outputs: `outputs/capc_chair64_gain_*_20260924/`
- POPE-32 CAP-C smoke:
  `outputs/acp_fallbackb_pope32_smoke_20260924/`
- ACP failure record:
  `.codex/work/20260924-cap-v2/artifacts/task3-decision.md`
