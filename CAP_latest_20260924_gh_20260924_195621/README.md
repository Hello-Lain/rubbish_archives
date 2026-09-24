# CAP Latest Method And Experiments

Snapshot date: 2026-09-24

This archive contains the latest CAP implementation, per-question attribution
repair, focused/full validation records, CAP-only experiments, and compatible
Vanilla/GTP Patch comparison artifacts.

## Contents

- `method/`: CAP implementation, runtime integration, configuration, and tests.
- `plans/`: CAP implementation and next-step plans.
- `reports/cap/`: task verification, gain/holdout decision, and final report.
- `experiments/main/`: CAP CHAIR-500 and POPE-3000 full-count results.
- `experiments/holdout/`: CAP CHAIR holdout using `start=64`, `limit=436`, `gain=0.05`.
- `experiments/gain_sweep/`: nine CAP-only CHAIR-64 gain runs.
- `experiments/baselines/`: existing protocol-compatible Vanilla and GTP Patch
  metrics and predictions. Vanilla was reused, not rerun, for the CAP full-count
  comparison.
- `diagnostics/`: compressed CHAIR diagnostics and compact summaries.
- `metadata/`: repository HEAD and worktree snapshot metadata.

## Reproducibility

The runs use the fixed local model/data resources and canonical runtime gates:

```text
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

The full CAP outputs are complete:

```text
CHAIR-500: status=PASS, count=500
POPE-3000: status=PASS, count=3000
```

## Result Summary

CAP with the full-run gain `0.20`:

```text
CHAIRs=0.542000
CHAIRi=0.171534
Recall=0.796584

POPE Accuracy=79.4000%
POPE Precision=84.1860%
POPE Recall=72.4000%
POPE F1=77.8495%
```

The CHAIR-64 exploration selected `gain=0.05`, but the 436-image holdout did
not improve over Vanilla or GTP Patch. The final report therefore marks the
current CAP formulation as `NO-GO` for a publishable CHAIR improvement while
recording the implementation and protocol as feasible.

## Size Limitation

The ZIP skill skips individual files larger than 3,000,000 uncompressed bytes.
The full POPE diagnostic JSON is larger than that limit even when compressed, so
this snapshot includes its compact diagnostic summary rather than the raw
17 MB compressed file. Full raw outputs remain in the local experiment
directories referenced by the reports.
