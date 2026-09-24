# CAP Next-Stage Final Report

- Mode: primary-agent
- Status: COMPLETE
- Plan path: `.codex/work/20260924-cap-next/plan.md`
- User plans: uploaded `CAP_implementation_plan.md` and `CAP_next_steps.md`

## Tasks Completed

- Task 1: repaired per-question attribution keying, row-aware prompt feature selection, diagnostics, and tests.
- Task 2: verified the repaired protocol on 32 POPE rows with six repeated images.
- Task 3: ran CAP-only CHAIR-500 and POPE-3000, reusing existing canonical Vanilla artifacts rather than rerunning Vanilla.
- Task 4: completed the nine-point CHAIR-64 gain sweep and CAP-only 436-row holdout decision experiment.

## Files Changed

- `scripts/vrsc_probe.py`
- `scripts/shield_vulnerability_runtime.py`
- `tests/test_vrsc_probe.py`
- Existing CAP implementation files remain: `src/methods/cap.py`, `src/methods/__init__.py`, `configs/method/cap.yaml`, `tests/test_cap.py`

## Verification

- `.venv/bin/python -m py_compile scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py` -> passed.
- `.venv/bin/ruff check scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py tests/test_vrsc_probe.py` -> passed.
- `.venv/bin/python -m pytest tests/test_cap.py tests/test_vrsc_probe.py -q` -> `36 passed`.
- `git diff --check` -> passed.
- `uv pip check --python .venv/bin/python` -> all installed packages compatible.
- Task 2: POPE-32 CAP diagnostics `32`, unique keys `32`, attribution length `576`, repeated-image attribution arrays distinct.
- Task 3: CAP CHAIR-500 and POPE-3000 have `status=PASS`, full counts, complete diagnostics, no caption injection, FA2, effective FP8, zero fallback calls, and empty fallback events.
- Task 4: all nine CHAIR-64 gain runs have complete counts and pass acceleration checks; CAP-only holdout has 436 predictions and 436 unique diagnostics.
- GPU cleanup: all launched experiments exited and both H100 GPUs returned to `14 MiB` used with no compute applications.

## Acceptance

- Per-question attribution is implemented and verified.
- CAP full-count protocol and acceleration gates pass.
- The CHAIR result is explicitly reported as negative rather than hidden.
- The requested next-stage plan was executed through gain sweep and holdout.

## Decision

- CHAIR-500 CAP gain 0.20: `CHAIRs=0.542000`, `CHAIRi=0.171534`, `Recall=0.796584`.
- CHAIR-500 Vanilla: `CHAIRs=0.518000`, `CHAIRi=0.163492`, `Recall=0.808083`.
- CHAIR holdout CAP gain 0.05: `CHAIRs=0.538991`, `CHAIRi=0.171333`, `Recall=0.805957`.
- CHAIR holdout GTP Patch: `CHAIRs=0.502294`, `CHAIRi=0.134150`, `Recall=0.842827`.
- Current CAP formulation is `NO-GO` for a publishable CHAIR improvement. POPE has a small F1 gain over Vanilla (`+0.554 pp`) but remains well below GTP Patch.

## Rework Cycles

- One static call-order and manifest-text fix cycle in Task 1.
- One bounded experimental decision cycle in Task 4; no further code rework was started after the holdout showed a negative method signal.

## Artifacts

- Task 1: `.codex/work/20260924-cap-next/artifacts/task1-verification.md`
- Task 2: `.codex/work/20260924-cap-next/artifacts/task2-verification.md`
- Task 3: `.codex/work/20260924-cap-next/artifacts/task3-verification.md`
- Task 4 decision: `.codex/work/20260924-cap-next/artifacts/task4-decision.md`
- Full CAP outputs: `outputs/cap_chair500_caponly_20260924/`, `outputs/cap_pope3000_caponly_20260924/`
- Gain sweep outputs: `outputs/cap_chair64_gain_*_20260924/`
- Holdout output: `outputs/cap_chair436_holdout_gain0.05_20260924/`

## Blockers / Risks

- Execution is complete; the separate method-effectiveness decision is `NO-GO`.
- The failed holdout is an experimental result, not a tooling blocker or an unresolved implementation failure.
- Partial interrupted directories `outputs/cap_chair500_20260924/` and `outputs/cap_pope3000_20260924/` are retained and excluded from formal conclusions.

## Next Action

Implement one bounded attribution reformulation (within-image centering and/or
projector-space probing), then rerun CHAIR-64 and only proceed to full CHAIR if
the holdout trend materially improves. Do not rerun Vanilla or GTP Patch; their
completed artifacts are protocol-compatible and should continue to be reused.

Raw captions, attribution vectors, and command logs are omitted from this report
and retained in the cited output directories.
