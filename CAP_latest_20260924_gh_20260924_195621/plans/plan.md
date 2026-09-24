# CAP 下一阶段执行计划

## Goal

修复 CAP 的 per-question attribution 协议，并在修复后的实现上完成 32 条 POPE smoke、CHAIR-500 与 POPE-3000 主实验；根据主实验结果继续执行 gain sweep、holdout 确认和 GTP Patch head-to-head，形成可复核的 CAP 可行性报告。

## Upstream Coverage

- 用户计划：`/data/lcq/.paseo/uploads/upload_1f86e88d-588f-46d7-b672-f2bd38db0733/CAP_next_steps.md`
- 项目约束：`AGENTS.md`
- 已完成实现与验证：`.codex/work/20260924-cap/`
- 当前 CAP runtime：`scripts/vrsc_probe.py`
- 当前 prompt helper：`scripts/shield_vulnerability_runtime.py`
- 固定资源与协议：`AGENTS.md` 中的模型、数据、解码和加速契约

## Task Breakdown

### Task 1: 修复 per-question attribution

- Description: 将 CAP attribution 从 per-image cache 改为 per-question cache，并允许同一 batch 中同图多问使用不同视觉 probe。
- Worker role: coding
- Wave: 1
- Concrete edits: Add a stable CAP row key; batch prompt-prefill rows; compute one hidden state and probe per question; add a row-aware image feature cache key to `build_vulnerability_prompt_batch`; project question-specific raw probes without path-key collisions; update manifest and diagnostics.
- Interfaces / contracts changed: Extend `build_vulnerability_prompt_batch` with an optional row-based image cache key; preserve all existing callers and default path-key behavior.
- Test cases: Same-image different-question keys differ; CAP CLI/routing remains stable; per-question diagnostics/cache cardinality is covered with a CPU fake-model helper test where practical.
- Pre-check commands: `.venv/bin/python -m pytest tests/test_cap.py tests/test_vrsc_probe.py -q`; `.venv/bin/py_compile scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py`
- Acceptance criteria: POPE CAP cache has one entry per selected row; same-image questions do not overwrite one another; diagnostics include `question_id`, question text, and 576 attribution scores; CHAIR remains one attribution per image.
- Verification: focused tests, Ruff, py_compile, and 32-row POPE smoke.
- Post-check commands: `.venv/bin/ruff check scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py tests/test_vrsc_probe.py`; `.venv/bin/python -m pytest tests/test_vrsc_probe.py -q`; `git diff --check`
- Dependencies: none
- Files likely touched: `scripts/vrsc_probe.py`; `scripts/shield_vulnerability_runtime.py`; `tests/test_vrsc_probe.py`
- Writable scope: `scripts/vrsc_probe.py`, `scripts/shield_vulnerability_runtime.py`, `tests/test_vrsc_probe.py`
- Output artifact: `.codex/work/20260924-cap-next/artifacts/task1-verification.md`
- Estimated scope: M

### Task 2: 32 条 POPE per-question smoke

- Description: Verify that the repaired CAP protocol runs with repeated images and produces one independent diagnostic per question.
- Worker role: validation
- Wave: 2
- Concrete edits: Create only the new smoke output directory and verification artifacts.
- Interfaces / contracts changed: None.
- Test cases: POPE rows `0:32`, `vanilla,cap`, canonical sampling, BF16, FlashAttention-2 and FP8.
- Pre-check commands: `nvidia-smi`; `nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv`; `uv pip check --python .venv/bin/python`
- Acceptance criteria: Both methods generate 32/32; CAP diagnostics count equals 32; repeated image questions have distinct cache keys and score arrays; acceleration gates pass.
- Verification: `CUDA_VISIBLE_DEVICES=<selected> .venv/bin/python scripts/vrsc_probe.py --mode generate --dataset pope --limit 32 --methods vanilla,cap --output outputs/cap_pope32_per_question_20260924 --cap-probe-gain 0.20`
- Post-check commands: inspect manifest, metrics, diagnostics, and output row counts; run a machine-readable acceptance script.
- Dependencies: Task 1
- Files likely touched: `outputs/cap_pope32_per_question_20260924/*`; `.codex/work/20260924-cap-next/artifacts/*`
- Writable scope: `outputs/cap_pope32_per_question_20260924/`, `.codex/work/20260924-cap-next/artifacts/`
- Output artifact: `.codex/work/20260924-cap-next/artifacts/task2-verification.md`
- Estimated scope: S

### Task 3: CHAIR-500 与 POPE-3000 主实验

- Description: Run the repaired CAP implementation against the canonical full CHAIR and POPE protocols and compare with vanilla.
- Worker role: validation
- Wave: 3
- Concrete edits: Create new dated output directories and experiment reports only.
- Interfaces / contracts changed: None.
- Test cases: Full CHAIR-500 and POPE adversarial-3000 with `vanilla,cap`, seed 42, sampling, max-new-tokens 512/8, batch 32/8, image batch 16, workers 4, prefetch 4.
- Pre-check commands: scheduler inspection on both GPUs; `uv pip check --python .venv/bin/python`; verify fixed model/data paths and no existing target directories.
- Acceptance criteria: Both runs complete with `status=PASS`, full count, `caption_token_injection=false`, and canonical acceleration gates.
- Verification: CHAIR and POPE metrics, manifest protocol fields, output row counts, and per-question/per-image CAP diagnostics.
- Post-check commands: machine-readable metrics/acceleration audit; GPU process cleanup check.
- Dependencies: Task 2
- Files likely touched: `outputs/cap_chair500_20260924/*`; `outputs/cap_pope3000_20260924/*`; `.codex/work/20260924-cap-next/artifacts/*`
- Writable scope: `outputs/cap_chair500_20260924/`, `outputs/cap_pope3000_20260924/`, `.codex/work/20260924-cap-next/artifacts/`
- Output artifact: `.codex/work/20260924-cap-next/artifacts/task3-verification.md`
- Estimated scope: L

### Task 4: Gain sweep、holdout 与 head-to-head

- Description: Use CHAIR-64 to explore probe gain, confirm the selected gain on CHAIR holdout, and compare CAP with GTP Patch under identical protocol.
- Worker role: validation
- Wave: 4
- Concrete edits: Create dated sweep/holdout/head-to-head outputs and a consolidated analysis report; do not modify model/data caches.
- Interfaces / contracts changed: None.
- Test cases: Gains `0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50`; holdout `start=64,limit=436`; methods `vanilla,gtp_patch,cap`.
- Pre-check commands: validate Task 3 outputs and choose gain from CHAIR-64 only; scheduler inspection before each parallel wave.
- Acceptance criteria: Sweep outputs are complete; selected gain is explicitly reported; holdout and head-to-head preserve protocol and acceleration fields.
- Verification: aggregate metrics table, gain curve data, holdout comparison, and decision report.
- Post-check commands: `git diff --check`; full test suite; execution validator.
- Dependencies: Task 3
- Files likely touched: `outputs/cap_chair64_gain_*/*`; `outputs/h2h_chair436_confirm_20260924/*`; `.codex/work/20260924-cap-next/artifacts/*`
- Writable scope: `outputs/cap_chair64_gain_*/`, `outputs/h2h_chair436_confirm_20260924/`, `.codex/work/20260924-cap-next/artifacts/`
- Output artifact: `.codex/work/20260924-cap-next/artifacts/final-report.md`
- Estimated scope: L

## Verification

1. Focused CAP/runtime tests and Ruff.
2. Full repository pytest.
3. 32-row per-question POPE smoke.
4. Full CHAIR-500 and POPE-3000 protocol audit.
5. Gain sweep and holdout/head-to-head audit.
6. `python /data/lcq/.codex/skills/plan2do/scripts/validate_execution.py .codex/work/20260924-cap-next`.

## Acceptance Criteria

- Per-question attribution is implemented, tested, and visible in diagnostics.
- Full CAP runs use the fixed project model/data/prompt/decoding/acceleration protocol.
- No full-count effectiveness claim is made unless output status, count, protocol and acceleration gates pass.
- Any weak or negative CHAIR result is reported as evidence and triggers the documented diagnostic path rather than being hidden.

## Rollback / Recovery Plan

- Never overwrite existing outputs; use the fixed dated directories in this plan.
- If a code defect appears, write rework guidance before the smallest in-scope fix and rerun focused checks.
- If a GPU/provider interruption occurs, preserve partial artifacts and rerun only the incomplete target directory after confirming it does not exist or is safely resumable.

## Abort Criteria

- Missing local model/data assets or broken `.venv`.
- No safe GPU placement under `gpu-scheduler`.
- Canonical acceleration gates fail repeatedly due to environment.
- Same code defect remains after two bounded rework cycles.

## Output Artifact

- Code: current repository CAP/runtime/test files
- Smoke: `outputs/cap_pope32_per_question_20260924/`
- Main experiments: `outputs/cap_chair500_20260924/`, `outputs/cap_pope3000_20260924/`
- Follow-up: `outputs/cap_chair64_gain_*`, `outputs/h2h_chair436_confirm_20260924/`
- Coordination: `.codex/work/20260924-cap-next/`

## Execution Handoff

The uploaded plan is advisory rather than compiler-ready. The previous CAP implementation and its `131 passed` baseline are authoritative upstream evidence. The local `context-engineering` skill path referenced by plan2do is absent on this host; this execution uses the plan2do contract and explicit context artifacts instead.
