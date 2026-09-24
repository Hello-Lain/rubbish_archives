# Task 1 Verification

- Status: complete
- Scope: per-question CAP attribution and row-aware feature-cache routing.
- Files: `scripts/vrsc_probe.py`, `scripts/shield_vulnerability_runtime.py`, `tests/test_vrsc_probe.py`
- Implementation:
  - `cap_row_key` hashes image, question ID, and question text into a stable row key.
  - CAP prompt-prefill attribution is computed independently for every selected row.
  - CAP raw probes and projected features are cached by row key.
  - The vulnerability prompt helper accepts an optional row-aware cache-key function while preserving path-keyed behavior for existing callers.
  - The manifest now records question-conditioned attribution rather than the previous per-image limitation.
- Verification: `.venv/bin/python -m py_compile scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py` -> passed.
- Verification: `.venv/bin/ruff check scripts/vrsc_probe.py scripts/shield_vulnerability_runtime.py tests/test_vrsc_probe.py` -> passed.
- Verification: `.venv/bin/python -m pytest tests/test_cap.py tests/test_vrsc_probe.py -q` -> `36 passed`.
- Verification: `git diff --check` -> passed.
- Verification: `uv pip check --python .venv/bin/python` -> all installed packages compatible.
- Acceptance: same-image different-question keys differ, identical rows remain stable, CAP routing remains `("base", "cap", None)`, and existing focused CAP/runtime tests pass.
- Follow-up: repeated-image runtime behavior is validated by Task 2's 32-row POPE smoke.
