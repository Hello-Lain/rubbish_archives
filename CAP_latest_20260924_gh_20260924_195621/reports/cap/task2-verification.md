# Task 2 Verification

- Status: complete
- Output: `outputs/cap_pope32_per_question_20260924/`
- Command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/vrsc_probe.py --mode generate --dataset pope --limit 32 --methods vanilla,cap --output outputs/cap_pope32_per_question_20260924 --cap-probe-gain 0.20`
- GPU scheduling: physical GPU `0` was selected with `81077 MiB` free; both H100 GPUs had no compute applications before launch. Post-run query showed no compute applications and `14 MiB` used on each GPU.
- Runtime: vanilla and CAP each generated `32/32`; manifest status is `SMOKE`; six unique images were reused by the 32 POPE rows.
- Per-question acceptance:
  - `probe_diagnostics.json` contains 32 question-scope diagnostics.
  - 32 distinct `cap_cache_key` values are present.
  - Every diagnostic includes `question_id` and question text.
  - Every attribution vector has length `576`.
  - All six repeated-image groups contain distinct attribution arrays across their questions.
- Metrics: vanilla and CAP both have `count=32`, `accuracy=0.8125`, `precision=0.8571428571428571`, `recall=0.75`, and `f1=0.8`. This is a routing smoke, not a full-count effectiveness claim.
- Acceleration: `effective_attention=flash_attention_2`, `fp8_effective=true`, `fp8_native_fallback_calls=0`, and `fallback_events=[]`.
- Protocol: `pope_answer_instruction=true`, `caption_token_injection=false`, `batch_size=8`, `image_batch_size=16`, `num_workers=4`, `prefetch_factor=4`, `max_new_tokens=8`, `cache=dynamic`.
- Environment: `uv pip check --python .venv/bin/python` passed.
