# Project Hyperparameter Structure

## Runtime Entry Points

The canonical inference and evaluation entry points are `src/infer.py::main`
and `src/eval.py::main`. They are configured through Hydra, with the top-level
composition roots in `configs/experiment/`. `configs/method/`, `configs/model/`,
`configs/data/`, `configs/eval_tasks/`, and the environment/path groups provide
the composed runtime settings.

The GTP/VRSC research evaluation used for this AHT session is a separate
CLI-based path: `scripts/vrsc_probe.py::main`. It loads the LLaVA runtime through
`scripts/shield_cumulative.py`, constructs method branches with
`scripts/shield_vulnerability_runtime.py`, and evaluates captions with
`src/adapters/chair_metrics.py` (through `ChairScorer`). This command does not
compose Hydra configs; its parser defaults and explicit CLI flags are the
effective configuration. Run Python commands with `.venv/bin/python`.

## Configuration Topology

Hydra experiment aliases live in `configs/experiment/`; method defaults live
in `configs/method/`; model and data settings live in their corresponding
groups. The existing Hydra Optuna example is
`configs/hparams_search/method_optuna.yaml`. For a Hydra run, inspect the
resolved composition with `src/eval.py --cfg job --resolve` and `--info`; run
artifacts and Hydra override records are stored beneath the configured Hydra
output directory in `.hydra/`.

The active GTP probe instead uses command-line values. Relevant defaults in
`scripts/vrsc_probe.py` include seed 42, sampling at temperature 1/top-p 1,
dynamic cache, BF16, FlashAttention-2, FP8 `text_mlp`, TF32, four data workers,
and image batch size 16. Dataset-specific defaults are batch size 32 and 512
new tokens for CHAIR, and batch size 8 and 8 new tokens for POPE. Model,
image, question, caption, and annotation inputs are resolved from the paths
recorded in the run manifest; source model and dataset assets must remain under
`/data/lcq/.cache/huggingface/hub`.

## TensorBoard Integration

The inspected inference/evaluation path does not use `SummaryWriter` and does
not emit TensorBoard event files. `src/utils/experiment_logger.py` supports
optional W&B tracking for canonical Hydra runs, but `scripts/vrsc_probe.py`
writes its own metrics JSON, prediction JSONL, diagnostics, and manifest into
the explicit `--output` directory. Those JSON metrics are the authoritative
source for GTP AHT objectives. The AHT-specific local SwanLab adapter is
optional and must not change normal inference behavior.

## Override Semantics

For the research CLI, pass values directly as flags; there is no Hydra
`key=value` override syntax:

```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset chair --limit 64 --methods gtp \
  --output outputs/gtp-chair-trial
```

The active counterfactual GTP parameters are:

- `--gtp-max-evidence-weight`: scales the evidence-specific correction.
- `--gtp-trust`: controls the chi-square projection radius.
- `--gtp-specificity-temperature`: smooths the visual-support gate.
- `--gtp-support-margin`: suppresses weak visual shifts before correction.
- `--gtp-correction-clip`: bounds per-token counterfactual corrections.

Legacy reliability-gate knobs (`--gtp-signal-scale`,
`--gtp-disagreement-scale`, `--gtp-sharpness`) are not used by the
control-logit projection path and should not be included in this search. Keep
the model, images, questions, captions, prompt, seed, decoding, and acceleration
settings fixed while tuning.

## Sweep / Experiment Orchestration

The CLI is non-Hydra, so AHT uses Optuna with a session-local
`search_space.yaml` and `scripts/optuna_cli_runner.py`. Search-space entries map
sampled values to CLI flags; the adapter writes the scalar and supporting
CHAIR metrics to a JSON metric file. Keep trial outputs isolated under the
session's `optuna/trial_NNNN/` directories. Run only one trial per available
GPU unless the live GPU scheduler verifies more safe placements.

## Config Debugging and Inspection

For Hydra entry points, use:

```bash
.venv/bin/python src/eval.py --cfg job --resolve
.venv/bin/python src/eval.py --info
```

For the GTP CLI, inspect `scripts/vrsc_probe.py --help`, each run's
`manifest.json`, and `<method>.metrics.json`. The manifest records the argv,
protocol, model/data hashes, method settings, selected physical GPU, and
acceleration state. AHT session history is stored under
`.codex/aht/sessions/<timestamp>/`, with the active session referenced by
`.codex/aht/latest`.

## Agent Workflow for Config-Related Tasks

Before changing evaluation behavior or tuning, inspect the command-specific
entry point, active config/CLI chain, GTP branch routing, metric implementation,
and an existing manifest. Reuse valid full-run artifacts instead of repeating
them. Treat CHAIR-64 as exploratory only; confirm selected settings on the
disjoint CHAIR rows 64-499 and the full 500-image set, then check POPE-3000
before accepting a CHAIR-focused change.
