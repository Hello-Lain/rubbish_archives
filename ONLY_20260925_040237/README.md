# ONLY: Canonical MLLM Inference and Evaluation

This repository contains a clean, configuration-driven implementation of the
ONLY inference method and the shared evaluation infrastructure for future
method development.

## Layout

- `src/`: model wrappers, methods, datasets, adapters, inference, and evaluation
- `configs/`: Hydra configuration groups
- `scripts/`: reproducible entry-point scripts
- `tests/`: contract and smoke tests
- `AGENTS.md`: fixed local inference protocol and latest verified metrics
- `PROJECT.md`: project structure and extension rules

## Environment

Use the repository-managed `uv` environment:

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
```

Model weights, datasets, and images are resolved from the local cache described
in `AGENTS.md`. Do not place model weights or generated experiment outputs in
the Git repository.

## POPE

Vanilla and ONLY use the same model, data, prompt, decoding, and acceleration
protocol:

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/eval.py experiment=pope_qwen25_vl method=vanilla method_name=vanilla

uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/eval.py experiment=pope_qwen25_vl method=only method_name=only
```

Before a real CUDA run, check device availability and other processes with the
local `gpu-scheduler` procedure. Reuse an existing verified output when its
manifest, protocol, model/data hashes, and metrics satisfy `AGENTS.md`.

## SHIELD Modules

This repository also contains composable SHIELD method plugins for LLaVA-1.5-7B:

- `shield_adaptive_plausibility`: clean-logit plausibility cutoff
- `shield_vulnerability_defense`: CLIP-space learnable attack plus contrastive decoding
- `shield_statistical_bias`: CLIP caption-token guided visual-token re-weighting
- `shield_inherent_bias`: noise-derived visual-prior estimation and subtraction

The composition is implemented by
`src/methods/shield_pipeline.py`. The recommended deployment composition
contains three components:

```text
Adaptive Plausibility
+ Vulnerability Defense
+ Statistical Bias
```

Its fixed SHIELD order is:

```text
raw visual features
  -> Statistical Bias
  -> multimodal projector
clean/adversarial logits
  -> Vulnerability Defense contrastive combination
  -> Adaptive Plausibility cutoff
```

`Inherent Bias` remains available as an optional research/ablation plugin, but
is not part of the default `shield_cumulative` method.

Each module has its own config and `BaseMethod` adapter:

- `configs/method/shield_adaptive_plausibility.yaml`
- `configs/method/shield_vulnerability_defense.yaml`
- `configs/method/shield_statistical_bias.yaml`
- `configs/method/shield_inherent_bias.yaml`
- `configs/method/shield_cumulative.yaml`

The cumulative evaluator selects a canonical ablation stage with
`--stage adaptive|vulnerability|statistical|inherent|full`, while the Hydra
method config accepts an explicit `components` list. The `statistical` stage
is the recommended three-component configuration. `inherent` and `full`
remain available for reproducing the four-component ablation. Component
manifests are written under `shield_components` in new cumulative metrics
files.

The fixed-protocol implementation and results are documented in
[`reports/shield_reproduction_20260922.md`](reports/shield_reproduction_20260922.md).
The strict component contribution analysis uses the complete `2^3` matrix
(singletons, pairs, and the full three-component method), and reports both
full leave-one-out and Shapley values:
`scripts/analyze_shield_contributions.py`,
[`reports/shield_strict_contributions_20260922.md`](reports/shield_strict_contributions_20260922.md).
The CHAIR Statistical Bias entry point defaults to SHIELD's pre-generated naive
caption file (`outputs/shield_llava15_chair_first_caption_20260922.jsonl`), not
the final Vanilla 512-token captions.

## ECP Research Candidate

Evidence Consensus Projection replaces the VRSC response heuristic with a
two-reference chi-square objective: both the clean-image distribution and the
image-plus-caption-memory distribution constrain probability changes. Its
harmonic reference and shared dual threshold jointly determine support and
token probabilities. The current configuration keeps all caption tokens in
the measured datasets; sparse joint affinities must not be interpreted as
validated semantic filtering.

Frozen-parameter LLaVA results, with the existing A+S baseline reused:

| Method | POPE Accuracy | POPE F1 | CHAIRs | CHAIRi | CHAIR Recall |
|---|---:|---:|---:|---:|---:|
| A+S | 82.700% | 81.511% | 0.432 | 0.129335 | 0.828539 |
| ECP | 83.067% | 81.779% | 0.422 | 0.124098 | 0.829856 |

ECP meets the declared point-estimate proximity criteria on POPE-3000 and
CHAIR-500, including after excluding this iteration's pilot images. This is
not a statistical equivalence or superiority claim. Default SHIELD A+V+S is
unchanged. Formulas, paired intervals, ablations, costs, and limitations:
[`idea.md`](idea.md) and
[`reports/evidence_consensus_20260923.md`](reports/evidence_consensus_20260923.md).

Implementation: `src/methods/evidence_consensus.py`,
`configs/method/evidence_consensus.yaml`. The actual LLaVA backend is the
research CLI `scripts/vrsc_probe.py --methods evidence_consensus`; the Hydra
method adapter alone does not implement ECP for arbitrary model wrappers.
POPE and CHAIR share `evidence_weight=0.75`, `trust=1`,
`transport_trust=1`, and `feature_gain=1`.

Reuse the existing full results for CPU-only comparison:

```bash
.venv/bin/python scripts/compare_evidence_results.py \
  --candidate outputs/vrsc_20260923/pope3000_ecp_frozen_v1/evidence_consensus.metrics.json \
  --reference outputs/pope_llava_7b_adversarial_subset_adaptive_statistical_official_20260922.metrics.json \
  --dataset pope --output outputs/ecp_pope_recheck.json
```

The output path must not exist. For a genuinely new inference configuration,
check GPU ownership/headroom first and select `CUDA_VISIBLE_DEVICES`, then use
the research CLI with `--mode generate --dataset pope --limit 3000` or
`--dataset chair --limit 500`, `--methods evidence_consensus`, and a new output
directory. Existing verified predictions do not need regeneration.

## VRSC Historical Prototype

VRSC is an experimental research plugin that combines an S-derived patch score
with a separate visual-response calibration rule. It uses an energy-preserving
visual probe, compares probe and clean next-token distributions under the same
prefix, and applies a chi-square trust-region projection to produce one sparse
distribution. It does not change the default SHIELD pipeline.

Implementation:

- `src/methods/vrsc.py`
- `configs/method/vrsc.yaml`
- `configs/method/vrsc_residual.yaml`
- `scripts/vrsc_probe.py`
- `scripts/analyze_vrsc_probe.py`

Run the paired first-token diagnostic in a new output directory:

```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode capture --dataset pope --limit 256 \
  --output outputs/vrsc_<date>/pope256_capture
.venv/bin/python scripts/analyze_vrsc_probe.py \
  outputs/vrsc_<date>/pope256_capture \
  --output outputs/vrsc_<date>/pope256_analysis
```

Run the protocol-matched autoregressive smoke:

```bash
.venv/bin/python scripts/vrsc_probe.py \
  --mode generate --dataset pope --limit 64 \
  --output outputs/vrsc_<date>/pope64_generate
```

VRSC remains research-only until a larger paired study shows that its semantic
visual response is better than shuffled and distribution-only controls. The
current result is documented in
[`reports/vrsc_feasibility_20260923.md`](reports/vrsc_feasibility_20260923.md).
The optional `vrsc_residual` config removes the response component aligned with
a shuffled probe. Earlier residual smokes had an invalid branch routing and
cannot support a mechanism conclusion; see
`reports/vrsc_residual_routing_correction_20260923.md`.
VRSC itself did not match A+S on the complete tasks, even after the residual
routing correction. ECP above is the subsequent candidate. The original design
is preserved in `reports/vrsc_design_history_20260923.md`.

## Citation

```bibtex
@article{wan2025only,
  title={ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models},
  author={Wan, Zifu and Zhang, Ce and Yong, Silong and Ma, Martin Q and Stepputtis, Simon and Morency, Louis-Philippe and Ramanan, Deva and Sycara, Katia and Xie, Yaqi},
  journal={arXiv preprint arXiv:2507.00898},
  year={2025}
}
```
