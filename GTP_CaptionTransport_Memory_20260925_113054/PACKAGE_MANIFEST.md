# GTP-CaptionTransport+Memory

This package contains the frozen GTP caption-token transport and grounded-only
caption-memory candidate used for the SHIELD-compatible CHAIR-500 confirmation
run.

## Frozen Configuration

```text
a_u=0.75
a_h=0.75
tau=1.0
tau_e=1.0
kappa=1.0
grounding_source=caption_tokens
inject_caption_memory=true
caption_memory_slots=32
```

## CHAIR-500 Confirmation

```text
CHAIRs=0.480000
CHAIRi=0.131422
Recall=0.842098
status=PASS
effective_attention=flash_attention_2
fp8_effective=true
fp8_native_fallback_calls=0
fallback_events=[]
```

## Included Evidence

- `evidence/chair500_frozen_metrics.json`
- `evidence/chair500_frozen_predictions.jsonl`
- `evidence/dev128_memory_metrics.json`
- `evidence/dev128_memory_predictions.jsonl`
- `evidence/original_gtp_chair_metrics.json`
- `evidence/pope3000_gtp_metrics.json`
- `evidence/pope3000_gtp_predictions.jsonl`

The POPE evidence is the current full GTP run with the base `coco_claims`
configuration. A full POPE run using the frozen CaptionTransport+Memory
configuration is not included because it was not run.
