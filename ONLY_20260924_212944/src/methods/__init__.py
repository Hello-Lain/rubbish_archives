"""Inference strategies with a stable ``setup`` and ``generate`` contract."""

from methods.cap import (
    CAP,
    CAPConfig,
    CAPMethod,
    attention_attribution_weights,
    causal_attribution_weights,
    eager_last_layer_attention,
    last_language_layer_index,
)

__all__ = [
    "CAP",
    "CAPConfig",
    "CAPMethod",
    "attention_attribution_weights",
    "causal_attribution_weights",
    "eager_last_layer_attention",
    "last_language_layer_index",
]
