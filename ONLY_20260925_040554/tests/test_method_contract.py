from __future__ import annotations

import torch

from methods.standard import StandardInferMethod
from methods.vanilla import VanillaPopeMethod
from models.mock import MockMLLMWrapper


def test_standard_method_contract() -> None:
    model = MockMLLMWrapper(
        path="mock://mllm",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    method = StandardInferMethod(
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,
    )
    model.setup(torch.device("cpu"))
    method.setup(model, torch.device("cpu"))

    outputs = method.generate(
        {
            "records": [
                {
                    "id": "sample-001",
                    "image_path": "mock://image-001",
                    "prompt": "Describe the scene.",
                }
            ]
        }
    )

    assert outputs[0]["id"] == "sample-001"
    assert outputs[0]["method"] == "StandardInferMethod"
    assert "mock_response" in outputs[0]["prediction"]


def test_explicit_cpu_device_is_honored() -> None:
    from utils import dist_utils

    assert dist_utils.get_device("cpu") == torch.device("cpu")


def test_vanilla_pope_method_has_generic_wrapper_fallback() -> None:
    model = MockMLLMWrapper(
        path="mock://mllm",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    method = VanillaPopeMethod()
    model.setup(torch.device("cpu"))
    method.setup(model, torch.device("cpu"))

    output = method.generate(
        {
            "records": [
                {
                    "id": "pope-1",
                    "prompt": "Is there a cat?",
                    "image_path": "mock://image",
                }
            ]
        }
    )

    assert output[0]["method"] == "vanilla"
    assert output[0]["backend"] == "generic_generate_response"
