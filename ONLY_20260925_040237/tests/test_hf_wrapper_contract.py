from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from models.hf import HFMLLMWrapper


def test_hf_wrapper_is_lazy_and_uses_explicit_loading_options() -> None:
    wrapper = HFMLLMWrapper(
        path="~/cached-model",
        model_family="auto",
        torch_dtype="bfloat16",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=False,
        local_files_only=True,
    )
    model = MagicMock()
    model.to.return_value = model
    processor = MagicMock()

    assert wrapper.model is None
    assert wrapper.processor is None
    with (
        patch("models.hf.AutoProcessor.from_pretrained", return_value=processor) as load_processor,
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as load_model,
    ):
        wrapper.setup(torch.device("cpu"))

    load_processor.assert_called_once_with(
        "~/cached-model",
        local_files_only=True,
        trust_remote_code=False,
    )
    load_model.assert_called_once_with(
        "~/cached-model",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()


def test_hf_wrapper_rejects_unknown_dtype() -> None:
    try:
        HFMLLMWrapper(path="mock://model", torch_dtype="int8")
    except ValueError as exc:
        assert "Unsupported torch_dtype" in str(exc)
    else:
        raise AssertionError("Expected unsupported dtype to be rejected")
