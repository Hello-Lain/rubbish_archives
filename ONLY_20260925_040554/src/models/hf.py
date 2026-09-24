from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from models.base import BaseMLLMWrapper

DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HFMLLMWrapper(BaseMLLMWrapper):
    """Lazy Hugging Face wrapper for Qwen2.5-VL, LLaVA, and causal LM models."""

    def __init__(
        self,
        path: str,
        model_family: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "auto",
        low_cpu_mem_usage: bool = True,
        device_map: str | dict[str, Any] | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        max_new_tokens: int = 8,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        cache_implementation: str = "dynamic",
    ) -> None:
        super().__init__(
            path,
            model_family=model_family,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            cache_implementation=cache_implementation,
        )
        self.model_family = model_family
        self.torch_dtype_name = torch_dtype
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.attn_implementation = attn_implementation
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.cache_implementation = cache_implementation
        self.model: torch.nn.Module | None = None
        self.processor: Any | None = None
        self.effective_attention: str | None = None

    def setup(self, device: torch.device) -> None:
        self.device = device
        self.effective_attention = self._resolve_attention(device)
        processor_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(self.path, **processor_kwargs)
        model_class = self._resolve_model_class()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.effective_attention is not None:
            model_kwargs["attn_implementation"] = self.effective_attention
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        self.model = model_class.from_pretrained(self.path, **model_kwargs)
        if self.device_map is None:
            self.model = self.model.to(device)
        self.model.eval()

    def generate_response(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> str:
        if self.model is None or self.processor is None or self.device is None:
            raise RuntimeError("HFMLLMWrapper.setup(device) must be called first.")
        inputs = self._prepare_inputs(prompt=prompt, image_path=image_path)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": True,
        }
        if self.cache_implementation:
            generation_kwargs["cache_implementation"] = self.cache_implementation
        if self.do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[:, prompt_length:]
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    def _prepare_inputs(
        self,
        *,
        prompt: str,
        image_path: str | None,
    ) -> dict[str, torch.Tensor]:
        if self.model_family in {"qwen25_vl", "qwen2.5-vl"}:
            if image_path is None:
                raise ValueError("Qwen2.5-VL inference requires image_path.")
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        elif self.model_family == "llava":
            if image_path is None:
                raise ValueError("LLaVA inference requires image_path.")
            with Image.open(Path(image_path)) as image:
                inputs = self.processor(
                    text=f"USER: <image>\n{prompt} ASSISTANT:",
                    images=image.convert("RGB"),
                    return_tensors="pt",
                )
        else:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
            )
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def _resolve_model_class(self) -> Any:
        if self.model_family in {"qwen25_vl", "qwen2.5-vl"}:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        if self.model_family == "llava":
            from transformers import LlavaForConditionalGeneration

            return LlavaForConditionalGeneration
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def _resolve_attention(self, device: torch.device) -> str:
        if self.attn_implementation != "auto":
            return self.attn_implementation
        if device.type == "cuda":
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except Exception:
                pass
        return "sdpa"

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        try:
            return DTYPE_BY_NAME[name]
        except KeyError as exc:
            valid = ", ".join(sorted(DTYPE_BY_NAME))
            raise ValueError(f"Unsupported torch_dtype={name!r}; expected one of: {valid}") from exc
