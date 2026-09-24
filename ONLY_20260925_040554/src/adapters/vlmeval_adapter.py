from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from methods.base_method import BaseMethod
from models.base import BaseMLLMWrapper

try:
    from vlmeval.vlm.base import BaseModel as VLMEvalBaseModel
except ModuleNotFoundError:

    class VLMEvalBaseModel:
        """Small fallback that permits adapter unit tests without VLMEvalKit."""

        INTERLEAVE = False

        def __init__(self) -> None:
            self.dump_image_func: Any | None = None

        def set_dump_image(self, dump_image_func: Any) -> None:
            self.dump_image_func = dump_image_func

        def generate(
            self,
            message: Sequence[Mapping[str, str]],
            dataset: str | None = None,
        ) -> str:
            return self.generate_inner(message, dataset)


class VLMEvalAdapter(VLMEvalBaseModel):
    """Translate VLMEvalKit messages to the project's method contract."""

    INTERLEAVE = False

    def __init__(
        self,
        model_wrapper: BaseMLLMWrapper,
        method: BaseMethod,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model_wrapper = model_wrapper
        self.method = method
        self.device = device
        self._call_count = 0

    def build_prompt(self, line: Any, dataset: str | None = None) -> Any:
        del line, dataset
        raise NotImplementedError(
            "Use VLMEvalKit dataset prompts unless a model-specific adapter overrides this."
        )

    def generate_inner(
        self,
        message: Sequence[Mapping[str, str]],
        dataset: str | None = None,
    ) -> str:
        record = self._message_to_record(message, dataset)
        outputs = self.method.generate({"records": [record]})
        if not outputs:
            raise RuntimeError("Method returned no output for VLMEvalKit message.")
        prediction = outputs[0].get("prediction")
        if prediction is None:
            raise KeyError("Method output must contain 'prediction'.")
        return str(prediction)

    def _message_to_record(
        self,
        message: Sequence[Mapping[str, str]],
        dataset: str | None,
    ) -> dict[str, Any]:
        prompts: list[str] = []
        image_paths: list[str] = []
        for item in message:
            item_type = item.get("type")
            value = item.get("value")
            if item_type == "text":
                prompts.append(str(value))
            elif item_type == "image":
                image_paths.append(str(value))
            elif item_type == "video":
                raise NotImplementedError(
                    "VLMEvalAdapter supports image benchmarks only; add a video adapter."
                )
            else:
                raise ValueError(f"Unsupported VLMEvalKit message type: {item_type!r}")
        if len(image_paths) > 1:
            raise NotImplementedError("VLMEvalAdapter does not support multi-image interleave yet.")
        self._call_count += 1
        return {
            "id": f"{dataset or 'vlmeval'}-{self._call_count:08d}",
            "prompt": "\n".join(prompts),
            "image_path": image_paths[0] if image_paths else None,
        }
