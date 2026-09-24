#!/usr/bin/env python3
"""Offline Qwen2.5-VL inference with measurable CUDA acceleration switches.

The default path is conservative: BF16, the fastest available attention
backend, and the normal dynamic KV cache. Optional switches are deliberately
explicit because static-cache compilation and FP8 have warm-up and numerical
trade-offs that are not useful for every workload.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from transformers import AutoProcessor, CompileConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = (
    "/data/lcq/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
    "/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)


@dataclass
class RuntimeState:
    requested_attention: str
    effective_attention: str
    flash_attn_version: str | None
    requested_cache: str
    cache: str
    compile_requested: bool
    compile_effective: bool
    compile_mode: str | None
    fp8_requested: bool
    fp8_effective: bool
    fp8_scope: str | None
    fp8_linear_count: int
    tf32: bool
    fallback_events: list[str]

    def as_dict(self) -> dict:
        return {
            "requested_attention": self.requested_attention,
            "effective_attention": self.effective_attention,
            "flash_attn_version": self.flash_attn_version,
            "requested_cache": self.requested_cache,
            "cache": self.cache,
            "compile_requested": self.compile_requested,
            "compile_effective": self.compile_effective,
            "compile_mode": self.compile_mode,
            "fp8_requested": self.fp8_requested,
            "fp8_effective": self.fp8_effective,
            "fp8_scope": self.fp8_scope,
            "fp8_linear_count": self.fp8_linear_count,
            "tf32": self.tf32,
            "fallback_events": self.fallback_events,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a local image")
    parser.add_argument("--prompt", required=True, help="Question or instruction")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--attention",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default="auto",
        help="Attention backend. auto prefers FA2 when it is importable on CUDA.",
    )
    parser.add_argument(
        "--cache",
        choices=("dynamic", "static"),
        default="dynamic",
        help="KV cache implementation. static is intended for repeated decoding.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use Transformers' static-cache generation compilation.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("reduce-overhead", "max-autotune", "default"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--fp8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive Transformer Engine FP8 when the local stack supports it.",
    )
    parser.add_argument(
        "--fp8-scope",
        choices=("text_mlp", "text_all"),
        default="text_mlp",
        help="text_mlp is the safer default; text_all also converts attention projections.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 for CUDA FP32 matmuls where applicable.",
    )
    parser.add_argument(
        "--fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry with a conservative backend when an optional acceleration fails.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run warm-up/repeated generation and emit timing JSON instead of only text.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Write runtime state and benchmark measurements to this JSON file.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def configure_cuda(device: torch.device, enabled: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def configure_compile() -> None:
    # Qwen attention compares a layer index stored on each module. Newer
    # Dynamo versions can treat that integer as unspecialized and avoid one
    # graph per decoder layer; older versions simply lack this option.
    try:
        import torch._dynamo as dynamo

        if hasattr(dynamo.config, "allow_unspec_int_on_nn_module"):
            dynamo.config.allow_unspec_int_on_nn_module = True
    except Exception:
        pass


def detect_flash_attention() -> tuple[bool, str | None]:
    try:
        import flash_attn
    except Exception:
        return False, None
    return True, str(getattr(flash_attn, "__version__", "unknown"))


def choose_attention(requested: str, device: torch.device) -> tuple[str, str | None]:
    available, version = detect_flash_attention()
    if requested == "auto":
        if device.type == "cuda" and available:
            return "flash_attention_2", version
        return "sdpa", version
    if requested == "flash_attention_2" and not available:
        raise RuntimeError("flash_attention_2 was requested but flash-attn is not importable")
    return requested, version


def _copy_linear_parameters(source: torch.nn.Linear, target: torch.nn.Module) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        if source.bias is not None:
            target.bias.copy_(source.bias)
    target.weight.requires_grad_(source.weight.requires_grad)
    if source.bias is not None:
        target.bias.requires_grad_(source.bias.requires_grad)


class AdaptiveTELinear(nn.Module):
    """Use TE FP8 only for shapes accepted by the H100 FP8 GEMM kernels."""

    def __init__(self, native: nn.Linear, te_linear: nn.Module) -> None:
        super().__init__()
        self.native = native
        self.te_linear = te_linear
        self.fp8_enabled = True
        self.fp8_calls = 0
        self.native_calls = 0
        self.in_features = native.in_features
        self.out_features = native.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.numel() // inputs.shape[-1]
        can_use_fp8 = (
            self.fp8_enabled
            and inputs.shape[-1] % 16 == 0
            and tokens % 8 == 0
        )
        if can_use_fp8:
            if not is_torch_compiling():
                self.fp8_calls += 1
            return self.te_linear(inputs)
        if not is_torch_compiling():
            self.native_calls += 1
        return self.native(inputs)


def is_torch_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except AttributeError:
        try:
            return torch._dynamo.is_compiling()
        except AttributeError:
            return False


def set_fp8_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, AdaptiveTELinear):
            module.fp8_enabled = enabled


def get_fp8_call_counts(model: nn.Module) -> tuple[int, int]:
    fp8_calls = 0
    native_calls = 0
    for module in model.modules():
        if isinstance(module, AdaptiveTELinear):
            fp8_calls += module.fp8_calls
            native_calls += module.native_calls
    return fp8_calls, native_calls


def enable_te_fp8(
    model: torch.nn.Module,
    device: torch.device,
    scope: str,
) -> tuple[object, int]:
    """Convert real Qwen language-model projections to TE Linear modules.

    Vision modules remain native PyTorch because their variable-length image
    packing is more sensitive to backend changes. The recipe/context is kept
    outside this function so every generation call uses the same FP8 state.
    """

    if device.type != "cuda":
        raise RuntimeError("Transformer Engine FP8 requires CUDA")

    from transformer_engine import pytorch as te
    from transformer_engine.common import recipe

    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if language_model is None:
        raise RuntimeError("Could not locate model.model.language_model for TE conversion")

    converted = 0

    def should_convert(path: str) -> bool:
        if ".mlp." in path:
            return True
        return scope == "text_all" and ".self_attn." in path

    def visit(parent: torch.nn.Module, prefix: str = "") -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear) and should_convert(path):
                if child.weight.device.type != "cuda":
                    raise RuntimeError(f"Cannot convert non-CUDA layer to TE: {path}")
                target = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                    name=path,
                )
                _copy_linear_parameters(child, target)
                setattr(parent, name, AdaptiveTELinear(child, target))
                converted += 1
            else:
                visit(child, path)

    visit(language_model)
    if converted == 0:
        raise RuntimeError(f"No Linear layers matched Transformer Engine scope {scope!r}")

    fp8_recipe = recipe.Float8CurrentScaling(fp8_format=recipe.Format.HYBRID)
    return fp8_recipe, converted


@contextlib.contextmanager
def fp8_context(state: RuntimeState, recipe: object) -> Iterator[None]:
    if not state.fp8_effective:
        yield
        return
    from transformer_engine.pytorch import fp8_autocast

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        yield


def load_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype | str,
    attention: str,
) -> Qwen2_5_VLForConditionalGeneration:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=str(device),
        attn_implementation=attention,
        local_files_only=True,
    )
    return model.eval()


def build_inputs(
    processor: AutoProcessor,
    image: Path,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generation_kwargs(args: argparse.Namespace, state: RuntimeState) -> dict:
    kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    if state.cache == "static" or state.compile_effective:
        kwargs["cache_implementation"] = "static"
    if state.compile_effective:
        kwargs["compile_config"] = CompileConfig(
            dynamic=False,
            mode=args.compile_mode,
        )
    else:
        # Transformers otherwise auto-compiles every static-cache generation.
        # Keep static allocation available without paying compile warm-up unless
        # the caller explicitly requested --compile.
        kwargs["disable_compile"] = True
    return kwargs


def generate_once(
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    state: RuntimeState,
    fp8_recipe: object | None,
) -> tuple[torch.Tensor, float]:
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), fp8_context(state, fp8_recipe):
        generated = model.generate(**inputs, **generation_kwargs(args, state))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return generated, time.perf_counter() - started


def decode_answer(
    processor: AutoProcessor,
    generated: torch.Tensor,
    inputs: dict[str, torch.Tensor],
) -> str:
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def load_runtime(
    args: argparse.Namespace,
    model_path: Path,
    device: torch.device,
) -> tuple[
    Qwen2_5_VLForConditionalGeneration,
    RuntimeState,
    object | None,
]:
    if args.compile:
        configure_compile()
    effective_attention, flash_version = choose_attention(args.attention, device)
    state = RuntimeState(
        requested_attention=args.attention,
        effective_attention=effective_attention,
        flash_attn_version=flash_version,
        requested_cache=args.cache,
        cache="static" if args.compile else args.cache,
        compile_requested=args.compile,
        compile_effective=args.compile,
        compile_mode=args.compile_mode if args.compile else None,
        fp8_requested=args.fp8,
        fp8_effective=False,
        fp8_scope=args.fp8_scope if args.fp8 else None,
        fp8_linear_count=0,
        tf32=args.tf32,
        fallback_events=[],
    )
    dtype: torch.dtype | str = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = load_model(model_path, device, dtype, effective_attention)

    fp8_recipe = None
    if args.fp8:
        try:
            fp8_recipe, converted = enable_te_fp8(model, device, args.fp8_scope)
            state.fp8_effective = True
            state.fp8_linear_count = converted
        except Exception as exc:
            if not args.fallback:
                raise
            state.fallback_events.append(f"FP8 disabled: {type(exc).__name__}: {exc}")
            state.fp8_scope = None
    return model, state, fp8_recipe


def write_report(path: Path | None, report: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    image = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Qwen model snapshot is incomplete: {model_path}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.benchmark and (args.warmup < 0 or args.runs < 1):
        raise ValueError("--warmup must be >= 0 and --runs must be >= 1")

    os.environ.setdefault("HF_HOME", "/data/lcq/.cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/data/lcq/.cache/huggingface/hub")
    device = resolve_device(args.device)
    configure_cuda(device, args.tf32)
    model, state, fp8_recipe = load_runtime(args, model_path, device)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    inputs = build_inputs(processor, image, args.prompt, device)

    measurements: list[float] = []
    generated: torch.Tensor | None = None
    total_calls = args.warmup + args.runs if args.benchmark else 1
    for call_index in range(total_calls):
        try:
            generated, elapsed = generate_once(model, inputs, args, state, fp8_recipe)
        except Exception as exc:
            can_retry = args.fallback and (
                state.compile_effective or state.cache == "static" or state.fp8_effective
            )
            if not can_retry:
                raise
            state.fallback_events.append(
                f"Optional path failed on call {call_index}: {type(exc).__name__}: {exc}"
            )
            state.compile_effective = False
            state.cache = "dynamic"
            state.fp8_effective = False
            set_fp8_enabled(model, False)
            fp8_recipe = None
            generated, elapsed = generate_once(model, inputs, args, state, fp8_recipe)
        if args.benchmark and call_index >= args.warmup:
            measurements.append(elapsed)

    if generated is None:
        raise RuntimeError("Generation produced no output")
    answer = decode_answer(processor, generated, inputs)
    report = {
        "model": str(model_path),
        "device": str(device),
        "dtype": args.dtype,
        "answer": answer,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
        "runtime": state.as_dict(),
    }
    if state.fp8_requested:
        fp8_calls, native_calls = get_fp8_call_counts(model)
        report["runtime"]["fp8_te_calls"] = fp8_calls
        report["runtime"]["fp8_native_fallback_calls"] = native_calls
    if measurements:
        report["benchmark"] = {
            "warmup": args.warmup,
            "runs": args.runs,
            "seconds": measurements,
            "mean_seconds": sum(measurements) / len(measurements),
            "tokens_per_second": (
                report["generated_tokens"] / (sum(measurements) / len(measurements))
            ),
        }
    write_report(args.report_json, report)

    if args.benchmark:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(answer)
        print(json.dumps({"runtime": state.as_dict()}, ensure_ascii=True), file=sys.stderr)


if __name__ == "__main__":
    main()
