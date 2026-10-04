#!/usr/bin/env python3
"""Convert LightX2V/PEFT Wan2.2-S2V LoRA weights to ComfyUI format.

LightX2V training exports keys such as::

    dit.blocks.0.self_attn.q.lora_A.weight
    dit.blocks.0.self_attn.q.lora_B.weight

ComfyUI's generic Wan key map expects::

    diffusion_model.blocks.0.self_attn.q.lora_down.weight
    diffusion_model.blocks.0.self_attn.q.lora_up.weight

The tensors are copied without rescaling. ComfyUI uses alpha/rank when an
``.alpha`` key is present; when it is absent, its LoRA adapter uses scale 1,
which is the convention used by the LightX2V S2V checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LightX2V Wan2.2-S2V LoRA checkpoint to ComfyUI keys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "--input-lora", dest="input_lora", required=True, help="Input .safetensors LoRA.")
    parser.add_argument("--output", "--output-lora", dest="output_lora", required=True, help="Output .safetensors LoRA.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional ComfyUI Wan/S2V base .safetensors used to validate all converted target keys and shapes.",
    )
    parser.add_argument(
        "--source-prefix",
        default="dit.",
        help="Prefix removed from the LightX2V key before adding the ComfyUI prefix.",
    )
    parser.add_argument(
        "--comfy-prefix",
        default="diffusion_model.",
        help="Prefix used by ComfyUI's Wan model key map.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("preserve", "fp16", "bf16", "fp32"),
        default="preserve",
        help="Optional dtype conversion for floating-point tensors.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    parser.add_argument(
        "--allow-other-keys",
        action="store_true",
        help="Keep non-LoRA keys instead of failing. LoRA A/B keys are always converted.",
    )
    return parser.parse_args(argv)


def _dtype(name: str) -> torch.dtype | None:
    return {"preserve": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def convert_key(key: str, source_prefix: str, comfy_prefix: str) -> str:
    """Map a LightX2V PEFT key to ComfyUI's generic Wan key."""
    if key.endswith(".lora_A.weight"):
        stem = key[: -len(".lora_A.weight")]
        suffix = ".lora_down.weight"
    elif key.endswith(".lora_B.weight"):
        stem = key[: -len(".lora_B.weight")]
        suffix = ".lora_up.weight"
    else:
        raise ValueError(f"Unsupported LoRA key (expected lora_A/lora_B): {key}")

    if source_prefix and stem.startswith(source_prefix):
        stem = stem[len(source_prefix) :]
    elif source_prefix:
        raise ValueError(f"LoRA key does not start with --source-prefix={source_prefix!r}: {key}")
    return f"{comfy_prefix}{stem}{suffix}"


def load_base_shapes(base_model: str) -> dict[str, tuple[int, ...]]:
    path = Path(base_model).expanduser()
    if not path.is_file() or path.suffix != ".safetensors":
        raise FileNotFoundError(f"--base-model must be an existing .safetensors file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}


def convert(args: argparse.Namespace) -> tuple[int, int, list[str]]:
    input_path = Path(args.input_lora).expanduser().resolve()
    output_path = Path(args.output_lora).expanduser().resolve()
    if not input_path.is_file() or input_path.suffix != ".safetensors":
        raise FileNotFoundError(f"--input must be an existing .safetensors file: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite to replace it.")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    output_dtype = _dtype(args.output_dtype)
    converted: dict[str, torch.Tensor] = {}
    converted_count = 0
    kept_count = 0
    warnings: list[str] = []
    with safe_open(input_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.endswith((".lora_A.weight", ".lora_B.weight")):
                new_key = convert_key(key, args.source_prefix, args.comfy_prefix)
                if new_key in converted:
                    raise ValueError(f"Key collision after conversion: {new_key}")
                if output_dtype is not None and tensor.dtype.is_floating_point:
                    tensor = tensor.to(output_dtype)
                converted[new_key] = tensor.contiguous()
                converted_count += 1
            elif key.endswith(".alpha"):
                # Preserve alpha under the converted stem when present.
                source_stem = key[: -len(".alpha")]
                if args.source_prefix and source_stem.startswith(args.source_prefix):
                    source_stem = source_stem[len(args.source_prefix) :]
                new_key = f"{args.comfy_prefix}{source_stem}.alpha"
                converted[new_key] = tensor.contiguous()
                kept_count += 1
            elif args.allow_other_keys:
                converted[key] = tensor.contiguous()
                kept_count += 1
            else:
                warnings.append(key)

    if converted_count == 0:
        raise ValueError("No .lora_A.weight/.lora_B.weight keys found in input.")
    if converted_count % 2:
        raise ValueError(f"Converted {converted_count} LoRA tensors; expected an even A/B count.")

    if args.base_model:
        base_shapes = load_base_shapes(args.base_model)
        missing: list[str] = []
        shape_errors: list[str] = []
        for key, tensor in converted.items():
            if not key.endswith((".lora_down.weight", ".lora_up.weight")):
                continue
            target = key.rsplit(".lora_", 1)[0] + ".weight"
            base_target = target
            if base_target not in base_shapes and args.comfy_prefix and base_target.startswith(args.comfy_prefix):
                base_target = base_target[len(args.comfy_prefix) :]
            if base_target not in base_shapes:
                missing.append(target)
                continue
            # ComfyUI supports flattened Conv1d/Conv3d LoRA matrices. Check
            # the output dimension and number of elements rather than requiring
            # the adapter tensor itself to already have the convolution shape.
            expected = base_shapes[base_target]
            if key.endswith(".lora_up.weight"):
                valid_shape = tensor.shape[0] == expected[0] and tensor.ndim >= 2
            else:
                valid_shape = tensor.shape[0] >= 1 and tensor.shape[1] == int(torch.tensor(expected[1:]).prod().item())
            if not valid_shape:
                shape_errors.append(f"{key}: adapter={tuple(tensor.shape)}, base={expected}")
        if missing or shape_errors:
            if missing:
                warnings.extend(f"missing base target: {key}" for key in missing[:20])
            if shape_errors:
                warnings.extend(shape_errors[:20])
            raise ValueError("Converted keys do not match the supplied base model:\n  " + "\n  ".join(warnings))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        save_file(converted, temporary, metadata=metadata)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return converted_count, kept_count, warnings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    converted, kept, warnings = convert(args)
    print(f"Converted {converted} LoRA tensors to ComfyUI keys.")
    if kept:
        print(f"Preserved {kept} metadata/extra tensors.")
    if warnings:
        print(f"Ignored {len(warnings)} unsupported extra keys.")
    print(f"Saved: {Path(args.output_lora).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
