#!/usr/bin/env python3
"""Merge a Wan2.2-S2V LoRA into its base DiT checkpoint.

The script is intentionally standalone and processes the official S2V
``diffusion_pytorch_model*.safetensors`` files one shard at a time.

Example:
    python tools/merge_s2v_lora.py \
        --base-model /path/to/Wan2.2-S2V-14B \
        --lora /path/to/pytorch_lora_weights.safetensors \
        --lora-weight 0.8 \
        --output-model /path/to/Wan2.2-S2V-14B-merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LORA_SUFFIX_PAIRS = (
    (".lora_B.weight", ".lora_A.weight"),  # PEFT / LightX2V training
    (".lora.up.weight", ".lora.down.weight"),  # Diffusers
    (".lora_up.weight", ".lora_down.weight"),  # LightX2V / ComfyUI
    ("_lora.up.weight", "_lora.down.weight"),
)

# S2V LoRA training may wrap the actual DiT in ``transformer``, ``backbone``
# and/or ``dit`` modules. Official Wan2.2-S2V checkpoint keys do not contain
# these wrapper prefixes.
S2V_WRAPPER_PREFIXES = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "backbone.",
    "module.",
    "model.",
    "dit.",
)

DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class LoraPair:
    base_name: str
    up_key: str
    down_key: str
    alpha: float | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA into a Wan2.2-S2V base model and export new safetensors weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="S2V base model directory, a safetensors shard, or a quoted shard glob.",
    )
    parser.add_argument(
        "--lora",
        required=True,
        help="S2V LoRA .safetensors file, such as pytorch_lora_weights.safetensors.",
    )
    parser.add_argument(
        "--lora-weight",
        type=float,
        default=1.0,
        help="LoRA strength. The merged delta is multiplied by this value.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="Optional global network alpha. Scale is alpha/rank; embedded per-layer alpha takes precedence.",
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Output directory for a sharded/directory base, or output .safetensors for a single-file base.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("preserve", *DTYPES.keys()),
        default="preserve",
        help="Floating-point dtype of exported weights. 'preserve' keeps each base tensor dtype.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=tuple(DTYPES.keys()),
        default="fp32",
        help="Dtype used to calculate B @ A.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Merge compute device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "weights-only"),
        default="copy",
        help=(
            "For a base directory: copy creates a standalone full model; hardlink saves disk space on the same "
            "filesystem; weights-only exports only DiT shards and top-level JSON files."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file/directory.")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Export even if some LoRA layers do not match the base model. Not recommended for S2V LoRAs.",
    )
    return parser.parse_args(argv)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def find_base_shards(base_model: str) -> tuple[list[Path], Path | None]:
    """Return S2V DiT shards and the model directory, if one was supplied."""
    path = Path(base_model).expanduser()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Base model must be safetensors, got: {path}")
        return [path.resolve()], None

    if path.is_dir():
        patterns = (
            "diffusion_pytorch_model*.safetensors",
            "transformer/diffusion_pytorch_model*.safetensors",
            "transformer/model*.safetensors",
        )
        for pattern in patterns:
            matches = _unique_paths(sorted(path.glob(pattern)))
            if matches:
                return matches, path.resolve()
        raise FileNotFoundError(
            f"No Wan S2V DiT shards found under {path}. Expected diffusion_pytorch_model*.safetensors."
        )

    matches = _unique_paths(Path(item) for item in sorted(glob(str(path))))
    if not matches:
        raise FileNotFoundError(f"Base model path/glob does not exist or has no matches: {base_model}")
    invalid = [item for item in matches if not item.is_file() or item.suffix != ".safetensors"]
    if invalid:
        raise ValueError(f"Base model glob contains non-safetensors entries: {invalid[:3]}")
    return matches, None


def load_lora(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    if not path.is_file() or path.suffix != ".safetensors":
        raise FileNotFoundError(f"LoRA must be an existing .safetensors file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        weights = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    return weights, metadata


def _alpha_for_base(weights: dict[str, torch.Tensor], base_name: str) -> float | None:
    candidates = (f"{base_name}.alpha", f"{base_name}.lora_alpha")
    for key in candidates:
        if key in weights:
            value = weights[key]
            if value.numel() != 1:
                raise ValueError(f"LoRA alpha must be scalar: {key} has shape {tuple(value.shape)}")
            return float(value.item())
    return None


def collect_lora_pairs(weights: dict[str, torch.Tensor]) -> tuple[list[LoraPair], list[str]]:
    pairs: list[LoraPair] = []
    used: set[str] = set()
    incomplete: list[str] = []

    for up_key in weights:
        for up_suffix, down_suffix in LORA_SUFFIX_PAIRS:
            if not up_key.endswith(up_suffix):
                continue
            base_name = up_key[: -len(up_suffix)]
            down_key = f"{base_name}{down_suffix}"
            if down_key not in weights:
                incomplete.append(f"{up_key} (missing {down_key})")
                break
            pairs.append(
                LoraPair(
                    base_name=base_name,
                    up_key=up_key,
                    down_key=down_key,
                    alpha=_alpha_for_base(weights, base_name),
                )
            )
            used.update((up_key, down_key))
            break

    alpha_keys = {key for key in weights if key.endswith((".alpha", ".lora_alpha"))}
    unused = sorted(set(weights) - used - alpha_keys)
    if incomplete:
        raise ValueError("Incomplete LoRA pairs:\n  " + "\n  ".join(incomplete[:20]))
    if not pairs:
        raise ValueError("No supported LoRA A/B pairs were found in the LoRA checkpoint.")
    return pairs, unused


def strip_s2v_wrappers(name: str) -> str:
    previous = None
    while previous != name:
        previous = name
        for prefix in S2V_WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    return name


def lora_to_base_key(base_name: str, base_keys: set[str]) -> str | None:
    normalized = strip_s2v_wrappers(base_name)
    candidates = [normalized if normalized.endswith(".weight") else f"{normalized}.weight"]

    # Some implementations fix the typo in the official S2V key while the
    # released checkpoint intentionally uses ``casual_audio_encoder``.
    if "causal_audio_encoder" in candidates[0]:
        candidates.append(candidates[0].replace("causal_audio_encoder", "casual_audio_encoder"))

    matches = [candidate for candidate in candidates if candidate in base_keys]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous S2V key mapping for {base_name}: {matches}")
    return matches[0] if matches else None


def index_base_keys(shards: list[Path]) -> tuple[dict[str, Path], dict[Path, dict[str, str] | None]]:
    key_to_shard: dict[str, Path] = {}
    shard_metadata: dict[Path, dict[str, str] | None] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            shard_metadata[shard] = handle.metadata()
            for key in handle.keys():
                if key in key_to_shard:
                    raise ValueError(f"Duplicate base-model tensor {key!r} in {key_to_shard[key]} and {shard}")
                key_to_shard[key] = shard
    return key_to_shard, shard_metadata


def map_pairs_to_base(
    pairs: list[LoraPair],
    key_to_shard: dict[str, Path],
    allow_unmatched: bool,
) -> tuple[dict[Path, dict[str, LoraPair]], list[str]]:
    by_shard: dict[Path, dict[str, LoraPair]] = {}
    unmatched: list[str] = []
    base_keys = set(key_to_shard)

    for pair in pairs:
        base_key = lora_to_base_key(pair.base_name, base_keys)
        if base_key is None:
            unmatched.append(pair.base_name)
            continue
        shard = key_to_shard[base_key]
        shard_pairs = by_shard.setdefault(shard, {})
        if base_key in shard_pairs:
            raise ValueError(f"Multiple LoRA pairs map to the same base tensor: {base_key}")
        shard_pairs[base_key] = pair

    if unmatched and not allow_unmatched:
        preview = "\n  ".join(unmatched[:20])
        extra = f"\n  ... and {len(unmatched) - 20} more" if len(unmatched) > 20 else ""
        raise ValueError(
            f"{len(unmatched)} LoRA layers do not match the S2V base model:\n  {preview}{extra}\n"
            "Use the matching Wan2.2-S2V base model, or pass --allow-unmatched only if this is intentional."
        )
    if not by_shard:
        raise ValueError("No LoRA layers matched the S2V base model; refusing to export an unchanged model.")
    return by_shard, unmatched


def _compose_lora_delta(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    if up.ndim == 2 and down.ndim == 2:
        return up @ down

    # S2V is transformer based, but support the common LoRA Conv2d/Conv3d
    # layouts when one side uses a 1x1 (or 1x1x1) kernel.
    if up.ndim == down.ndim and up.ndim in (3, 4, 5):
        up_spatial = up.shape[2:]
        down_spatial = down.shape[2:]
        if all(size == 1 for size in up_spatial):
            return torch.einsum("or,ri...->oi...", up.reshape(up.shape[0], up.shape[1]), down)
        if all(size == 1 for size in down_spatial):
            return torch.einsum("or...,ri->oi...", up, down.reshape(down.shape[0], down.shape[1]))

    raise ValueError(f"Unsupported LoRA shapes: up={tuple(up.shape)}, down={tuple(down.shape)}")


def merge_tensor(
    base: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    strength: float,
    pair_alpha: float | None,
    global_alpha: float | None,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not base.dtype.is_floating_point:
        raise TypeError(f"Cannot merge LoRA into non-floating base tensor with dtype {base.dtype}")
    if down.ndim < 2:
        raise ValueError(f"LoRA down tensor must have at least 2 dimensions, got {tuple(down.shape)}")

    rank = down.shape[0]
    alpha = pair_alpha if pair_alpha is not None else global_alpha
    network_scale = alpha / rank if alpha is not None else 1.0
    up_compute = up.to(device=device, dtype=compute_dtype)
    down_compute = down.to(device=device, dtype=compute_dtype)
    delta = _compose_lora_delta(up_compute, down_compute)
    if delta.shape != base.shape:
        # DiffSynth/ComfyUI S2V LoRAs store Conv1d/Conv3d adapters as
        # flattened linear matrices. Restore the official Wan checkpoint
        # kernel layout after B @ A, e.g. (1280, 3072) -> (1280, 1024, 3).
        if delta.numel() == base.numel() and delta.shape[0] == base.shape[0]:
            delta = delta.reshape(base.shape)
        else:
            raise ValueError(
                f"Merged LoRA shape {tuple(delta.shape)} does not match base shape {tuple(base.shape)}"
            )
    merged = base.to(device=device, dtype=compute_dtype).add_(delta, alpha=float(strength * network_scale))
    return merged.to(device="cpu", dtype=base.dtype).contiguous()


def _output_dtype(name: str) -> torch.dtype | None:
    return None if name == "preserve" else DTYPES[name]


def _prepare_output(
    base_dir: Path | None,
    shards: list[Path],
    output_arg: str,
    copy_mode: str,
    overwrite: bool,
) -> tuple[Path, dict[Path, Path]]:
    output = Path(output_arg).expanduser().resolve()
    is_directory_output = base_dir is not None or len(shards) > 1 or output.is_dir()

    if is_directory_output:
        if output.suffix == ".safetensors":
            raise ValueError("Multiple/directory base shards require --output-model to be a directory, not a .safetensors file.")
        if base_dir is not None and (output == base_dir or base_dir in output.parents):
            raise ValueError("Output directory must not be the base directory or a child of it.")
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()

        if base_dir is not None and copy_mode in ("copy", "hardlink"):
            copy_function = shutil.copy2 if copy_mode == "copy" else os.link
            print(f"Preparing full output model ({copy_mode}): {output}", flush=True)
            shutil.copytree(base_dir, output, copy_function=copy_function)
        else:
            output.mkdir(parents=True)
            if base_dir is not None:
                for json_file in base_dir.glob("*.json"):
                    shutil.copy2(json_file, output / json_file.name)

        destinations: dict[Path, Path] = {}
        for shard in shards:
            relative = shard.relative_to(base_dir) if base_dir is not None else Path(shard.name)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destinations[shard] = destination
        return output, destinations

    if output.suffix != ".safetensors":
        output = output.with_suffix(".safetensors")
    if output == shards[0]:
        raise ValueError("Output file must differ from the base model file.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, {shards[0]: output}


def _atomic_save(tensors: dict[str, torch.Tensor], destination: Path, metadata: dict[str, str] | None) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_shards(
    shards: list[Path],
    destinations: dict[Path, Path],
    shard_pairs: dict[Path, dict[str, LoraPair]],
    lora_weights: dict[str, torch.Tensor],
    metadata: dict[Path, dict[str, str] | None],
    args: argparse.Namespace,
) -> int:
    compute_dtype = DTYPES[args.compute_dtype]
    output_dtype = _output_dtype(args.output_dtype)
    device = torch.device(args.device)
    merged_count = 0

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA merge device requested but CUDA is unavailable: {device}")

    for shard_index, shard in enumerate(shards, start=1):
        pairs = shard_pairs.get(shard, {})
        print(
            f"[{shard_index}/{len(shards)}] Loading {shard.name} "
            f"({len(pairs)} LoRA layers)",
            flush=True,
        )
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}

        for base_key, pair in pairs.items():
            try:
                tensors[base_key] = merge_tensor(
                    base=tensors[base_key],
                    up=lora_weights[pair.up_key],
                    down=lora_weights[pair.down_key],
                    strength=args.lora_weight,
                    pair_alpha=pair.alpha,
                    global_alpha=args.lora_alpha,
                    compute_dtype=compute_dtype,
                    device=device,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to merge {pair.base_name!r} into {base_key!r}: {exc}") from exc
            merged_count += 1

        if output_dtype is not None:
            tensors = {
                key: (tensor.to(output_dtype).contiguous() if tensor.dtype.is_floating_point else tensor.contiguous())
                for key, tensor in tensors.items()
            }
        else:
            tensors = {key: tensor.contiguous() for key, tensor in tensors.items()}

        destination = destinations[shard]
        print(f"[{shard_index}/{len(shards)}] Saving {destination}", flush=True)
        _atomic_save(tensors, destination, metadata[shard])
        del tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return merged_count


def write_merge_info(output: Path, args: argparse.Namespace, merged_count: int, unmatched: list[str]) -> None:
    if not output.is_dir():
        return
    info = {
        "base_model": str(Path(args.base_model).expanduser()),
        "lora": str(Path(args.lora).expanduser()),
        "lora_weight": args.lora_weight,
        "lora_alpha": args.lora_alpha,
        "merged_layers": merged_count,
        "unmatched_layers": unmatched,
        "output_dtype": args.output_dtype,
    }
    (output / "lora_merge_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shards, base_dir = find_base_shards(args.base_model)
    lora_path = Path(args.lora).expanduser().resolve()

    print(f"Base S2V shards: {len(shards)}", flush=True)
    print(f"LoRA: {lora_path}", flush=True)
    print(f"LoRA weight: {args.lora_weight}", flush=True)

    lora_weights, _ = load_lora(lora_path)
    pairs, unused = collect_lora_pairs(lora_weights)
    if unused:
        preview = ", ".join(unused[:5])
        print(f"Warning: ignoring {len(unused)} non-pair LoRA tensors: {preview}", file=sys.stderr)

    key_to_shard, metadata = index_base_keys(shards)
    shard_pairs, unmatched = map_pairs_to_base(pairs, key_to_shard, args.allow_unmatched)
    matched_count = sum(len(value) for value in shard_pairs.values())
    print(f"LoRA layers: {len(pairs)} total, {matched_count} matched, {len(unmatched)} unmatched", flush=True)

    output, destinations = _prepare_output(
        base_dir=base_dir,
        shards=shards,
        output_arg=args.output_model,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    merged_count = merge_shards(
        shards=shards,
        destinations=destinations,
        shard_pairs=shard_pairs,
        lora_weights=lora_weights,
        metadata=metadata,
        args=args,
    )
    write_merge_info(output, args, merged_count, unmatched)
    print(f"Done: merged {merged_count} LoRA layers into {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
