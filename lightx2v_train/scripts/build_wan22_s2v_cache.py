import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

from lightx2v.infer import WanS2VRunner  # noqa: F401
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.base.global_var import AI_DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="Build cached Wan2.2-S2V conditions for 4-step LoRA DMD training.")
    parser.add_argument("--model_path", required=True, help="Path to Wan2.2-S2V model directory.")
    parser.add_argument("--config_json", required=True, help="Wan2.2-S2V inference JSON, usually configs/wan22/wan_s2v.json.")
    parser.add_argument("--metadata_path", required=True, nargs="+", help="JSONL/JSON/CSV metadata files.")
    parser.add_argument("--output_dir", required=True, help="Directory to write .pt cache files.")
    parser.add_argument("--prompt_column", default="prompt")
    parser.add_argument("--image_column", default="image_path")
    parser.add_argument("--audio_column", default="audio_path")
    parser.add_argument("--negative_prompt_column", default="negative_prompt")
    parser.add_argument("--src_pose_column", default="src_pose_path")
    parser.add_argument("--seed_column", default="seed")
    parser.add_argument("--default_negative_prompt", default=" ")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_clips_per_sample", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_rows(metadata_paths):
    for metadata_path in metadata_paths:
        path = Path(metadata_path)
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield path, json.loads(line)
            continue
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if isinstance(records, dict):
                records = records.get("data", records.get("samples", [records]))
            for row in records:
                yield path, row
            continue
        csv.field_size_limit(sys.maxsize)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield path, row


def resolve_path(metadata_path, value):
    if value is None or str(value).strip() == "":
        return ""
    path = Path(str(value).strip())
    if path.is_absolute():
        return str(path)
    return str((metadata_path.parent / path).resolve())


def build_runner(args):
    torch.set_grad_enabled(False)
    seed_all(args.seed)
    config_args = SimpleNamespace(
        model_cls="wan2.2_s2v",
        task="s2v",
        model_path=args.model_path,
        config_json=args.config_json,
        support_tasks=[],
        use_prompt_enhancer=False,
    )
    config = set_config(config_args)
    validate_config_paths(config)
    if config.get("parallel"):
        raise RuntimeError("S2V cache builder only supports single-process encoding. Use a sharded metadata split for multi-GPU cache construction.")
    runner = RUNNER_REGISTER["wan2.2_s2v"](config)
    runner.text_encoders = runner.load_text_encoder()
    runner.vae_encoder, runner.vae_decoder = runner.load_vae()
    runner.audio_encoder = runner.load_audio_encoder()
    runner.run_input_encoder = runner._run_input_encoder_local_s2v
    runner.config.lock()
    return runner


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def encode_cache_items(runner, row, metadata_path, args):
    image_path = resolve_path(metadata_path, row.get(args.image_column))
    audio_path = resolve_path(metadata_path, row.get(args.audio_column))
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"image_path not found: {image_path}")
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio_path not found: {audio_path}")

    input_info = init_empty_input_info("s2v")
    update_input_info_from_dict(
        input_info,
        {
            "prompt": str(row.get(args.prompt_column, "")),
            "negative_prompt": str(row.get(args.negative_prompt_column, args.default_negative_prompt) or args.default_negative_prompt),
            "image_path": image_path,
            "audio_path": audio_path,
            "src_pose_path": resolve_path(metadata_path, row.get(args.src_pose_column, "")),
            "seed": int(row.get(args.seed_column, args.seed) or args.seed),
        },
    )

    runner.input_info = input_info
    inputs = runner._run_input_encoder_local_s2v()
    height, width = int(inputs["height"]), int(inputs["width"])
    motion_frames = int(runner.config["motion_frames"])
    infer_frames = int(runner.config["infer_frames"])
    lat_motion_frames = (motion_frames + 3) // 4
    lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
    latent_shape = (16, lat_target_frames, height // 8, width // 8)

    offload = runner.config.get("cpu_offload", False)
    if offload:
        runner.vae_encoder.to_cuda()
    with torch.no_grad(), torch.amp.autocast(str(AI_DEVICE), dtype=runner.vae_encoder.dtype):
        ref_latents = runner.vae_encoder.encode(inputs["ref_pixel_values"]).unsqueeze(0)
        motion_latents = runner.vae_encoder.encode(inputs["motion_latents"]).unsqueeze(0)
    if offload:
        runner.vae_encoder.to_cpu()

    num_repeat = int(inputs["num_repeat"])
    max_clips = min(num_repeat, max(1, int(args.max_clips_per_sample)))
    pose_conds = None
    src_pose_path = getattr(input_info, "src_pose_path", "") or ""
    if src_pose_path and os.path.isfile(src_pose_path):
        pose_conds = runner.load_pose_cond(src_pose_path, num_repeat, infer_frames, height, width)

    for clip_idx in range(max_clips):
        if pose_conds is not None:
            cond_latents = pose_conds[clip_idx].to(dtype=runner.param_dtype, device=AI_DEVICE)
        else:
            cond_latents = runner.build_cond_latents(height, width)
        left_idx = clip_idx * infer_frames
        right_idx = left_idx + infer_frames
        audio_input = inputs["audio_emb"][..., left_idx:right_idx]
        yield {
            "prompt": input_info.prompt,
            "negative_prompt": input_info.negative_prompt,
            "image_path": image_path,
            "audio_path": audio_path,
            "src_pose_path": src_pose_path,
            "height": height,
            "width": width,
            "clip_index": clip_idx,
            "latent_shape": torch.tensor(latent_shape, dtype=torch.long),
            "context": to_cpu(inputs["context"]),
            "context_null": to_cpu(inputs["context_null"]),
            "ref_latents": to_cpu(ref_latents),
            "motion_latents": to_cpu(motion_latents.clone()),
            "cond_latents": to_cpu(cond_latents),
            "audio_input": to_cpu(audio_input),
            "motion_frames": torch.tensor([motion_frames, lat_motion_frames], dtype=torch.long),
            "drop_motion_frames": bool(runner.config["drop_first_motion"] and clip_idx == 0),
            "add_last_motion": 2,
        }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = build_runner(args)

    total = 0
    for row_idx, (metadata_path, row) in enumerate(iter_rows(args.metadata_path)):
        if args.max_samples is not None and row_idx >= args.max_samples:
            break
        try:
            for item in encode_cache_items(runner, row, metadata_path, args):
                cache_path = output_dir / f"sample_{row_idx:06d}_clip_{int(item['clip_index']):03d}.pt"
                if cache_path.exists() and not args.overwrite:
                    logger.info("skip existing {}", cache_path)
                    continue
                torch.save(item, cache_path)
                total += 1
                logger.info("saved {}", cache_path)
        except Exception as exc:
            logger.warning("failed row {} from {}: {}", row_idx, metadata_path, exc)
    logger.info("finished, wrote {} cache files to {}", total, output_dir)


if __name__ == "__main__":
    main()
