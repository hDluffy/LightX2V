#!/usr/bin/env python3
"""Convert DiffSynth Wan-S2V CSV metadata for LightX2V cache building."""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Wan-S2V CSV metadata and extract video first frames."
    )
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data-training/train_data_5s"),
        help="Base directory for video, input_audio, and s2v_pose_video paths.",
    )
    parser.add_argument("--output", type=Path, default=Path("metadata.json"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("metadata_images"),
        help="Directory where reference frames are written.",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def resolve_source_path(source_root, value):
    path = Path(str(value).strip())
    return path if path.is_absolute() else source_root / path


def output_relative_path(path, output_dir):
    return Path(os.path.relpath(path, output_dir)).as_posix()


def source_relative_path(path, source_root):
    return Path(os.path.relpath(path, source_root)).as_posix()


def extract_first_frame(video_path, image_path, jpeg_quality, overwrite):
    if image_path.is_file() and not overwrite:
        return

    image_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame: {video_path}")
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise RuntimeError(f"Failed to write first frame: {image_path}")


def main():
    args = parse_args()
    metadata_csv = args.metadata_csv.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    csv.field_size_limit(os.sys.maxsize)
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required_columns = {"video", "input_audio", "prompt"}
    missing_columns = required_columns.difference(rows[0] if rows else {})
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")

    jobs = []
    for row in rows:
        video_value = row["video"].strip()
        video_path = resolve_source_path(source_root, video_value)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        image_path = image_dir / Path(video_value).with_suffix(".jpg")
        jobs.append((video_path, image_path))

    def run_job(job):
        extract_first_frame(
            *job,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite_images,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run_job, jobs))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row, (_, image_path) in zip(rows, jobs):
        audio_path = resolve_source_path(source_root, row["input_audio"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        pose_value = (row.get("s2v_pose_video") or "").strip()
        pose_path = resolve_source_path(source_root, pose_value) if pose_value else None
        if pose_path is not None and not pose_path.is_file():
            raise FileNotFoundError(f"Pose video not found: {pose_path}")

        records.append(
            {
                "prompt": row["prompt"],
                "image_path": output_relative_path(image_path, output_path.parent),
                "audio_path": source_relative_path(audio_path, source_root),
                "negative_prompt": " ",
                "src_pose_path": (
                    source_relative_path(pose_path, source_root)
                    if pose_path is not None
                    else ""
                ),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        if output_path.suffix == ".jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Reference images: {image_dir}")


if __name__ == "__main__":
    main()
