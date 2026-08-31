#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-/path/to/Wan2.2-S2V-14B}
CONFIG_JSON=${CONFIG_JSON:-../configs/wan22/wan_s2v.json}
METADATA_PATH=${METADATA_PATH:-/path/to/wan22_s2v_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./data_cache/wan22_s2v}
MAX_CLIPS_PER_SAMPLE=${MAX_CLIPS_PER_SAMPLE:-1}

python scripts/build_wan22_s2v_cache.py \
    --model_path "${MODEL_PATH}" \
    --config_json "${CONFIG_JSON}" \
    --metadata_path "${METADATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_clips_per_sample "${MAX_CLIPS_PER_SAMPLE}" \
    "$@"
