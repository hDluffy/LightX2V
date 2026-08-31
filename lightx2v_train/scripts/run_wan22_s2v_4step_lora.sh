#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
if [ -n "${DIFFSYNTH_STUDIO_PATH:-}" ]; then
    export PYTHONPATH="${DIFFSYNTH_STUDIO_PATH}:${PYTHONPATH}"
fi

CONFIG=${CONFIG:-configs/train/dmd/wan22_s2v_4step_lora.yaml}

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py --config "${CONFIG}"
