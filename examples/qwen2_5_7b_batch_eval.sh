#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/disk4/yiran/yaoqi/checkpoints/ContinualLearning}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/multi_stage_grpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/qwen2_5_7b_batch_eval}"
DATASETS="${DATASETS:-deepscaler aime25 deepcoder humaneval livecodebench mbpp apps}"
SELECTION="${SELECTION:-all}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
CODE_REWARD_WORKERS="${CODE_REWARD_WORKERS:-8}"
CODE_REWARD_MAX_TESTS="${CODE_REWARD_MAX_TESTS:-8}"
PROJECT_NAME="${PROJECT_NAME:-ContinualLearningEval}"
VAL_LIMIT_ARG=()
CUDA_VISIBLE_DEVICES_ARG=()

if [[ -n "${VAL_LIMIT:-}" ]]; then
    VAL_LIMIT_ARG=(--val_limit "${VAL_LIMIT}")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES_ARG=(--cuda_devices "${CUDA_VISIBLE_DEVICES}")
fi

python3 "${ROOT_DIR}/scripts/batch_eval_models.py" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --data-dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_ROOT}" \
    --datasets "${DATASETS}" \
    --selection "${SELECTION}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --code-reward-workers "${CODE_REWARD_WORKERS}" \
    --code-reward-max-tests "${CODE_REWARD_MAX_TESTS}" \
    --project-name "${PROJECT_NAME}" \
    "${VAL_LIMIT_ARG[@]}" \
    "${CUDA_VISIBLE_DEVICES_ARG[@]}" \
    "$@"