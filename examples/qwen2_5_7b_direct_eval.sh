#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${CONDA_PREFIX}/lib:"*) ;;
        *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/disk4/yiran/yaoqi/checkpoints/ContinualLearning}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/multi_stage_grpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/qwen2_5_7b_direct_eval}"
DATASETS="${DATASETS:-deepscaler aime24 aime25 amc math500 minerva olympiadbench deepcoder humaneval humanevalplus livecodebench mbpp apps taco}"
SELECTION="${SELECTION:-all}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
CODE_REWARD_WORKERS="${CODE_REWARD_WORKERS:-8}"
CODE_REWARD_MAX_TESTS="${CODE_REWARD_MAX_TESTS:-8}"
PROJECT_NAME="${PROJECT_NAME:-ContinualLearningDirectEval}"
SAMPLE_DETAILS="${SAMPLE_DETAILS:-5}"
VAL_LIMIT_ARG=()
CUDA_VISIBLE_DEVICES_ARG=()
SAVE_FULL_RESPONSE_ARG=()
SAVE_ALL_GENERATIONS_ARG=()

if [[ -n "${VAL_LIMIT:-}" ]]; then
    VAL_LIMIT_ARG=(--val_limit "${VAL_LIMIT}")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES_ARG=(--cuda_devices "${CUDA_VISIBLE_DEVICES}")
fi

if [[ "${SAVE_FULL_RESPONSE:-0}" == "1" ]]; then
    SAVE_FULL_RESPONSE_ARG=(--save-full-response)
fi

if [[ "${SAVE_ALL_GENERATIONS:-0}" == "1" ]]; then
    SAVE_ALL_GENERATIONS_ARG=(--save-all-generations)
fi

python3 "${ROOT_DIR}/scripts/direct_batch_eval_vllm.py" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --data-dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_ROOT}" \
    --datasets "${DATASETS}" \
    --selection "${SELECTION}" \
    --batch-size "${BATCH_SIZE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --code-reward-workers "${CODE_REWARD_WORKERS}" \
    --code-reward-max-tests "${CODE_REWARD_MAX_TESTS}" \
    --project-name "${PROJECT_NAME}" \
    --save-generations \
    --sample-details "${SAMPLE_DETAILS}" \
    "${SAVE_ALL_GENERATIONS_ARG[@]}" \
    "${SAVE_FULL_RESPONSE_ARG[@]}" \
    "${VAL_LIMIT_ARG[@]}" \
    "${CUDA_VISIBLE_DEVICES_ARG[@]}" \
    "$@"