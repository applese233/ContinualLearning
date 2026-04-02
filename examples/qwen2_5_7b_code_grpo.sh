#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT_DIR}/examples/qwen2_5_7b_stage_common.sh"

# Editable defaults
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/multi_stage_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_7b_code_grpo}"
MODEL_PATH="${MODEL_PATH:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_PATH}}"
PREV_STAGE_EXPERIMENT_NAME="${PREV_STAGE_EXPERIMENT_NAME:-}"
PREV_STAGE_CHECKPOINT_ROOT="${PREV_STAGE_CHECKPOINT_ROOT:-}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-best}"
PROJECT_NAME="${PROJECT_NAME:-easy_r1}"
CHECKPOINT_DIR_ROOT="${CHECKPOINT_DIR_ROOT:-/hy-tmp/checkpoints/${PROJECT_NAME}}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-}"
VAL_FREQ="${VAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-2}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-${CHECKPOINT_DIR_ROOT}/${EXPERIMENT_NAME}}"
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"
FIND_LAST_CHECKPOINT="${FIND_LAST_CHECKPOINT:-true}"

ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_UPDATE="${MICRO_BATCH_SIZE_UPDATE:-4}"
MICRO_BATCH_SIZE_EXPERIENCE="${MICRO_BATCH_SIZE_EXPERIENCE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ACTOR_LR="${ACTOR_LR:-5.0e-7}"

# Eval dataset limits loaded at runtime. Leave empty for full datasets.
PRIMARY_VAL_LIMIT="${PRIMARY_VAL_LIMIT:-128}"
APPS_VAL_LIMIT="${APPS_VAL_LIMIT:-32}"
HUMANEVAL_VAL_LIMIT="${HUMANEVAL_VAL_LIMIT:-32}"
LIVECODEBENCH_VAL_LIMIT="${LIVECODEBENCH_VAL_LIMIT:-32}"
AIME25_VAL_LIMIT="${AIME25_VAL_LIMIT:-30}"

# Add raw OmegaConf CLI args here if needed, for example:
# EXTRA_ARGS=("trainer.val_before_train=false" "worker.rollout.n=2")
# If you want to resume from a specific checkpoint, set:
# LOAD_CHECKPOINT_PATH="/path/to/.../global_step_XX"
EXTRA_ARGS=()

if [[ -z "${BASE_MODEL_PATH}" && -z "${PREV_STAGE_EXPERIMENT_NAME}" && -z "${PREV_STAGE_CHECKPOINT_ROOT}" ]]; then
    BASE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
fi

BASE_MODEL_PATH=$(resolve_base_model_path "${BASE_MODEL_PATH}" "${PREV_STAGE_EXPERIMENT_NAME}" "${PREV_STAGE_CHECKPOINT_ROOT}" "${CHECKPOINT_SELECTION}" "${PROJECT_NAME}")

TRAIN_ARGS=(
    "config=${ROOT_DIR}/examples/configs/qwen2_5_7b_stage2_code_grpo.yaml"
    "worker.actor.model.model_path=${BASE_MODEL_PATH}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
)

[[ -n "${TOTAL_EPOCHS}" ]] && TRAIN_ARGS+=("trainer.total_epochs=${TOTAL_EPOCHS}")
[[ -n "${MAX_STEPS}" ]] && TRAIN_ARGS+=("trainer.max_steps=${MAX_STEPS}")
[[ -n "${VAL_FREQ}" ]] && TRAIN_ARGS+=("trainer.val_freq=${VAL_FREQ}")
[[ -n "${SAVE_FREQ}" ]] && TRAIN_ARGS+=("trainer.save_freq=${SAVE_FREQ}")
[[ -n "${SAVE_LIMIT}" ]] && TRAIN_ARGS+=("trainer.save_limit=${SAVE_LIMIT}")
[[ -n "${SAVE_CHECKPOINT_PATH}" ]] && TRAIN_ARGS+=("trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH}")
[[ -n "${LOAD_CHECKPOINT_PATH}" ]] && TRAIN_ARGS+=("trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH}")
[[ -n "${FIND_LAST_CHECKPOINT}" ]] && TRAIN_ARGS+=("trainer.find_last_checkpoint=${FIND_LAST_CHECKPOINT}")
[[ -n "${ACTOR_GLOBAL_BATCH_SIZE}" ]] && TRAIN_ARGS+=("worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}")
[[ -n "${ROLLOUT_BATCH_SIZE}" ]] && TRAIN_ARGS+=("data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}")
[[ -n "${VAL_BATCH_SIZE}" ]] && TRAIN_ARGS+=("data.val_batch_size=${VAL_BATCH_SIZE}")
[[ -n "${MICRO_BATCH_SIZE_UPDATE}" ]] && TRAIN_ARGS+=("worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_UPDATE}")
[[ -n "${MICRO_BATCH_SIZE_EXPERIENCE}" ]] && TRAIN_ARGS+=("worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_EXPERIENCE}")
[[ -n "${GPU_MEMORY_UTILIZATION}" ]] && TRAIN_ARGS+=("worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}")
[[ -n "${MAX_PROMPT_LENGTH}" ]] && TRAIN_ARGS+=("data.max_prompt_length=${MAX_PROMPT_LENGTH}")
[[ -n "${MAX_RESPONSE_LENGTH}" ]] && TRAIN_ARGS+=("data.max_response_length=${MAX_RESPONSE_LENGTH}")
[[ -n "${ACTOR_LR}" ]] && TRAIN_ARGS+=("worker.actor.optim.lr=${ACTOR_LR}")

TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

DATA_DIR="${DATA_DIR}" \
PRIMARY_VAL_LIMIT="${PRIMARY_VAL_LIMIT}" \
APPS_VAL_LIMIT="${APPS_VAL_LIMIT}" \
HUMANEVAL_VAL_LIMIT="${HUMANEVAL_VAL_LIMIT}" \
LIVECODEBENCH_VAL_LIMIT="${LIVECODEBENCH_VAL_LIMIT}" \
AIME25_VAL_LIMIT="${AIME25_VAL_LIMIT}" \
python3 -m verl.trainer.main "${TRAIN_ARGS[@]}"