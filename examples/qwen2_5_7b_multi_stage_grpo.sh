#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Default sequence keeps the existing math -> code workflow.
# Alternative examples:
#   STAGE_SEQUENCE=code,math bash examples/qwen2_5_7b_multi_stage_grpo.sh
#   STAGE_SEQUENCE=math,code,math bash examples/qwen2_5_7b_multi_stage_grpo.sh
# Stage-specific overrides are also supported, for example:
#   STAGE1_TOTAL_EPOCHS=1 STAGE2_TOTAL_EPOCHS=3 bash examples/qwen2_5_7b_multi_stage_grpo.sh
#   STAGE2_ACTOR_LR=3.0e-7 STAGE2_MAX_RESPONSE_LENGTH=1024 bash examples/qwen2_5_7b_multi_stage_grpo.sh
# Validation/save presets:
#   Current defaults are the medium preset.
#   Light:  STAGE1_PRIMARY_VAL_LIMIT=128 STAGE2_PRIMARY_VAL_LIMIT=64 STAGE1_VAL_FREQ=100 STAGE2_VAL_FREQ=100 STAGE1_SAVE_FREQ=100 STAGE2_SAVE_FREQ=100
#   Medium: STAGE1_PRIMARY_VAL_LIMIT=256 STAGE2_PRIMARY_VAL_LIMIT=128 STAGE1_VAL_FREQ=50  STAGE2_VAL_FREQ=50  STAGE1_SAVE_FREQ=50  STAGE2_SAVE_FREQ=50
#   Heavy:  STAGE1_PRIMARY_VAL_LIMIT=512 STAGE2_PRIMARY_VAL_LIMIT=256 STAGE1_VAL_FREQ=50  STAGE2_VAL_FREQ=50  STAGE1_SAVE_FREQ=50  STAGE2_SAVE_FREQ=50

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/multi_stage_grpo}"
PREPARE_DATA="${PREPARE_DATA:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
STAGE_SEQUENCE="${STAGE_SEQUENCE:-math,code}"
PROJECT_NAME="${PROJECT_NAME:-ContinueLearning}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-best}"
CHECKPOINT_DIR_ROOT="${CHECKPOINT_DIR_ROOT:-/disk4/yiran/yaoqi/ContinualLearning/checkpoints/${PROJECT_NAME}}"
MATH_STAGE_BASH="${MATH_STAGE_BASH:-${ROOT_DIR}/examples/qwen2_5_7b_math_grpo.sh}"
CODE_STAGE_BASH="${CODE_STAGE_BASH:-${ROOT_DIR}/examples/qwen2_5_7b_code_grpo.sh}"

STAGE1_EXPERIMENT_NAME="${STAGE1_EXPERIMENT_NAME:-qwen2_5_7b_stage1_math_grpo}"
STAGE2_EXPERIMENT_NAME="${STAGE2_EXPERIMENT_NAME:-qwen2_5_7b_stage2_code_grpo}"
STAGE3_EXPERIMENT_NAME="${STAGE3_EXPERIMENT_NAME:-qwen2_5_7b_stage3_math_grpo}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-qwen2_5_7b_multi_stage_grpo}"

# Editable per-stage defaults.
# These are prefilled for the default math -> code schedule.
# You can still override them at runtime, for example:
#   STAGE2_ACTOR_LR=3.0e-7 bash examples/qwen2_5_7b_multi_stage_grpo.sh
STAGE1_TOTAL_EPOCHS="${STAGE1_TOTAL_EPOCHS:-2}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-400}"
STAGE1_VAL_FREQ="${STAGE1_VAL_FREQ:-50}"
STAGE1_SAVE_FREQ="${STAGE1_SAVE_FREQ:-50}"
STAGE1_SAVE_LIMIT="${STAGE1_SAVE_LIMIT:-2}"
STAGE1_FIND_LAST_CHECKPOINT="${STAGE1_FIND_LAST_CHECKPOINT:-true}"
STAGE1_ACTOR_GLOBAL_BATCH_SIZE="${STAGE1_ACTOR_GLOBAL_BATCH_SIZE:-32}"
STAGE1_ROLLOUT_BATCH_SIZE="${STAGE1_ROLLOUT_BATCH_SIZE:-32}"
STAGE1_VAL_BATCH_SIZE="${STAGE1_VAL_BATCH_SIZE:-32}"
STAGE1_MICRO_BATCH_SIZE_UPDATE="${STAGE1_MICRO_BATCH_SIZE_UPDATE:-4}"
STAGE1_MICRO_BATCH_SIZE_EXPERIENCE="${STAGE1_MICRO_BATCH_SIZE_EXPERIENCE:-4}"
STAGE1_GPU_MEMORY_UTILIZATION="${STAGE1_GPU_MEMORY_UTILIZATION:-0.8}"
STAGE1_MAX_PROMPT_LENGTH="${STAGE1_MAX_PROMPT_LENGTH:-1536}"
STAGE1_MAX_RESPONSE_LENGTH="${STAGE1_MAX_RESPONSE_LENGTH:-1024}"
STAGE1_ACTOR_LR="${STAGE1_ACTOR_LR:-1.0e-6}"
STAGE1_PRIMARY_VAL_LIMIT="${STAGE1_PRIMARY_VAL_LIMIT:-256}"
STAGE1_AIME25_VAL_LIMIT="${STAGE1_AIME25_VAL_LIMIT:-30}"
STAGE1_MBPP_VAL_LIMIT="${STAGE1_MBPP_VAL_LIMIT:-16}"
STAGE1_LIVECODEBENCH_VAL_LIMIT="${STAGE1_LIVECODEBENCH_VAL_LIMIT:-16}"

STAGE2_TOTAL_EPOCHS="${STAGE2_TOTAL_EPOCHS:-2}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-400}"
STAGE2_VAL_FREQ="${STAGE2_VAL_FREQ:-50}"
STAGE2_SAVE_FREQ="${STAGE2_SAVE_FREQ:-50}"
STAGE2_SAVE_LIMIT="${STAGE2_SAVE_LIMIT:-2}"
STAGE2_FIND_LAST_CHECKPOINT="${STAGE2_FIND_LAST_CHECKPOINT:-true}"
STAGE2_ACTOR_GLOBAL_BATCH_SIZE="${STAGE2_ACTOR_GLOBAL_BATCH_SIZE:-32}"
STAGE2_ROLLOUT_BATCH_SIZE="${STAGE2_ROLLOUT_BATCH_SIZE:-32}"
STAGE2_VAL_BATCH_SIZE="${STAGE2_VAL_BATCH_SIZE:-32}"
STAGE2_MICRO_BATCH_SIZE_UPDATE="${STAGE2_MICRO_BATCH_SIZE_UPDATE:-4}"
STAGE2_MICRO_BATCH_SIZE_EXPERIENCE="${STAGE2_MICRO_BATCH_SIZE_EXPERIENCE:-4}"
STAGE2_GPU_MEMORY_UTILIZATION="${STAGE2_GPU_MEMORY_UTILIZATION:-0.8}"
STAGE2_MAX_PROMPT_LENGTH="${STAGE2_MAX_PROMPT_LENGTH:-1536}"
STAGE2_MAX_RESPONSE_LENGTH="${STAGE2_MAX_RESPONSE_LENGTH:-1024}"
STAGE2_ACTOR_LR="${STAGE2_ACTOR_LR:-5.0e-7}"
STAGE2_PRIMARY_VAL_LIMIT="${STAGE2_PRIMARY_VAL_LIMIT:-128}"
STAGE2_APPS_VAL_LIMIT="${STAGE2_APPS_VAL_LIMIT:-16}"
STAGE2_HUMANEVAL_VAL_LIMIT="${STAGE2_HUMANEVAL_VAL_LIMIT:-16}"
STAGE2_LIVECODEBENCH_VAL_LIMIT="${STAGE2_LIVECODEBENCH_VAL_LIMIT:-16}"
STAGE2_AIME25_VAL_LIMIT="${STAGE2_AIME25_VAL_LIMIT:-30}"

STAGE3_TOTAL_EPOCHS="${STAGE3_TOTAL_EPOCHS:-2}"
STAGE3_MAX_STEPS="${STAGE3_MAX_STEPS:-}"
STAGE3_VAL_FREQ="${STAGE3_VAL_FREQ:-50}"
STAGE3_SAVE_FREQ="${STAGE3_SAVE_FREQ:-50}"
STAGE3_SAVE_LIMIT="${STAGE3_SAVE_LIMIT:-2}"
STAGE3_FIND_LAST_CHECKPOINT="${STAGE3_FIND_LAST_CHECKPOINT:-true}"
STAGE3_ACTOR_GLOBAL_BATCH_SIZE="${STAGE3_ACTOR_GLOBAL_BATCH_SIZE:-32}"
STAGE3_ROLLOUT_BATCH_SIZE="${STAGE3_ROLLOUT_BATCH_SIZE:-32}"
STAGE3_VAL_BATCH_SIZE="${STAGE3_VAL_BATCH_SIZE:-32}"
STAGE3_MICRO_BATCH_SIZE_UPDATE="${STAGE3_MICRO_BATCH_SIZE_UPDATE:-4}"
STAGE3_MICRO_BATCH_SIZE_EXPERIENCE="${STAGE3_MICRO_BATCH_SIZE_EXPERIENCE:-4}"
STAGE3_GPU_MEMORY_UTILIZATION="${STAGE3_GPU_MEMORY_UTILIZATION:-0.8}"
STAGE3_MAX_PROMPT_LENGTH="${STAGE3_MAX_PROMPT_LENGTH:-1536}"
STAGE3_MAX_RESPONSE_LENGTH="${STAGE3_MAX_RESPONSE_LENGTH:-1024}"
STAGE3_ACTOR_LR="${STAGE3_ACTOR_LR:-1.0e-6}"
STAGE3_PRIMARY_VAL_LIMIT="${STAGE3_PRIMARY_VAL_LIMIT:-256}"
STAGE3_AIME25_VAL_LIMIT="${STAGE3_AIME25_VAL_LIMIT:-30}"
STAGE3_MBPP_VAL_LIMIT="${STAGE3_MBPP_VAL_LIMIT:-32}"
STAGE3_LIVECODEBENCH_VAL_LIMIT="${STAGE3_LIVECODEBENCH_VAL_LIMIT:-32}"

# Dataset preparation limits
DEEPSCALER_TRAIN_LIMIT="${DEEPSCALER_TRAIN_LIMIT:-}"
DEEPSCALER_VAL_LIMIT="${DEEPSCALER_VAL_LIMIT:-${STAGE1_PRIMARY_VAL_LIMIT}}"
DEEPCODER_TRAIN_LIMIT="${DEEPCODER_TRAIN_LIMIT:-}"
DEEPCODER_VAL_LIMIT="${DEEPCODER_VAL_LIMIT:-${STAGE2_PRIMARY_VAL_LIMIT}}"
APPS_TRAIN_LIMIT="${APPS_TRAIN_LIMIT:-5000}"
AIME25_VAL_LIMIT="${AIME25_VAL_LIMIT:-}"
MBPP_TEST_LIMIT="${MBPP_TEST_LIMIT:-}"
APPS_TEST_LIMIT="${APPS_TEST_LIMIT:-}"
HUMANEVAL_VAL_LIMIT="${HUMANEVAL_VAL_LIMIT:-}"
LIVECODEBENCH_VAL_LIMIT="${LIVECODEBENCH_VAL_LIMIT:-}"

STAGE_OVERRIDE_KEYS=(
    BASE_MODEL_PATH
    PREV_STAGE_EXPERIMENT_NAME
    PREV_STAGE_CHECKPOINT_ROOT
    TOTAL_EPOCHS
    MAX_STEPS
    VAL_FREQ
    SAVE_FREQ
    SAVE_LIMIT
    SAVE_CHECKPOINT_PATH
    LOAD_CHECKPOINT_PATH
    FIND_LAST_CHECKPOINT
    ACTOR_GLOBAL_BATCH_SIZE
    ROLLOUT_BATCH_SIZE
    VAL_BATCH_SIZE
    MICRO_BATCH_SIZE_UPDATE
    MICRO_BATCH_SIZE_EXPERIENCE
    GPU_MEMORY_UTILIZATION
    MAX_PROMPT_LENGTH
    MAX_RESPONSE_LENGTH
    ACTOR_LR
    PRIMARY_VAL_LIMIT
    AIME25_VAL_LIMIT
    MBPP_VAL_LIMIT
    APPS_VAL_LIMIT
    HUMANEVAL_VAL_LIMIT
    LIVECODEBENCH_VAL_LIMIT
)

get_stage_override_value() {
    local stage_index="${1}"
    local key="${2}"
    local override_var="STAGE${stage_index}_${key}"
    printf '%s\n' "${!override_var:-}"
}

append_stage_override_envs() {
    local stage_index="${1}"
    local env_array_name="${2}"
    local key
    local value
    for key in "${STAGE_OVERRIDE_KEYS[@]}"; do
        value=$(get_stage_override_value "${stage_index}" "${key}")
        if [[ -n "${value}" ]]; then
            eval "${env_array_name}+=(\"${key}=${value}\")"
        fi
    done
}

get_stage_experiment_name() {
    local stage_index="${1}"
    local stage_type="${2}"
    local stage_name_var="STAGE${stage_index}_EXPERIMENT_NAME"
    local stage_name="${!stage_name_var:-}"

    if [[ -n "${stage_name}" ]]; then
        printf '%s\n' "${stage_name}"
        return 0
    fi

    printf '%s_%02d_%s\n' "${EXPERIMENT_PREFIX}" "${stage_index}" "${stage_type}"
}

if [[ "${PREPARE_DATA}" == "1" ]]; then
    PREPARE_ARGS=(
        "--output_dir" "${DATA_DIR}"
        "--deepscaler_val_limit" "${DEEPSCALER_VAL_LIMIT}"
        "--deepcoder_val_limit" "${DEEPCODER_VAL_LIMIT}"
        "--apps_train_limit" "${APPS_TRAIN_LIMIT}"
    )
    [[ -n "${DEEPSCALER_TRAIN_LIMIT}" ]] && PREPARE_ARGS+=("--deepscaler_train_limit" "${DEEPSCALER_TRAIN_LIMIT}")
    [[ -n "${DEEPCODER_TRAIN_LIMIT}" ]] && PREPARE_ARGS+=("--deepcoder_train_limit" "${DEEPCODER_TRAIN_LIMIT}")
    [[ -n "${AIME25_VAL_LIMIT}" ]] && PREPARE_ARGS+=("--aime25_val_limit" "${AIME25_VAL_LIMIT}")
    [[ -n "${MBPP_TEST_LIMIT}" ]] && PREPARE_ARGS+=("--mbpp_test_limit" "${MBPP_TEST_LIMIT}")
    [[ -n "${APPS_TEST_LIMIT}" ]] && PREPARE_ARGS+=("--apps_test_limit" "${APPS_TEST_LIMIT}")
    [[ -n "${HUMANEVAL_VAL_LIMIT}" ]] && PREPARE_ARGS+=("--humaneval_val_limit" "${HUMANEVAL_VAL_LIMIT}")
    [[ -n "${LIVECODEBENCH_VAL_LIMIT}" ]] && PREPARE_ARGS+=("--livecodebench_val_limit" "${LIVECODEBENCH_VAL_LIMIT}")

    python3 "${ROOT_DIR}/scripts/prepare_two_stage_datasets.py" "${PREPARE_ARGS[@]}"
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
    exit 0
fi

IFS=',' read -r -a STAGES <<< "${STAGE_SEQUENCE}"

LAST_EXPERIMENT_NAME=""
for stage_idx in "${!STAGES[@]}"; do
    stage_number=$((stage_idx + 1))
    stage_type="${STAGES[stage_idx]//[[:space:]]/}"
    experiment_name=$(get_stage_experiment_name "${stage_number}" "${stage_type}")
    base_model_path=$(get_stage_override_value "${stage_number}" "BASE_MODEL_PATH")
    previous_experiment_name=$(get_stage_override_value "${stage_number}" "PREV_STAGE_EXPERIMENT_NAME")
    previous_checkpoint_root=$(get_stage_override_value "${stage_number}" "PREV_STAGE_CHECKPOINT_ROOT")
    stage_env=(
        "EXPERIMENT_NAME=${experiment_name}"
        "PROJECT_NAME=${PROJECT_NAME}"
        "DATA_DIR=${DATA_DIR}"
        "NNODES=${NNODES}"
        "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
        "CHECKPOINT_DIR_ROOT=${CHECKPOINT_DIR_ROOT}"
        "CHECKPOINT_SELECTION=${CHECKPOINT_SELECTION}"
    )

    if (( stage_number == 1 )); then
        [[ -z "${base_model_path}" ]] && base_model_path="${MODEL_PATH}"
    else
        [[ -z "${previous_experiment_name}" ]] && previous_experiment_name="${LAST_EXPERIMENT_NAME}"
    fi

    [[ -n "${base_model_path}" ]] && stage_env+=("BASE_MODEL_PATH=${base_model_path}")
    [[ -n "${previous_experiment_name}" ]] && stage_env+=("PREV_STAGE_EXPERIMENT_NAME=${previous_experiment_name}")
    [[ -n "${previous_checkpoint_root}" ]] && stage_env+=("PREV_STAGE_CHECKPOINT_ROOT=${previous_checkpoint_root}")
    append_stage_override_envs "${stage_number}" stage_env

    case "${stage_type}" in
        math)
            env "${stage_env[@]}" bash "${MATH_STAGE_BASH}"
            ;;
        code)
            env "${stage_env[@]}" bash "${CODE_STAGE_BASH}"
            ;;
        *)
            echo "Unsupported stage type: ${stage_type}. Expected one of: math, code." >&2
            exit 1
            ;;
    esac

    LAST_EXPERIMENT_NAME="${experiment_name}"
done