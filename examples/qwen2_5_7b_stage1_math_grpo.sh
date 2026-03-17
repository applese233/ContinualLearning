#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_7b_stage1_math_grpo}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_PATH}}"

EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
bash "${ROOT_DIR}/examples/qwen2_5_7b_math_grpo.sh"