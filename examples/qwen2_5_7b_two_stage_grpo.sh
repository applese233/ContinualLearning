#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

STAGE_SEQUENCE="${STAGE_SEQUENCE:-math,code}"

STAGE_SEQUENCE="${STAGE_SEQUENCE}" \
bash "${ROOT_DIR}/examples/qwen2_5_7b_multi_stage_grpo.sh"