#!/bin/bash

resolve_base_model_path() {
    local model_path="${1:-}"
    local previous_experiment_name="${2:-}"
    local previous_checkpoint_root="${3:-}"
    local checkpoint_selection="${4:-best}"
    local project_name="${5:-easy_r1}"

    if [[ -n "${model_path}" ]]; then
        printf '%s\n' "${model_path}"
        return 0
    fi

    if [[ -z "${previous_checkpoint_root}" && -z "${previous_experiment_name}" ]]; then
        printf '%s\n' ""
        return 0
    fi

    if [[ -z "${previous_checkpoint_root}" ]]; then
        previous_checkpoint_root="/hy-tmp/checkpoints/${project_name}/${previous_experiment_name}"
    fi

    local previous_actor_dir
    previous_actor_dir=$(python3 "${ROOT_DIR}/scripts/resolve_checkpoint.py" --checkpoint_root "${previous_checkpoint_root}" --selection "${checkpoint_selection}" --artifact actor)
    if [[ ! -d "${previous_actor_dir}/huggingface" ]]; then
        python3 "${ROOT_DIR}/scripts/model_merger.py" --local_dir "${previous_actor_dir}"
    fi

    printf '%s\n' "${previous_actor_dir}/huggingface"
}