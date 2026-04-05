#!/bin/bash

has_huggingface_model_weights() {
    local hf_dir="${1:-}"

    [[ -f "${hf_dir}/model.safetensors" ]] && return 0
    [[ -f "${hf_dir}/model.safetensors.index.json" ]] && return 0
    [[ -f "${hf_dir}/pytorch_model.bin" ]] && return 0
    [[ -f "${hf_dir}/pytorch_model.bin.index.json" ]] && return 0

    return 1
}

resolve_base_model_path() {
    local model_path="${1:-}"
    local previous_experiment_name="${2:-}"
    local previous_checkpoint_root="${3:-}"
    local checkpoint_selection="${4:-best}"
    local project_name="${5:-easy_r1}"
    local checkpoint_dir_root="${CHECKPOINT_DIR_ROOT:-/hy-tmp/checkpoints/${project_name}}"

    if [[ -n "${model_path}" ]]; then
        printf '%s\n' "${model_path}"
        return 0
    fi

    if [[ -z "${previous_checkpoint_root}" && -z "${previous_experiment_name}" ]]; then
        printf '%s\n' ""
        return 0
    fi

    if [[ -z "${previous_checkpoint_root}" ]]; then
        previous_checkpoint_root="${checkpoint_dir_root}/${previous_experiment_name}"
    fi

    if [[ ! -f "${previous_checkpoint_root}/checkpoint_tracker.json" ]]; then
        echo "Checkpoint tracker not found: ${previous_checkpoint_root}/checkpoint_tracker.json" >&2
        return 1
    fi

    local previous_actor_dir
    previous_actor_dir=$(python3 "${ROOT_DIR}/scripts/resolve_checkpoint.py" --checkpoint_root "${previous_checkpoint_root}" --selection "${checkpoint_selection}" --artifact actor)
    if [[ -z "${previous_actor_dir}" || ! -d "${previous_actor_dir}" ]]; then
        echo "Resolved actor checkpoint directory is invalid: ${previous_actor_dir}" >&2
        return 1
    fi

    if [[ ! -d "${previous_actor_dir}/huggingface" ]] || ! has_huggingface_model_weights "${previous_actor_dir}/huggingface"; then
        python3 "${ROOT_DIR}/scripts/model_merger.py" --local_dir "${previous_actor_dir}" >&2
    fi

    if [[ ! -d "${previous_actor_dir}/huggingface" ]]; then
        echo "Merged huggingface checkpoint directory not found: ${previous_actor_dir}/huggingface" >&2
        return 1
    fi

    if ! has_huggingface_model_weights "${previous_actor_dir}/huggingface"; then
        echo "Merged huggingface checkpoint directory is missing model weights: ${previous_actor_dir}/huggingface" >&2
        return 1
    fi

    printf '%s\n' "${previous_actor_dir}/huggingface"
}