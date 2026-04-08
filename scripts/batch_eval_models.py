#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_CONFIG = ROOT_DIR / "examples/configs/qwen2_5_7b_stage2_code_grpo.yaml"
DEFAULT_DATA_DIR = ROOT_DIR / "data/multi_stage_grpo"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "results" / "qwen2_5_7b_batch_eval"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    group: str
    val_file: str
    prompt_template: str
    reward_function: str
    val_batch_size: int
    default_limit: int | None = None
    train_file: str | None = None
    reward_function_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    model_path: str
    source: str
    selection: str
    load_checkpoint_path: str | None = None
    checkpoint_root: str | None = None
    global_step: int | None = None


DATASET_ORDER = (
    "deepscaler",
    "aime25",
    "aime24",
    "amc",
    "math500",
    "minerva",
    "olympiadbench",
    "deepcoder",
    "humaneval",
    "humanevalplus",
    "livecodebench",
    "mbpp",
    "apps",
    "taco",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate ContinualLearning checkpoints and the base model on math/code datasets."
    )
    parser.add_argument(
        "--checkpoint-root",
        "--checkpoint_root",
        default="/disk4/yiran/yaoqi/checkpoints/ContinualLearning",
        help="Root directory containing experiment folders with checkpoint_tracker.json.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model path or Hugging Face model id used to instantiate the actor.",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing prepared parquet evaluation datasets.",
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        "--output_dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where per-model evaluation folders and summary files are written.",
    )
    parser.add_argument(
        "--template-config",
        default=str(DEFAULT_TEMPLATE_CONFIG),
        help="Template config used as the base for val_only evaluation runs.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_ORDER),
        help="Dataset names separated by commas or spaces. Missing parquet files are skipped automatically.",
    )
    parser.add_argument(
        "--selection",
        default="best",
        help="Checkpoint selections separated by commas or spaces. Supported: best,last,all,all_saved.",
    )
    parser.add_argument(
        "--experiment-filter",
        default="",
        help="Optional regex. Only matching experiment folder names are evaluated.",
    )
    parser.add_argument(
        "--val-limit",
        "--val_limit",
        type=int,
        default=None,
        help="Global validation limit applied to every selected dataset unless overridden by --dataset-limit.",
    )
    parser.add_argument(
        "--dataset-limit",
        action="append",
        default=[],
        help="Override dataset limits, for example --dataset-limit deepscaler=500. Use NAME=all for no limit.",
    )
    parser.add_argument(
        "--full-datasets",
        action="store_true",
        help="Ignore built-in default limits and evaluate all rows in each available dataset.",
    )
    parser.add_argument(
        "--include-base-model",
        action="store_true",
        default=True,
        help="Include the base model in evaluation. Enabled by default.",
    )
    parser.add_argument(
        "--exclude-base-model",
        action="store_true",
        help="Do not evaluate the base model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run models even if results.json already exists in the output directory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one model evaluation fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize configs and summaries without launching evaluation commands.",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="trainer.nnodes override.")
    parser.add_argument("--n-gpus-per-node", type=int, default=4, help="trainer.n_gpus_per_node override.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="worker.rollout.gpu_memory_utilization override.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1536,
        help="Shared max prompt length used for offline evaluation.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=3072,
        help="Shared max response length used for offline evaluation.",
    )
    parser.add_argument(
        "--code-reward-workers",
        type=int,
        default=8,
        help="Parallel workers for code reward execution.",
    )
    parser.add_argument(
        "--code-reward-max-tests",
        type=int,
        default=8,
        help="Maximum tests per sample for code reward execution.",
    )
    parser.add_argument(
        "--project-name",
        default="ContinualLearningEval",
        help="Project name written into eval configs and result metadata.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        "--cuda_devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES passed to each evaluation subprocess.",
    )
    return parser.parse_args()


def build_dataset_registry(data_dir: Path, code_reward_workers: int, code_reward_max_tests: int) -> dict[str, DatasetSpec]:
    code_kwargs = {
        "timeout": 5.0,
        "parallel_workers": code_reward_workers,
        "max_tests_per_sample": code_reward_max_tests,
    }
    math_prompt = str((ROOT_DIR / "examples/format_prompt/math.jinja").resolve())
    code_prompt = str((ROOT_DIR / "examples/format_prompt/code.jinja").resolve())
    math_reward = str((ROOT_DIR / "examples/reward_function/math.py").resolve()) + ":compute_score"
    code_reward = str((ROOT_DIR / "examples/reward_function/code.py").resolve()) + ":compute_score"

    return {
        "deepscaler": DatasetSpec(
            name="deepscaler",
            group="math",
            val_file=str((data_dir / "deepscaler_val.parquet").resolve()),
            train_file=str((data_dir / "deepscaler_train.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=256,
        ),
        "aime25": DatasetSpec(
            name="aime25",
            group="math",
            val_file=str((data_dir / "aime25_val.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=30,
        ),
        "aime24": DatasetSpec(
            name="aime24",
            group="math",
            val_file=str((data_dir / "aime24_val.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=30,
        ),
        "amc": DatasetSpec(
            name="amc",
            group="math",
            val_file=str((data_dir / "amc_val.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=64,
        ),
        "math500": DatasetSpec(
            name="math500",
            group="math",
            val_file=str((data_dir / "math500_val.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=128,
        ),
        "minerva": DatasetSpec(
            name="minerva",
            group="math",
            val_file=str((data_dir / "minerva_val.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=64,
        ),
        "olympiadbench": DatasetSpec(
            name="olympiadbench",
            group="math",
            val_file=str((data_dir / "olympiadbench_val.parquet").resolve()),
            prompt_template=math_prompt,
            reward_function=math_reward,
            val_batch_size=64,
            default_limit=64,
        ),
        "deepcoder": DatasetSpec(
            name="deepcoder",
            group="code",
            val_file=str((data_dir / "deepcoder_val.parquet").resolve()),
            train_file=str((data_dir / "deepcoder_train.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=32,
            default_limit=128,
        ),
        "humaneval": DatasetSpec(
            name="humaneval",
            group="code",
            val_file=str((data_dir / "humaneval_val.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=32,
            default_limit=32,
        ),
        "humanevalplus": DatasetSpec(
            name="humanevalplus",
            group="code",
            val_file=str((data_dir / "humanevalplus_val.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=32,
            default_limit=32,
        ),
        "livecodebench": DatasetSpec(
            name="livecodebench",
            group="code",
            val_file=str((data_dir / "livecodebench_val.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=32,
            default_limit=32,
        ),
        "mbpp": DatasetSpec(
            name="mbpp",
            group="code",
            val_file=str((data_dir / "mbpp_test.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=16,
            default_limit=32,
        ),
        "apps": DatasetSpec(
            name="apps",
            group="code",
            val_file=str((data_dir / "apps_test.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=8,
            default_limit=32,
        ),
        "taco": DatasetSpec(
            name="taco",
            group="code",
            val_file=str((data_dir / "taco_test.parquet").resolve()),
            prompt_template=code_prompt,
            reward_function=code_reward,
            reward_function_kwargs=code_kwargs,
            val_batch_size=8,
            default_limit=64,
        ),
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\s,]+", value) if item.strip()]


def parse_limit_overrides(raw_items: list[str]) -> dict[str, int | None]:
    overrides: dict[str, int | None] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-limit value: {item}. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        dataset_name = name.strip()
        value = raw_value.strip().lower()
        if value in {"all", "none", "full"}:
            overrides[dataset_name] = None
            continue
        overrides[dataset_name] = int(raw_value)
    return overrides


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_requested_datasets(args: argparse.Namespace, registry: dict[str, DatasetSpec]) -> tuple[list[DatasetSpec], list[str]]:
    requested = parse_csv(args.datasets)
    limit_overrides = parse_limit_overrides(args.dataset_limit)
    resolved: list[DatasetSpec] = []
    skipped: list[str] = []

    for dataset_name in requested:
        if dataset_name not in registry:
            skipped.append(f"unknown:{dataset_name}")
            continue
        spec = registry[dataset_name]
        if not Path(spec.val_file).exists():
            skipped.append(f"missing:{dataset_name}")
            continue
        override_limit = limit_overrides.get(dataset_name)
        default_limit = None if args.full_datasets else (args.val_limit if args.val_limit is not None else spec.default_limit)
        effective_limit = override_limit if dataset_name in limit_overrides else default_limit
        resolved.append(
            DatasetSpec(
                name=spec.name,
                group=spec.group,
                val_file=spec.val_file,
                train_file=spec.train_file,
                prompt_template=spec.prompt_template,
                reward_function=spec.reward_function,
                reward_function_kwargs=copy.deepcopy(spec.reward_function_kwargs),
                val_batch_size=spec.val_batch_size,
                default_limit=effective_limit,
            )
        )

    return resolved, skipped


def choose_train_file(datasets: list[DatasetSpec], registry: dict[str, DatasetSpec]) -> str:
    for spec in datasets:
        if spec.train_file and Path(spec.train_file).exists():
            return spec.train_file

    for dataset_name in ("deepscaler", "deepcoder"):
        spec = registry.get(dataset_name)
        if spec and spec.train_file and Path(spec.train_file).exists():
            return spec.train_file

    raise FileNotFoundError("No usable train parquet was found. val_only still requires train_files to exist.")


def discover_models(args: argparse.Namespace) -> list[ModelSpec]:
    checkpoint_root = Path(args.checkpoint_root)
    selections = parse_csv(args.selection)
    unsupported = [selection for selection in selections if selection not in {"best", "last", "all", "all_saved"}]
    if unsupported:
        raise ValueError(f"Unsupported --selection values: {unsupported}")

    all_saved = any(selection in {"all", "all_saved"} for selection in selections)
    tracked_selections = [selection for selection in selections if selection in {"best", "last"}]

    models: list[ModelSpec] = []
    include_base_model = args.include_base_model and not args.exclude_base_model
    if include_base_model:
        models.append(
            ModelSpec(
                name="base_model",
                slug=sanitize_slug(f"base__{Path(args.base_model).name or args.base_model}"),
                model_path=args.base_model,
                source="base_model",
                selection="base",
            )
        )

    pattern = re.compile(args.experiment_filter) if args.experiment_filter else None
    if not checkpoint_root.exists():
        return models

    for experiment_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if pattern and not pattern.search(experiment_dir.name):
            continue

        tracker_path = experiment_dir / "checkpoint_tracker.json"
        tracker = load_json(tracker_path) if tracker_path.exists() else {}

        if all_saved:
            step_dirs = sorted(
                [path for path in experiment_dir.iterdir() if path.is_dir() and re.fullmatch(r"global_step_\d+", path.name)],
                key=lambda path: int(path.name.split("global_step_", 1)[1]),
            )
            for step_dir in step_dirs:
                global_step = int(step_dir.name.split("global_step_", 1)[1])
                models.append(
                    ModelSpec(
                        name=experiment_dir.name,
                        slug=sanitize_slug(f"{experiment_dir.name}__global_step_{global_step}"),
                        model_path=args.base_model,
                        source="checkpoint",
                        selection=f"global_step_{global_step}",
                        checkpoint_root=str(experiment_dir.resolve()),
                        load_checkpoint_path=str(step_dir.resolve()),
                        global_step=global_step,
                    )
                )
            if step_dirs:
                continue

        if not tracker:
            continue

        for selection in tracked_selections:
            global_step_key = f"{selection}_global_step"
            global_step = tracker.get(global_step_key)
            if global_step is None:
                continue
            load_checkpoint_path = experiment_dir / f"global_step_{global_step}"
            if not load_checkpoint_path.exists():
                continue

            models.append(
                ModelSpec(
                    name=experiment_dir.name,
                    slug=sanitize_slug(f"{experiment_dir.name}__{selection}"),
                    model_path=args.base_model,
                    source="checkpoint",
                    selection=selection,
                    checkpoint_root=str(experiment_dir.resolve()),
                    load_checkpoint_path=str(load_checkpoint_path.resolve()),
                    global_step=int(global_step),
                )
            )

    return models


def make_primary_and_evaluations(datasets: list[DatasetSpec]) -> tuple[DatasetSpec, list[DatasetSpec]]:
    if not datasets:
        raise ValueError("No datasets available for evaluation.")
    return datasets[0], datasets[1:]


def build_eval_config(
    template_config_path: Path,
    model: ModelSpec,
    primary_dataset: DatasetSpec,
    evaluation_datasets: list[DatasetSpec],
    train_file: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    os.environ.setdefault("DATA_DIR", str(Path(args.data_dir).resolve()))
    base_config = OmegaConf.to_container(OmegaConf.load(str(template_config_path)), resolve=True)
    if not isinstance(base_config, dict):
        raise TypeError("Template config did not resolve to a mapping.")

    config = copy.deepcopy(base_config)
    data_cfg = config.setdefault("data", {})
    worker_cfg = config.setdefault("worker", {})
    actor_cfg = worker_cfg.setdefault("actor", {})
    actor_model_cfg = actor_cfg.setdefault("model", {})
    reward_cfg = worker_cfg.setdefault("reward", {})
    rollout_cfg = worker_cfg.setdefault("rollout", {})
    trainer_cfg = config.setdefault("trainer", {})

    data_cfg["train_files"] = train_file
    data_cfg["val_files"] = primary_dataset.val_file
    data_cfg["val_limit"] = primary_dataset.default_limit
    data_cfg["format_prompt"] = primary_dataset.prompt_template
    data_cfg["val_batch_size"] = primary_dataset.val_batch_size
    data_cfg["max_prompt_length"] = args.max_prompt_length
    data_cfg["max_response_length"] = args.max_response_length

    actor_model_cfg["model_path"] = model.model_path
    reward_cfg["reward_function"] = primary_dataset.reward_function
    reward_cfg["reward_function_kwargs"] = copy.deepcopy(primary_dataset.reward_function_kwargs or {})
    rollout_cfg["gpu_memory_utilization"] = args.gpu_memory_utilization
    rollout_cfg["val_override_config"] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
    }

    trainer_cfg["project_name"] = args.project_name
    trainer_cfg["experiment_name"] = model.slug
    trainer_cfg["nnodes"] = args.nnodes
    trainer_cfg["n_gpus_per_node"] = args.n_gpus_per_node
    trainer_cfg["logger"] = ["console", "file"]
    trainer_cfg["val_before_train"] = True
    trainer_cfg["val_only"] = True
    trainer_cfg["val_freq"] = -1
    trainer_cfg["val_generations_to_log"] = 0
    trainer_cfg["primary_val_name"] = primary_dataset.name
    trainer_cfg["save_freq"] = -1
    trainer_cfg["save_limit"] = 1
    trainer_cfg["find_last_checkpoint"] = False
    trainer_cfg["save_checkpoint_path"] = str(run_dir.resolve())
    trainer_cfg["load_checkpoint_path"] = model.load_checkpoint_path
    trainer_cfg["max_steps"] = 0
    trainer_cfg["total_epochs"] = 0

    config["evaluations"] = [
        {
            "name": dataset.name,
            "val_files": dataset.val_file,
            "val_limit": dataset.default_limit,
            "format_prompt": dataset.prompt_template,
            "reward_function": dataset.reward_function,
            "reward_function_kwargs": copy.deepcopy(dataset.reward_function_kwargs or {}),
            "val_batch_size": dataset.val_batch_size,
        }
        for dataset in evaluation_datasets
    ]
    return config


def read_last_validation(log_path: Path, primary_dataset_name: str) -> dict[str, Any]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing experiment log: {log_path}")

    last_record: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "val" in record:
            last_record = record

    if last_record is None:
        raise ValueError(f"No validation payload found in {log_path}")

    val_section = last_record["val"]
    if primary_dataset_name not in val_section:
        primary_metrics = {
            key: value
            for key, value in val_section.items()
            if key not in {"val_response_length", "val_prompt_length"}
            and not isinstance(value, dict)
        }
        if primary_metrics:
            val_section[primary_dataset_name] = primary_metrics
    return last_record


def collect_dataset_metrics(val_record: dict[str, Any], datasets: list[DatasetSpec]) -> dict[str, Any]:
    val_section = val_record["val"]
    metrics: dict[str, Any] = {}
    for dataset in datasets:
        if dataset.name in val_section:
            metrics[dataset.name] = val_section[dataset.name]
    return metrics


def build_results_payload(
    model: ModelSpec,
    datasets: list[DatasetSpec],
    primary_dataset: DatasetSpec,
    run_dir: Path,
    last_record: dict[str, Any] | None,
    command: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": asdict(model),
        "status": "dry_run" if args.dry_run else "ok",
        "primary_dataset": primary_dataset.name,
        "datasets": [
            {
                "name": dataset.name,
                "group": dataset.group,
                "val_file": dataset.val_file,
                "limit": dataset.default_limit,
                "val_batch_size": dataset.val_batch_size,
            }
            for dataset in datasets
        ],
        "run_dir": str(run_dir.resolve()),
        "command": command,
    }
    if last_record is not None:
        payload["step"] = last_record.get("step")
        payload["metrics"] = collect_dataset_metrics(last_record, datasets)
        payload["raw_validation"] = last_record["val"]
    return payload


def write_markdown_summary(summary_path: Path, model_results: list[dict[str, Any]]) -> None:
    lines = [
        "# ContinualLearning Batch Eval Summary",
        "",
        "| Model | Source | Dataset | Group | Reward | Accuracy | Format | Syntax | Step | Status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for result in model_results:
        metrics = result.get("metrics", {})
        step = result.get("step", "-")
        status = result.get("status", "unknown")
        model_name = result["model"]["name"]
        source = result["model"]["source"]
        if not metrics:
            lines.append(f"| {model_name} | {source} | - | - | - | - | - | - | {step} | {status} |")
            continue

        for dataset_name, dataset_metrics in metrics.items():
            reward_score = dataset_metrics.get("reward_score", dataset_metrics.get("overall_reward", "-"))
            accuracy = dataset_metrics.get("accuracy_reward", "-")
            format_reward = dataset_metrics.get("format_reward", "-")
            syntax = dataset_metrics.get("syntax_reward", "-")
            group = next((item["group"] for item in result["datasets"] if item["name"] == dataset_name), "-")
            lines.append(
                "| {model} | {source} | {dataset} | {group} | {reward} | {accuracy} | {format_reward} | {syntax} | {step} | {status} |".format(
                    model=model_name,
                    source=source,
                    dataset=dataset_name,
                    group=group,
                    reward=_format_metric(reward_score),
                    accuracy=_format_metric(accuracy),
                    format_reward=_format_metric(format_reward),
                    syntax=_format_metric(syntax),
                    step=step,
                    status=status,
                )
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def write_failure(run_dir: Path, model: ModelSpec, command: list[str], exc: Exception) -> None:
    dump_json(
        run_dir / "error.json",
        {
            "model": asdict(model),
            "status": "failed",
            "command": command,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def run_single_model(
    model: ModelSpec,
    datasets: list[DatasetSpec],
    registry: dict[str, DatasetSpec],
    args: argparse.Namespace,
    template_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    model_dir = output_root / model.slug
    run_dir = model_dir / "raw_run"
    results_path = model_dir / "results.json"

    if results_path.exists() and not args.force:
        return load_json(results_path)

    if model_dir.exists() and args.force:
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    primary_dataset, evaluation_datasets = make_primary_and_evaluations(datasets)
    train_file = choose_train_file(datasets, registry)
    config = build_eval_config(template_config_path, model, primary_dataset, evaluation_datasets, train_file, args, run_dir)
    config_path = model_dir / "eval_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dump_json(model_dir / "model_info.json", asdict(model))

    command = [sys.executable, "-m", "verl.trainer.main", f"config={config_path}"]
    last_record: dict[str, Any] | None = None

    if not args.dry_run:
        env = os.environ.copy()
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        subprocess.run(command, cwd=str(ROOT_DIR), env=env, check=True)
        last_record = read_last_validation(run_dir / "experiment_log.jsonl", primary_dataset.name)

    payload = build_results_payload(model, datasets, primary_dataset, run_dir, last_record, command, args)
    dump_json(results_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()

    registry = build_dataset_registry(data_dir, args.code_reward_workers, args.code_reward_max_tests)
    datasets, skipped_datasets = resolve_requested_datasets(args, registry)
    if not datasets:
        raise SystemExit(f"No usable datasets were found. Skipped: {skipped_datasets}")

    models = discover_models(args)
    if not models:
        raise SystemExit("No models were discovered. Check --checkpoint-root and --experiment-filter.")

    summary_payload = {
        "project_name": args.project_name,
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "base_model": args.base_model,
        "template_config": str(Path(args.template_config).resolve()),
        "output_root": str(output_root),
        "datasets": [dataset.name for dataset in datasets],
        "skipped_datasets": skipped_datasets,
        "models": [],
    }

    model_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    template_config_path = Path(args.template_config).resolve()

    for model in models:
        try:
            result = run_single_model(model, datasets, registry, args, template_config_path, output_root)
            model_results.append(result)
        except Exception as exc:  # noqa: BLE001
            model_dir = output_root / model.slug
            model_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "verl.trainer.main", f"config={model_dir / 'eval_config.json'}"]
            write_failure(model_dir, model, command, exc)
            failure_payload = {
                "model": asdict(model),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure_payload)
            model_results.append(failure_payload)
            if args.fail_fast:
                raise

    summary_payload["models"] = model_results
    if failures:
        summary_payload["failures"] = failures

    dump_json(output_root / "summary.json", summary_payload)
    write_markdown_summary(output_root / "summary.md", [result for result in model_results if isinstance(result, dict)])
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())