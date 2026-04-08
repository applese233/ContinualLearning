#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from jinja2 import Template
from tqdm import tqdm
from transformers import AutoTokenizer

from batch_eval_models import DatasetSpec, ModelSpec, build_dataset_registry, discover_models, resolve_requested_datasets


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data/multi_stage_grpo"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "results" / "qwen2_5_7b_direct_eval"
CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def get_vllm_classes() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("vLLM is required for direct_batch_eval_vllm.py. Install it first.") from exc
    return LLM, SamplingParams


def get_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct vLLM batch evaluation for ContinualLearning checkpoints.")
    parser.add_argument("--checkpoint-root", "--checkpoint_root", default="/disk4/yiran/yaoqi/checkpoints/ContinualLearning")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", "--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", "--output_root", "--output_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--datasets",
        default=(
            "deepscaler,aime25,aime24,amc,math500,minerva,olympiadbench,"
            "deepcoder,humaneval,humanevalplus,livecodebench,mbpp,apps,taco"
        ),
    )
    parser.add_argument("--selection", default="best", help="Supported: best,last,all,all_saved")
    parser.add_argument("--experiment-filter", default="")
    parser.add_argument("--val-limit", "--val_limit", type=int, default=None)
    parser.add_argument("--dataset-limit", action="append", default=[])
    parser.add_argument("--full-datasets", action="store_true")
    parser.add_argument("--include-base-model", action="store_true", default=True)
    parser.add_argument("--exclude-base-model", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-response-length", type=int, default=3072)
    parser.add_argument("--code-reward-workers", type=int, default=8)
    parser.add_argument("--code-reward-max-tests", type=int, default=8)
    parser.add_argument("--project-name", default="ContinualLearningDirectEval")
    parser.add_argument("--cuda-visible-devices", "--cuda_devices", default="")
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument(
        "--sample-details",
        type=int,
        default=5,
        help="Number of per-sample detailed examples to keep in *_eval_results.json when --save-generations is enabled.",
    )
    parser.add_argument(
        "--save-all-generations",
        action="store_true",
        help="Write all per-sample generation rows to *_generations.jsonl. Disabled by default to reduce disk usage.",
    )
    parser.add_argument(
        "--save-full-response",
        action="store_true",
        help="Include raw full_response text in full generation exports. Sampled details always include it.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def has_huggingface_model_weights(hf_dir: Path) -> bool:
    return any(
        (hf_dir / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def resolve_inference_model_path(model: ModelSpec) -> str:
    if model.source == "base_model":
        return model.model_path

    if model.load_checkpoint_path is None:
        raise ValueError(f"Checkpoint model {model.name} is missing load_checkpoint_path")

    actor_dir = Path(model.load_checkpoint_path) / "actor"
    hf_dir = actor_dir / "huggingface"
    if hf_dir.exists() and has_huggingface_model_weights(hf_dir):
        return str(hf_dir.resolve())

    subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts/model_merger.py"), "--local_dir", str(actor_dir.resolve())],
        cwd=str(ROOT_DIR),
        check=True,
    )

    if not hf_dir.exists() or not has_huggingface_model_weights(hf_dir):
        raise FileNotFoundError(f"Merged huggingface weights not found under {hf_dir}")
    return str(hf_dir.resolve())


def load_reward_function(reward_spec: str) -> Callable[..., list[dict[str, float]]]:
    module_path, function_name = reward_spec.split(":", 1)
    module_file = Path(module_path)
    module_name = f"direct_eval_{module_file.stem}_{abs(hash(str(module_file.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def build_prompt_renderer(template_path: str) -> Template:
    return Template(Path(template_path).read_text(encoding="utf-8").strip())


def build_chat_prompt(tokenizer: Any, renderer: Template, content: str) -> str:
    rendered = renderer.render(content=content)
    messages = [{"role": "user", "content": rendered}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def compute_length_metrics(lengths: list[int], max_length: int) -> dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "clip_ratio": 0.0}
    return {
        "mean": sum(lengths) / len(lengths),
        "max": float(max(lengths)),
        "min": float(min(lengths)),
        "clip_ratio": sum(1 for item in lengths if item >= max_length) / len(lengths),
    }


def aggregate_reward_metrics(score_rows: list[dict[str, float]]) -> dict[str, float]:
    if not score_rows:
        return {"reward_score": 0.0}

    metrics: dict[str, float] = {}
    keys = sorted({key for row in score_rows for key in row})
    for key in keys:
        values = [float(row[key]) for row in score_rows if key in row]
        if not values:
            continue
        if key == "overall":
            metrics["reward_score"] = sum(values) / len(values)
        metrics[f"{key}_reward"] = sum(values) / len(values)
    if "reward_score" not in metrics and "overall_reward" in metrics:
        metrics["reward_score"] = metrics["overall_reward"]
    return metrics


def _extract_balanced_braced_content(text: str, marker: str) -> str:
    marker_index = text.rfind(marker)
    if marker_index < 0:
        return ""

    start_index = marker_index + len(marker)
    depth = 1
    current_index = start_index
    while current_index < len(text):
        char = text[current_index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:current_index].strip()
        current_index += 1
    return ""


def extract_math_answer(response: str) -> str:
    try:
        from mathruler.grader import extract_boxed_content

        answer = extract_boxed_content(response)
        if answer is None:
            return ""
        if isinstance(answer, list):
            return " | ".join(str(item).strip() for item in answer if str(item).strip())
        return str(answer).strip()
    except Exception:
        return _extract_balanced_braced_content(response, r"\boxed{")


def extract_code_answer(response: str) -> str:
    cleaned = THINK_PATTERN.sub("", response).strip()
    matches = CODE_BLOCK_PATTERN.findall(cleaned)
    if matches:
        return max(matches, key=len).strip()
    return cleaned


def extract_predicted_answer(dataset: DatasetSpec, response: str) -> str:
    if dataset.group == "math":
        return extract_math_answer(response)
    if dataset.group == "code":
        return extract_code_answer(response)
    return response.strip()


def maybe_build_ground_truth_preview(dataset: DatasetSpec, ground_truth: Any, max_chars: int = 400) -> dict[str, Any]:
    if dataset.group != "code":
        return {"ground_truth": ground_truth}

    serialized = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth, ensure_ascii=False)
    preview = serialized[:max_chars]
    if len(serialized) > max_chars:
        preview += "..."
    return {
        "ground_truth_preview": preview,
        "ground_truth_bytes": len(serialized.encode("utf-8")),
    }


def build_generation_record(
    dataset: DatasetSpec,
    row_index: int,
    raw_prompt: str,
    predicted_answer: str,
    ground_truth: Any,
    score_row: dict[str, float],
    prompt_length: int,
    response_length: int,
    generation_time: float,
    full_response: str | None = None,
) -> dict[str, Any]:
    record = {
        "idx": row_index,
        "question": raw_prompt,
        "predicted_answer": predicted_answer,
        "reward_score": score_row.get("overall"),
        "accuracy": score_row.get("accuracy"),
        "format": score_row.get("format"),
        "syntax": score_row.get("syntax"),
        "correct": bool(score_row.get("accuracy", 0.0) >= 0.999999),
        "prompt_length": prompt_length,
        "response_length": response_length,
        "generation_time": generation_time,
        "scores": score_row,
    }
    record.update(maybe_build_ground_truth_preview(dataset, ground_truth))
    if full_response is not None:
        record["full_response"] = full_response
    return record


def sample_generation_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not records:
        return []
    if len(records) <= limit:
        return records
    indices = sorted(random.Random(42).sample(range(len(records)), limit))
    return [records[index] for index in indices]


def build_dataset_metrics(
    dataset: DatasetSpec,
    score_rows: list[dict[str, float]],
    prompt_lengths: list[int],
    response_lengths: list[int],
    generation_times: list[float],
    filtered_examples: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reward_metrics = aggregate_reward_metrics(score_rows)
    accuracy_values = [float(row.get("accuracy", 0.0)) for row in score_rows if "accuracy" in row]
    exact_accuracy = None
    if accuracy_values:
        exact_accuracy = sum(1 for value in accuracy_values if value >= 0.999999) / len(accuracy_values)

    metrics: dict[str, Any] = {
        "dataset": Path(dataset.val_file).name,
        "num_samples": len(score_rows),
        "num_filtered_overlong_prompts": filtered_examples,
        "avg_reward_score": reward_metrics.get("reward_score", 0.0),
        "reward_score": reward_metrics.get("reward_score", 0.0),
        "avg_accuracy": reward_metrics.get("accuracy_reward"),
        "accuracy_reward": reward_metrics.get("accuracy_reward"),
        "exact_accuracy": exact_accuracy,
        "avg_format": reward_metrics.get("format_reward"),
        "format_reward": reward_metrics.get("format_reward"),
        "avg_syntax": reward_metrics.get("syntax_reward"),
        "syntax_reward": reward_metrics.get("syntax_reward"),
        "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0.0,
        "total_time": sum(generation_times),
        "prompt_length": compute_length_metrics(prompt_lengths, args.max_prompt_length),
        "response_length": compute_length_metrics(response_lengths, args.max_response_length),
    }
    if reward_metrics.get("overall_reward") is not None:
        metrics["overall_reward"] = reward_metrics.get("overall_reward")
    return metrics


def evaluate_dataset(
    llm: Any,
    tokenizer: Any,
    dataset: DatasetSpec,
    args: argparse.Namespace,
    model_dir: Path,
) -> dict[str, Any]:
    data_frame = pd.read_parquet(dataset.val_file)
    if dataset.default_limit is not None:
        data_frame = data_frame.iloc[: dataset.default_limit]

    renderer = build_prompt_renderer(dataset.prompt_template)
    reward_fn = load_reward_function(dataset.reward_function)
    reward_kwargs = copy.deepcopy(dataset.reward_function_kwargs or {})

    prompts: list[str] = []
    raw_prompts: list[str] = []
    ground_truths: list[Any] = []
    prompt_lengths: list[int] = []
    row_indices: list[int] = []
    filtered_examples = 0

    for row_index, row in enumerate(data_frame.to_dict("records")):
        raw_prompt = str(row["prompt"])
        prompt = build_chat_prompt(tokenizer, renderer, raw_prompt)
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_length > args.max_prompt_length:
            filtered_examples += 1
            continue
        prompts.append(prompt)
        raw_prompts.append(raw_prompt)
        ground_truths.append(row["answer"])
        prompt_lengths.append(prompt_length)
        row_indices.append(row_index)

    _, SamplingParams = get_vllm_classes()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
    )

    sampled_generation_candidates: list[dict[str, Any]] = []
    full_generation_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, float]] = []
    response_lengths: list[int] = []
    generation_times: list[float] = []

    progress = tqdm(total=len(prompts), desc=dataset.name, unit="sample")
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_raw_prompts = raw_prompts[start : start + args.batch_size]
        batch_ground_truths = ground_truths[start : start + args.batch_size]
        batch_prompt_lengths = prompt_lengths[start : start + args.batch_size]
        batch_row_indices = row_indices[start : start + args.batch_size]

        batch_started_at = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        batch_elapsed = time.perf_counter() - batch_started_at
        per_sample_generation_time = batch_elapsed / len(batch_prompts) if batch_prompts else 0.0
        responses = [output.outputs[0].text if output.outputs else "" for output in outputs]
        reward_inputs = [
            {"response": response, "ground_truth": ground_truth}
            for response, ground_truth in zip(responses, batch_ground_truths)
        ]
        batch_scores = reward_fn(reward_inputs, **reward_kwargs)

        for row_index, raw_prompt, prompt, response, ground_truth, score_row, prompt_length in zip(
            batch_row_indices,
            batch_raw_prompts,
            batch_prompts,
            responses,
            batch_ground_truths,
            batch_scores,
            batch_prompt_lengths,
        ):
            response_length = len(tokenizer.encode(response, add_special_tokens=False))
            predicted_answer = extract_predicted_answer(dataset, response)
            response_lengths.append(response_length)
            score_rows.append(score_row)
            generation_times.append(per_sample_generation_time)
            if args.save_generations:
                sampled_generation_candidates.append(
                    build_generation_record(
                        dataset=dataset,
                        row_index=row_index,
                        raw_prompt=raw_prompt,
                        predicted_answer=predicted_answer,
                        ground_truth=ground_truth,
                        score_row=score_row,
                        prompt_length=prompt_length,
                        response_length=response_length,
                        generation_time=per_sample_generation_time,
                        full_response=response,
                    )
                )
            if args.save_all_generations:
                full_generation_rows.append(
                    build_generation_record(
                        dataset=dataset,
                        row_index=row_index,
                        raw_prompt=raw_prompt,
                        predicted_answer=predicted_answer,
                        ground_truth=ground_truth,
                        score_row=score_row,
                        prompt_length=prompt_length,
                        response_length=response_length,
                        generation_time=per_sample_generation_time,
                        full_response=response if args.save_full_response else None,
                    )
                )
        progress.update(len(batch_prompts))
    progress.close()

    metrics = build_dataset_metrics(
        dataset=dataset,
        score_rows=score_rows,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        generation_times=generation_times,
        filtered_examples=filtered_examples,
        args=args,
    )

    dataset_dir = model_dir / dataset.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sampled_results = sample_generation_records(sampled_generation_candidates, args.sample_details)
    dump_json(dataset_dir / f"{dataset.name}_metrics.json", metrics)
    dump_json(
        dataset_dir / f"{dataset.name}_eval_results.json",
        {
            "model": model_dir.name,
            "dataset": dataset.name,
            "metrics": metrics,
            "num_total_results": len(score_rows),
            "num_saved_results": len(sampled_results),
            "results": sampled_results if args.save_generations else [],
        },
    )
    if args.save_all_generations:
        with (dataset_dir / f"{dataset.name}_generations.jsonl").open("w", encoding="utf-8") as handle:
            for row in full_generation_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "name": dataset.name,
        "group": dataset.group,
        "val_file": dataset.val_file,
        "limit": dataset.default_limit,
        "metrics": metrics,
    }


def build_model_summary(model: ModelSpec, resolved_model_path: str, dataset_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": asdict(model),
        "resolved_model_path": resolved_model_path,
        "status": "ok",
        "datasets": dataset_results,
    }


def build_evaluation_summary(model_result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint": model_result["resolved_model_path"],
        "model": model_result["model"],
        "config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
        },
        "datasets": [dataset["metrics"] for dataset in model_result["datasets"]],
    }


def write_markdown_summary(summary_path: Path, model_results: list[dict[str, Any]]) -> None:
    lines = [
        "# Direct vLLM Eval Summary",
        "",
        "| Model | Selection | Dataset | Group | Reward | Accuracy | Format | Syntax | Samples | Status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in model_results:
        if result.get("status") != "ok":
            model = result["model"]
            lines.append(
                f"| {model['name']} | {model['selection']} | - | - | - | - | - | - | - | {result.get('status', 'failed')} |"
            )
            continue

        for dataset in result["datasets"]:
            metrics = dataset["metrics"]
            lines.append(
                "| {model} | {selection} | {dataset} | {group} | {reward} | {accuracy} | {format_reward} | {syntax} | {samples} | ok |".format(
                    model=result["model"]["name"],
                    selection=result["model"]["selection"],
                    dataset=dataset["name"],
                    group=dataset["group"],
                    reward=_fmt(metrics.get("reward_score")),
                    accuracy=_fmt(metrics.get("accuracy_reward")),
                    format_reward=_fmt(metrics.get("format_reward")),
                    syntax=_fmt(metrics.get("syntax_reward")),
                    samples=metrics.get("num_samples", 0),
                )
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def evaluate_model(model: ModelSpec, datasets: list[DatasetSpec], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    model_dir = output_root / model.slug
    results_path = model_dir / "results.json"
    if results_path.exists() and not args.force:
        return json.loads(results_path.read_text(encoding="utf-8"))

    if model_dir.exists() and args.force:
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_path = resolve_inference_model_path(model)
    tokenizer = get_tokenizer(resolved_model_path, trust_remote_code=args.trust_remote_code)
    LLM, _ = get_vllm_classes()
    llm = LLM(
        model=resolved_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    dataset_results = [evaluate_dataset(llm, tokenizer, dataset, args, model_dir) for dataset in datasets]
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    payload = build_model_summary(model, resolved_model_path, dataset_results)
    dump_json(results_path, payload)
    dump_json(model_dir / "evaluation_summary.json", build_evaluation_summary(payload, args))
    return payload


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    registry = build_dataset_registry(Path(args.data_dir).resolve(), args.code_reward_workers, args.code_reward_max_tests)
    datasets, skipped_datasets = resolve_requested_datasets(args, registry)
    if not datasets:
        raise SystemExit(f"No usable datasets were found. Skipped: {skipped_datasets}")

    models = discover_models(args)
    if not models:
        raise SystemExit("No models were discovered. Check --checkpoint-root and --experiment-filter.")

    model_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for model in models:
        try:
            print(f"Evaluating model: {model.slug}")
            model_results.append(evaluate_model(model, datasets, args, output_root))
        except Exception as exc:  # noqa: BLE001
            traceback_text = traceback.format_exc()
            print(f"Evaluation failed for {model.slug}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(traceback_text, file=sys.stderr)
            (output_root / model.slug).mkdir(parents=True, exist_ok=True)
            failure_payload = {
                "model": asdict(model),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback_text,
            }
            failures.append(failure_payload)
            model_results.append(failure_payload)
            dump_json(output_root / model.slug / "error.json", failure_payload)
            if args.fail_fast:
                raise

    summary = {
        "project_name": args.project_name,
        "output_root": str(output_root),
        "datasets": [dataset.name for dataset in datasets],
        "skipped_datasets": skipped_datasets,
        "models": model_results,
    }
    if failures:
        summary["failures"] = failures

    dump_json(output_root / "summary.json", summary)
    write_markdown_summary(output_root / "summary.md", model_results)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())