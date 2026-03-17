#!/usr/bin/env python3

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import json
import os
import pickle
import re
import zlib
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare math and coding datasets for two-stage EasyR1 training.")
    parser.add_argument("--output_dir", default="data/two_stage_grpo", help="Directory to save parquet files.")
    parser.add_argument("--apps_train_limit", type=int, default=5000, help="Maximum APPS training samples to keep.")
    parser.add_argument("--gsm8k_test_limit", type=int, default=None, help="Maximum GSM8K test samples to keep.")
    parser.add_argument("--aime25_val_limit", type=int, default=None, help="Maximum AIME 2025 validation samples to keep.")
    parser.add_argument("--mbpp_test_limit", type=int, default=None, help="Maximum MBPP test samples to keep.")
    parser.add_argument("--apps_test_limit", type=int, default=None, help="Maximum APPS test samples to keep.")
    parser.add_argument("--humaneval_val_limit", type=int, default=None, help="Maximum HumanEval validation samples to keep.")
    parser.add_argument(
        "--livecodebench_val_limit",
        type=int,
        default=None,
        help="Maximum LiveCodeBench validation samples to keep.",
    )
    parser.add_argument(
        "--apps_difficulties",
        nargs="*",
        default=["introductory", "interview"],
        help="APPS difficulty levels to keep for the conservative route.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_first(record: dict, keys: list[str]):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    raise KeyError(f"None of the candidate keys exist: {keys}")


def extract_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+)", answer)
    if match:
        return match.group(1).strip()
    return answer.strip()


def save_parquet(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Saved {len(rows)} rows to {path}")


def log_step(message: str) -> None:
    print(message, flush=True)


def download_hf_dataset_file(dataset_name: str, filename: str) -> str:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    log_step(f"Downloading {dataset_name}/{filename} to local cache...")
    return hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=filename)


def download_hf_dataset_snapshot(dataset_name: str, allow_patterns: list[str]) -> Path:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    log_step(f"Downloading {dataset_name} snapshot to local cache...")
    return Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            allow_patterns=allow_patterns,
        )
    )


def find_local_files(root: Path, pattern: str) -> list[str]:
    files = sorted(str(path) for path in root.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found under {root}")
    return files


def build_code_prompt(question: str, starter_code: str | None = None) -> str:
    parts = ["Solve the following Python programming task.", question.strip()]
    if starter_code:
        parts.extend(["Starter code:", "```python", starter_code.rstrip(), "```"])
    parts.append("Return only executable Python code inside a single ```python``` block.")
    return "\n\n".join(parts)


def prepare_gsm8k(output_dir: Path, test_limit: int | None = None) -> None:
    log_step("Preparing GSM8K...")
    snapshot_dir = download_hf_dataset_snapshot("openai/gsm8k", ["main/*.parquet", "README.md"])
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": find_local_files(snapshot_dir / "main", "train-*.parquet"),
            "test": find_local_files(snapshot_dir / "main", "test-*.parquet"),
        },
    )

    for split_name, output_name in (("train", "gsm8k_train.parquet"), ("test", "gsm8k_test.parquet")):
        rows = []
        for sample in dataset[split_name]:
            rows.append(
                {
                    "prompt": sample["question"].strip(),
                    "answer": extract_gsm8k_answer(sample["answer"]),
                }
            )
            if split_name == "test" and test_limit is not None and len(rows) >= test_limit:
                break
        save_parquet(rows, output_dir / output_name)


def prepare_aime25(output_dir: Path, val_limit: int | None = None) -> None:
    log_step("Preparing AIME 2025...")
    split_i = load_dataset(
        "json",
        data_files={"test": download_hf_dataset_file("opencompass/AIME2025", "aime2025-I.jsonl")},
        split="test",
    )
    split_ii = load_dataset(
        "json",
        data_files={"test": download_hf_dataset_file("opencompass/AIME2025", "aime2025-II.jsonl")},
        split="test",
    )

    rows = []
    for dataset in (split_i, split_ii):
        for sample in dataset:
            rows.append(
                {
                    "prompt": find_first(sample, ["problem", "question"]),
                    "answer": str(find_first(sample, ["answer", "final_answer", "solution"])).strip(),
                }
            )
            if val_limit is not None and len(rows) >= val_limit:
                break
        if val_limit is not None and len(rows) >= val_limit:
            break

    save_parquet(rows, output_dir / "aime25_val.parquet")


def build_mbpp_prompt(text: str) -> str:
    return "\n\n".join(
        [
            "Solve the following Python task.",
            text.strip(),
        ]
    )


def prepare_mbpp(output_dir: Path, test_limit: int | None = None) -> tuple[list[dict], list[dict]]:
    log_step("Preparing MBPP...")
    snapshot_dir = download_hf_dataset_snapshot("google-research-datasets/mbpp", ["sanitized/*.parquet", "README.md"])
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": find_local_files(snapshot_dir / "sanitized", "train-*.parquet"),
            "test": find_local_files(snapshot_dir / "sanitized", "test-*.parquet"),
        },
    )
    split_mapping = {
        "train": "mbpp_train.parquet",
        "test": "mbpp_test.parquet",
    }

    split_rows = {}
    for split_name, output_name in split_mapping.items():
        rows = []
        for sample in dataset[split_name]:
            prompt_text = find_first(sample, ["text", "prompt"])
            answer = {
                "test_type": "assert_list",
                "tests": sample["test_list"],
            }
            rows.append(
                {
                    "prompt": build_mbpp_prompt(prompt_text),
                    "answer": json.dumps(answer, ensure_ascii=False),
                }
            )
            if split_name == "test" and test_limit is not None and len(rows) >= test_limit:
                break
        split_rows[split_name] = rows
        save_parquet(rows, output_dir / output_name)

    return split_rows["train"], split_rows["test"]


def prepare_apps(
    output_dir: Path, train_limit: int, test_limit: int | None, allowed_difficulties: list[str]
) -> tuple[list[dict], list[dict]]:
    log_step("Preparing APPS...")

    def build_rows(data_file: str, split_name: str, limit: int | None = None) -> list[dict]:
        dataset = load_dataset("json", data_files={split_name: data_file}, streaming=True)
        rows = []
        scanned = 0
        for sample in dataset[split_name]:
            scanned += 1
            difficulty = sample.get("difficulty")
            if allowed_difficulties and difficulty not in allowed_difficulties:
                continue

            raw_tests = sample.get("input_output")
            if not raw_tests:
                continue

            try:
                test_spec = json.loads(raw_tests) if isinstance(raw_tests, str) else raw_tests
            except Exception:
                continue

            if not isinstance(test_spec, dict) or not test_spec.get("inputs") or not test_spec.get("outputs"):
                continue

            question = sample.get("question") or sample.get("problem")
            if not question:
                continue

            answer = {
                "test_type": "input_output",
                "inputs": test_spec["inputs"],
                "outputs": test_spec["outputs"],
                "fn_name": test_spec.get("fn_name"),
            }
            rows.append(
                {
                    "prompt": build_code_prompt(question, sample.get("starter_code")),
                    "answer": json.dumps(answer, ensure_ascii=False),
                }
            )

            if len(rows) == 1 or len(rows) % 250 == 0:
                log_step(f"APPS {split_name}: kept {len(rows)} samples after scanning {scanned} rows")

            if limit is not None and len(rows) >= limit:
                break

            if scanned % 1000 == 0:
                log_step(f"APPS {split_name}: scanned {scanned} rows, kept {len(rows)}")

        return rows

    train_path = download_hf_dataset_file("codeparrot/apps", "train.jsonl")
    train_rows = build_rows(train_path, "train", limit=train_limit)
    test_path = download_hf_dataset_file("codeparrot/apps", "test.jsonl")
    test_rows = build_rows(test_path, "test", limit=test_limit)
    save_parquet(train_rows, output_dir / "apps_train.parquet")
    save_parquet(test_rows, output_dir / "apps_test.parquet")
    return train_rows, test_rows


def build_humaneval_prompt(prompt: str) -> str:
    return "\n\n".join(
        [
            "Complete the following Python function.",
            "```python",
            prompt.rstrip(),
            "```",
        ]
    )


def is_livecodebench_target_date(contest_date: str | datetime | None) -> bool:
    if contest_date is None:
        return False
    if isinstance(contest_date, datetime):
        year_month = (contest_date.year, contest_date.month)
    elif isinstance(contest_date, str):
        match = re.match(r"(\d{4})-(\d{2})", contest_date)
        if not match:
            return False
        year_month = (int(match.group(1)), int(match.group(2)))
    else:
        return False

    return (2024, 8) <= year_month < (2025, 1)


def prepare_humaneval(output_dir: Path, val_limit: int | None = None) -> None:
    log_step("Preparing HumanEval...")
    snapshot_dir = download_hf_dataset_snapshot("openai/openai_humaneval", ["openai_humaneval/*.parquet", "README.md"])
    dataset = load_dataset(
        "parquet",
        data_files={"test": find_local_files(snapshot_dir / "openai_humaneval", "test-*.parquet")},
        split="test",
    )

    rows = []
    for sample in dataset:
        answer = {
            "test_type": "humaneval",
            "prompt_prefix": sample["prompt"],
            "test_code": sample["test"],
            "entry_point": sample["entry_point"],
        }
        rows.append(
            {
                "prompt": build_humaneval_prompt(sample["prompt"]),
                "answer": json.dumps(answer, ensure_ascii=False),
            }
        )
        if val_limit is not None and len(rows) >= val_limit:
            break

    save_parquet(rows, output_dir / "humaneval_val.parquet")


def prepare_livecodebench(output_dir: Path, val_limit: int | None = None) -> list[dict]:
    log_step("Preparing LiveCodeBench...")
    data_files = [
        download_hf_dataset_file("livecodebench/code_generation_lite", "test4.jsonl"),
        download_hf_dataset_file("livecodebench/code_generation_lite", "test5.jsonl"),
        download_hf_dataset_file("livecodebench/code_generation_lite", "test6.jsonl"),
    ]
    dataset = load_dataset("json", data_files={"test": data_files}, split="test", streaming=True)

    rows = []
    for sample in dataset:
        contest_date = sample.get("contest_date")
        if not is_livecodebench_target_date(contest_date):
            continue

        public_test_cases = json.loads(sample["public_test_cases"])
        try:
            private_test_cases = json.loads(sample["private_test_cases"])
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(sample["private_test_cases"].encode("utf-8"))))
            )

        full_test_cases = public_test_cases + private_test_cases
        metadata = json.loads(sample["metadata"])
        answer = {
            "test_type": "input_output",
            "inputs": [case["input"] for case in full_test_cases],
            "outputs": [case["output"] for case in full_test_cases],
            "fn_name": metadata.get("func_name"),
        }
        rows.append(
            {
                "prompt": build_code_prompt(sample["question_content"], sample.get("starter_code")),
                "answer": json.dumps(answer, ensure_ascii=False),
            }
        )
        if val_limit is not None and len(rows) >= val_limit:
            break

    save_parquet(rows, output_dir / "livecodebench_val.parquet")
    return rows


def prepare_coding_mix(output_dir: Path, mbpp_train: list[dict], apps_train: list[dict]) -> None:
    log_step("Preparing coding train mix...")
    coding_train_rows = mbpp_train + apps_train
    save_parquet(coding_train_rows, output_dir / "coding_train_mix.parquet")


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    prepare_gsm8k(output_dir, test_limit=args.gsm8k_test_limit)
    prepare_aime25(output_dir, val_limit=args.aime25_val_limit)
    mbpp_train, _ = prepare_mbpp(output_dir, test_limit=args.mbpp_test_limit)
    apps_train, _ = prepare_apps(
        output_dir,
        train_limit=args.apps_train_limit,
        test_limit=args.apps_test_limit,
        allowed_difficulties=args.apps_difficulties,
    )
    prepare_humaneval(output_dir, val_limit=args.humaneval_val_limit)
    prepare_livecodebench(output_dir, val_limit=args.livecodebench_val_limit)
    prepare_coding_mix(output_dir, mbpp_train=mbpp_train, apps_train=apps_train)


if __name__ == "__main__":
    main()