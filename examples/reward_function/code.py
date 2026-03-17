import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REWARD_NAME = "code"
REWARD_TYPE = "batch"

CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _normalize_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth

    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            return {"test_type": "assert_list", "tests": [ground_truth]}
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Unsupported coding ground truth type: {type(ground_truth)}")


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _extract_code(response: str) -> tuple[str, float]:
    cleaned = THINK_PATTERN.sub("", response).strip()
    matches = CODE_BLOCK_PATTERN.findall(cleaned)
    if matches:
        code = max(matches, key=len).strip()
        return code, 1.0

    return cleaned, 0.0


def _build_candidate_program(code: str, spec: dict[str, Any]) -> str:
    prompt_prefix = spec.get("prompt_prefix", "")
    entry_point = spec.get("entry_point")

    if prompt_prefix and entry_point and f"def {entry_point}" not in code:
        return f"{prompt_prefix}{code}"

    if prompt_prefix and spec.get("prepend_prompt_prefix", False):
        return f"{prompt_prefix}\n{code}"

    return code


def _syntax_reward(program: str) -> float:
    if not program.strip():
        return 0.0

    try:
        ast.parse(program)
    except SyntaxError:
        return 0.0

    return 1.0


def _run_python(code: str, timeout: float) -> bool:
    with tempfile.TemporaryDirectory(prefix="easyr1_code_reward_") as temp_dir:
        script_path = Path(temp_dir) / "check.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    return True


def _run_python_with_stdio(code: str, stdin_text: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="easyr1_code_reward_") as temp_dir:
        script_path = Path(temp_dir) / "check.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True,
                input=stdin_text,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False, ""

    return True, result.stdout


def _normalize_text_output(text: Any) -> str:
    return str(text).replace("\r\n", "\n").strip()


def _compare_values(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True

    if isinstance(expected, list) and len(expected) == 1 and actual == expected[0]:
        return True

    if isinstance(actual, tuple) and list(actual) == expected:
        return True

    if isinstance(actual, list) and isinstance(expected, tuple) and actual == list(expected):
        return True

    return _normalize_text_output(actual) == _normalize_text_output(expected)


def _run_call_based_case(program: str, fn_name: str, test_input: Any, expected_output: Any, timeout: float) -> bool:
    payload = _safe_json_loads(test_input)
    expected = _safe_json_loads(expected_output)
    harness = "\n".join(
        [
            "import json",
            program,
            f"candidate = {fn_name}",
            f"payload = json.loads({json.dumps(json.dumps(payload))})",
            f"expected = json.loads({json.dumps(json.dumps(expected))})",
            "if isinstance(payload, dict):",
            "    result = candidate(**payload)",
            "elif isinstance(payload, list):",
            "    result = candidate(*payload)",
            "else:",
            "    result = candidate(payload)",
            "print(json.dumps(result))",
            "print('__EASYR1_EXPECTED__')",
            "print(json.dumps(expected))",
        ]
    )
    ok, stdout = _run_python_with_stdio(harness, stdin_text="", timeout=timeout)
    if not ok:
        return False

    parts = stdout.split("__EASYR1_EXPECTED__")
    if len(parts) != 2:
        return False

    actual = _safe_json_loads(parts[0].strip())
    expected = _safe_json_loads(parts[1].strip())
    return _compare_values(actual, expected)


def _run_stdio_case(program: str, test_input: Any, expected_output: Any, timeout: float) -> bool:
    ok, stdout = _run_python_with_stdio(program, stdin_text=str(test_input), timeout=timeout)
    if not ok:
        return False

    return _normalize_text_output(stdout) == _normalize_text_output(expected_output)


def _input_output_pass_ratio(spec: dict[str, Any], program: str, timeout: float) -> float:
    inputs = spec.get("inputs", [])
    outputs = spec.get("outputs", [])
    fn_name = spec.get("fn_name")
    if not inputs or not outputs or len(inputs) != len(outputs):
        return 0.0

    passed = 0
    for test_input, expected_output in zip(inputs, outputs):
        if fn_name:
            passed += int(_run_call_based_case(program, fn_name, test_input, expected_output, timeout=timeout))
        else:
            passed += int(_run_stdio_case(program, test_input, expected_output, timeout=timeout))

    return passed / len(inputs)


def _pass_ratio(program: str, spec: dict[str, Any], timeout: float) -> float:
    setup_code = spec.get("setup_code", "")
    test_type = spec.get("test_type", "assert_list")

    if test_type == "humaneval":
        test_code = spec.get("test_code", "")
        entry_point = spec.get("entry_point", "candidate")
        runner = "\n\n".join(
            part
            for part in (
                program,
                setup_code,
                test_code,
                f"check({entry_point})",
            )
            if part
        )
        return 1.0 if _run_python(runner, timeout=timeout) else 0.0

    if test_type == "input_output":
        runner = "\n\n".join(part for part in (program, setup_code) if part)
        return _input_output_pass_ratio(spec, runner, timeout=timeout)

    tests = spec.get("tests", [])
    if isinstance(tests, str):
        tests = [tests]

    if not tests:
        return 0.0

    passed = 0
    for test in tests:
        runner = "\n\n".join(part for part in (program, setup_code, test) if part)
        passed += int(_run_python(runner, timeout=timeout))

    return passed / len(tests)


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.05,
    syntax_weight: float = 0.05,
    timeout: float = 5.0,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        spec = _normalize_ground_truth(reward_input["ground_truth"])
        code, format_score = _extract_code(reward_input["response"])
        program = _build_candidate_program(code, spec)
        syntax_score = _syntax_reward(program)
        accuracy_score = _pass_ratio(program, spec, timeout=timeout) if syntax_score else 0.0
        overall = max(0.0, 1.0 - format_weight - syntax_weight) * accuracy_score
        overall += format_weight * format_score + syntax_weight * syntax_score
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "syntax": syntax_score,
                "accuracy": accuracy_score,
            }
        )

    return scores