from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ARTIFACT_PATH = Path("benchmarks/suite/artifacts/B4_api_endpoint.py")
LINT_VIOLATION_SCALE = 20


def check_correctness() -> float:
    try:
        spec = importlib.util.spec_from_file_location(
            "B4_api_endpoint", ARTIFACT_PATH
        )
        if spec is None or spec.loader is None:
            return 0.0
        module = importlib.util.module_from_spec(spec)
        artifacts_dir = str(ARTIFACT_PATH.parent.resolve())
        if artifacts_dir not in sys.path:
            sys.path.insert(0, artifacts_dir)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        result = module._run_self_test()
        return 1.0 if result is True else 0.0
    except Exception:
        return 0.0


def check_lint() -> tuple[int, float]:
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                str(ARTIFACT_PATH),
                "--output-format=json",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        violations_data = json.loads(proc.stdout) if proc.stdout.strip() else []
        violations = len(violations_data)
        score = max(0.0, 1.0 - (violations / LINT_VIOLATION_SCALE))
        return violations, score
    except Exception:
        return 0, 0.0


def check_type_coverage() -> float:
    try:
        source = ARTIFACT_PATH.read_text()
        tree = ast.parse(source)

        total_functions = 0
        annotated_returns = 0
        total_params = 0
        annotated_params = 0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            total_functions += 1
            if node.returns is not None:
                annotated_returns += 1

            args = node.args
            all_args = (
                args.posonlyargs
                + args.args
                + args.kwonlyargs
                + ([args.vararg] if args.vararg else [])
                + ([args.kwarg] if args.kwarg else [])
            )
            # exclude self
            non_self_args = [a for a in all_args if a.arg != "self"]
            total_params += len(non_self_args)
            annotated_params += sum(
                1 for a in non_self_args if a.annotation is not None
            )

        if total_functions == 0:
            return 0.0

        return_coverage = annotated_returns / total_functions
        param_coverage = (
            annotated_params / total_params if total_params > 0 else 1.0
        )
        return (return_coverage + param_coverage) / 2.0
    except Exception:
        return 0.0


def main() -> None:
    correctness = check_correctness()
    lint_violations, lint_score = check_lint()
    type_coverage = check_type_coverage()

    composite = 0.5 * correctness + 0.3 * lint_score + 0.2 * type_coverage

    print(f"correctness: {correctness:.1f}")
    print(f"lint_violations: {lint_violations}")
    print(f"lint_score: {lint_score:.2f}")
    print(f"type_coverage: {type_coverage:.2f}")
    print(f"composite: {composite:.3f}")


if __name__ == "__main__":
    main()
