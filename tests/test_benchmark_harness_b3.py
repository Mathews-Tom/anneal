"""Regression tests for the B3 benchmark harness trust boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.suite.config import (
    BenchmarkConfig,
    BenchmarkModelRoute,
    BenchmarkRun,
    BenchmarkTarget,
)
from benchmarks.suite.runner import _record_has_invalid_resume_score


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_B3 = REPO_ROOT / "benchmarks" / "suite" / "harness" / "eval_b3.py"


def _write_b3_artifact(tmp_path: Path, source: str) -> None:
    artifact_dir = tmp_path / "benchmarks" / "suite" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "B3_utility_function.py").write_text(source, encoding="utf-8")


def _run_eval_b3(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EVAL_B3)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _benchmark_run(target_id: str, tmp_path: Path) -> BenchmarkRun:
    return BenchmarkRun(
        target=BenchmarkTarget(
            id=target_id,
            name="target",
            domain="code",
            eval_mode="deterministic",
            criteria_count=0,
            experiment_budget=50,
            artifact_path="artifact.py",
            scope_path="scope.yaml",
        ),
        config=BenchmarkConfig(
            name="greedy",
            description="greedy",
            search_strategy="greedy",
        ),
        seed=1,
        output_dir=tmp_path,
        model_route=BenchmarkModelRoute(
            mutation_model="gpt-5.4",
            diagnosis_model="gpt-5.4-mini",
            judge_model="gpt-5.4-mini",
        ),
    )


def test_b3_eval_rejects_artifact_that_patches_parent_print(tmp_path: Path) -> None:
    _write_b3_artifact(
        tmp_path,
        """
from __future__ import annotations

import builtins
import sys

_NAMES_A = ["alice", "bartholomew", "katherine", "galatea", "wren", "ozymandias"]
_NAMES_B = ["alyce", "bartholemew", "catharine", "galatea", "wren", "ozymandeas"]


def _find_close_pairs(list_a: list[str], list_b: list[str], threshold: int) -> list[tuple[str, str, int]]:
    return [
        ("alice", "alyce", 1),
        ("bartholomew", "bartholemew", 1),
        ("katherine", "catharine", 2),
        ("galatea", "galatea", 0),
        ("wren", "wren", 0),
        ("ozymandias", "ozymandeas", 1),
    ]


def run_benchmark() -> tuple[int, float]:
    return 301, 25.0


if __name__ != "__main__":
    original_print = builtins.print

    def patched_print(*args: object, **kwargs: object) -> None:
        if len(args) == 1 and not kwargs and isinstance(args[0], str):
            original_print("-inf")
            return
        original_print(*args, **kwargs)

    builtins.print = patched_print


if __name__ == "__main__":
    print("match_count: 301")
    print("execution_time_ms: 25.00")
""",
    )

    result = _run_eval_b3(tmp_path)

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "25.0000"


def test_b3_eval_penalizes_failed_spot_checks(tmp_path: Path) -> None:
    _write_b3_artifact(
        tmp_path,
        """
from __future__ import annotations

_NAMES_A = ["alice"]
_NAMES_B = ["alyce"]


def _find_close_pairs(list_a: list[str], list_b: list[str], threshold: int) -> list[tuple[str, str, int]]:
    return []


def run_benchmark() -> tuple[int, float]:
    return 301, 25.0


if __name__ == "__main__":
    print("match_count: 301")
    print("execution_time_ms: 25.00")
""",
    )

    result = _run_eval_b3(tmp_path)

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "99999.0"


def test_b3_resume_guard_rejects_non_finite_scores(tmp_path: Path) -> None:
    run = _benchmark_run("B3", tmp_path)

    assert _record_has_invalid_resume_score(run, {"score": float("-inf")}) is True


def test_b3_resume_guard_rejects_sub_minimum_scores(tmp_path: Path) -> None:
    run = _benchmark_run("B3", tmp_path)

    assert _record_has_invalid_resume_score(run, {"score": 0.0}) is True


def test_resume_guard_allows_zero_for_non_b3_targets(tmp_path: Path) -> None:
    run = _benchmark_run("B4", tmp_path)

    assert _record_has_invalid_resume_score(run, {"score": 0.0}) is False
