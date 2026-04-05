"""Eval harness for B3: Python performance benchmark.

Runs the B3 artifact, verifies correctness independently, and prints
the execution time. Returns a penalty score (99999.0) if:
- The script crashes or times out
- The match count is wrong (correctness violation)
- Spot-check pairs are missing or have wrong distances
- The execution time is unphysical (< 1ms for this workload)
- The output format is unparseable

This script is immutable — the agent cannot modify it.

Usage: uv run python benchmarks/suite/harness/eval_b3.py
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

PENALTY = 99999.0
EXPECTED_MATCH_COUNT = 301
TIMEOUT_SECONDS = 60
TIME_PATTERN = re.compile(r"^execution_time_ms:\s*([0-9]+(?:\.[0-9]*)?)$")
COUNT_PATTERN = re.compile(r"^match_count:\s*([0-9]+)$")
# Minimum plausible execution time: 260×260 Levenshtein pairs cannot run in < 1ms
MIN_EXEC_TIME_MS = 1.0

# Resolve relative to CWD (the worktree), not __file__ (the main repo).
# The eval engine sets cwd=worktree_path, so this reads the worktree's artifact.
ARTIFACT = Path("benchmarks/suite/artifacts/B3_utility_function.py")

# Spot-check pairs: (name_a, name_b, expected_distance)
# Verified against the reference Levenshtein implementation.
# The artifact must produce these exact pairs to prove computation is real.
_SPOT_CHECKS: list[tuple[str, str, int]] = [
    ("alice", "alyce", 1),
    ("bartholomew", "bartholemew", 1),
    ("katherine", "catharine", 2),
    ("galatea", "galatea", 0),
    ("wren", "wren", 0),
    ("ozymandias", "ozymandeas", 1),
]


def _verify_spot_checks(artifact_path: Path) -> bool:
    """Import the artifact module and verify spot-check pairs exist in output."""
    spec = importlib.util.spec_from_file_location("b3_artifact", artifact_path)
    if spec is None or spec.loader is None:
        return False
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    run_fn = getattr(mod, "run_benchmark", None)
    if run_fn is None:
        return False

    # The function should also expose _find_close_pairs or produce full results.
    # We call run_benchmark and check by re-importing the internal function.
    find_fn = getattr(mod, "_find_close_pairs", None)
    names_a = getattr(mod, "_NAMES_A", None)
    names_b = getattr(mod, "_NAMES_B", None)
    if find_fn is None or names_a is None or names_b is None:
        return False

    pairs = find_fn(names_a, names_b, 2)
    pair_set = {(a, b, d) for a, b, d in pairs}
    for a, b, expected_d in _SPOT_CHECKS:
        if (a, b, expected_d) not in pair_set:
            return False
    return True


def main() -> None:
    try:
        result = subprocess.run(
            [sys.executable, str(ARTIFACT)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(PENALTY)
        return

    if result.returncode != 0:
        print(PENALTY)
        return

    stdout = result.stdout

    match_count: int | None = None
    exec_time: float | None = None

    for line in stdout.splitlines():
        stripped = line.strip()
        m = COUNT_PATTERN.match(stripped)
        if m:
            match_count = int(m.group(1))
        m = TIME_PATTERN.match(stripped)
        if m:
            exec_time = float(m.group(1))

    if match_count != EXPECTED_MATCH_COUNT:
        print(PENALTY)
        return

    if exec_time is None or exec_time < MIN_EXEC_TIME_MS:
        print(PENALTY)
        return

    # Spot-check: verify the artifact actually computes distances
    if not _verify_spot_checks(ARTIFACT):
        print(PENALTY)
        return

    print(f"{exec_time:.4f}")


if __name__ == "__main__":
    main()
