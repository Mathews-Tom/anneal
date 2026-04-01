"""Eval harness for B3: Python performance benchmark.

Runs the B3 artifact, verifies correctness independently, and prints
the execution time. Returns a penalty score (99999.0) if:
- The script crashes or times out
- The match count is wrong (correctness violation)
- The output format is unparseable

This script is immutable — the agent cannot modify it.

Usage: uv run python benchmarks/suite/harness/eval_b3.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PENALTY = 99999.0
EXPECTED_MATCH_COUNT = 301
TIMEOUT_SECONDS = 60
TIME_PATTERN = re.compile(r"^execution_time_ms:\s*([0-9]+(?:\.[0-9]*)?)$")
COUNT_PATTERN = re.compile(r"^match_count:\s*([0-9]+)$")

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "B3_utility_function.py"


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
        # Script crashed — penalty
        print(PENALTY)
        return

    stdout = result.stdout

    # Parse match_count and execution_time_ms from output
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

    # Correctness gate: match count must be exactly right
    if match_count != EXPECTED_MATCH_COUNT:
        print(PENALTY)
        return

    # Sanity: execution time must be positive
    if exec_time is None or exec_time <= 0.0:
        print(PENALTY)
        return

    print(f"{exec_time:.2f}")


if __name__ == "__main__":
    main()
