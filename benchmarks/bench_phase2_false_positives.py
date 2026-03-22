"""Phase 2 Gate: Verify Holm-Bonferroni correction reduces false positive rate.

Runs 100 synthetic 50-experiment windows with known null distribution
(no real improvement). Counts how many false positives occur with and
without the correction. Expected: correction reduces FP rate by >50%.

Usage: uv run python benchmarks/bench_phase2_false_positives.py
"""
from __future__ import annotations

import random
import sys

from anneal.engine.search import GreedySearch
from anneal.engine.types import Direction, EvalResult


def run_null_experiment(
    n_samples: int = 10,
    use_correction: bool = False,
    window_size: int = 50,
) -> int:
    """Run a window of experiments with null distribution (no real improvement).

    Returns number of false positives (mutations incorrectly accepted).
    """
    search = GreedySearch()
    false_positives = 0
    base_alpha = 0.05

    for experiment_idx in range(window_size):
        # Generate null data: both challenger and baseline from same distribution
        baseline_raw = [random.gauss(0.5, 0.1) for _ in range(n_samples)]
        challenger_raw = [random.gauss(0.5, 0.1) for _ in range(n_samples)]

        if use_correction:
            adjusted_alpha = GreedySearch._adjusted_alpha(base_alpha, experiment_idx, window_size)
            confidence = 1 - adjusted_alpha
        else:
            confidence = 0.95

        result = EvalResult(
            score=sum(challenger_raw) / len(challenger_raw),
            raw_scores=challenger_raw,
        )

        kept = search.should_keep(
            result,
            baseline_score=sum(baseline_raw) / len(baseline_raw),
            baseline_raw_scores=baseline_raw,
            direction=Direction.HIGHER_IS_BETTER,
            min_improvement_threshold=0.0,
            confidence=confidence,
        )

        if kept:
            false_positives += 1

    return false_positives


def main() -> None:
    random.seed(42)
    n_runs = 100

    fp_without: list[int] = []
    fp_with: list[int] = []

    for _ in range(n_runs):
        fp_without.append(run_null_experiment(use_correction=False))
        fp_with.append(run_null_experiment(use_correction=True))

    avg_without = sum(fp_without) / n_runs
    avg_with = sum(fp_with) / n_runs
    reduction = (avg_without - avg_with) / max(avg_without, 1e-10) * 100

    print("=" * 60)
    print("Phase 2 Gate: False Positive Rate Validation")
    print("=" * 60)
    print(f"Runs: {n_runs} x 50-experiment windows")
    print(f"Null distribution: N(0.5, 0.1), n=10 paired samples")
    print()
    print(f"Without correction: {avg_without:.1f} avg false positives per window")
    print(f"With Holm-Bonferroni: {avg_with:.1f} avg false positives per window")
    print(f"Reduction: {reduction:.0f}%")
    print()

    passed = reduction > 50
    print(f"Gate (>50% reduction): {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
