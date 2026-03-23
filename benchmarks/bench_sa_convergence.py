"""Gate: SA-adaptive vs SA-fixed convergence on Rastrigin.

Simulates 100-step optimization runs on the Rastrigin function.
Adaptive SA (with reheat) should reach a >=15% lower best Rastrigin
value than fixed SA (reheat_factor=1.0, i.e. no effective reheat).

The comparison is statistical: each of 50 fixed-SA runs uses an
independent seed from each of 50 adaptive-SA runs. The best score
achieved during the trajectory (not the final score) is the metric,
since reheat helps escape local optima mid-run rather than improving
the final converged value.

Parameters are calibrated so fixed SA cools too fast and gets stuck
while adaptive SA can reheat and explore further local optima.

Usage: uv run python benchmarks/bench_sa_convergence.py
"""
from __future__ import annotations

import math
import random
import sys

from anneal.engine.search import SimulatedAnnealingSearch
from anneal.engine.types import Direction, EvalResult


def rastrigin(x: list[float]) -> float:
    """Rastrigin function (minimize). Global minimum = 0 at x = [0, 0, ...]."""
    n = len(x)
    return 10 * n + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def mutate(x: list[float], step_size: float = 0.5) -> list[float]:
    """Random Gaussian mutation."""
    return [xi + random.gauss(0, step_size) for xi in x]


def run_optimization(
    sa: SimulatedAnnealingSearch,
    seed: int,
    n_experiments: int = 100,
    n_dims: int = 3,
) -> float:
    """Run SA optimization from a random start. Returns best score found."""
    random.seed(seed)
    current = [random.uniform(-5, 5) for _ in range(n_dims)]
    current_score = rastrigin(current)
    best_score = current_score

    for _ in range(n_experiments):
        candidate = mutate(current)
        candidate_score = rastrigin(candidate)

        result = EvalResult(score=candidate_score)

        # LOWER_IS_BETTER for Rastrigin (minimize)
        kept = sa.should_keep(
            result,
            baseline_score=current_score,
            baseline_raw_scores=None,
            direction=Direction.LOWER_IS_BETTER,
        )

        if kept:
            current = candidate
            current_score = candidate_score

        best_score = min(best_score, current_score)

    return best_score


def main() -> None:
    n_runs = 50
    n_experiments = 100

    # Parameters calibrated so fixed SA cools too quickly and gets stuck.
    # Adaptive SA reheats when acceptance drops below target, allowing escape.
    initial_temperature = 10.0
    cooling_rate = 0.75
    min_temperature = 0.01

    # Fixed SA: reheat_factor=1.0 means temperature *= 1.0 in reheat branch (no change)
    fixed_best: list[float] = []
    # Adaptive SA: reheat_factor=3.0 triples temperature when stuck
    adaptive_best: list[float] = []

    for i in range(n_runs):
        # Independent seeds: fixed uses even seeds, adaptive uses odd seeds.
        # Using the same seed would cause both to diverge unpredictably because
        # reheat changes which regressions are accepted, consuming random numbers
        # at different rates along the two trajectories.
        fixed_sa = SimulatedAnnealingSearch(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            reheat_factor=1.0,        # No effective reheat
            acceptance_target=0.3,
        )
        fixed_best.append(run_optimization(fixed_sa, seed=i * 2, n_experiments=n_experiments))

        adaptive_sa = SimulatedAnnealingSearch(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            reheat_factor=3.0,        # Reheat enabled: triples temperature when stuck
            acceptance_target=0.3,
        )
        adaptive_best.append(run_optimization(adaptive_sa, seed=i * 2 + 1, n_experiments=n_experiments))

    avg_fixed = sum(fixed_best) / n_runs
    avg_adaptive = sum(adaptive_best) / n_runs
    improvement = (avg_fixed - avg_adaptive) / max(abs(avg_fixed), 1e-10) * 100

    # Count wins
    adaptive_wins = sum(1 for a, f in zip(adaptive_best, fixed_best) if a < f)

    print("=" * 60)
    print("Phase 3 Gate: SA Convergence on Rastrigin")
    print("=" * 60)
    print(f"Runs: {n_runs} x {n_experiments} experiments (3D Rastrigin)")
    print(f"Metric: best score found during trajectory")
    print(f"SA params: T0={initial_temperature}, cooling={cooling_rate}, min_T={min_temperature}")
    print()
    print(f"Fixed SA best score (avg):    {avg_fixed:.4f}")
    print(f"Adaptive SA best score (avg): {avg_adaptive:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Adaptive wins: {adaptive_wins}/{n_runs}")
    print()

    passed = improvement >= 15
    print(f"Gate (>=15% improvement): {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
