"""Search strategies for accept/reject decisions in the optimization loop.

Greedy strategy (MVP) uses direct comparison for deterministic evals
and paired Wilcoxon signed-rank test for stochastic evals.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from scipy.stats import wilcoxon

from anneal.engine.types import Direction, EvalResult


class SearchStrategy(Protocol):
    """Protocol allowing future strategies (e.g. simulated annealing) to be swapped in."""

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float,
        confidence: float,
    ) -> bool: ...


class GreedySearch:
    """Accept only strict improvements, verified by statistical test when possible."""

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
    ) -> bool:
        challenger_raw = challenger_result.raw_scores

        # Fall back to deterministic when paired comparison isn't possible
        if (
            baseline_raw_scores is None
            or challenger_raw is None
            or len(baseline_raw_scores) != len(challenger_raw)
        ):
            return self._deterministic_compare(
                challenger_result.score,
                baseline_score,
                direction,
                min_improvement_threshold,
            )

        return self._stochastic_compare(
            challenger_raw,
            baseline_raw_scores,
            direction,
            confidence,
        )

    @staticmethod
    def _deterministic_compare(
        challenger_score: float,
        baseline_score: float,
        direction: Direction,
        threshold: float,
    ) -> bool:
        if direction is Direction.HIGHER_IS_BETTER:
            return challenger_score > baseline_score + threshold
        return challenger_score < baseline_score - threshold

    @staticmethod
    def _stochastic_compare(
        challenger_raw: list[float],
        baseline_raw: list[float],
        direction: Direction,
        confidence: float,
    ) -> bool:
        differences = [c - b for c, b in zip(challenger_raw, baseline_raw)]
        challenger_mean = sum(challenger_raw) / len(challenger_raw)
        baseline_mean = sum(baseline_raw) / len(baseline_raw)

        # Quick rejection: mean must be in the right direction
        if direction is Direction.HIGHER_IS_BETTER and challenger_mean <= baseline_mean:
            return False
        if direction is Direction.LOWER_IS_BETTER and challenger_mean >= baseline_mean:
            return False

        alternative = (
            "greater" if direction is Direction.HIGHER_IS_BETTER else "less"
        )

        try:
            _, p_value = wilcoxon(differences, alternative=alternative)
        except ValueError:
            # All differences are zero (plateau) — no evidence of improvement
            return False

        return float(p_value) < (1 - confidence)


class SimulatedAnnealingSearch:
    """Simulated annealing: accepts regressions with decreasing probability.

    Escapes local optima by occasionally accepting worse mutations early
    in the search, then cooling to greedy behavior as the run progresses.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
    ) -> None:
        self._initial_temperature = initial_temperature
        self._temperature = initial_temperature
        self._cooling_rate = cooling_rate
        self._min_temperature = min_temperature

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
    ) -> bool:
        """Accept improvements always. Accept regressions with probability
        exp(-delta / temperature). Cool temperature after each call."""
        challenger_score = challenger_result.score

        # For stochastic evals with raw scores, use the mean
        challenger_raw = challenger_result.raw_scores
        if challenger_raw is not None and len(challenger_raw) > 0:
            challenger_score = sum(challenger_raw) / len(challenger_raw)
        if (
            baseline_raw_scores is not None
            and len(baseline_raw_scores) > 0
        ):
            baseline_score = sum(baseline_raw_scores) / len(baseline_raw_scores)

        # Compute delta so positive = improvement
        if direction is Direction.HIGHER_IS_BETTER:
            delta = challenger_score - baseline_score
        else:
            delta = baseline_score - challenger_score

        if delta > 0:
            accept = True
        else:
            # delta <= 0 (regression): accept with probability exp(delta / temperature)
            accept = random.random() < math.exp(delta / self._temperature)

        self.cool()
        return accept

    def cool(self) -> None:
        """Apply cooling: temperature *= cooling_rate."""
        self._temperature = max(
            self._temperature * self._cooling_rate,
            self._min_temperature,
        )

    @property
    def temperature(self) -> float:
        return self._temperature

    def reset(self) -> None:
        """Reset temperature to initial value."""
        self._temperature = self._initial_temperature
