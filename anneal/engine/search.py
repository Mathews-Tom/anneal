"""Search strategies for accept/reject decisions in the optimization loop.

Greedy strategy (MVP) uses direct comparison for deterministic evals
and paired Wilcoxon signed-rank test for stochastic evals.
"""

from __future__ import annotations

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
