"""Search strategies for accept/reject decisions in the optimization loop.

Greedy strategy (MVP) uses direct comparison for deterministic evals
and paired Wilcoxon signed-rank test for stochastic evals.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Protocol

logger = logging.getLogger(__name__)

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
            logger.warning(
                "Falling back to deterministic comparison: "
                "baseline_raw=%s, challenger_raw=%s",
                len(baseline_raw_scores) if baseline_raw_scores else None,
                len(challenger_raw) if challenger_raw else None,
            )
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


class PopulationSearch:
    """Population-based search with tournament selection.

    Maintains a population of candidates. Uses tournament selection
    (random pairs, keep higher-scoring member). No crossover.
    """

    def __init__(
        self,
        population_size: int = 4,
        tournament_size: int = 2,
    ) -> None:
        self._population_size = population_size
        self._tournament_size = tournament_size
        self._population: list[tuple[str, float]] = []

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
    ) -> bool:
        """Compare challenger against baseline per direction.

        Returns True if the challenger beats the baseline score.
        The runner manages multi-branch logic; this handles per-experiment decisions.
        """
        challenger_score = challenger_result.score
        if direction is Direction.HIGHER_IS_BETTER:
            return challenger_score > baseline_score
        return challenger_score < baseline_score

    def add_candidate(self, branch: str, score: float) -> None:
        """Add a candidate to the population. Cull via tournament if oversized."""
        self._population.append((branch, score))
        if len(self._population) > self._population_size:
            self._population = self.tournament_select(Direction.HIGHER_IS_BETTER)

    def tournament_select(self, direction: Direction) -> list[tuple[str, float]]:
        """Run tournament selection. Returns surviving candidates.

        Repeatedly samples ``tournament_size`` candidates, keeps the best,
        until the population is reduced to ``population_size``.
        """
        survivors: list[tuple[str, float]] = list(self._population)
        while len(survivors) > self._population_size:
            pool_size = min(self._tournament_size, len(survivors))
            contestants = random.sample(survivors, pool_size)
            if direction is Direction.HIGHER_IS_BETTER:
                loser = min(contestants, key=lambda c: c[1])
            else:
                loser = max(contestants, key=lambda c: c[1])
            survivors.remove(loser)
        return survivors

    def is_population_full(self) -> bool:
        """Check if population has reached target size."""
        return len(self._population) >= self._population_size

    @property
    def population(self) -> list[tuple[str, float]]:
        """Current population."""
        return list(self._population)
