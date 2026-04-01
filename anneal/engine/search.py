"""Search strategies for accept/reject decisions in the optimization loop.

Greedy strategy (MVP) uses direct comparison for deterministic evals
and paired Wilcoxon signed-rank test for stochastic evals.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Protocol

from scipy.stats import wilcoxon

from anneal.engine.types import Direction, EvalResult

logger = logging.getLogger(__name__)


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
        experiment_index: int = ...,
        holm_bonferroni: bool = ...,
    ) -> bool: ...


class GreedySearch:
    """Accept only strict improvements, verified by statistical test when possible."""

    MIN_PAIRED_SAMPLES: int = 6

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
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
            experiment_index,
            holm_bonferroni,
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
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
    ) -> bool:
        n = len(challenger_raw)
        differences = [c - b for c, b in zip(challenger_raw, baseline_raw)]
        challenger_mean = sum(challenger_raw) / n
        baseline_mean = sum(baseline_raw) / n

        # Quick rejection: mean must be in the right direction
        if direction is Direction.HIGHER_IS_BETTER and challenger_mean <= baseline_mean:
            return False
        if direction is Direction.LOWER_IS_BETTER and challenger_mean >= baseline_mean:
            return False

        # Insufficient samples: fall back to effect-size threshold
        if n < GreedySearch.MIN_PAIRED_SAMPLES:
            logger.warning(
                "Only %d paired samples (need %d for Wilcoxon). "
                "Using effect-size threshold instead.",
                n, GreedySearch.MIN_PAIRED_SAMPLES,
            )
            mean_diff = sum(differences) / n
            std_diff = (sum((d - mean_diff) ** 2 for d in differences) / max(n - 1, 1)) ** 0.5
            if std_diff == 0:
                return mean_diff != 0  # All identical differences
            effect_size = abs(mean_diff) / std_diff
            return effect_size > 0.5  # Medium effect (Cohen's d)

        alternative = (
            "greater" if direction is Direction.HIGHER_IS_BETTER else "less"
        )

        try:
            _, p_value = wilcoxon(differences, alternative=alternative)
        except ValueError:
            # All differences are zero (plateau) — no evidence of improvement
            return False

        alpha = 1 - confidence
        if holm_bonferroni:
            alpha = GreedySearch._adjusted_alpha(alpha, experiment_index)
        return float(p_value) < alpha

    @staticmethod
    def _adjusted_alpha(
        base_alpha: float,
        experiment_index: int,
        window_size: int = 50,
    ) -> float:
        """Holm-Bonferroni adjusted alpha for sequential testing.
        Divides alpha by remaining comparisons in the window.
        """
        remaining = max(1, window_size - experiment_index)
        return base_alpha / remaining


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
        reheat_factor: float = 2.0,
        acceptance_target: float = 0.3,
    ) -> None:
        self._initial_temperature = initial_temperature
        self._temperature = initial_temperature
        self._cooling_rate = cooling_rate
        self._min_temperature = min_temperature
        self._reheat_factor = reheat_factor
        self._acceptance_target = acceptance_target
        self._accept_history: list[bool] = []
        self._window_size = 10

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
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

        self._accept_history.append(accept)
        self.cool()
        return accept

    def cool(self) -> None:
        """Adaptive cooling: adjust rate based on acceptance ratio."""
        self._temperature = max(
            self._temperature * self._cooling_rate,
            self._min_temperature,
        )
        # Adaptive: if acceptance ratio drops below target, reheat
        if len(self._accept_history) >= self._window_size:
            recent = self._accept_history[-self._window_size:]
            acceptance_ratio = sum(recent) / len(recent)
            if acceptance_ratio < self._acceptance_target * 0.5:
                self._temperature = min(
                    self._temperature * self._reheat_factor,
                    self._initial_temperature,
                )
                logger.info("SA reheat: T=%.4f (acceptance=%.2f)", self._temperature, acceptance_ratio)

    @property
    def temperature(self) -> float:
        return self._temperature

    def reset(self) -> None:
        """Reset temperature to initial value."""
        self._temperature = self._initial_temperature

    @property
    def initial_temperature(self) -> float:
        return self._initial_temperature

    @property
    def temperature_ratio(self) -> float:
        """Current temperature as fraction of initial temperature (0.0 to 1.0)."""
        if self._initial_temperature <= 0:
            return 0.0
        return self._temperature / self._initial_temperature


class PopulationSearch:
    """Population-based search with tournament selection.

    Maintains a population of candidates. Uses tournament selection
    (random pairs, keep higher-scoring member). No crossover.
    """

    def __init__(
        self,
        population_size: int = 4,
        tournament_size: int = 2,
        crossover_rate: float = 0.3,
    ) -> None:
        self._population_size = population_size
        self._tournament_size = tournament_size
        self._crossover_rate = crossover_rate
        self._population: list[tuple[str, float]] = []
        self._hypotheses: dict[str, str] = {}

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
    ) -> bool:
        """Compare challenger against baseline per direction.

        Returns True if the challenger beats the baseline score.
        The runner manages multi-branch logic; this handles per-experiment decisions.
        """
        challenger_score = challenger_result.score
        if direction is Direction.HIGHER_IS_BETTER:
            return challenger_score > baseline_score
        return challenger_score < baseline_score

    def add_candidate(self, branch: str, score: float, hypothesis: str = "") -> None:
        """Add a candidate with its hypothesis for crossover."""
        self._population.append((branch, score))
        if hypothesis:
            self._hypotheses[branch] = hypothesis
        if len(self._population) > self._population_size:
            self._population = self.tournament_select(Direction.HIGHER_IS_BETTER)

    def get_crossover_parents(self) -> tuple[str, str] | None:
        """Select two parents for crossover if population has >= 2 candidates.

        Returns (hypothesis_a, hypothesis_b) or None.
        """
        if len(self._population) < 2 or random.random() > self._crossover_rate:
            return None
        sorted_pop = sorted(self._population, key=lambda c: c[1], reverse=True)
        parent_a = sorted_pop[0][0]
        parent_b = sorted_pop[1][0]
        hyp_a = self._hypotheses.get(parent_a, "")
        hyp_b = self._hypotheses.get(parent_b, "")
        if hyp_a and hyp_b:
            return (hyp_a, hyp_b)
        return None

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


class ParetoSearch:
    """Multi-objective search using Pareto dominance.

    Instead of comparing scalar scores, compares vectors of per-criterion
    scores. A challenger dominates if it's >= on all criteria and > on at
    least one. Non-dominated solutions are always kept.
    """

    def __init__(self, criterion_weights: dict[str, float] | None = None) -> None:
        self._criterion_weights = criterion_weights or {}
        self._pareto_front: list[dict[str, float]] = []

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
    ) -> bool:
        """Accept if challenger Pareto-dominates baseline or is non-dominated.

        Falls back to scalar comparison if per_criterion_scores unavailable.
        """
        challenger_criteria = challenger_result.per_criterion_scores
        if not challenger_criteria:
            # No per-criterion data: fall back to scalar comparison
            if direction is Direction.HIGHER_IS_BETTER:
                return challenger_result.score > baseline_score + min_improvement_threshold
            return challenger_result.score < baseline_score - min_improvement_threshold

        # Check Pareto dominance against current front
        if not self._pareto_front:
            self._pareto_front.append(challenger_criteria)
            return True

        dominated_by_any = False
        dominates_any = False

        for front_point in self._pareto_front:
            if self._dominates(front_point, challenger_criteria, direction):
                dominated_by_any = True
                break
            if self._dominates(challenger_criteria, front_point, direction):
                dominates_any = True

        if dominated_by_any:
            return False

        # Non-dominated: add to front and remove dominated points
        if dominates_any:
            self._pareto_front = [
                p for p in self._pareto_front
                if not self._dominates(challenger_criteria, p, direction)
            ]
        self._pareto_front.append(challenger_criteria)
        return True

    @staticmethod
    def _dominates(
        a: dict[str, float],
        b: dict[str, float],
        direction: Direction,
    ) -> bool:
        """Return True if a Pareto-dominates b.

        a dominates b iff a >= b on all shared criteria and a > b on at least one.
        """
        shared_keys = set(a.keys()) & set(b.keys())
        if not shared_keys:
            return False

        all_geq = True
        any_gt = False
        for key in shared_keys:
            if direction is Direction.HIGHER_IS_BETTER:
                if a[key] < b[key]:
                    all_geq = False
                    break
                if a[key] > b[key]:
                    any_gt = True
            else:
                if a[key] > b[key]:
                    all_geq = False
                    break
                if a[key] < b[key]:
                    any_gt = True

        return all_geq and any_gt

    def scalarize(self, criteria: dict[str, float]) -> float:
        """Weighted sum scalarization for ranking."""
        if not self._criterion_weights:
            return sum(criteria.values()) / max(len(criteria), 1)
        total = 0.0
        weight_sum = 0.0
        for name, value in criteria.items():
            w = self._criterion_weights.get(name, 1.0)
            total += w * value
            weight_sum += w
        return total / max(weight_sum, 1e-10)

    def save_front(self, path: Path) -> None:
        """Persist current Pareto front to JSON for dashboard consumption."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._pareto_front))

    def load_front(self, path: Path) -> None:
        """Restore Pareto front from JSON."""
        if path.exists():
            self._pareto_front = json.loads(path.read_text())

    @property
    def pareto_front(self) -> list[dict[str, float]]:
        """Current Pareto front."""
        return list(self._pareto_front)


class IslandPopulationSearch:
    """Island-based population search with periodic migration.

    Maintains N independent PopulationSearch instances ("islands").
    Every K experiments, migrates the best individual from each island
    to every other island.
    """

    def __init__(
        self,
        island_count: int = 2,
        population_per_island: int = 4,
        tournament_size: int = 2,
        migration_interval: int = 10,
        crossover_rate: float = 0.3,
    ) -> None:
        self._islands = [
            PopulationSearch(
                population_size=population_per_island,
                tournament_size=tournament_size,
                crossover_rate=crossover_rate,
            )
            for _ in range(island_count)
        ]
        self._migration_interval = migration_interval
        self._experiment_count = 0
        self._current_island: int = 0

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
    ) -> bool:
        """Delegate to current island's should_keep."""
        return self._current_island_instance.should_keep(
            challenger_result, baseline_score, baseline_raw_scores,
            direction, min_improvement_threshold, confidence,
        )

    def select_island(self) -> int:
        """Round-robin island selection. Returns island index."""
        self._current_island = self._experiment_count % len(self._islands)
        self._experiment_count += 1
        return self._current_island

    def add_candidate(self, branch: str, score: float, hypothesis: str = "") -> None:
        """Add candidate to current island."""
        self._current_island_instance.add_candidate(branch, score, hypothesis)

    def migrate(self, direction: Direction) -> int:
        """Migrate best individual from each island to every other island.

        Returns number of migrations performed.
        """
        if len(self._islands) < 2:
            return 0

        migrations = 0
        bests: list[tuple[str, float, str]] = []

        for island in self._islands:
            if not island.population:
                continue
            if direction is Direction.HIGHER_IS_BETTER:
                best = max(island.population, key=lambda c: c[1])
            else:
                best = min(island.population, key=lambda c: c[1])
            hyp = island._hypotheses.get(best[0], "")
            bests.append((best[0], best[1], hyp))

        for i, (branch, score, hyp) in enumerate(bests):
            for j, island in enumerate(self._islands):
                if i != j:
                    island.add_candidate(branch, score, hyp)
                    migrations += 1

        logger.info(
            "Island migration: %d candidates transferred across %d islands",
            migrations, len(self._islands),
        )
        return migrations

    def should_migrate(self) -> bool:
        """Check if migration is due based on experiment count."""
        return (
            self._experiment_count > 0
            and self._experiment_count % self._migration_interval == 0
        )

    @property
    def _current_island_instance(self) -> PopulationSearch:
        return self._islands[self._current_island]

    def get_crossover_parents(self) -> tuple[str, str] | None:
        """Get crossover parents from current island."""
        return self._current_island_instance.get_crossover_parents()

    @property
    def island_summaries(self) -> list[dict[str, int | float]]:
        """Summary stats per island for logging."""
        return [
            {
                "island": i,
                "population_size": len(island.population),
                "best_score": max((s for _, s in island.population), default=0.0),
            }
            for i, island in enumerate(self._islands)
        ]


class HybridSearch:
    """Simple-mode default: greedy for the first N experiments, then simulated annealing.

    Provides a sensible default for new users who do not want to think about
    search strategy selection.  The greedy phase quickly establishes a good
    baseline; the annealing phase then explores the neighbourhood.
    """

    def __init__(
        self,
        greedy_phase_length: int = 10,
    ) -> None:
        self._greedy_phase_length = greedy_phase_length
        self._call_count = 0
        self._greedy = GreedySearch()
        self._annealing = SimulatedAnnealingSearch()

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
        experiment_index: int = 0,
        holm_bonferroni: bool = False,
    ) -> bool:
        """Delegate to greedy for the first N calls, then to simulated annealing."""
        self._call_count += 1
        active = self._greedy if self._call_count <= self._greedy_phase_length else self._annealing
        return active.should_keep(
            challenger_result,
            baseline_score,
            baseline_raw_scores,
            direction,
            min_improvement_threshold,
            confidence,
            experiment_index,
            holm_bonferroni,
        )
