"""Tests for population-based search (F7): PopulationSearch add/select/keep."""

from __future__ import annotations

import pytest

from anneal.engine.search import PopulationSearch
from anneal.engine.types import Direction, EvalResult


class TestShouldKeep:
    def test_higher_is_better_keep(self) -> None:
        ps = PopulationSearch()
        result = EvalResult(score=0.9)
        assert ps.should_keep(
            result, baseline_score=0.8, baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
        ) is True

    def test_higher_is_better_reject(self) -> None:
        ps = PopulationSearch()
        result = EvalResult(score=0.7)
        assert ps.should_keep(
            result, baseline_score=0.8, baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
        ) is False

    def test_higher_is_better_equal_reject(self) -> None:
        ps = PopulationSearch()
        result = EvalResult(score=0.8)
        assert ps.should_keep(
            result, baseline_score=0.8, baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
        ) is False

    def test_lower_is_better_keep(self) -> None:
        ps = PopulationSearch()
        result = EvalResult(score=0.5)
        assert ps.should_keep(
            result, baseline_score=0.8, baseline_raw_scores=None,
            direction=Direction.LOWER_IS_BETTER,
        ) is True

    def test_lower_is_better_reject(self) -> None:
        ps = PopulationSearch()
        result = EvalResult(score=0.9)
        assert ps.should_keep(
            result, baseline_score=0.8, baseline_raw_scores=None,
            direction=Direction.LOWER_IS_BETTER,
        ) is False


class TestAddCandidate:
    def test_add_within_capacity(self) -> None:
        ps = PopulationSearch(population_size=4)
        ps.add_candidate("branch-1", 0.8)
        ps.add_candidate("branch-2", 0.9)
        assert len(ps.population) == 2

    def test_add_beyond_capacity_culls(self) -> None:
        ps = PopulationSearch(population_size=3, tournament_size=2)
        for i in range(5):
            ps.add_candidate(f"branch-{i}", float(i) / 10)
        assert len(ps.population) <= 3

    def test_population_contents(self) -> None:
        ps = PopulationSearch(population_size=4)
        ps.add_candidate("a", 0.5)
        ps.add_candidate("b", 0.7)
        branches = [b for b, _ in ps.population]
        assert "a" in branches
        assert "b" in branches


class TestTournamentSelect:
    def test_reduces_to_population_size(self) -> None:
        ps = PopulationSearch(population_size=3, tournament_size=2)
        for i in range(6):
            ps._population.append((f"branch-{i}", float(i)))

        survivors = ps.tournament_select(Direction.HIGHER_IS_BETTER)
        assert len(survivors) == 3

    def test_higher_is_better_keeps_higher_scores(self) -> None:
        ps = PopulationSearch(population_size=2, tournament_size=2)
        ps._population = [("low", 0.1), ("mid", 0.5), ("high", 0.9)]
        survivors = ps.tournament_select(Direction.HIGHER_IS_BETTER)
        scores = [s for _, s in survivors]
        # The lowest scorer should be the one eliminated
        assert 0.9 in scores

    def test_lower_is_better_keeps_lower_scores(self) -> None:
        ps = PopulationSearch(population_size=2, tournament_size=2)
        ps._population = [("low", 0.1), ("mid", 0.5), ("high", 0.9)]
        survivors = ps.tournament_select(Direction.LOWER_IS_BETTER)
        scores = [s for _, s in survivors]
        assert 0.1 in scores

    def test_already_at_size_no_change(self) -> None:
        ps = PopulationSearch(population_size=3)
        ps._population = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
        survivors = ps.tournament_select(Direction.HIGHER_IS_BETTER)
        assert len(survivors) == 3


class TestIsPopulationFull:
    def test_empty_not_full(self) -> None:
        ps = PopulationSearch(population_size=4)
        assert ps.is_population_full() is False

    def test_partial_not_full(self) -> None:
        ps = PopulationSearch(population_size=4)
        ps.add_candidate("a", 0.5)
        ps.add_candidate("b", 0.6)
        assert ps.is_population_full() is False

    def test_at_capacity_is_full(self) -> None:
        ps = PopulationSearch(population_size=3)
        ps.add_candidate("a", 0.5)
        ps.add_candidate("b", 0.6)
        ps.add_candidate("c", 0.7)
        assert ps.is_population_full() is True

    def test_over_capacity_still_full(self) -> None:
        ps = PopulationSearch(population_size=2)
        ps._population = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
        assert ps.is_population_full() is True


class TestCrossoverParents:
    """Tests for PopulationSearch crossover parent selection."""

    def test_crossover_parents_selected_returns_hypotheses(self) -> None:
        pop = PopulationSearch(population_size=4, crossover_rate=1.0)
        pop.add_candidate("branch-a", 0.9, hypothesis="Improve clarity")
        pop.add_candidate("branch-b", 0.8, hypothesis="Add examples")
        result = pop.get_crossover_parents()
        assert result is not None
        assert result == ("Improve clarity", "Add examples")

    def test_crossover_respects_rate_zero_returns_none(self) -> None:
        pop = PopulationSearch(population_size=4, crossover_rate=0.0)
        pop.add_candidate("branch-a", 0.9, hypothesis="Improve clarity")
        pop.add_candidate("branch-b", 0.8, hypothesis="Add examples")
        result = pop.get_crossover_parents()
        assert result is None

    def test_crossover_needs_two_candidates_returns_none(self) -> None:
        pop = PopulationSearch(population_size=4, crossover_rate=1.0)
        pop.add_candidate("branch-a", 0.9, hypothesis="Improve clarity")
        result = pop.get_crossover_parents()
        assert result is None
