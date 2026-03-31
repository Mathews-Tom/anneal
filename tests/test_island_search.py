"""Tests for IslandPopulationSearch island-based population search."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from anneal.engine.search import IslandPopulationSearch
from anneal.engine.types import Direction, EvalResult


def _make_eval_result(score: float, raw_scores: list[float] | None = None) -> EvalResult:
    return EvalResult(
        score=score,
        ci_lower=None,
        ci_upper=None,
        raw_scores=raw_scores,
        cost_usd=0.0,
    )


class TestIslandPopulationSearch:
    """Tests for IslandPopulationSearch."""

    def test_island_round_robin_selection_distributes_evenly(self) -> None:
        # Arrange
        search = IslandPopulationSearch(island_count=3, population_per_island=4)

        # Act
        selections = [search.select_island() for _ in range(6)]

        # Assert
        assert selections == [0, 1, 2, 0, 1, 2]

    def test_island_migration_transfers_best(self) -> None:
        # Arrange
        search = IslandPopulationSearch(island_count=2, population_per_island=4)

        # Add candidates to island 0
        search._current_island = 0
        search.add_candidate("branch_a", 0.9)
        search.add_candidate("branch_b", 0.7)

        # Add candidates to island 1
        search._current_island = 1
        search.add_candidate("branch_c", 0.8)
        search.add_candidate("branch_d", 0.6)

        # Act
        migrations = search.migrate(Direction.HIGHER_IS_BETTER)

        # Assert
        assert migrations == 2
        island_0_branches = [b for b, _ in search._islands[0].population]
        island_1_branches = [b for b, _ in search._islands[1].population]
        # Island 0 should now contain branch_c (best from island 1)
        assert "branch_c" in island_0_branches
        # Island 1 should now contain branch_a (best from island 0)
        assert "branch_a" in island_1_branches

    def test_island_migration_interval_returns_true_at_interval(self) -> None:
        # Arrange
        search = IslandPopulationSearch(
            island_count=2,
            population_per_island=4,
            migration_interval=5,
        )

        # Act / Assert
        for _ in range(4):
            search.select_island()
        assert search.should_migrate() is False

        search.select_island()  # count = 5
        assert search.should_migrate() is True

        for _ in range(4):
            search.select_island()
        assert search.should_migrate() is False

        search.select_island()  # count = 10
        assert search.should_migrate() is True

    def test_island_count_1_no_migration(self) -> None:
        # Arrange
        search = IslandPopulationSearch(island_count=1, population_per_island=4)
        search._current_island = 0
        search.add_candidate("branch_x", 0.5)

        # Act
        result = search.migrate(Direction.HIGHER_IS_BETTER)

        # Assert
        assert result == 0

    def test_island_crossover_within_island(self) -> None:
        # Arrange
        search = IslandPopulationSearch(
            island_count=2,
            population_per_island=4,
            crossover_rate=1.0,  # guarantee crossover triggers
        )
        search._current_island = 0
        search.add_candidate("branch_a", 0.9, hypothesis="hyp_a")
        search.add_candidate("branch_b", 0.7, hypothesis="hyp_b")

        # Island 1 gets different candidates
        search._current_island = 1
        search.add_candidate("branch_c", 0.6, hypothesis="hyp_c")
        search.add_candidate("branch_d", 0.4, hypothesis="hyp_d")

        # Act
        search._current_island = 0
        parents = search.get_crossover_parents()

        # Assert — parents come from island 0's hypotheses
        assert parents is not None
        assert parents == ("hyp_a", "hyp_b")

    def test_island_summary_stats(self) -> None:
        # Arrange
        search = IslandPopulationSearch(island_count=2, population_per_island=4)

        search._current_island = 0
        search.add_candidate("branch_a", 0.9)
        search.add_candidate("branch_b", 0.7)

        search._current_island = 1
        search.add_candidate("branch_c", 0.8)

        # Act
        summaries = search.island_summaries

        # Assert
        assert len(summaries) == 2
        assert summaries[0]["population_size"] == 2
        assert summaries[0]["best_score"] == 0.9
        assert summaries[1]["population_size"] == 1
        assert summaries[1]["best_score"] == 0.8

    def test_island_should_keep_delegates_to_current(self) -> None:
        # Arrange
        search = IslandPopulationSearch(island_count=2, population_per_island=4)
        search._current_island = 0

        challenger = _make_eval_result(score=0.9)

        # Act — challenger 0.9 > baseline 0.5, HIGHER_IS_BETTER
        result = search.should_keep(
            challenger_result=challenger,
            baseline_score=0.5,
            baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
        )

        # Assert
        assert result is True

        # Act — challenger 0.3 < baseline 0.5, HIGHER_IS_BETTER
        worse = _make_eval_result(score=0.3)
        result = search.should_keep(
            challenger_result=worse,
            baseline_score=0.5,
            baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
        )

        # Assert
        assert result is False
