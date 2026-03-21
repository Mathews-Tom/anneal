"""Tests for Thompson Sampling strategy selector."""
from __future__ import annotations

import random

from anneal.engine.strategy_selector import StrategyArm, StrategySelector


class TestStrategyArm:
    """Tests for individual StrategyArm behavior."""

    def test_arm_update_success_increments_alpha(self) -> None:
        arm = StrategyArm(name="greedy")
        arm.update(reward=True)
        assert arm.alpha == 2.0
        assert arm.beta == 1.0

    def test_arm_update_failure_increments_beta(self) -> None:
        arm = StrategyArm(name="greedy")
        arm.update(reward=False)
        assert arm.alpha == 1.0
        assert arm.beta == 2.0

    def test_arm_sample_returns_float_in_unit_interval(self) -> None:
        arm = StrategyArm(name="greedy", alpha=5.0, beta=5.0)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0


class TestStrategySelector:
    """Tests for Thompson Sampling selector."""

    def test_thompson_explores_initially_all_strategies_selected(self) -> None:
        """With uniform priors, all strategies should be selected within first calls."""
        random.seed(42)
        selector = StrategySelector(["greedy", "sa", "population"])
        counts: dict[str, int] = {"greedy": 0, "sa": 0, "population": 0}
        for _ in range(100):
            choice = selector.select()
            counts[choice] += 1
        # Each should be picked at least once in 100 draws with uniform priors
        assert all(c > 0 for c in counts.values())
        # And roughly uniform: each should be picked at least 15 times
        assert all(c >= 15 for c in counts.values())

    def test_thompson_exploits_winner_greedy_dominates(self) -> None:
        """After strong signal, the winning arm should be selected most often."""
        random.seed(42)
        selector = StrategySelector(["greedy", "sa", "population"])
        # Give greedy 50 successes, others 50 failures
        for _ in range(50):
            selector.update("greedy", True)
            selector.update("sa", False)
            selector.update("population", False)

        counts: dict[str, int] = {"greedy": 0, "sa": 0, "population": 0}
        for _ in range(100):
            counts[selector.select()] += 1
        assert counts["greedy"] > 80  # Should dominate

    def test_summary_reflects_history_correct_means(self) -> None:
        selector = StrategySelector(["a", "b"])
        # a: 10 successes, 0 failures → alpha=11, beta=1 → mean≈0.917
        for _ in range(10):
            selector.update("a", True)
        # b: 0 successes, 10 failures → alpha=1, beta=11 → mean≈0.083
        for _ in range(10):
            selector.update("b", False)

        summary = selector.summary()
        assert summary["a"]["mean"] > 0.85
        assert summary["b"]["mean"] < 0.15
        assert summary["a"]["n"] == 10.0
        assert summary["b"]["n"] == 10.0

    def test_update_unknown_strategy_no_error(self) -> None:
        """Updating a non-existent strategy name is a no-op."""
        selector = StrategySelector(["greedy"])
        selector.update("nonexistent", True)  # Should not raise
        summary = selector.summary()
        assert summary["greedy"]["n"] == 0.0  # Unchanged
