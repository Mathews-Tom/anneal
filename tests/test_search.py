"""Tests for GreedySearch statistical comparison logic."""

from __future__ import annotations

import pytest

from anneal.engine.search import GreedySearch, SimulatedAnnealingSearch
from anneal.engine.types import Direction, EvalResult


def _make_eval_result(score: float, raw_scores: list[float] | None = None) -> EvalResult:
    return EvalResult(
        score=score,
        ci_lower=None,
        ci_upper=None,
        raw_scores=raw_scores,
        cost_usd=0.0,
    )


class TestSmallSampleEffectSize:
    """Tests for _stochastic_compare small-sample fallback (n < MIN_PAIRED_SAMPLES)."""

    def test_small_sample_uses_effect_size_large_effect_returns_true(self) -> None:
        # Arrange: n=4, large effect — challenger clearly better
        search = GreedySearch()
        challenger_raw = [0.9, 0.85, 0.88, 0.92]
        baseline_raw = [0.5, 0.55, 0.52, 0.48]

        # Act
        result = search._stochastic_compare(
            challenger_raw, baseline_raw, Direction.HIGHER_IS_BETTER, 0.95
        )

        # Assert
        assert result is True

    def test_small_sample_weak_effect_rejected_returns_false(self) -> None:
        # Arrange: n=4, tiny differences — effect size below threshold
        search = GreedySearch()
        challenger_raw = [0.51, 0.52, 0.49, 0.50]
        baseline_raw = [0.50, 0.50, 0.50, 0.50]

        # Act
        result = search._stochastic_compare(
            challenger_raw, baseline_raw, Direction.HIGHER_IS_BETTER, 0.95
        )

        # Assert
        assert result is False

    def test_sufficient_samples_uses_wilcoxon_clearly_better_returns_true(self) -> None:
        # Arrange: n=10, clearly better challenger — Wilcoxon path
        search = GreedySearch()
        challenger_raw = [0.9, 0.88, 0.91, 0.87, 0.92, 0.89, 0.90, 0.93, 0.86, 0.91]
        baseline_raw = [0.5, 0.52, 0.49, 0.51, 0.50, 0.48, 0.53, 0.50, 0.51, 0.49]

        # Act
        result = search._stochastic_compare(
            challenger_raw, baseline_raw, Direction.HIGHER_IS_BETTER, 0.95
        )

        # Assert
        assert result is True


class TestHolmBonferroniAdjustedAlpha:
    """Tests for GreedySearch._adjusted_alpha Holm-Bonferroni correction."""

    def test_holm_bonferroni_first_experiment_divides_by_full_window(self) -> None:
        # Arrange / Act
        result = GreedySearch._adjusted_alpha(0.05, 0, 50)

        # Assert: alpha / (50 - 0) = 0.05 / 50 = 0.001
        assert result == pytest.approx(0.001)

    def test_holm_bonferroni_last_experiment_divides_by_one(self) -> None:
        # Arrange / Act
        result = GreedySearch._adjusted_alpha(0.05, 49, 50)

        # Assert: alpha / max(1, 50 - 49) = 0.05 / 1 = 0.05
        assert result == pytest.approx(0.05)

    def test_holm_bonferroni_mid_window_divides_by_remaining(self) -> None:
        # Arrange / Act
        result = GreedySearch._adjusted_alpha(0.05, 25, 50)

        # Assert: alpha / (50 - 25) = 0.05 / 25 = 0.002
        assert result == pytest.approx(0.002)


class TestSimulatedAnnealingAdaptive:
    """Tests for SA adaptive cooling with reheat."""

    def test_sa_reheat_on_low_acceptance_temperature_increases(self) -> None:
        """When acceptance drops below threshold, temperature reheats."""
        sa = SimulatedAnnealingSearch(
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=0.01,
            reheat_factor=2.0,
            acceptance_target=0.3,
        )
        # Simulate 10 rejections to trigger reheat
        sa._accept_history = [False] * 10
        temp_before = sa.temperature
        sa.cool()
        # acceptance_ratio = 0.0 < 0.3 * 0.5 = 0.15 → reheat
        assert sa.temperature > temp_before * sa._cooling_rate

    def test_sa_no_reheat_above_target_cools_normally(self) -> None:
        """When acceptance is healthy, normal cooling proceeds."""
        sa = SimulatedAnnealingSearch(
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=0.01,
            reheat_factor=2.0,
            acceptance_target=0.3,
        )
        # Simulate 10 acceptances — acceptance_ratio = 1.0 > 0.15
        sa._accept_history = [True] * 10
        temp_before = sa.temperature
        sa.cool()
        expected = temp_before * 0.95
        assert abs(sa.temperature - expected) < 1e-10

    def test_sa_temperature_bounded_never_exceeds_initial(self) -> None:
        """Reheat never pushes temperature above initial_temperature."""
        sa = SimulatedAnnealingSearch(
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=0.01,
            reheat_factor=100.0,  # Very aggressive reheat
            acceptance_target=0.3,
        )
        sa._accept_history = [False] * 10
        sa.cool()
        assert sa.temperature <= 1.0
