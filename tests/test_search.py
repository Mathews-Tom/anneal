"""Tests for GreedySearch statistical comparison logic."""

from __future__ import annotations

import pytest

from anneal.engine.search import GreedySearch
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
