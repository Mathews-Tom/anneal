from __future__ import annotations

import pytest

from anneal.engine.search import (
    GreedySearch,
    ParetoSearch,
    PopulationSearch,
    SimulatedAnnealingSearch,
)
from anneal.engine.types import Direction, EvalResult


def _make_eval_result(
    score: float,
    raw_scores: list[float] | None = None,
) -> EvalResult:
    return EvalResult(
        score=score,
        ci_lower=None,
        ci_upper=None,
        raw_scores=raw_scores,
        cost_usd=0.0,
    )


class TestAdjustedAlpha:

    def test_adjusted_alpha_reduces_over_experiments(self) -> None:
        # Arrange / Act
        alpha_first = GreedySearch._adjusted_alpha(0.05, 0, 50)
        alpha_last = GreedySearch._adjusted_alpha(0.05, 49, 50)

        # Assert
        assert alpha_first == pytest.approx(0.05 / 50)
        assert alpha_last == pytest.approx(0.05 / 1)

    def test_adjusted_alpha_mid_window(self) -> None:
        # Arrange / Act
        alpha_mid = GreedySearch._adjusted_alpha(0.05, 25, 50)

        # Assert — remaining = 50 - 25 = 25
        assert alpha_mid == pytest.approx(0.05 / 25)


class TestGreedyStochasticHolmBonferroni:

    # Paired samples where Wilcoxon gives a borderline p-value
    CHALLENGER_RAW = [0.6, 0.7, 0.65, 0.72, 0.68, 0.71]
    BASELINE_RAW = [0.5, 0.55, 0.52, 0.58, 0.54, 0.56]

    def test_greedy_stochastic_with_holm_bonferroni_more_conservative_early(self) -> None:
        # Arrange
        search = GreedySearch()
        challenger = _make_eval_result(
            score=sum(self.CHALLENGER_RAW) / len(self.CHALLENGER_RAW),
            raw_scores=self.CHALLENGER_RAW,
        )

        # Act — without correction the test should accept
        result_uncorrected = search.should_keep(
            challenger,
            baseline_score=sum(self.BASELINE_RAW) / len(self.BASELINE_RAW),
            baseline_raw_scores=self.BASELINE_RAW,
            direction=Direction.HIGHER_IS_BETTER,
            confidence=0.95,
            experiment_index=0,
            holm_bonferroni=False,
        )

        # Act — with correction at experiment_index=0 alpha shrinks drastically
        result_corrected = search.should_keep(
            challenger,
            baseline_score=sum(self.BASELINE_RAW) / len(self.BASELINE_RAW),
            baseline_raw_scores=self.BASELINE_RAW,
            direction=Direction.HIGHER_IS_BETTER,
            confidence=0.95,
            experiment_index=0,
            holm_bonferroni=True,
        )

        # Assert — uncorrected accepts, corrected rejects (more conservative)
        assert result_uncorrected is True
        assert result_corrected is False

    def test_greedy_stochastic_without_holm_bonferroni_unchanged(self) -> None:
        # Arrange
        search = GreedySearch()
        challenger = _make_eval_result(
            score=sum(self.CHALLENGER_RAW) / len(self.CHALLENGER_RAW),
            raw_scores=self.CHALLENGER_RAW,
        )
        kwargs: dict = dict(
            baseline_score=sum(self.BASELINE_RAW) / len(self.BASELINE_RAW),
            baseline_raw_scores=self.BASELINE_RAW,
            direction=Direction.HIGHER_IS_BETTER,
            confidence=0.95,
            holm_bonferroni=False,
        )

        # Act — varying experiment_index should have no effect
        result_idx0 = search.should_keep(challenger, experiment_index=0, **kwargs)
        result_idx25 = search.should_keep(challenger, experiment_index=25, **kwargs)
        result_idx49 = search.should_keep(challenger, experiment_index=49, **kwargs)

        # Assert
        assert result_idx0 == result_idx25 == result_idx49


class TestOtherStrategiesAcceptNewParams:

    def test_simulated_annealing_accepts_new_params(self) -> None:
        # Arrange
        sa = SimulatedAnnealingSearch()
        challenger = _make_eval_result(score=0.8, raw_scores=[0.8])

        # Act / Assert — no TypeError
        sa.should_keep(
            challenger,
            baseline_score=0.7,
            baseline_raw_scores=[0.7],
            direction=Direction.HIGHER_IS_BETTER,
            experiment_index=5,
            holm_bonferroni=True,
        )

    def test_population_search_accepts_new_params(self) -> None:
        # Arrange
        ps = PopulationSearch()
        challenger = _make_eval_result(score=0.8)

        # Act / Assert — no TypeError
        ps.should_keep(
            challenger,
            baseline_score=0.7,
            baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
            experiment_index=5,
            holm_bonferroni=True,
        )

    def test_pareto_search_accepts_new_params(self) -> None:
        # Arrange
        ps = ParetoSearch()
        challenger = _make_eval_result(score=0.8)

        # Act / Assert — no TypeError
        ps.should_keep(
            challenger,
            baseline_score=0.7,
            baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
            experiment_index=5,
            holm_bonferroni=True,
        )
