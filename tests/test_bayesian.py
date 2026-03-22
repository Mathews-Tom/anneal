"""Tests for Bayesian surrogate model."""
from __future__ import annotations

import pytest
from anneal.engine.bayesian import SurrogateModel


class TestSurrogateModel:
    def test_is_available_reflects_sklearn_presence(self) -> None:
        # scikit-learn should be installable in test env
        # This test just verifies the method runs without error
        result = SurrogateModel.is_available()
        assert isinstance(result, bool)

    def test_predict_unfitted_returns_defaults(self) -> None:
        model = SurrogateModel()
        mean, std = model.predict([0.5, 0.3, 0.8, 0.6])
        assert mean == 0.0
        assert std == float("inf")

    def test_add_observation_increments_count(self) -> None:
        model = SurrogateModel()
        model.add_observation([1.0, 2.0, 3.0, 4.0], 0.8)
        assert model.observation_count == 1
        model.add_observation([1.5, 2.5, 3.5, 4.5], 0.9)
        assert model.observation_count == 2

    def test_fit_requires_min_observations(self) -> None:
        model = SurrogateModel(min_observations=5)
        for i in range(4):
            model.add_observation([float(i), 0.0, 0.0, 0.0], float(i) * 0.1)
        assert model.fit() is False or not SurrogateModel.is_available()

    def test_extract_features_returns_four_floats(self) -> None:
        features = SurrogateModel.extract_features(
            hypothesis="improve the prompt clarity",
            tags=["prompt", "clarity"],
            baseline_score=0.75,
            per_criterion_scores={"clarity": 0.8, "accuracy": 0.7},
        )
        assert len(features) == 4
        assert all(isinstance(f, float) for f in features)

    def test_extract_features_no_criteria(self) -> None:
        features = SurrogateModel.extract_features(
            hypothesis="test",
            tags=[],
            baseline_score=0.5,
        )
        assert features[3] == 0.0  # criterion_mean defaults to 0

    def test_expected_improvement_unfitted_returns_zero(self) -> None:
        model = SurrogateModel()
        ei = model.expected_improvement([0.5, 0.3, 0.8, 0.6], best_score=0.8)
        assert ei == 0.0

    @pytest.mark.skipif(
        not SurrogateModel.is_available(),
        reason="scikit-learn not installed",
    )
    def test_fit_and_predict_with_sufficient_data(self) -> None:
        model = SurrogateModel(min_observations=10)
        # Generate synthetic training data: score = sum(features) / 4
        for i in range(20):
            f = [float(i) / 20.0] * 4
            model.add_observation(f, sum(f) / 4.0)

        assert model.fit() is True
        mean, std = model.predict([0.5, 0.5, 0.5, 0.5])
        assert abs(mean - 0.5) < 0.3  # Rough prediction
        assert std >= 0  # Non-negative uncertainty
