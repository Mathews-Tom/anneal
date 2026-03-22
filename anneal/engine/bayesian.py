"""Bayesian surrogate model for mutation score prediction.

Uses Gaussian Process regression (scikit-learn) to predict mutation
scores from experiment history features. Falls back gracefully when
scikit-learn is not installed.
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SKLEARN_AVAILABLE = False
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    _SKLEARN_AVAILABLE = True
except ImportError:
    GaussianProcessRegressor = None  # type: ignore[misc,assignment]
    Matern = None  # type: ignore[misc,assignment]


@dataclass
class SurrogateModel:
    """GP-based surrogate for predicting mutation scores.

    Trains on (feature_vector, score) pairs extracted from experiment
    history. Features are simple: normalized tag counts, hypothesis
    length, and per-criterion baselines.
    """

    _observations_X: list[list[float]] = field(default_factory=list)
    _observations_y: list[float] = field(default_factory=list)
    _model: object | None = None  # GaussianProcessRegressor when available
    _fitted: bool = False
    min_observations: int = 10  # Don't predict until we have enough data

    @staticmethod
    def is_available() -> bool:
        """Check if scikit-learn is installed."""
        return _SKLEARN_AVAILABLE

    def add_observation(self, features: list[float], score: float) -> None:
        """Record a (features, score) observation."""
        self._observations_X.append(features)
        self._observations_y.append(score)
        self._fitted = False  # Invalidate model

    def fit(self) -> bool:
        """Fit the GP model on current observations. Returns True if successful."""
        if not _SKLEARN_AVAILABLE:
            return False
        if len(self._observations_X) < self.min_observations:
            return False

        kernel = Matern(nu=2.5)
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=42,
        )
        X = np.array(self._observations_X)
        y = np.array(self._observations_y)
        self._model.fit(X, y)
        self._fitted = True
        return True

    def predict(self, features: list[float]) -> tuple[float, float]:
        """Predict (mean, std) for a feature vector.

        Returns (0.0, float('inf')) if model is not fitted or unavailable.
        """
        if not self._fitted or self._model is None:
            return (0.0, float("inf"))

        X = np.array([features])
        mean, std = self._model.predict(X, return_std=True)
        return (float(mean[0]), float(std[0]))

    def expected_improvement(
        self, features: list[float], best_score: float, xi: float = 0.01
    ) -> float:
        """Compute expected improvement over best_score.

        EI = (mu - best - xi) * Phi(Z) + sigma * phi(Z)
        where Z = (mu - best - xi) / sigma

        Returns 0.0 if model unavailable or unfitted.
        """
        if not self._fitted or self._model is None:
            return 0.0

        from scipy.stats import norm

        mu, sigma = self.predict(features)
        if sigma <= 0:
            return 0.0

        z = (mu - best_score - xi) / sigma
        ei = (mu - best_score - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return max(0.0, float(ei))

    @property
    def observation_count(self) -> int:
        return len(self._observations_y)

    @staticmethod
    def extract_features(
        hypothesis: str,
        tags: list[str],
        baseline_score: float,
        per_criterion_scores: dict[str, float] | None = None,
    ) -> list[float]:
        """Extract a simple feature vector from experiment metadata.

        Features: [hypothesis_word_count, num_tags, baseline_score,
                   mean_criterion_score (or 0)]
        """
        word_count = len(hypothesis.split()) / 50.0  # Normalize
        num_tags = len(tags) / 10.0
        criterion_mean = 0.0
        if per_criterion_scores:
            vals = list(per_criterion_scores.values())
            criterion_mean = sum(vals) / len(vals) if vals else 0.0
        return [word_count, num_tags, baseline_score, criterion_mean]
