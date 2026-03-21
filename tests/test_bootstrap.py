"""Tests for bootstrap CI seed reproducibility and correctness."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from anneal.engine.eval import _bootstrap_ci


def _compute_seed(scores: list[float]) -> int:
    """Replicate the seed computation from StochasticEvaluator."""
    score_bytes = ",".join(f"{s:.6f}" for s in scores).encode()
    return int(hashlib.sha256(score_bytes).hexdigest(), 16) % 2**32


class TestBootstrapSeedDeterministic:
    """Same scores produce identical CI bounds across repeated calls."""

    def test_bootstrap_seed_deterministic(self) -> None:
        scores = [0.5, 0.7, 0.8, 0.6, 0.9]
        seed = _compute_seed(scores)

        results = [_bootstrap_ci(scores, seed=seed) for _ in range(5)]

        assert all(r == results[0] for r in results)

    def test_bootstrap_seed_deterministic_different_resample_counts(self) -> None:
        scores = [0.3, 0.4, 0.5, 0.6]
        seed = _compute_seed(scores)

        a = _bootstrap_ci(scores, n_resamples=5000, seed=seed)
        b = _bootstrap_ci(scores, n_resamples=5000, seed=seed)

        assert a == b


class TestBootstrapSeedRoundTrip:
    """Float precision differences must not change the seed."""

    def test_bootstrap_seed_round_trip(self) -> None:
        scores_precise = [0.50000000001, 0.70000000002]
        scores_clean = [0.5, 0.7]

        seed_precise = _compute_seed(scores_precise)
        seed_clean = _compute_seed(scores_clean)

        assert seed_precise == seed_clean

    def test_bootstrap_seed_round_trip_negative_values(self) -> None:
        scores_a = [-0.10000000001, 0.30000000002, 0.99999999999]
        scores_b = [-0.1, 0.3, 1.0]

        assert _compute_seed(scores_a) == _compute_seed(scores_b)

    def test_bootstrap_ci_same_seed_for_rounded_scores(self) -> None:
        """Seeds match, so the bootstrap resampling indices are identical.

        The CI values may differ by a tiny epsilon because _bootstrap_ci
        operates on the raw (unrounded) score arrays.  The important
        invariant is that the *seed* is the same across platforms.
        """
        scores_precise = [0.50000000001, 0.70000000002, 0.80000000003]
        scores_clean = [0.5, 0.7, 0.8]

        seed_a = _compute_seed(scores_precise)
        seed_b = _compute_seed(scores_clean)

        assert seed_a == seed_b

        ci_a = _bootstrap_ci(scores_precise, seed=seed_a)
        ci_b = _bootstrap_ci(scores_clean, seed=seed_b)

        assert ci_a[0] == pytest.approx(ci_b[0], abs=1e-6)
        assert ci_a[1] == pytest.approx(ci_b[1], abs=1e-6)


class TestBootstrapCIValuesReasonable:
    """CI bounds must bracket the mean."""

    def test_bootstrap_ci_values_reasonable(self) -> None:
        scores = [0.4, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7]
        mean = float(np.mean(scores))
        seed = _compute_seed(scores)

        ci_lower, ci_upper = _bootstrap_ci(scores, seed=seed)

        assert ci_lower <= mean
        assert ci_upper >= mean
        assert ci_lower < ci_upper

    def test_bootstrap_ci_bounds_within_score_range(self) -> None:
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        seed = _compute_seed(scores)

        ci_lower, ci_upper = _bootstrap_ci(scores, seed=seed)

        assert ci_lower >= min(scores)
        assert ci_upper <= max(scores)

    def test_bootstrap_ci_identical_scores_collapse(self) -> None:
        scores = [0.5, 0.5, 0.5, 0.5]
        seed = _compute_seed(scores)

        ci_lower, ci_upper = _bootstrap_ci(scores, seed=seed)

        assert ci_lower == 0.5
        assert ci_upper == 0.5
