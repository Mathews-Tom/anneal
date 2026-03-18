"""Tests for evaluator drift monitoring (F3): KnowledgeStore drift and _variance."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from anneal.engine.knowledge import KnowledgeStore, _variance
from anneal.engine.types import ConsolidationRecord, DriftEntry, ExperimentRecord, Outcome


def _make_record(
    *,
    idx: int = 0,
    outcome: Outcome = Outcome.KEPT,
    score: float = 0.8,
    baseline_score: float = 0.75,
    raw_scores: list[float] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=f"exp-{idx:04d}",
        target_id="target-1",
        git_sha=f"abc{idx:04d}",
        pre_experiment_sha=f"pre{idx:04d}",
        timestamp=datetime(2026, 1, 1, 0, idx % 60),
        hypothesis=f"hypothesis {idx}",
        hypothesis_source="agent",
        mutation_diff_summary=f"diff {idx}",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=raw_scores,
        baseline_score=baseline_score,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=1.0,
        tags=["prompt"],
        learnings=f"learning {idx}",
        cost_usd=0.01,
        bootstrap_seed=42,
    )


class TestVarianceHelper:
    def test_empty_list(self) -> None:
        assert _variance([]) == 0.0

    def test_single_element(self) -> None:
        assert _variance([5.0]) == 0.0

    def test_identical_values(self) -> None:
        assert _variance([3.0, 3.0, 3.0]) == 0.0

    def test_known_variance(self) -> None:
        # [1, 2, 3] -> mean=2, variance = ((1+0+1)/3) = 2/3
        assert _variance([1.0, 2.0, 3.0]) == pytest.approx(2.0 / 3.0)

    def test_two_elements(self) -> None:
        # [0, 10] -> mean=5, variance = (25+25)/2 = 25
        assert _variance([0.0, 10.0]) == pytest.approx(25.0)


class TestConsolidateScoreVariance:
    """Verify consolidate() computes score_variance."""

    def test_score_variance_computed(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        scores = [0.75 + i * 0.01 for i in range(50)]
        for i, s in enumerate(scores):
            store.append_record(_make_record(idx=i, score=s))

        cr = store.consolidate()
        assert cr.score_variance > 0.0
        assert cr.score_variance == pytest.approx(_variance(scores))

    def test_constant_scores_zero_variance(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        for i in range(50):
            store.append_record(_make_record(idx=i, score=0.8))

        cr = store.consolidate()
        assert cr.score_variance == 0.0


class TestConsolidateCriterionVariances:
    """Verify consolidate() computes criterion_variances from raw_scores."""

    def test_criterion_variances_computed(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        for i in range(50):
            # Two criteria: first varies, second is constant
            raw = [float(i % 5), 1.0]
            store.append_record(_make_record(idx=i, raw_scores=raw))

        cr = store.consolidate()
        assert "criterion_0" in cr.criterion_variances
        assert "criterion_1" in cr.criterion_variances
        assert cr.criterion_variances["criterion_0"] > 0.0
        assert cr.criterion_variances["criterion_1"] == 0.0

    def test_no_raw_scores_means_no_criterion_variances(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        for i in range(50):
            store.append_record(_make_record(idx=i, raw_scores=None))

        cr = store.consolidate()
        assert cr.criterion_variances == {}


class TestGetDriftReport:
    """Verify get_drift_report() returns DriftEntry for high-variance criteria."""

    def test_drift_detected_above_threshold(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        for i in range(50):
            # High variance on criterion_0, low on criterion_1
            raw = [float(i % 10), 0.5]
            store.append_record(_make_record(idx=i, raw_scores=raw))

        store.consolidate()
        entries = store.get_drift_report(variance_threshold=0.01)
        criterion_names = [e.criterion_name for e in entries]
        assert "criterion_0" in criterion_names
        for e in entries:
            assert isinstance(e, DriftEntry)
            assert e.variance > 0.01
            assert e.window_size > 0

    def test_no_drift_below_threshold(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        for i in range(50):
            raw = [0.5, 0.5]
            store.append_record(_make_record(idx=i, raw_scores=raw))

        store.consolidate()
        entries = store.get_drift_report(variance_threshold=0.1)
        assert entries == []

    def test_empty_store_returns_empty(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        entries = store.get_drift_report()
        assert entries == []

    def test_drift_report_mean_score_correct(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        values = []
        for i in range(50):
            val = float(i % 10)
            values.append(val)
            store.append_record(_make_record(idx=i, raw_scores=[val]))

        store.consolidate()
        entries = store.get_drift_report(variance_threshold=0.0)
        assert len(entries) == 1
        expected_mean = sum(values) / len(values)
        assert entries[0].mean_score == pytest.approx(expected_mean)
