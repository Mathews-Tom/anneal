"""Tests for anneal.engine.learning_pool."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from anneal.engine.learning_pool import (
    Learning,
    LearningPool,
    LearningScope,
    LearningSignal,
    PersistentLearningPool,
    extract_learning,
)
from anneal.engine.types import ExperimentRecord, Outcome


def _make_record(
    *,
    outcome: Outcome = Outcome.KEPT,
    score: float = 0.8,
    baseline_score: float = 0.5,
    hypothesis: str = "Refactor prompt structure",
    record_id: str = "42",
    tags: list[str] | None = None,
    raw_scores: list[float] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=record_id,
        target_id="target-1",
        git_sha="abc123",
        pre_experiment_sha="abc122",
        timestamp=datetime(2026, 1, 1),
        hypothesis=hypothesis,
        hypothesis_source="agent",
        mutation_diff_summary="diff --git ...",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=raw_scores,
        baseline_score=baseline_score,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=1.0,
        tags=tags or [],
        learnings="",
        cost_usd=0.01,
        bootstrap_seed=0,
    )


def _make_learning(
    *,
    score_delta: float = 0.3,
    signal: LearningSignal = LearningSignal.POSITIVE,
    source_condition: str = "guided",
    source_target: str = "target-1",
    observation: str = "obs",
) -> Learning:
    return Learning(
        observation=observation,
        signal=signal,
        source_condition=source_condition,
        source_target=source_target,
        source_experiment_ids=[1],
        score_delta=score_delta,
        criterion_deltas={},
        confidence=1.0,
        tags=[],
    )


# ---------------------------------------------------------------------------
# extract_learning
# ---------------------------------------------------------------------------


class TestExtractLearning:
    def test_kept_produces_positive_signal(self) -> None:
        record = _make_record(outcome=Outcome.KEPT)
        learning = extract_learning(record)
        assert learning.signal is LearningSignal.POSITIVE

    def test_discarded_produces_negative_signal(self) -> None:
        record = _make_record(outcome=Outcome.DISCARDED)
        learning = extract_learning(record)
        assert learning.signal is LearningSignal.NEGATIVE

    def test_score_delta_computed_correctly(self) -> None:
        record = _make_record(score=0.9, baseline_score=0.6)
        learning = extract_learning(record)
        assert learning.score_delta == pytest.approx(0.3)

    def test_score_delta_negative(self) -> None:
        record = _make_record(score=0.3, baseline_score=0.5)
        learning = extract_learning(record)
        assert learning.score_delta == pytest.approx(-0.2)

    def test_observation_contains_hypothesis(self) -> None:
        record = _make_record(hypothesis="Use chain-of-thought prompting")
        learning = extract_learning(record)
        assert "Use chain-of-thought prompting" in learning.observation

    def test_source_condition_set(self) -> None:
        record = _make_record()
        learning = extract_learning(record, source_condition="bayesian")
        assert learning.source_condition == "bayesian"

    def test_source_target_set(self) -> None:
        record = _make_record()
        learning = extract_learning(record, source_target="target-7")
        assert learning.source_target == "target-7"

    def test_criterion_deltas_computed(self) -> None:
        record = _make_record(raw_scores=[0.9, 0.7])
        previous = {"clarity": 0.8, "accuracy": 0.5}
        learning = extract_learning(record, previous_per_criterion=previous)
        assert learning.criterion_deltas["clarity"] == pytest.approx(0.1)
        assert learning.criterion_deltas["accuracy"] == pytest.approx(0.2)

    def test_experiment_ids_from_numeric_id(self) -> None:
        record = _make_record(record_id="99")
        learning = extract_learning(record)
        assert learning.source_experiment_ids == [99]

    def test_experiment_ids_empty_for_non_numeric_id(self) -> None:
        record = _make_record(record_id="abc")
        learning = extract_learning(record)
        assert learning.source_experiment_ids == []


# ---------------------------------------------------------------------------
# LearningPool
# ---------------------------------------------------------------------------


class TestLearningPool:
    def test_add_increases_count(self) -> None:
        pool = LearningPool()
        assert pool.count == 0
        pool.add(_make_learning())
        assert pool.count == 1
        pool.add(_make_learning())
        assert pool.count == 2

    def test_retrieve_sorted_by_abs_score_delta_descending(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(score_delta=0.1, observation="small"))
        pool.add(_make_learning(score_delta=-0.5, observation="large-neg"))
        pool.add(_make_learning(score_delta=0.3, observation="medium"))

        results = pool.retrieve(scope=LearningScope.PROJECT, k=10)
        deltas = [abs(r.score_delta) for r in results]
        assert deltas == sorted(deltas, reverse=True)
        assert results[0].observation == "large-neg"

    def test_retrieve_exclude_condition(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(source_condition="guided"))
        pool.add(_make_learning(source_condition="random"))
        pool.add(_make_learning(source_condition="guided"))

        results = pool.retrieve(
            scope=LearningScope.PROJECT, k=10, exclude_condition="guided"
        )
        assert len(results) == 1
        assert results[0].source_condition == "random"

    def test_retrieve_signal_filter(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(signal=LearningSignal.POSITIVE))
        pool.add(_make_learning(signal=LearningSignal.NEGATIVE))
        pool.add(_make_learning(signal=LearningSignal.POSITIVE))

        results = pool.retrieve(
            scope=LearningScope.PROJECT, k=10, signal=LearningSignal.NEGATIVE
        )
        assert len(results) == 1
        assert results[0].signal is LearningSignal.NEGATIVE

    def test_retrieve_k_limits_results(self) -> None:
        pool = LearningPool()
        for i in range(10):
            pool.add(_make_learning(score_delta=float(i)))

        results = pool.retrieve(scope=LearningScope.PROJECT, k=3)
        assert len(results) == 3

    def test_retrieve_empty_pool(self) -> None:
        pool = LearningPool()
        results = pool.retrieve(scope=LearningScope.PROJECT)
        assert results == []

    def test_summarize_contains_header(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(score_delta=0.5, source_condition="guided", source_target="t1"))

        text = pool.summarize(scope=LearningScope.PROJECT)
        assert "## Cross-Condition Insights" in text

    def test_summarize_empty_pool_returns_empty_string(self) -> None:
        pool = LearningPool()
        text = pool.summarize(scope=LearningScope.PROJECT)
        assert text == ""

    def test_summarize_signal_markers(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(signal=LearningSignal.POSITIVE, score_delta=0.2))
        pool.add(_make_learning(signal=LearningSignal.NEGATIVE, score_delta=-0.1))

        text = pool.summarize(scope=LearningScope.PROJECT, k=10)
        assert "[+]" in text
        assert "[-]" in text


# ---------------------------------------------------------------------------
# PersistentLearningPool
# ---------------------------------------------------------------------------


class TestPersistentLearningPool:
    def test_learnings_persist_to_disk(self, tmp_path: Path) -> None:
        pool1 = PersistentLearningPool(tmp_path)
        pool1.add(_make_learning(observation="first"))
        pool1.add(_make_learning(observation="second"))
        assert pool1.count == 2

        pool2 = PersistentLearningPool(tmp_path)
        assert pool2.count == 2

    def test_jsonl_file_created_on_first_add(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "learnings-pool.jsonl"
        assert not jsonl_path.exists()

        pool = PersistentLearningPool(tmp_path)
        pool.add(_make_learning())
        assert jsonl_path.exists()
        assert jsonl_path.stat().st_size > 0

    def test_persisted_data_matches_original(self, tmp_path: Path) -> None:
        pool1 = PersistentLearningPool(tmp_path)
        pool1.add(
            _make_learning(
                score_delta=0.42,
                signal=LearningSignal.NEGATIVE,
                source_condition="random",
                source_target="t-9",
                observation="persisted obs",
            )
        )

        pool2 = PersistentLearningPool(tmp_path)
        results = pool2.retrieve(scope=LearningScope.PROJECT, k=1)
        assert len(results) == 1
        r = results[0]
        assert r.score_delta == pytest.approx(0.42)
        assert r.signal is LearningSignal.NEGATIVE
        assert r.source_condition == "random"
        assert r.source_target == "t-9"
        assert r.observation == "persisted obs"


# ---------------------------------------------------------------------------
# Eviction diversity
# ---------------------------------------------------------------------------


def test_eviction_breaks_ties_randomly_varies_across_runs(tmp_path: Path) -> None:
    """Eviction with identical scores does not always keep the same subset."""
    from datetime import UTC, datetime

    fixed_ts = datetime(2026, 1, 1, tzinfo=UTC)

    surviving_sets: list[frozenset[str]] = []
    for _ in range(10):
        pool = LearningPool(max_size=10)
        for i in range(20):
            pool.add(
                Learning(
                    observation=f"obs-{i}",
                    signal=LearningSignal.POSITIVE,
                    source_condition="guided",
                    source_target="target-1",
                    source_experiment_ids=[i],
                    score_delta=0.1,
                    criterion_deltas={},
                    confidence=1.0,
                    tags=[],
                    created_at=fixed_ts,
                )
            )
        surviving = frozenset(l.observation for l in pool._learnings)  # noqa: SLF001
        surviving_sets.append(surviving)

    # With random tiebreaking, not all 10 runs can produce the same survivor set
    assert len(set(surviving_sets)) > 1


# ---------------------------------------------------------------------------
# Domain-aware retrieval (Step 5.2)
# ---------------------------------------------------------------------------


class TestDomainAwareRetrieval:
    """Tests for domain-aware filtering in retrieve."""

    def test_same_domain_preferred_over_cross_domain(self, tmp_path: Path) -> None:
        """Same-domain learning ranked above equal-score cross-domain."""
        pool = LearningPool(max_size=100)
        same_domain = Learning(
            observation="improve prompt clarity",
            signal=LearningSignal.POSITIVE,
            source_condition="cond1",
            source_target="target1",
            source_experiment_ids=[1],
            score_delta=0.1,
            criterion_deltas={},
            confidence=0.9,
            tags=["prompt"],
            domain="prompt-tuning",
        )
        cross_domain = Learning(
            observation="optimize batch size",
            signal=LearningSignal.POSITIVE,
            source_condition="cond2",
            source_target="target2",
            source_experiment_ids=[2],
            score_delta=0.1,
            criterion_deltas={},
            confidence=0.9,
            tags=["code"],
            domain="code-optimization",
        )
        pool.add(same_domain)
        pool.add(cross_domain)

        results = pool.retrieve(LearningScope.GLOBAL, k=2, domain="prompt-tuning")
        assert results[0].observation == "improve prompt clarity"

    def test_cross_domain_high_score_still_ranked_above_weak_same_domain(
        self, tmp_path: Path
    ) -> None:
        """Cross-domain with 10x score beats weak same-domain even after penalty."""
        pool = LearningPool(max_size=100)
        weak_same = Learning(
            observation="minor tweak",
            signal=LearningSignal.POSITIVE,
            source_condition="cond1",
            source_target="target1",
            source_experiment_ids=[1],
            score_delta=0.05,
            criterion_deltas={},
            confidence=0.9,
            tags=[],
            domain="prompt-tuning",
        )
        strong_cross = Learning(
            observation="major optimization",
            signal=LearningSignal.POSITIVE,
            source_condition="cond2",
            source_target="target2",
            source_experiment_ids=[2],
            score_delta=0.5,  # 10x stronger; 0.5 * 0.5 = 0.25 > 0.05
            criterion_deltas={},
            confidence=0.9,
            tags=[],
            domain="code-optimization",
        )
        pool.add(weak_same)
        pool.add(strong_cross)

        results = pool.retrieve(LearningScope.GLOBAL, k=2, domain="prompt-tuning")
        assert results[0].observation == "major optimization"

    def test_no_domain_filter_preserves_original_score_order(self) -> None:
        """Without domain param, ordering is unchanged (no penalty applied)."""
        pool = LearningPool(max_size=100)
        pool.add(
            Learning(
                observation="high score",
                signal=LearningSignal.POSITIVE,
                source_condition="cond1",
                source_target="t1",
                source_experiment_ids=[1],
                score_delta=0.8,
                criterion_deltas={},
                confidence=1.0,
                tags=[],
                domain="code-optimization",
            )
        )
        pool.add(
            Learning(
                observation="low score",
                signal=LearningSignal.POSITIVE,
                source_condition="cond2",
                source_target="t2",
                source_experiment_ids=[2],
                score_delta=0.1,
                criterion_deltas={},
                confidence=1.0,
                tags=[],
                domain="prompt-tuning",
            )
        )

        results = pool.retrieve(LearningScope.GLOBAL, k=2)
        assert results[0].observation == "high score"

    def test_domain_field_defaults_to_empty_string(self) -> None:
        """Learning without explicit domain defaults to empty string."""
        learning = _make_learning()
        assert learning.domain == ""

    def test_empty_domain_on_learning_not_penalized(self) -> None:
        """Learning with no domain set is not penalized even when domain filter active."""
        pool = LearningPool(max_size=100)
        no_domain = Learning(
            observation="no domain learning",
            signal=LearningSignal.POSITIVE,
            source_condition="cond1",
            source_target="t1",
            source_experiment_ids=[1],
            score_delta=0.1,
            criterion_deltas={},
            confidence=1.0,
            tags=[],
            domain="",
        )
        pool.add(no_domain)

        results = pool.retrieve(LearningScope.GLOBAL, k=1, domain="prompt-tuning")
        assert len(results) == 1
        assert results[0].observation == "no domain learning"


# ---------------------------------------------------------------------------
# Criterion delta exposure in summarize (Step 5.3)
# ---------------------------------------------------------------------------


class TestSummarizeCriterionDeltas:
    def test_summarize_includes_criterion_deltas_in_output(self) -> None:
        """Summarize output includes top criterion deltas when available."""
        pool = LearningPool(max_size=100)
        learning = Learning(
            observation="improved clarity",
            signal=LearningSignal.POSITIVE,
            source_condition="cond1",
            source_target="target1",
            source_experiment_ids=[1],
            score_delta=0.1,
            criterion_deltas={"clarity": 0.15, "accuracy": -0.05, "tone": 0.02},
            confidence=0.9,
            tags=[],
        )
        pool.add(learning)

        output = pool.summarize(scope=LearningScope.GLOBAL)
        assert "Criteria:" in output
        assert "clarity: +0.15" in output
        assert "accuracy: -0.05" in output

    def test_summarize_omits_criteria_line_when_no_criterion_deltas(self) -> None:
        """Summarize does not append Criteria line when criterion_deltas is empty."""
        pool = LearningPool(max_size=100)
        pool.add(_make_learning(score_delta=0.3))

        output = pool.summarize(scope=LearningScope.GLOBAL)
        assert "Criteria:" not in output

    def test_summarize_top_three_criteria_by_abs_value(self) -> None:
        """Only the top 3 criteria by absolute delta are included."""
        pool = LearningPool(max_size=100)
        learning = Learning(
            observation="multi-criteria",
            signal=LearningSignal.POSITIVE,
            source_condition="cond1",
            source_target="t1",
            source_experiment_ids=[1],
            score_delta=0.2,
            criterion_deltas={
                "a": 0.01,
                "b": 0.5,
                "c": -0.4,
                "d": 0.3,
            },
            confidence=1.0,
            tags=[],
        )
        pool.add(learning)

        output = pool.summarize(scope=LearningScope.GLOBAL)
        assert "b: +0.50" in output
        assert "c: -0.40" in output
        assert "d: +0.30" in output
        assert "a: +0.01" not in output
