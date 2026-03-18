"""Tests for learning pool confidence decay (F4): created_at, _effective_score, retrieve ordering."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from anneal.engine.learning_pool import (
    Learning,
    LearningPool,
    LearningScope,
    LearningSignal,
    _dict_to_learning,
    _learning_to_dict,
)


def _make_learning(
    *,
    score_delta: float = 0.3,
    signal: LearningSignal = LearningSignal.POSITIVE,
    observation: str = "obs",
    created_at: datetime | None = None,
    project_id: str = "",
) -> Learning:
    return Learning(
        observation=observation,
        signal=signal,
        source_condition="guided",
        source_target="target-1",
        source_experiment_ids=[1],
        score_delta=score_delta,
        criterion_deltas={},
        confidence=1.0,
        tags=[],
        created_at=created_at or datetime.now(UTC),
        project_id=project_id,
    )


class TestLearningCreatedAt:
    def test_default_created_at_is_utc_now(self) -> None:
        before = datetime.now(UTC)
        learning = _make_learning()
        after = datetime.now(UTC)
        assert before <= learning.created_at <= after

    def test_explicit_created_at_preserved(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
        learning = _make_learning(created_at=ts)
        assert learning.created_at == ts

    def test_project_id_field(self) -> None:
        learning = _make_learning(project_id="my-project")
        assert learning.project_id == "my-project"

    def test_default_project_id_empty(self) -> None:
        learning = _make_learning()
        assert learning.project_id == ""


class TestEffectiveScore:
    def test_recent_learning_has_full_score(self) -> None:
        pool = LearningPool(decay_rate=0.05)
        learning = _make_learning(score_delta=0.5, created_at=datetime.now(UTC))
        effective = pool._effective_score(learning)
        assert effective == pytest.approx(0.5, abs=0.01)

    def test_old_learning_has_decayed_score(self) -> None:
        pool = LearningPool(decay_rate=0.05)
        old_time = datetime.now(UTC) - timedelta(days=30)
        learning = _make_learning(score_delta=0.5, created_at=old_time)
        effective = pool._effective_score(learning)
        expected = 0.5 * math.exp(-0.05 * 30)
        assert effective == pytest.approx(expected, rel=0.01)

    def test_zero_age_no_decay(self) -> None:
        pool = LearningPool(decay_rate=0.1)
        learning = _make_learning(score_delta=1.0, created_at=datetime.now(UTC))
        effective = pool._effective_score(learning)
        assert effective == pytest.approx(1.0, abs=0.01)

    def test_higher_decay_rate_faster_decay(self) -> None:
        old_time = datetime.now(UTC) - timedelta(days=10)
        learning = _make_learning(score_delta=1.0, created_at=old_time)

        slow = LearningPool(decay_rate=0.01)
        fast = LearningPool(decay_rate=0.1)
        assert slow._effective_score(learning) > fast._effective_score(learning)


class TestRetrieveDecayOrdering:
    def test_recent_smaller_delta_beats_old_larger_delta(self) -> None:
        pool = LearningPool(decay_rate=0.1)
        old = _make_learning(
            score_delta=1.0,
            observation="old-big",
            created_at=datetime.now(UTC) - timedelta(days=60),
        )
        recent = _make_learning(
            score_delta=0.3,
            observation="recent-small",
            created_at=datetime.now(UTC),
        )
        pool.add(old)
        pool.add(recent)

        results = pool.retrieve(scope=LearningScope.PROJECT, k=2)
        assert results[0].observation == "recent-small"

    def test_same_age_sorted_by_abs_delta(self) -> None:
        pool = LearningPool(decay_rate=0.05)
        now = datetime.now(UTC)
        pool.add(_make_learning(score_delta=0.1, observation="small", created_at=now))
        pool.add(_make_learning(score_delta=-0.5, observation="big-neg", created_at=now))
        pool.add(_make_learning(score_delta=0.3, observation="medium", created_at=now))

        results = pool.retrieve(scope=LearningScope.PROJECT, k=3)
        assert results[0].observation == "big-neg"

    def test_decayed_confidence_in_returned_learnings(self) -> None:
        pool = LearningPool(decay_rate=0.05)
        old_time = datetime.now(UTC) - timedelta(days=20)
        pool.add(_make_learning(score_delta=0.5, created_at=old_time))

        results = pool.retrieve(scope=LearningScope.PROJECT, k=1)
        assert len(results) == 1
        # Confidence should be decayed from 1.0
        assert results[0].confidence < 1.0
        expected = 1.0 * math.exp(-0.05 * 20)
        assert results[0].confidence == pytest.approx(expected, rel=0.01)


class TestDictToLearningBackwardCompat:
    def test_missing_created_at_uses_current_time(self) -> None:
        d: dict[str, object] = {
            "observation": "test",
            "signal": "positive",
            "source_condition": "guided",
            "source_target": "t-1",
            "source_experiment_ids": [1],
            "score_delta": 0.2,
            "criterion_deltas": {},
            "confidence": 1.0,
            "tags": [],
        }
        before = datetime.now(UTC)
        learning = _dict_to_learning(d)
        after = datetime.now(UTC)
        assert before <= learning.created_at <= after

    def test_present_created_at_parsed(self) -> None:
        ts = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
        d: dict[str, object] = {
            "observation": "test",
            "signal": "positive",
            "source_condition": "guided",
            "source_target": "t-1",
            "source_experiment_ids": [1],
            "score_delta": 0.2,
            "criterion_deltas": {},
            "confidence": 1.0,
            "tags": [],
            "created_at": ts.isoformat(),
        }
        learning = _dict_to_learning(d)
        assert learning.created_at == ts

    def test_missing_project_id_defaults_empty(self) -> None:
        d: dict[str, object] = {
            "observation": "test",
            "signal": "negative",
            "source_condition": "random",
            "source_target": "t-2",
            "source_experiment_ids": [],
            "score_delta": -0.1,
            "criterion_deltas": {},
            "confidence": 0.8,
            "tags": ["tag1"],
            "created_at": datetime.now(UTC).isoformat(),
        }
        learning = _dict_to_learning(d)
        assert learning.project_id == ""

    def test_present_project_id_preserved(self) -> None:
        d: dict[str, object] = {
            "observation": "test",
            "signal": "positive",
            "source_condition": "guided",
            "source_target": "t-1",
            "source_experiment_ids": [1],
            "score_delta": 0.5,
            "criterion_deltas": {},
            "confidence": 1.0,
            "tags": [],
            "created_at": datetime.now(UTC).isoformat(),
            "project_id": "my-project",
        }
        learning = _dict_to_learning(d)
        assert learning.project_id == "my-project"

    def test_roundtrip_serialization(self) -> None:
        original = _make_learning(score_delta=0.42, project_id="proj-a")
        d = _learning_to_dict(original)
        restored = _dict_to_learning(d)
        assert restored.score_delta == pytest.approx(original.score_delta)
        assert restored.project_id == original.project_id
        assert restored.signal == original.signal
