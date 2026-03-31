"""Tests for structured episodic memory: Lesson model and extract_lesson."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anneal.engine.knowledge import extract_lesson
from anneal.engine.learning_pool import LearningPool, LearningScope, LearningSignal, Learning
from anneal.engine.types import ExperimentRecord, Lesson, Outcome


def _make_record(
    score: float = 0.7,
    baseline: float = 0.5,
    outcome: Outcome = Outcome.KEPT,
    hypothesis: str = "Improve error handling",
    per_criterion: dict[str, float] | None = None,
    tags: list[str] | None = None,
    mutation_diff: str = "Refactored error handling logic",
) -> ExperimentRecord:
    return ExperimentRecord(
        id="1",
        target_id="t1",
        git_sha="abc123",
        pre_experiment_sha="def456",
        timestamp=datetime.now(timezone.utc),
        hypothesis=hypothesis,
        hypothesis_source="agent",
        mutation_diff_summary=mutation_diff,
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=[score],
        baseline_score=baseline,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=10.0,
        tags=tags or ["error_handling"],
        learnings="",
        cost_usd=0.01,
        bootstrap_seed=42,
        per_criterion_scores=per_criterion,
    )


class TestLesson:
    def test_lesson_model_fields(self) -> None:
        lesson = Lesson(
            what_changed="Refactored error handling",
            what_improved=["style: PASS (was FAIL)"],
            what_regressed=[],
            transferable_insight="Better error messages improve debuggability",
            domain_tags=["error_handling", "code_style"],
        )
        assert lesson.what_changed == "Refactored error handling"
        assert len(lesson.what_improved) == 1
        assert lesson.domain_tags == ["error_handling", "code_style"]

    def test_lesson_round_trip(self) -> None:
        lesson = Lesson(
            what_changed="Added retry logic",
            what_improved=["reliability: PASS (was FAIL)"],
            what_regressed=["latency: FAIL (was PASS)"],
            transferable_insight="Retries improve reliability at cost of latency",
            domain_tags=["reliability"],
        )
        json_str = lesson.model_dump_json()
        restored = Lesson.model_validate_json(json_str)
        assert restored == lesson

    def test_backward_compat_empty_learnings(self) -> None:
        """Old records with learnings='' don't break Lesson parsing."""
        record = _make_record()
        assert record.learnings == ""
        # Parsing empty string should raise, not silently succeed
        with pytest.raises(Exception):
            Lesson.model_validate_json("")


class TestExtractLesson:
    @pytest.mark.asyncio
    async def test_extract_lesson_with_improvements(self) -> None:
        previous = _make_record(
            per_criterion={"style": 0.3, "correctness": 0.8},
        )
        current = _make_record(
            per_criterion={"style": 0.7, "correctness": 0.4},
        )
        with patch("anneal.engine.knowledge.make_client") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({
                "transferable_insight": "Style improvements came at correctness cost",
                "domain_tags": ["style", "tradeoff"],
            })
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            lesson = await extract_lesson(current, previous, "openai/gpt-4.1-mini")

        assert "style: PASS (was FAIL)" in lesson.what_improved
        assert "correctness: FAIL (was PASS)" in lesson.what_regressed
        assert lesson.transferable_insight == "Style improvements came at correctness cost"

    @pytest.mark.asyncio
    async def test_extract_lesson_deterministic_fallback(self) -> None:
        record = _make_record(hypothesis="Test hypothesis for fallback")
        with patch("anneal.engine.knowledge.make_client", side_effect=RuntimeError("No API")):
            lesson = await extract_lesson(record, None, "openai/gpt-4.1-mini")

        assert lesson.transferable_insight == "Test hypothesis for fallback"
        assert lesson.domain_tags == ["error_handling"]

    @pytest.mark.asyncio
    async def test_lesson_serialized_to_learnings_field(self) -> None:
        record = _make_record()
        with patch("anneal.engine.knowledge.make_client", side_effect=RuntimeError("No API")):
            lesson = await extract_lesson(record, None, "openai/gpt-4.1-mini")

        record.learnings = lesson.model_dump_json()
        parsed = Lesson.model_validate_json(record.learnings)
        assert parsed.what_changed == record.mutation_diff_summary[:200]


class TestLearningPoolDomainTags:
    def test_domain_tag_boost(self) -> None:
        pool = LearningPool()
        # Add learnings with JSON observations containing domain_tags
        lesson_data = json.dumps({
            "what_changed": "x",
            "transferable_insight": "y",
            "domain_tags": ["error_handling", "logging"],
        })
        tagged_learning = Learning(
            observation=lesson_data,
            signal=LearningSignal.POSITIVE,
            source_condition="guided",
            source_target="t1",
            source_experiment_ids=[1],
            score_delta=0.1,
            criterion_deltas={},
            confidence=1.0,
            tags=["error_handling"],
        )
        untagged_learning = Learning(
            observation="Plain text observation with no JSON",
            signal=LearningSignal.POSITIVE,
            source_condition="guided",
            source_target="t1",
            source_experiment_ids=[2],
            score_delta=0.2,  # Higher delta but no matching tags
            criterion_deltas={},
            confidence=1.0,
            tags=[],
        )
        pool.add(tagged_learning)
        pool.add(untagged_learning)

        results = pool.retrieve(
            scope=LearningScope.GLOBAL,
            k=5,
            domain_tags=["error_handling"],
        )
        # Tagged learning should come first despite lower score_delta
        assert len(results) == 2
        # The first result should have the matching tag observation
        assert "error_handling" in results[0].observation

    def test_retrieve_without_domain_tags_unchanged(self) -> None:
        """When domain_tags is None, retrieve behavior is unchanged."""
        pool = LearningPool()
        learning = Learning(
            observation="Test observation",
            signal=LearningSignal.POSITIVE,
            source_condition="guided",
            source_target="t1",
            source_experiment_ids=[1],
            score_delta=0.1,
            criterion_deltas={},
            confidence=1.0,
            tags=[],
        )
        pool.add(learning)
        results = pool.retrieve(scope=LearningScope.GLOBAL, k=5)
        assert len(results) == 1

    def test_extract_tags_from_json_observation(self) -> None:
        tags = LearningPool._extract_tags(json.dumps({"domain_tags": ["a", "b"]}))
        assert tags == ["a", "b"]

    def test_extract_tags_from_plain_text(self) -> None:
        tags = LearningPool._extract_tags("Just plain text")
        assert tags == []
