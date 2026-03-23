"""Tests for failure taxonomy classification, distribution, and blind spots."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from anneal.engine.taxonomy import FailureTaxonomy, SEED_CATEGORIES
from anneal.engine.types import ExperimentRecord, FailureClassification, Outcome


def _make_record(
    outcome: Outcome = Outcome.DISCARDED,
    failure_mode: str | None = None,
    failure_classification: FailureClassification | None = None,
    hypothesis: str = "test hypothesis",
) -> ExperimentRecord:
    return ExperimentRecord(
        id="test", target_id="t", git_sha="abc", pre_experiment_sha="abc",
        timestamp=datetime.now(tz=timezone.utc), hypothesis=hypothesis,
        hypothesis_source="agent", mutation_diff_summary="", score=0.5,
        score_ci_lower=None, score_ci_upper=None, raw_scores=None,
        baseline_score=0.5, outcome=outcome, failure_mode=failure_mode,
        duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
        bootstrap_seed=0,
        failure_classification=failure_classification,
    )


class TestFailureTaxonomyInit:
    def test_seed_categories_loaded(self) -> None:
        t = FailureTaxonomy()
        assert len(t.categories) == 8
        assert "output_format" in t.category_names
        assert "logic_error" in t.category_names

    def test_custom_categories_appended(self) -> None:
        custom = [{"category": "custom_cat", "description": "Custom failure"}]
        t = FailureTaxonomy(custom_categories=custom)
        assert len(t.categories) == 9
        assert "custom_cat" in t.category_names

    def test_seed_categories_immutable(self) -> None:
        assert len(SEED_CATEGORIES) == 8


class TestFallbackClassify:
    def test_scope_keyword(self) -> None:
        t = FailureTaxonomy()
        c = t._fallback_classify("h", "scope_violation:artifact.md")
        assert c.category == "scope_violation"

    def test_syntax_keyword(self) -> None:
        t = FailureTaxonomy()
        c = t._fallback_classify("h", "syntax error in output")
        assert c.category == "syntax_error"

    def test_verifier_keyword(self) -> None:
        t = FailureTaxonomy()
        c = t._fallback_classify("h", "verifier:typecheck")
        assert c.category == "syntax_error"

    def test_constraint_keyword(self) -> None:
        t = FailureTaxonomy()
        c = t._fallback_classify("h", "constraint_violated:metric")
        assert c.category == "regression"

    def test_unknown_defaults_to_logic_error(self) -> None:
        t = FailureTaxonomy()
        c = t._fallback_classify("h", "something unknown")
        assert c.category == "logic_error"
        assert c.confidence == 0.3


class TestDistribution:
    def test_counts_classifications(self) -> None:
        records = [
            _make_record(failure_classification=FailureClassification(
                category="output_format", description="d", fix_direction="f",
            )),
            _make_record(failure_classification=FailureClassification(
                category="output_format", description="d", fix_direction="f",
            )),
            _make_record(failure_classification=FailureClassification(
                category="logic_error", description="d", fix_direction="f",
            )),
        ]
        dist = FailureTaxonomy.distribution(records)
        assert dist == {"output_format": 2, "logic_error": 1}

    def test_ignores_records_without_classification(self) -> None:
        records = [
            _make_record(failure_classification=None),
            _make_record(failure_classification=FailureClassification(
                category="regression", description="d", fix_direction="f",
            )),
        ]
        dist = FailureTaxonomy.distribution(records)
        assert dist == {"regression": 1}

    def test_empty_records(self) -> None:
        assert FailureTaxonomy.distribution([]) == {}


class TestBlindSpotCheck:
    def test_detects_unattributed_categories(self) -> None:
        t = FailureTaxonomy()
        records = [
            _make_record(failure_classification=FailureClassification(
                category="output_format", description="d", fix_direction="f",
            ))
            for _ in range(15)
        ]
        blind = t.blind_spot_check(records)
        # All 7 other categories should be blind spots
        assert len(blind) == 7
        assert "output_format" not in blind
        assert "logic_error" in blind

    def test_no_blind_spots_when_insufficient_failures(self) -> None:
        t = FailureTaxonomy()
        records = [
            _make_record(failure_classification=FailureClassification(
                category="output_format", description="d", fix_direction="f",
            ))
            for _ in range(5)
        ]
        blind = t.blind_spot_check(records)
        assert blind == []

    def test_no_blind_spots_when_all_attributed(self) -> None:
        t = FailureTaxonomy()
        records = []
        for cat_info in SEED_CATEGORIES:
            records.extend([
                _make_record(failure_classification=FailureClassification(
                    category=cat_info["category"], description="d", fix_direction="f",
                ))
                for _ in range(2)
            ])
        blind = t.blind_spot_check(records)
        assert blind == []


class TestClassifyLLM:
    @pytest.mark.asyncio
    async def test_classify_parses_json_response(self) -> None:
        t = FailureTaxonomy()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"category": "output_format", "description": "bad json", "fix_direction": "fix schema", "confidence": 0.9}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.taxonomy.make_client", return_value=mock_client):
            classification, cost = await t.classify(
                hypothesis="test", failure_mode="format error", score=0.3, model="gpt-4.1-mini",
            )

        assert classification.category == "output_format"
        assert classification.description == "bad json"
        assert classification.confidence == 0.9

    @pytest.mark.asyncio
    async def test_classify_handles_markdown_fences(self) -> None:
        t = FailureTaxonomy()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"category": "logic_error", "description": "wrong calc", "fix_direction": "fix math", "confidence": 0.7}\n```'
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.taxonomy.make_client", return_value=mock_client):
            classification, cost = await t.classify(
                hypothesis="test", failure_mode="wrong", score=0.2, model="gpt-4.1-mini",
            )

        assert classification.category == "logic_error"
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_classify_falls_back_on_api_error(self) -> None:
        t = FailureTaxonomy()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with patch("anneal.engine.taxonomy.make_client", return_value=mock_client):
            classification, cost = await t.classify(
                hypothesis="test", failure_mode="verifier:typecheck", score=0.0, model="gpt-4.1-mini",
            )

        assert classification.category == "syntax_error"
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_classify_falls_back_on_invalid_json(self) -> None:
        t = FailureTaxonomy()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I'm not sure what happened"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.taxonomy.make_client", return_value=mock_client):
            classification, cost = await t.classify(
                hypothesis="test", failure_mode="constraint_violated:score", score=0.1, model="gpt-4.1-mini",
            )

        assert classification.category == "regression"

    @pytest.mark.asyncio
    async def test_classify_corrects_unknown_category(self) -> None:
        t = FailureTaxonomy()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"category": "output-format", "description": "d", "fix_direction": "f", "confidence": 0.8}'
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.taxonomy.make_client", return_value=mock_client):
            classification, _ = await t.classify(
                hypothesis="test", failure_mode="format", score=0.3, model="gpt-4.1-mini",
            )

        assert classification.category == "output_format"


class TestConsolidationFailureTracking:
    def test_consolidation_includes_failure_distribution(self, tmp_path) -> None:
        from anneal.engine.knowledge import KnowledgeStore

        ks = KnowledgeStore(tmp_path)
        for i in range(50):
            fc = None
            if i < 20:
                fc = FailureClassification(
                    category="output_format", description="d", fix_direction="f",
                )
            elif i < 30:
                fc = FailureClassification(
                    category="logic_error", description="d", fix_direction="f",
                )
            record = ExperimentRecord(
                id=str(i), target_id="t", git_sha="abc", pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc), hypothesis=f"h{i}",
                hypothesis_source="agent", mutation_diff_summary="", score=float(i),
                score_ci_lower=None, score_ci_upper=None, raw_scores=None,
                baseline_score=0.0,
                outcome=Outcome.DISCARDED if i < 30 else Outcome.KEPT,
                failure_mode=None, failure_classification=fc,
                duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
                bootstrap_seed=0,
            )
            ks.append_record(record)

        cr = ks.consolidate()
        assert cr.failure_distribution == {"output_format": 20, "logic_error": 10}
        assert len(cr.blind_spots) > 0
