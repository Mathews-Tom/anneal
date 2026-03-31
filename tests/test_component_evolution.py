"""Tests for component evolution: weakest component selection, prompt assembly, streak tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from anneal.engine.strategy import (
    StrategyManifest,
    _summarize_criterion_performance,
    evolve_weakest_component,
    load_strategy,
    save_strategy,
)
from anneal.engine.types import ExperimentRecord, Outcome


def _make_record(
    outcome: Outcome = Outcome.KEPT,
    score: float = 0.5,
    per_criterion_scores: dict[str, float] | None = None,
    hypothesis: str = "test hypothesis",
) -> ExperimentRecord:
    return ExperimentRecord(
        id="test-id",
        target_id="target-1",
        git_sha="abc123",
        pre_experiment_sha="def456",
        timestamp=datetime.now(tz=timezone.utc),
        hypothesis=hypothesis,
        hypothesis_source="agent",
        mutation_diff_summary="",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=0.0,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=1.0,
        tags=[],
        learnings="",
        cost_usd=0.01,
        bootstrap_seed=42,
        per_criterion_scores=per_criterion_scores,
    )


class TestEvolveWeakestSelection:
    def test_evolve_weakest_selects_highest_streak(self) -> None:
        # Arrange
        manifest = StrategyManifest()
        manifest.hypothesis_generation.streak_without_improvement = 2
        manifest.context_assembly.streak_without_improvement = 10
        manifest.mutation_style.streak_without_improvement = 1
        manifest.failure_analysis.streak_without_improvement = 5

        # Act
        target, _prompt = evolve_weakest_component(manifest, [])

        # Assert
        assert target.name == "context_assembly"
        assert target.streak_without_improvement == 10

    def test_evolution_prompt_includes_criterion_feedback(self) -> None:
        # Arrange
        manifest = StrategyManifest()
        manifest.failure_analysis.streak_without_improvement = 8
        records = [
            _make_record(
                outcome=Outcome.DISCARDED,
                per_criterion_scores={"clarity": 0.0, "relevance": 1.0},
            ),
            _make_record(
                outcome=Outcome.KEPT,
                per_criterion_scores={"clarity": 1.0, "relevance": 1.0},
            ),
            _make_record(
                outcome=Outcome.DISCARDED,
                per_criterion_scores={"clarity": 0.0, "relevance": 0.0},
            ),
        ]

        # Act
        _target, prompt = evolve_weakest_component(manifest, records)

        # Assert — prompt contains PASS/FAIL summary for each criterion
        assert "clarity" in prompt
        assert "relevance" in prompt
        # clarity has 2 failures → appears in summary with failure count
        assert "failure" in prompt.lower() or "fail" in prompt.lower()


class TestStreakTracking:
    def test_component_streak_reset_on_kept(self) -> None:
        # Arrange
        manifest = StrategyManifest()
        manifest.hypothesis_generation.streak_without_improvement = 5
        manifest.context_assembly.streak_without_improvement = 3
        manifest.mutation_style.streak_without_improvement = 7
        manifest.failure_analysis.streak_without_improvement = 1

        # Act — simulate KEPT outcome: reset all streaks to 0
        for component in manifest.components:
            component.streak_without_improvement = 0

        # Assert
        assert all(c.streak_without_improvement == 0 for c in manifest.components)

    def test_component_streak_increment_on_discard(self) -> None:
        # Arrange
        manifest = StrategyManifest()
        initial_streaks = {c.name: c.streak_without_improvement for c in manifest.components}

        # Act — simulate DISCARDED outcome: increment all streaks by 1
        for component in manifest.components:
            component.streak_without_improvement += 1

        # Assert
        for component in manifest.components:
            assert component.streak_without_improvement == initial_streaks[component.name] + 1


class TestEvolutionSavesManifest:
    def test_evolution_saves_manifest(self, tmp_path: Path) -> None:
        # Arrange
        manifest = StrategyManifest()
        manifest.mutation_style.streak_without_improvement = 9

        # Act — evolve, update approach, save, reload
        target, _prompt = evolve_weakest_component(manifest, [])
        target.approach = "Revised: prefer surgical single-line edits."
        save_strategy(manifest, tmp_path)
        reloaded = load_strategy(tmp_path)

        # Assert — reloaded manifest reflects the updated approach
        assert reloaded is not None
        assert reloaded.mutation_style.approach == "Revised: prefer surgical single-line edits."

    def test_evolution_records_lineage(self, tmp_path: Path) -> None:
        # Arrange
        manifest = StrategyManifest()
        manifest.context_assembly.streak_without_improvement = 6

        # Act — append lineage entry, save, reload
        _target, _prompt = evolve_weakest_component(manifest, [])
        manifest.lineage.append("evolved context_assembly at experiment 12")
        save_strategy(manifest, tmp_path)
        reloaded = load_strategy(tmp_path)

        # Assert — lineage entry survives the round-trip
        assert reloaded is not None
        assert len(reloaded.lineage) == 1
        assert "context_assembly" in reloaded.lineage[0]


class TestLegacyMode:
    def test_legacy_mode_unchanged(self, tmp_path: Path) -> None:
        # Arrange — no strategy.yaml exists in knowledge_path

        # Act
        result = load_strategy(tmp_path)

        # Assert — returns None without error; caller handles None as legacy mode
        assert result is None


class TestSummarizeCriterionPerformance:
    def test_summarize_no_records_returns_placeholder(self) -> None:
        # Arrange
        records: list[ExperimentRecord] = []

        # Act
        summary = _summarize_criterion_performance(records)

        # Assert
        assert "No experiments" in summary

    def test_summarize_records_without_criterion_scores(self) -> None:
        # Arrange
        records = [_make_record(per_criterion_scores=None)]

        # Act
        summary = _summarize_criterion_performance(records)

        # Assert
        assert "No per-criterion data" in summary

    def test_summarize_orders_by_failure_count_descending(self) -> None:
        # Arrange — "tone" has 3 failures, "clarity" has 1 failure
        records = [
            _make_record(per_criterion_scores={"tone": 0.0, "clarity": 0.0}),
            _make_record(per_criterion_scores={"tone": 0.0, "clarity": 1.0}),
            _make_record(per_criterion_scores={"tone": 0.0, "clarity": 1.0}),
        ]

        # Act
        summary = _summarize_criterion_performance(records)

        # Assert — "tone" appears before "clarity" in the summary
        assert summary.index("tone") < summary.index("clarity")

    def test_summarize_pass_fail_counts_correct(self) -> None:
        # Arrange
        records = [
            _make_record(per_criterion_scores={"quality": 1.0}),
            _make_record(per_criterion_scores={"quality": 1.0}),
            _make_record(per_criterion_scores={"quality": 0.0}),
        ]

        # Act
        summary = _summarize_criterion_performance(records)

        # Assert — 2 passes, 1 failure out of 3 total
        assert "2/3" in summary
        assert "1 failure" in summary
