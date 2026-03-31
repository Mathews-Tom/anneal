from __future__ import annotations

from datetime import datetime, timezone

from anneal.engine.context import (
    _deduplicate_criteria,
    _format_experiment_summary,
    _format_recent_history,
    estimate_tokens,
)
from anneal.engine.types import ExperimentRecord, Outcome


def _make_record(
    idx: int,
    score: float = 0.5,
    baseline: float = 0.4,
    outcome: Outcome = Outcome.KEPT,
    hypothesis: str = "test hypothesis",
    per_criterion: dict[str, float] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=f"rec-{idx:04d}-abcd-efgh",
        target_id="test",
        git_sha=f"sha{idx}",
        pre_experiment_sha=f"sha{idx - 1}" if idx > 0 else "sha0",
        timestamp=datetime.now(tz=timezone.utc),
        hypothesis=hypothesis,
        hypothesis_source="agent",
        mutation_diff_summary="changed something",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=baseline,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=10.0,
        tags=[],
        learnings="",
        cost_usd=0.1,
        bootstrap_seed=42,
        per_criterion_scores=per_criterion,
    )


def test_compression_none_matches_original_behavior() -> None:
    records = [_make_record(i) for i in range(5)]
    result_none = _format_recent_history(records, compression="none")
    result_default = _format_recent_history(records)
    assert result_none == result_default


def test_compression_moderate_last_three_full() -> None:
    records = [_make_record(i) for i in range(5)]
    result = _format_recent_history(records, compression="moderate")
    # First 2 records should be one-line summaries
    for rec in records[:2]:
        assert f"- Exp {rec.id[:8]}" in result
    # Last 3 records should be full format
    for rec in records[2:]:
        assert f"## Experiment {rec.id}" in result


def test_compression_aggressive_last_two_full() -> None:
    records = [_make_record(i) for i in range(5)]
    result = _format_recent_history(records, compression="aggressive")
    # First 3 records should be one-line summaries
    for rec in records[:3]:
        assert f"- Exp {rec.id[:8]}" in result
    # Last 2 records should be full format
    for rec in records[3:]:
        assert f"## Experiment {rec.id}" in result


def test_experiment_summary_format() -> None:
    record = _make_record(0, score=0.7, baseline=0.4, hypothesis="improve clarity of output text")
    summary = _format_experiment_summary(record)
    assert record.id[:8] in summary
    assert "KEPT" in summary
    assert "0.7000" in summary
    assert "+0.3000" in summary
    assert "improve clarity" in summary


def test_deduplicate_criteria_collapses_consecutive() -> None:
    records = [_make_record(i, per_criterion={"clarity": 0.3}) for i in range(5)]
    result = _deduplicate_criteria(records)
    assert "clarity: FAIL (exp 0-4)" in result


def test_deduplicate_criteria_alternating() -> None:
    records = []
    for i in range(5):
        score = 0.8 if i % 2 == 0 else 0.3
        records.append(_make_record(i, per_criterion={"clarity": score}))
    result = _deduplicate_criteria(records)
    # Each should be listed separately since statuses alternate
    assert "clarity: PASS (exp 0)" in result
    assert "clarity: FAIL (exp 1)" in result
    assert "clarity: PASS (exp 2)" in result
    assert "clarity: FAIL (exp 3)" in result
    assert "clarity: PASS (exp 4)" in result


def test_deduplicate_criteria_empty_history() -> None:
    result = _deduplicate_criteria([])
    assert result == ""


def test_token_reduction_moderate_vs_none() -> None:
    records = [
        _make_record(
            i,
            hypothesis=f"hypothesis number {i} with extra detail to increase token count",
            per_criterion={"clarity": 0.8, "relevance": 0.6},
        )
        for i in range(5)
    ]
    tokens_none = estimate_tokens(_format_recent_history(records, compression="none"))
    tokens_moderate = estimate_tokens(_format_recent_history(records, compression="moderate"))
    assert tokens_moderate < tokens_none
