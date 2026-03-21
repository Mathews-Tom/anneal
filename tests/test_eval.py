"""Tests for anneal.engine.eval — EvalResult type contracts."""

from __future__ import annotations

from anneal.engine.types import EvalResult


def test_eval_result_accepts_criterion_names() -> None:
    """EvalResult carries criterion_names field."""
    result = EvalResult(
        score=0.8,
        criterion_names=["clarity", "accuracy", "relevance"],
    )
    assert result.criterion_names == ["clarity", "accuracy", "relevance"]


def test_eval_result_criterion_names_defaults_none() -> None:
    """Backward compat: criterion_names defaults to None."""
    result = EvalResult(score=0.8)
    assert result.criterion_names is None


def test_eval_result_criterion_names_length_matches_criteria_count() -> None:
    """criterion_names length equals the number of criteria passed in."""
    names = ["fluency", "conciseness", "tone"]
    result = EvalResult(score=0.75, criterion_names=names)
    assert len(result.criterion_names) == 3  # type: ignore[arg-type]


def test_eval_result_criterion_names_order_preserved() -> None:
    """criterion_names preserves insertion order (original criteria order, not shuffled)."""
    names = ["z_criterion", "a_criterion", "m_criterion"]
    result = EvalResult(score=0.9, criterion_names=names)
    assert result.criterion_names == ["z_criterion", "a_criterion", "m_criterion"]


def test_eval_result_backward_compat_all_optional_fields_default() -> None:
    """EvalResult with only score uses None/0.0 defaults for all optional fields."""
    result = EvalResult(score=0.5)
    assert result.ci_lower is None
    assert result.ci_upper is None
    assert result.raw_scores is None
    assert result.cost_usd == 0.0
    assert result.criterion_names is None


def test_eval_result_per_criterion_scores_computed() -> None:
    """EvalResult carries per_criterion_scores dict."""
    result = EvalResult(
        score=0.8,
        per_criterion_scores={"clarity": 0.9, "accuracy": 0.7},
    )
    assert result.per_criterion_scores == {"clarity": 0.9, "accuracy": 0.7}


def test_eval_result_per_criterion_scores_defaults_none() -> None:
    """Backward compat: per_criterion_scores defaults to None."""
    result = EvalResult(score=0.8)
    assert result.per_criterion_scores is None
