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


# ---------------------------------------------------------------------------
# Step 7.2 — Bradley-Terry scorer
# ---------------------------------------------------------------------------


class TestBradleyTerryScorer:
    """Tests for Bradley-Terry strength estimation."""

    def test_estimate_strength_all_yes_near_one(self) -> None:
        from anneal.engine.eval import BradleyTerryScorer
        mean, uncertainty = BradleyTerryScorer.estimate_strength(10, 10)
        assert mean > 0.85
        assert uncertainty < 0.2

    def test_estimate_strength_all_no_near_zero(self) -> None:
        from anneal.engine.eval import BradleyTerryScorer
        mean, uncertainty = BradleyTerryScorer.estimate_strength(0, 10)
        assert mean < 0.15
        assert uncertainty < 0.2

    def test_estimate_strength_even_split_near_half(self) -> None:
        from anneal.engine.eval import BradleyTerryScorer
        mean, uncertainty = BradleyTerryScorer.estimate_strength(5, 10)
        assert abs(mean - 0.5) < 0.1

    def test_should_stop_early_confident_above(self) -> None:
        from anneal.engine.eval import BradleyTerryScorer
        # 9/10 yes votes: mean ≈ 0.833, uncertainty small
        mean, uncertainty = BradleyTerryScorer.estimate_strength(9, 10)
        assert BradleyTerryScorer.should_stop_early(mean, uncertainty) is True

    def test_should_stop_early_uncertain_no_stop(self) -> None:
        from anneal.engine.eval import BradleyTerryScorer
        # 3/5 yes votes: mean ≈ 0.571, still uncertain
        mean, uncertainty = BradleyTerryScorer.estimate_strength(3, 5)
        assert BradleyTerryScorer.should_stop_early(mean, uncertainty) is False


# ---------------------------------------------------------------------------
# Step 7.3 — FidelityStage dataclass
# ---------------------------------------------------------------------------


_FIDELITY_PLACEHOLDER = True
