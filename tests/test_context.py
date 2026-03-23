from __future__ import annotations

from datetime import datetime

from anneal.engine.context import ContextBudget, TokenCache, _format_experiment_record, estimate_tokens
from anneal.engine.types import ExperimentRecord, Outcome


class TestEstimateTokens:
    def test_english_text_tokenizes_accurately(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        tokens = estimate_tokens(text)
        # tiktoken cl100k_base: 9 tokens for this sentence
        assert tokens == 9

    def test_code_tokenizes(self) -> None:
        text = "def hello():\n    return 'world'"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_empty_string_returns_zero(self) -> None:
        assert estimate_tokens("") == 0


class TestContextBudgetAddAndAssemble:
    def test_required_slots_always_included(self) -> None:
        sys_text = "System prompt with instructions for the agent."
        art_text = "Artifact content that must always be included."
        budget = ContextBudget(max_tokens=5)  # Tiny budget — required still included
        budget.add_slot("sys", sys_text, priority=1, required=True)
        budget.add_slot("artifact", art_text, priority=2, required=True)
        result = budget.assemble()
        assert sys_text in result
        assert art_text in result

    def test_optional_slots_added_in_priority_order(self) -> None:
        budget = ContextBudget(max_tokens=10_000)
        budget.add_slot("req", "Required content here.", priority=1, required=True)
        low_text = "Low priority optional content for the agent."
        high_text = "High priority optional content for the agent."
        budget.add_slot("low_pri", low_text, priority=5, required=False)
        budget.add_slot("high_pri", high_text, priority=3, required=False)
        result = budget.assemble()
        h_pos = result.index(high_text)
        l_pos = result.index(low_text)
        assert h_pos < l_pos

    def test_optional_slot_skipped_when_budget_exhausted(self) -> None:
        req_text = "This is a required prompt that takes up most of the token budget for this test."
        req_tokens = estimate_tokens(req_text)
        small_text = "Fits."
        large_text = "This optional content is very long and will not fit. " * 20
        huge_text = "This should be entirely skipped because no budget remains. " * 20
        budget = ContextBudget(max_tokens=req_tokens + 5)
        budget.add_slot("req", req_text, priority=1, required=True)
        budget.add_slot("opt1", small_text, priority=2, required=False)
        budget.add_slot("opt2", large_text, priority=3, required=False)
        budget.add_slot("opt3", huge_text, priority=4, required=False)
        result = budget.assemble()
        assert small_text in result
        assert huge_text not in result

    def test_partial_truncation_of_last_fitting_optional(self) -> None:
        req_text = "Required content that fills most of the budget for testing truncation behavior."
        req_tokens = estimate_tokens(req_text)
        opt_text = "Repeated optional content. " * 50
        budget = ContextBudget(max_tokens=req_tokens + 10)
        budget.add_slot("req", req_text, priority=1, required=True)
        budget.add_slot("opt", opt_text, priority=2, required=False)
        result = budget.assemble()
        # Optional should be present but truncated
        assert "Repeated optional" in result
        assert opt_text not in result


class TestContextBudgetProperties:
    def test_total_tokens_and_budget_remaining(self) -> None:
        content = "The quick brown fox jumps over the lazy dog near the river bank."
        tokens = estimate_tokens(content)
        budget = ContextBudget(max_tokens=tokens * 3)
        budget.add_slot("a", content, priority=1, required=True)
        budget.assemble()
        assert budget.total_tokens == tokens
        assert budget.budget_remaining == tokens * 3 - tokens

    def test_summary_returns_readable_breakdown(self) -> None:
        budget = ContextBudget(max_tokens=1000)
        budget.add_slot("sys", "x" * 100, priority=1, required=True)
        budget.add_slot("notes", "y" * 100, priority=3, required=False)
        budget.assemble()
        summary = budget.summary()
        assert "sys" in summary
        assert "notes" in summary
        assert "REQUIRED" in summary
        assert "optional" in summary
        assert "remaining" in summary


class TestTokenCache:
    def test_get_returns_consistent_count(self) -> None:
        cache = TokenCache()
        content = "hello world test"
        first = cache.get("k1", content)
        second = cache.get("k1", content)
        assert first == second
        assert first == estimate_tokens(content)

    def test_invalidate_clears_cache(self) -> None:
        cache = TokenCache()
        cache.get("k1", "some content")
        cache.invalidate("k1")
        # After invalidation, re-calling get recomputes (still same value, but exercises the path)
        result = cache.get("k1", "some content")
        assert result == estimate_tokens("some content")

    def test_detects_content_staleness(self) -> None:
        cache = TokenCache()
        v1 = cache.get("k1", "short")
        v2 = cache.get("k1", "a much longer string that has more tokens")
        # Different content under the same key produces a different token count
        assert v1 != v2
        assert v2 == estimate_tokens("a much longer string that has more tokens")


def _make_experiment_record(**overrides: object) -> ExperimentRecord:
    defaults: dict[str, object] = dict(
        id="exp-0001",
        target_id="target-1",
        git_sha="abc0001",
        pre_experiment_sha="pre0001",
        timestamp=datetime(2026, 1, 1, 0, 0),
        hypothesis="test hypothesis",
        hypothesis_source="agent",
        mutation_diff_summary="diff summary",
        score=0.75,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=0.70,
        outcome=Outcome.KEPT,
        failure_mode=None,
        duration_seconds=1.0,
        tags=["prompt"],
        learnings="some learning",
        cost_usd=0.01,
        bootstrap_seed=42,
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)  # type: ignore[arg-type]


def test_per_criterion_in_context_appears_in_formatted_record() -> None:
    """Per-criterion breakdown appears in formatted experiment record."""
    record = _make_experiment_record(
        per_criterion_scores={"clarity": 0.9, "accuracy": 0.3},
    )
    output = _format_experiment_record(record)
    assert "clarity: 0.90 (PASS)" in output
    assert "accuracy: 0.30 (FAIL)" in output


def test_format_experiment_record_no_per_criterion_omits_section() -> None:
    """Records without per_criterion_scores do not include Per-criterion section."""
    record = _make_experiment_record(per_criterion_scores=None)
    output = _format_experiment_record(record)
    assert "Per-criterion" not in output
