from __future__ import annotations

from anneal.engine.context import ContextBudget, TokenCache, estimate_tokens


class TestEstimateTokens:
    def test_400_chars_returns_approx_100(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_empty_string_returns_zero(self) -> None:
        assert estimate_tokens("") == 0


class TestContextBudgetAddAndAssemble:
    def test_required_slots_always_included(self) -> None:
        budget = ContextBudget(max_tokens=50)
        budget.add_slot("sys", "x" * 80, priority=1, required=True)
        budget.add_slot("artifact", "y" * 80, priority=2, required=True)
        result = budget.assemble()
        assert "x" * 80 in result
        assert "y" * 80 in result

    def test_optional_slots_added_in_priority_order(self) -> None:
        budget = ContextBudget(max_tokens=10_000)
        budget.add_slot("req", "r" * 40, priority=1, required=True)
        budget.add_slot("low_pri", "L" * 40, priority=5, required=False)
        budget.add_slot("high_pri", "H" * 40, priority=3, required=False)
        result = budget.assemble()
        # Both optional slots fit; verify priority ordering in output
        h_pos = result.index("H" * 40)
        l_pos = result.index("L" * 40)
        assert h_pos < l_pos

    def test_optional_slot_skipped_when_budget_exhausted(self) -> None:
        budget = ContextBudget(max_tokens=30)
        # Required slot uses 25 tokens (100 chars / 4)
        budget.add_slot("req", "r" * 100, priority=1, required=True)
        # Optional slot needs 25 tokens — only 5 remain, but truncation will
        # partially fit. Use a very large optional to prove skip after budget gone.
        budget.add_slot("opt1", "a" * 20, priority=2, required=False)  # 5 tokens, fits
        budget.add_slot("opt2", "b" * 400, priority=3, required=False)  # 100 tokens, won't fully fit
        budget.add_slot("opt3", "c" * 400, priority=4, required=False)  # should be skipped entirely
        result = budget.assemble()
        assert "a" * 20 in result
        assert "c" * 400 not in result

    def test_partial_truncation_of_last_fitting_optional(self) -> None:
        budget = ContextBudget(max_tokens=30)
        # Required: 25 tokens
        budget.add_slot("req", "r" * 100, priority=1, required=True)
        # Optional needs 100 tokens but only 5 remain → truncated to ~20 chars
        budget.add_slot("opt", "abcd" * 100, priority=2, required=False)
        result = budget.assemble()
        # The optional content should be present but truncated
        assert "abcd" in result
        assert "abcd" * 100 not in result


class TestContextBudgetProperties:
    def test_total_tokens_and_budget_remaining(self) -> None:
        budget = ContextBudget(max_tokens=200)
        budget.add_slot("a", "x" * 400, priority=1, required=True)  # 100 tokens
        budget.assemble()
        assert budget.total_tokens == 100
        assert budget.budget_remaining == 100

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
