"""Tests for anneal.engine.eval_cache — EvalCache."""
from __future__ import annotations

from anneal.engine.eval_cache import CacheEntry, EvalCache


class TestEvalCacheHitMiss:
    """Tests for cache get/put behavior."""

    def test_cache_hit_returns_result(self) -> None:
        cache = EvalCache(max_size=10)
        cache.put("artifact content", ["clarity", "accuracy"], 0.85, [0.9, 0.8])

        result = cache.get("artifact content", ["clarity", "accuracy"])

        assert result is not None
        assert result.score == 0.85
        assert result.raw_scores == (0.9, 0.8)
        assert result.criterion_names == ("accuracy", "clarity")  # stored sorted

    def test_cache_miss_returns_none(self) -> None:
        cache = EvalCache(max_size=10)
        cache.put("artifact A", ["clarity"], 0.85, [0.9])

        result = cache.get("artifact B", ["clarity"])

        assert result is None

    def test_cache_key_includes_criteria_different_criteria_miss(self) -> None:
        cache = EvalCache(max_size=10)
        cache.put("same content", ["clarity"], 0.85, [0.9])

        result = cache.get("same content", ["accuracy"])

        assert result is None

    def test_cache_key_criteria_order_independent(self) -> None:
        cache = EvalCache(max_size=10)
        cache.put("content", ["b", "a"], 0.85, [0.9, 0.8])

        # Different order, same criteria → should hit
        result = cache.get("content", ["a", "b"])

        assert result is not None


class TestEvalCacheLRU:
    """Tests for LRU eviction behavior."""

    def test_lru_eviction_oldest_entry_removed(self) -> None:
        cache = EvalCache(max_size=2)
        cache.put("first", ["c"], 0.1, [0.1])
        cache.put("second", ["c"], 0.2, [0.2])
        cache.put("third", ["c"], 0.3, [0.3])  # Evicts "first"

        assert cache.get("first", ["c"]) is None
        assert cache.get("second", ["c"]) is not None
        assert cache.get("third", ["c"]) is not None

    def test_lru_access_refreshes_entry(self) -> None:
        cache = EvalCache(max_size=2)
        cache.put("first", ["c"], 0.1, [0.1])
        cache.put("second", ["c"], 0.2, [0.2])

        # Access "first" to make it recently used
        cache.get("first", ["c"])

        cache.put("third", ["c"], 0.3, [0.3])  # Evicts "second" (LRU)

        assert cache.get("first", ["c"]) is not None
        assert cache.get("second", ["c"]) is None


class TestEvalCacheMetrics:
    """Tests for cache hit rate tracking."""

    def test_hit_rate_initially_zero(self) -> None:
        cache = EvalCache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_tracks_correctly(self) -> None:
        cache = EvalCache()
        cache.put("content", ["c"], 0.5, [0.5])
        cache.get("content", ["c"])  # hit
        cache.get("other", ["c"])    # miss

        assert cache.hit_rate == 0.5

    def test_size_tracks_entries(self) -> None:
        cache = EvalCache(max_size=10)
        assert cache.size == 0
        cache.put("a", ["c"], 0.1, [0.1])
        cache.put("b", ["c"], 0.2, [0.2])
        assert cache.size == 2
