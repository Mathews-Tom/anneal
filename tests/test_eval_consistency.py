"""Tests for eval cache consistency monitoring."""
from __future__ import annotations

from anneal.engine.eval_cache import CacheEntry, EvalCache


class TestCacheEntryScoreHistory:
    def test_default_empty_history(self) -> None:
        entry = CacheEntry(
            content_hash="abc",
            score=0.5,
            raw_scores=(0.5,),
            criterion_names=("style",),
        )
        assert entry.score_history == ()

    def test_score_history_accumulated(self) -> None:
        cache = EvalCache()
        cache.put("content", ["style"], 0.5, [0.5])
        cache.put("content", ["style"], 0.6, [0.6])
        cache.put("content", ["style"], 0.7, [0.7])
        entry = cache.get("content", ["style"])
        assert entry is not None
        assert entry.score_history == (0.5, 0.6, 0.7)

    def test_different_content_independent_history(self) -> None:
        cache = EvalCache()
        cache.put("content_a", ["style"], 0.5, [0.5])
        cache.put("content_b", ["style"], 0.8, [0.8])
        cache.put("content_a", ["style"], 0.6, [0.6])

        entry_a = cache.get("content_a", ["style"])
        entry_b = cache.get("content_b", ["style"])
        assert entry_a is not None and entry_b is not None
        assert entry_a.score_history == (0.5, 0.6)
        assert entry_b.score_history == (0.8,)

    def test_backward_compat_empty_history(self) -> None:
        """CacheEntry with default score_history=() works normally."""
        entry = CacheEntry(
            content_hash="abc",
            score=0.5,
            raw_scores=(0.5,),
            criterion_names=("style",),
        )
        assert entry.score_history == ()
        assert entry.score == 0.5


class TestConsistencyReport:
    def test_flags_high_variance(self) -> None:
        cache = EvalCache()
        cache.put("content", ["style"], 0.3, [0.3])
        cache.put("content", ["style"], 0.8, [0.8])
        cache.put("content", ["style"], 0.2, [0.2])
        report = cache.consistency_report()
        assert len(report) == 1
        assert report[0]["n_evals"] == 3
        assert report[0]["std_dev"] > 0.1

    def test_ignores_low_variance(self) -> None:
        cache = EvalCache()
        cache.put("content", ["style"], 0.50, [0.50])
        cache.put("content", ["style"], 0.51, [0.51])
        cache.put("content", ["style"], 0.49, [0.49])
        report = cache.consistency_report()
        assert len(report) == 0

    def test_skips_single_eval(self) -> None:
        cache = EvalCache()
        cache.put("content", ["style"], 0.5, [0.5])
        report = cache.consistency_report()
        assert len(report) == 0

    def test_multiple_entries_some_flagged(self) -> None:
        cache = EvalCache()
        # Stable entry
        cache.put("stable", ["style"], 0.5, [0.5])
        cache.put("stable", ["style"], 0.51, [0.51])
        # Unstable entry
        cache.put("unstable", ["style"], 0.2, [0.2])
        cache.put("unstable", ["style"], 0.9, [0.9])
        report = cache.consistency_report()
        assert len(report) == 1
        assert report[0]["content_hash"] == cache._hash_content("unstable", ["style"])

    def test_empty_cache_empty_report(self) -> None:
        cache = EvalCache()
        assert cache.consistency_report() == []
