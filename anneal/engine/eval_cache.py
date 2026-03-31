"""LRU cache for stochastic evaluation results."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypedDict


class ConsistencyEntry(TypedDict):
    """A single flagged entry from the consistency report."""

    content_hash: str
    mean_score: float
    std_dev: float
    n_evals: int


@dataclass(frozen=True)
class CacheEntry:
    """Immutable cache entry storing evaluation results."""
    content_hash: str
    score: float
    raw_scores: tuple[float, ...]
    criterion_names: tuple[str, ...]
    hit_count: int = 0
    score_history: tuple[float, ...] = ()


class EvalCache:
    """Content-hash based LRU cache for eval results."""

    def __init__(self, max_size: int = 200) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits: int = 0
        self._misses: int = 0

    def _hash_content(self, artifact_content: str, criteria_names: list[str]) -> str:
        """Hash artifact + criteria to produce cache key."""
        combined = artifact_content + "|" + "|".join(sorted(criteria_names))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, artifact_content: str, criteria_names: list[str]) -> CacheEntry | None:
        """Look up cached result. Returns None on miss."""
        key = self._hash_content(artifact_content, criteria_names)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(
        self,
        artifact_content: str,
        criteria_names: list[str],
        score: float,
        raw_scores: list[float],
    ) -> None:
        """Store evaluation result. Evicts LRU entry if at capacity."""
        key = self._hash_content(artifact_content, criteria_names)
        existing = self._cache.get(key)
        if existing is not None:
            history: tuple[float, ...] = existing.score_history + (score,)
        else:
            history = (score,)
        self._cache[key] = CacheEntry(
            content_hash=key,
            score=score,
            raw_scores=tuple(raw_scores),
            criterion_names=tuple(sorted(criteria_names)),
            score_history=history,
        )
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # Evict LRU

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a float between 0.0 and 1.0."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    def consistency_report(self) -> list[ConsistencyEntry]:
        """Return entries with high score variance (potential evaluator drift).

        Only reports entries with 2 or more evaluations.
        """
        flagged: list[ConsistencyEntry] = []
        for entry in self._cache.values():
            if len(entry.score_history) < 2:
                continue
            scores = list(entry.score_history)
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
            std_dev = variance ** 0.5
            if std_dev > 0.1:
                flagged.append({
                    "content_hash": entry.content_hash,
                    "mean_score": mean,
                    "std_dev": std_dev,
                    "n_evals": len(scores),
                })
        return flagged
