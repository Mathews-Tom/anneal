"""MAP-Elites quality-diversity archive for maintaining diverse solutions.

Maintains a grid of best solutions indexed by behavioral descriptors
(e.g., per-criterion scores). Each cell holds the highest-fitness
solution that maps to that behavioral region.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveEntry:
    """A single entry in the MAP-Elites archive."""

    record_id: str
    behavior: tuple[float, ...]  # Discretized behavior descriptor
    fitness: float
    raw_behavior: dict[str, float]  # Original per-criterion scores
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MapElitesArchive:
    """MAP-Elites archive mapping behavior descriptors to best solutions.

    Behavior descriptors are per-criterion scores discretized into bins.
    Each bin holds only the highest-fitness solution that maps to it.
    """

    def __init__(
        self,
        n_bins: int = 10,
        persist_path: Path | None = None,
    ) -> None:
        self._n_bins = n_bins
        self._archive: dict[tuple[float, ...], ArchiveEntry] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load()

    def _discretize(self, behavior: dict[str, float]) -> tuple[float, ...]:
        """Discretize continuous behavior values into bin indices.

        Each value in [0, 1] is mapped to a bin in [0, n_bins-1].
        Values outside [0, 1] are clamped.
        """
        bins: list[float] = []
        for key in sorted(behavior.keys()):
            val = max(0.0, min(1.0, behavior[key]))
            bin_idx = min(int(val * self._n_bins), self._n_bins - 1)
            bins.append(float(bin_idx))
        return tuple(bins)

    def add(
        self,
        record_id: str,
        behavior: dict[str, float],
        fitness: float,
    ) -> bool:
        """Add a solution to the archive.

        Returns True if the solution was added (new cell or better fitness).
        """
        cell = self._discretize(behavior)
        existing = self._archive.get(cell)

        if existing is not None and existing.fitness >= fitness:
            return False  # Existing solution is better

        entry = ArchiveEntry(
            record_id=record_id,
            behavior=cell,
            fitness=fitness,
            raw_behavior=behavior,
        )
        self._archive[cell] = entry

        if self._persist_path:
            self._save()

        return True

    def get(self, behavior: dict[str, float]) -> ArchiveEntry | None:
        """Look up the best solution for a behavior region."""
        cell = self._discretize(behavior)
        return self._archive.get(cell)

    def get_diverse_solutions(self, k: int = 5) -> list[ArchiveEntry]:
        """Return top-k highest-fitness entries from distinct cells."""
        entries = sorted(
            self._archive.values(),
            key=lambda e: e.fitness,
            reverse=True,
        )
        return entries[:k]

    @property
    def coverage(self) -> int:
        """Number of occupied cells in the archive."""
        return len(self._archive)

    @property
    def max_cells(self) -> int:
        """Theoretical maximum cells (n_bins ^ n_dimensions).

        Returns 0 if no entries exist (dimensions unknown).
        """
        if not self._archive:
            return 0
        sample = next(iter(self._archive.values()))
        n_dims = len(sample.behavior)
        return self._n_bins ** n_dims

    def _save(self) -> None:
        """Persist archive to JSONL file."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._persist_path) + ".lock")
        with lock:
            with open(self._persist_path, "w") as f:
                for entry in self._archive.values():
                    data = {
                        "record_id": entry.record_id,
                        "behavior": list(entry.behavior),
                        "fitness": entry.fitness,
                        "raw_behavior": entry.raw_behavior,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                    f.write(json.dumps(data) + "\n")

    def _load(self) -> None:
        """Load archive from JSONL file."""
        if not self._persist_path or not self._persist_path.exists():
            return
        lock = FileLock(str(self._persist_path) + ".lock")
        with lock:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = ArchiveEntry(
                        record_id=data["record_id"],
                        behavior=tuple(data["behavior"]),
                        fitness=data["fitness"],
                        raw_behavior=data["raw_behavior"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                    )
                    self._archive[entry.behavior] = entry
