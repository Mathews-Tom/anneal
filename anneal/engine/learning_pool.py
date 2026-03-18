"""Cross-experiment knowledge transfer with scope-based filtering.

The Learning Pool provides structured knowledge sharing between search
conditions and optimization targets. Learnings are extracted deterministically
from ExperimentRecord fields — no LLM summarization.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from filelock import FileLock

from anneal.engine.types import ExperimentRecord, Outcome


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LearningScope(Enum):
    """Scope of knowledge retrieval."""

    CONDITION = "condition"
    TARGET = "target"
    PROJECT = "project"
    GLOBAL = "global"


class LearningSignal(Enum):
    """Whether the learning represents a positive or negative outcome."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


# ---------------------------------------------------------------------------
# Learning dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Learning:
    """A distilled observation extracted from an ExperimentRecord."""

    observation: str
    signal: LearningSignal
    source_condition: str
    source_target: str
    source_experiment_ids: list[int]
    score_delta: float
    criterion_deltas: dict[str, float]
    confidence: float
    tags: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    project_id: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_learning(
    record: ExperimentRecord,
    previous_per_criterion: dict[str, float] | None = None,
    source_condition: str = "",
    source_target: str = "",
) -> Learning:
    """Extract a Learning from a completed experiment. Deterministic — no LLM.

    Args:
        record: The completed experiment record.
        previous_per_criterion: Per-criterion scores from the previous best,
            keyed by criterion name. Used to compute criterion_deltas.
        source_condition: The search condition that produced this experiment
            (e.g. "guided", "random", "bayesian").
        source_target: The optimization target ID.

    Returns:
        A frozen Learning instance.
    """
    score_delta = record.score - record.baseline_score

    if record.outcome is Outcome.KEPT:
        signal = LearningSignal.POSITIVE
    else:
        signal = LearningSignal.NEGATIVE

    # Compute criterion deltas
    criterion_deltas: dict[str, float] = {}
    if previous_per_criterion is not None and record.raw_scores is not None:
        for name, prev_score in previous_per_criterion.items():
            # raw_scores indexes align with criteria ordering; criterion_deltas
            # uses the same keys provided by the caller
            if name in previous_per_criterion:
                criterion_deltas[name] = 0.0  # placeholder, overwritten below

        # When raw_scores length matches previous_per_criterion, zip them
        criterion_names = list(previous_per_criterion.keys())
        for i, name in enumerate(criterion_names):
            if i < len(record.raw_scores):
                criterion_deltas[name] = record.raw_scores[i] - previous_per_criterion[name]

    # Build deterministic observation text
    parts: list[str] = []
    parts.append(f"Hypothesis: {record.hypothesis}")
    parts.append(f"Outcome: {record.outcome.value}")
    parts.append(f"Score delta: {score_delta:+.4f} ({record.baseline_score:.4f} -> {record.score:.4f})")

    if criterion_deltas:
        sorted_criteria = sorted(criterion_deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        top = sorted_criteria[:3]
        criterion_parts = [f"{name}: {delta:+.4f}" for name, delta in top]
        parts.append(f"Top criterion changes: {', '.join(criterion_parts)}")

    observation = " | ".join(parts)

    return Learning(
        observation=observation,
        signal=signal,
        source_condition=source_condition,
        source_target=source_target,
        source_experiment_ids=[int(record.id)] if record.id.isdigit() else [],
        score_delta=score_delta,
        criterion_deltas=criterion_deltas,
        confidence=1.0,
        tags=list(record.tags),
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _learning_to_dict(learning: Learning) -> dict[str, object]:
    """Convert a Learning to a JSON-serializable dict."""
    d = asdict(learning)
    d["signal"] = learning.signal.value
    d["created_at"] = learning.created_at.isoformat()
    return d


def _dict_to_learning(d: dict[str, object]) -> Learning:
    """Reconstruct a Learning from a deserialized dict."""
    # Handle missing created_at for backward compatibility
    raw_created_at = d.get("created_at")
    if raw_created_at is not None:
        created_at = datetime.fromisoformat(str(raw_created_at))
    else:
        created_at = datetime.now(UTC)

    return Learning(
        observation=str(d["observation"]),
        signal=LearningSignal(d["signal"]),
        source_condition=str(d["source_condition"]),
        source_target=str(d["source_target"]),
        source_experiment_ids=list(d["source_experiment_ids"]),  # type: ignore[arg-type]
        score_delta=float(d["score_delta"]),  # type: ignore[arg-type]
        criterion_deltas={str(k): float(v) for k, v in d["criterion_deltas"].items()},  # type: ignore[union-attr]
        confidence=float(d["confidence"]),  # type: ignore[arg-type]
        tags=[str(t) for t in d["tags"]],  # type: ignore[union-attr]
        created_at=created_at,
        project_id=str(d.get("project_id", "")),
    )


# ---------------------------------------------------------------------------
# LearningPool
# ---------------------------------------------------------------------------


class LearningPool:
    """In-memory pool of cross-experiment learnings with scope-based retrieval."""

    def __init__(self, decay_rate: float = 0.05) -> None:
        self._learnings: list[Learning] = []
        self._decay_rate = decay_rate

    def add(self, learning: Learning) -> None:
        """Add a learning to the pool."""
        self._learnings.append(learning)

    def _effective_score(self, learning: Learning) -> float:
        """Compute decay-adjusted effective score for ranking."""
        now = datetime.now(UTC)
        age_days = (now - learning.created_at).total_seconds() / 86400.0
        return abs(learning.score_delta) * math.exp(-self._decay_rate * age_days)

    def _decay_confidence(self, learning: Learning) -> Learning:
        """Return a new Learning with decay-adjusted confidence."""
        now = datetime.now(UTC)
        age_days = (now - learning.created_at).total_seconds() / 86400.0
        decayed = learning.confidence * math.exp(-self._decay_rate * age_days)
        return replace(learning, confidence=decayed)

    def retrieve(
        self,
        scope: LearningScope,
        k: int = 5,
        exclude_condition: str | None = None,
        signal: LearningSignal | None = None,
        source_condition: str | None = None,
        source_target: str | None = None,
        project_id: str | None = None,
    ) -> list[Learning]:
        """Retrieve top-K learnings by decay-adjusted |score_delta| descending.

        Filtering:
            - exclude_condition: omit learnings from this condition
            - signal: keep only POSITIVE or NEGATIVE
            - source_condition: keep only learnings from this condition
            - source_target: keep only learnings from this target
            - project_id: keep only learnings from this project
        """
        candidates = self._learnings

        if exclude_condition is not None:
            candidates = [l for l in candidates if l.source_condition != exclude_condition]

        if signal is not None:
            candidates = [l for l in candidates if l.signal is signal]

        if source_condition is not None:
            candidates = [l for l in candidates if l.source_condition == source_condition]

        if source_target is not None:
            candidates = [l for l in candidates if l.source_target == source_target]

        if project_id is not None:
            candidates = [l for l in candidates if l.project_id == project_id]

        # Scope filtering: narrow by scope semantics
        # CONDITION scope requires source_condition filter (caller must provide)
        # TARGET scope requires source_target filter (caller must provide)
        # PROJECT and GLOBAL return everything matching other filters

        # Sort by decay-adjusted |score_delta| descending
        candidates = sorted(candidates, key=self._effective_score, reverse=True)

        # Return with decayed confidence values
        return [self._decay_confidence(l) for l in candidates[:k]]

    def summarize(
        self,
        scope: LearningScope,
        k: int = 5,
        exclude_condition: str | None = None,
        **kwargs: object,
    ) -> str:
        """Format retrieved learnings as a '## Cross-Condition Insights' text block."""
        learnings = self.retrieve(
            scope=scope,
            k=k,
            exclude_condition=exclude_condition,
            signal=kwargs.get("signal"),  # type: ignore[arg-type]
            source_condition=kwargs.get("source_condition"),  # type: ignore[arg-type]
            source_target=kwargs.get("source_target"),  # type: ignore[arg-type]
        )

        if not learnings:
            return ""

        lines: list[str] = ["## Cross-Condition Insights", ""]

        for i, learning in enumerate(learnings, 1):
            signal_marker = "+" if learning.signal is LearningSignal.POSITIVE else "-"
            lines.append(
                f"{i}. [{signal_marker}] (from {learning.source_condition}/{learning.source_target}, "
                f"delta={learning.score_delta:+.4f}) {learning.observation}"
            )

        lines.append("")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        """Total number of learnings in the pool."""
        return len(self._learnings)


# ---------------------------------------------------------------------------
# PersistentLearningPool
# ---------------------------------------------------------------------------


class PersistentLearningPool(LearningPool):
    """File-backed learning pool that persists across sessions.

    Backs to ``{path}/learnings-pool.jsonl``. Loads existing learnings on
    init and appends on each ``add`` call. Uses filelock for concurrent safety.
    """

    def __init__(self, path: Path, decay_rate: float = 0.05) -> None:
        super().__init__(decay_rate=decay_rate)
        self._dir = path
        self._file = path / "learnings-pool.jsonl"
        self._lock = FileLock(str(self._file) + ".lock")

        # Load existing learnings
        if self._file.exists():
            with self._lock:
                with self._file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        self._learnings.append(_dict_to_learning(d))

    def add(self, learning: Learning) -> None:
        """Add to memory and append to disk atomically."""
        super().add(learning)

        self._dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(_learning_to_dict(learning))

        with self._lock:
            with self._file.open("a") as f:
                f.write(serialized + "\n")


# ---------------------------------------------------------------------------
# Global (cross-project) pool
# ---------------------------------------------------------------------------


def get_global_pool_path() -> Path:
    """Return the path to the global cross-project learnings file."""
    return Path.home() / ".anneal" / "global-learnings.jsonl"


class GlobalLearningPool(PersistentLearningPool):
    """Cross-project learning pool stored at ``~/.anneal/global-learnings.jsonl``."""

    def __init__(self, decay_rate: float = 0.05) -> None:
        pool_path = get_global_pool_path()
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path=pool_path.parent, decay_rate=decay_rate)
        # Override the file path to use the global-specific filename
        self._file = pool_path
        self._lock = FileLock(str(self._file) + ".lock")
        # Reload from the correct file
        self._learnings.clear()
        if self._file.exists():
            with self._lock:
                with self._file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        self._learnings.append(_dict_to_learning(d))
