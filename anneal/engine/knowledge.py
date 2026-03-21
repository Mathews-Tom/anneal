"""Knowledge store: experiment records, vector retrieval, and consolidation.

Manages the per-target knowledge directory containing experiments.jsonl,
hypotheses.jsonl, learnings-structured.jsonl, and learnings.md.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from filelock import FileLock

from anneal.engine.types import ConsolidationRecord, DriftEntry, ExperimentRecord, Outcome

logger = logging.getLogger(__name__)


class KnowledgeError(Exception):
    """Raised on knowledge store failures."""


def _serialize_value(obj: object) -> object:
    """Convert non-JSON-serializable types for JSON encoding."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot serialize {type(obj)}: {obj}")


def _record_to_json(record: ExperimentRecord) -> str:
    """Serialize an ExperimentRecord to a single JSON line."""
    data = asdict(record)
    return json.dumps(data, default=_serialize_value, separators=(",", ":"))


def _json_to_record(line: str) -> ExperimentRecord:
    """Deserialize a JSON line into an ExperimentRecord."""
    data = json.loads(line)
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    data["outcome"] = Outcome(data["outcome"])
    return ExperimentRecord(**data)


def _consolidation_to_json(record: ConsolidationRecord) -> str:
    """Serialize a ConsolidationRecord to a single JSON line."""
    data = asdict(record)
    return json.dumps(data, default=_serialize_value, separators=(",", ":"))


def _json_to_consolidation(line: str) -> ConsolidationRecord:
    """Deserialize a JSON line into a ConsolidationRecord."""
    data = json.loads(line)
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    data["experiment_range"] = tuple(data["experiment_range"])
    data.setdefault("criterion_variances", {})
    data.setdefault("score_variance", 0.0)
    return ConsolidationRecord(**data)


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _variance(xs: list[float]) -> float:
    """Population variance of a list of floats. Returns 0.0 for empty/single."""
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    result = sum((x - mean) ** 2 for x in xs) / len(xs)
    return 0.0 if result < 1e-15 else result


class KnowledgeStore:
    """Manages experiment records, similarity retrieval, and consolidation.

    All file paths are derived from knowledge_path, which is the per-target
    knowledge directory (e.g., targets/<target-id>/).
    """

    CONSOLIDATION_INTERVAL: int = 50
    COLD_START_THRESHOLD: int = 10

    def __init__(self, knowledge_path: Path) -> None:
        """Initialize with the target's knowledge directory path."""
        self._path = knowledge_path
        self._path.mkdir(parents=True, exist_ok=True)

        self._experiments_file = self._path / "experiments.jsonl"
        self._hypotheses_file = self._path / "hypotheses.jsonl"
        self._consolidations_file = self._path / "learnings-structured.jsonl"
        self._learnings_file = self._path / "learnings.md"
        self._lock_file = self._path / ".knowledge.lock"

    # ------------------------------------------------------------------
    # 2.10 — JSONL Corruption Recovery
    # ------------------------------------------------------------------

    def validate_and_repair(self) -> int:
        """Validate experiments.jsonl on startup. Truncate last line if invalid.

        Returns the valid record count.
        """
        if not self._experiments_file.exists():
            return 0

        lines = self._experiments_file.read_text().splitlines()
        if not lines:
            return 0

        valid_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                _json_to_record(stripped)
                valid_lines.append(stripped)
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                # Truncate from this point — only tolerate corruption at the tail
                break

        self._experiments_file.write_text(
            "\n".join(valid_lines) + ("\n" if valid_lines else "")
        )
        return len(valid_lines)

    # ------------------------------------------------------------------
    # 2.11 — JSONL Records
    # ------------------------------------------------------------------

    def append_record(self, record: ExperimentRecord) -> None:
        """Append an experiment record to experiments.jsonl. File-locked."""
        lock = FileLock(str(self._lock_file))
        with lock:
            with open(self._experiments_file, "a") as f:
                f.write(_record_to_json(record) + "\n")

    def load_records(self, limit: int | None = None) -> list[ExperimentRecord]:
        """Load records from experiments.jsonl. Optional limit reads last N."""
        if not self._experiments_file.exists():
            return []

        lines = self._experiments_file.read_text().splitlines()
        non_empty = [ln for ln in lines if ln.strip()]

        if limit is not None:
            non_empty = non_empty[-limit:]

        return [_json_to_record(ln) for ln in non_empty]

    def record_count(self) -> int:
        """Count records without loading all into memory."""
        if not self._experiments_file.exists():
            return 0

        count = 0
        with open(self._experiments_file) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    # ------------------------------------------------------------------
    # 2.12, 2.13 — Vector Index (MVP: Jaccard similarity)
    # ------------------------------------------------------------------

    def update_index(self, record: ExperimentRecord) -> None:
        """Append the hypothesis to the hypothesis index."""
        entry = json.dumps(
            {"id": record.id, "hypothesis": record.hypothesis},
            separators=(",", ":"),
        )
        lock = FileLock(str(self._lock_file))
        with lock:
            with open(self._hypotheses_file, "a") as f:
                f.write(entry + "\n")

    def retrieve_similar(
        self, query: str, k: int = 5
    ) -> list[ExperimentRecord]:
        """Return K most similar past experiments by hypothesis similarity.

        Uses Jaccard word overlap. Returns empty list when fewer than
        COLD_START_THRESHOLD records exist (noise > signal).
        """
        count = self.record_count()
        if count < self.COLD_START_THRESHOLD:
            return []

        if not self._hypotheses_file.exists():
            return []

        # Load hypothesis index
        entries: list[dict[str, str]] = []
        with open(self._hypotheses_file) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))

        if not entries:
            return []

        # Score by Jaccard similarity
        scored: list[tuple[float, str]] = []
        for entry in entries:
            sim = _jaccard_similarity(query, entry["hypothesis"])
            scored.append((sim, entry["id"]))

        # Sort descending by similarity, take top K
        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [sid for _, sid in scored[:k]]

        # Load matching records
        all_records = self.load_records()
        records_by_id = {r.id: r for r in all_records}
        return [records_by_id[rid] for rid in top_ids if rid in records_by_id]

    # ------------------------------------------------------------------
    # 2.16, 2.17, 2.18 — Consolidation
    # ------------------------------------------------------------------

    def should_consolidate(self) -> bool:
        """Return True if 50+ experiments since last consolidation."""
        total = self.record_count()
        if total < self.CONSOLIDATION_INTERVAL:
            return False

        consolidations = self.load_consolidations()
        if not consolidations:
            return total >= self.CONSOLIDATION_INTERVAL

        last = consolidations[-1]
        last_end = last.experiment_range[1]
        experiments_since = total - last_end
        return experiments_since >= self.CONSOLIDATION_INTERVAL

    def consolidate(self) -> ConsolidationRecord:
        """Extract a ConsolidationRecord from recent experiments.

        Deterministic — no LLM. Runs inside the caller's lock
        (not self-locking — the runner holds the lock).
        """
        all_records = self.load_records()
        total = len(all_records)

        consolidations = self.load_consolidations()
        if consolidations:
            start_idx = consolidations[-1].experiment_range[1]
        else:
            start_idx = 0

        window = all_records[start_idx:total]
        if not window:
            raise KnowledgeError(
                f"No experiments in consolidation window [{start_idx}, {total})"
            )

        # Counts
        kept_count = sum(1 for r in window if r.outcome == Outcome.KEPT)
        discarded_count = sum(
            1 for r in window if r.outcome == Outcome.DISCARDED
        )
        crashed_count = sum(1 for r in window if r.outcome == Outcome.CRASHED)

        # Score range
        score_start = window[0].baseline_score
        score_end = window[-1].score

        # Top improvements: KEPT records sorted by delta descending
        kept_records = [r for r in window if r.outcome == Outcome.KEPT]
        kept_with_delta = [
            (r, r.score - r.baseline_score) for r in kept_records
        ]
        kept_with_delta.sort(key=lambda x: x[1], reverse=True)
        top_improvements: list[dict[str, str | float]] = [
            {
                "hypothesis": r.hypothesis,
                "score_delta": delta,
                "git_sha": r.git_sha,
            }
            for r, delta in kept_with_delta[:5]
        ]

        # Failed approaches: DISCARDED records sorted by delta ascending (worst first)
        discarded_records = [
            r for r in window if r.outcome == Outcome.DISCARDED
        ]
        discarded_with_delta = [
            (r, r.score - r.baseline_score) for r in discarded_records
        ]
        discarded_with_delta.sort(key=lambda x: x[1])
        failed_approaches: list[dict[str, str | float]] = [
            {
                "hypothesis": r.hypothesis,
                "score_delta": delta,
            }
            for r, delta in discarded_with_delta[:5]
        ]

        # Tag frequency
        tag_counter: Counter[str] = Counter()
        for r in window:
            tag_counter.update(r.tags)

        # Score variance across the consolidation window
        scores = [r.score for r in window]
        score_variance = _variance(scores)

        # Per-criterion variance from raw_scores across experiments
        criterion_variances: dict[str, float] = {}
        records_with_raw = [r for r in window if r.raw_scores]
        if records_with_raw:
            max_len = max(len(r.raw_scores) for r in records_with_raw)  # type: ignore[arg-type]
            for i in range(max_len):
                values = [
                    r.raw_scores[i]  # type: ignore[index]
                    for r in records_with_raw
                    if r.raw_scores is not None and len(r.raw_scores) > i
                ]
                if len(values) >= 2:
                    criterion_variances[f"criterion_{i}"] = _variance(values)

        record = ConsolidationRecord(
            experiment_range=(start_idx, total),
            timestamp=datetime.now(),
            total_experiments=len(window),
            kept_count=kept_count,
            discarded_count=discarded_count,
            crashed_count=crashed_count,
            score_start=score_start,
            score_end=score_end,
            top_improvements=top_improvements,
            failed_approaches=failed_approaches,
            tags_frequency=dict(tag_counter),
            criterion_variances=criterion_variances,
            score_variance=score_variance,
        )

        drifting = [name for name, var in criterion_variances.items() if var > 0.1]
        if drifting:
            logger.warning("Evaluator drift detected in criteria: %s", drifting)

        # Append to learnings-structured.jsonl
        with open(self._consolidations_file, "a") as f:
            f.write(_consolidation_to_json(record) + "\n")

        return record

    def get_drift_report(self, variance_threshold: float = 0.1) -> list[DriftEntry]:
        """Return criteria with variance above threshold from recent consolidations.

        Analyzes the most recent consolidation record's criterion_variances
        and returns entries where variance exceeds the threshold.
        """
        consolidations = self.load_consolidations()
        if not consolidations:
            return []

        latest = consolidations[-1]
        if not latest.criterion_variances:
            return []

        # Collect raw_scores from the latest consolidation window to compute means
        all_records = self.load_records()
        start_idx, end_idx = latest.experiment_range
        window = all_records[start_idx:end_idx]
        records_with_raw = [r for r in window if r.raw_scores]

        entries: list[DriftEntry] = []
        for name, var in latest.criterion_variances.items():
            if var <= variance_threshold:
                continue
            # Extract criterion index from name
            idx = int(name.split("_")[1])
            values = [
                r.raw_scores[idx]  # type: ignore[index]
                for r in records_with_raw
                if r.raw_scores is not None and len(r.raw_scores) > idx
            ]
            mean_score = sum(values) / len(values) if values else 0.0
            entries.append(
                DriftEntry(
                    criterion_name=name,
                    variance=var,
                    mean_score=mean_score,
                    window_size=len(values),
                )
            )

        return entries

    def load_consolidations(self) -> list[ConsolidationRecord]:
        """Load all consolidation records from learnings-structured.jsonl."""
        if not self._consolidations_file.exists():
            return []

        lines = self._consolidations_file.read_text().splitlines()
        return [
            _json_to_consolidation(ln)
            for ln in lines
            if ln.strip()
        ]

    # ------------------------------------------------------------------
    # 2.17 — Narrative Learnings
    # ------------------------------------------------------------------

    def regenerate_learnings(self) -> None:
        """Regenerate learnings.md from consolidation records.

        Atomic via os.replace. Simple structured text (not LLM-generated).
        """
        consolidations = self.load_consolidations()
        if not consolidations:
            return

        lines: list[str] = ["# Learnings\n"]

        for cr in consolidations:
            range_str = f"{cr.experiment_range[0]}-{cr.experiment_range[1]}"
            lines.append(f"## Experiments {range_str}\n")
            lines.append(
                f"Total: {cr.total_experiments} | "
                f"Kept: {cr.kept_count} | "
                f"Discarded: {cr.discarded_count} | "
                f"Crashed: {cr.crashed_count}\n"
            )
            lines.append(
                f"Score: {cr.score_start:.4f} -> {cr.score_end:.4f}\n"
            )

            if cr.top_improvements:
                lines.append("### Top Improvements\n")
                for imp in cr.top_improvements:
                    lines.append(
                        f"- (+{imp['score_delta']:.4f}) {imp['hypothesis']}\n"
                    )

            if cr.failed_approaches:
                lines.append("### Failed Approaches\n")
                for fail in cr.failed_approaches:
                    lines.append(
                        f"- ({fail['score_delta']:.4f}) {fail['hypothesis']}\n"
                    )

            if cr.tags_frequency:
                tags_sorted = sorted(
                    cr.tags_frequency.items(), key=lambda x: x[1], reverse=True
                )
                tag_str = ", ".join(f"{t}({c})" for t, c in tags_sorted)
                lines.append(f"### Tags: {tag_str}\n")

            lines.append("")

        tmp_path = self._path / ".learnings.md.tmp"
        tmp_path.write_text("\n".join(lines))
        os.replace(tmp_path, self._learnings_file)

    # ------------------------------------------------------------------
    # 2.14, 2.15 — Context Assembly
    # ------------------------------------------------------------------

    def get_context(self, current_hypothesis: str | None = None) -> str:
        """Assemble knowledge context for the agent prompt.

        - Experiments 1-10: cold-start exploration prompt
        - Experiments 11+: last 5 records + top 5 similar + learnings summary
        """
        count = self.record_count()

        if count == 0:
            return (
                "No prior experiments. This is the first run. "
                "Explore broadly — try a meaningful change and observe the effect."
            )

        if count < self.COLD_START_THRESHOLD:
            recent = self.load_records()
            sections = [
                f"Early exploration phase ({count}/{self.COLD_START_THRESHOLD} before retrieval activates).\n",
                "## Recent Experiments\n",
            ]
            for r in recent:
                sections.append(self._format_record(r))
            sections.append(
                "\nContinue exploring diverse approaches. "
                "Not enough data for pattern matching yet."
            )
            return "\n".join(sections)

        # Full context: recent + similar + learnings
        sections: list[str] = []

        # Last 5 records
        recent = self.load_records(limit=5)
        sections.append("## Recent Experiments (last 5)\n")
        for r in recent:
            sections.append(self._format_record(r))

        # Top 5 similar
        if current_hypothesis:
            similar = self.retrieve_similar(current_hypothesis, k=5)
            if similar:
                # Deduplicate against recent
                recent_ids = {r.id for r in recent}
                similar = [s for s in similar if s.id not in recent_ids]
                if similar:
                    sections.append("\n## Similar Past Experiments\n")
                    for r in similar:
                        sections.append(self._format_record(r))

        # Learnings summary
        if self._learnings_file.exists():
            learnings_text = self._learnings_file.read_text().strip()
            if learnings_text:
                sections.append(f"\n## Consolidated Learnings\n\n{learnings_text}")

        return "\n".join(sections)

    @staticmethod
    def _format_record(record: ExperimentRecord) -> str:
        """Format a single ExperimentRecord for context injection."""
        delta = record.score - record.baseline_score
        sign = "+" if delta >= 0 else ""
        tags_str = ", ".join(record.tags) if record.tags else "none"
        return (
            f"- [{record.outcome.value}] {record.hypothesis}\n"
            f"  Score: {record.score:.4f} ({sign}{delta:.4f}) | "
            f"Tags: {tags_str}\n"
            f"  {record.learnings}\n"
        )
