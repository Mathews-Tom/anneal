"""Centralized experiment display formatting.

Single source of truth for how experiment records, scores, outcomes,
costs, and durations are rendered across all consumers:

  - CLI progress (Rich markup)
  - Web dashboard (JSON/SSE)
  - Comparison tables (Rich tables)
  - Knowledge store (plain text for prompt injection)
  - Benchmark suite runner (Rich + JSONL file polling)

All formatters accept both ``ExperimentRecord`` (Pydantic model) and
raw ``dict`` (from JSONL) via the ``_get()`` helper.
"""

from __future__ import annotations

import json
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from rich.console import Console

from anneal.engine.types import Outcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output modes
# ---------------------------------------------------------------------------


class OutputMode(Enum):
    """Target format for display output."""

    RICH = "rich"
    PLAIN = "plain"
    JSON = "json"


# ---------------------------------------------------------------------------
# Outcome styling — single source of truth
# ---------------------------------------------------------------------------


class OutcomeStyle(NamedTuple):
    """Presentation attributes for an experiment outcome."""

    rich_color: str
    abbreviation: str
    hex_color: str


OUTCOME_STYLES: dict[Outcome, OutcomeStyle] = {
    Outcome.KEPT: OutcomeStyle("green", "KEPT", "#3fb950"),
    Outcome.DISCARDED: OutcomeStyle("red", "DISC", "#f85149"),
    Outcome.BLOCKED: OutcomeStyle("yellow", "BLKD", "#d29922"),
    Outcome.KILLED: OutcomeStyle("red", "KILL", "#f85149"),
    Outcome.CRASHED: OutcomeStyle("red", "CRASH", "#f85149"),
}

# Map string values to styles for dict-based consumers.
_OUTCOME_STYLES_BY_VALUE: dict[str, OutcomeStyle] = {
    o.value: s for o, s in OUTCOME_STYLES.items()
}

_FALLBACK_STYLE = OutcomeStyle("dim", "????", "#484f58")


# ---------------------------------------------------------------------------
# Record accessor — dual ExperimentRecord / dict support
# ---------------------------------------------------------------------------


def _get(record: Any, field: str, default: Any = None) -> Any:
    """Extract a field from an ExperimentRecord or a raw dict."""
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def _outcome_str(record: Any) -> str:
    """Extract the outcome as a plain string."""
    raw = _get(record, "outcome", "")
    if isinstance(raw, Outcome):
        return raw.value
    return str(raw).upper()


def _style_for(outcome_value: str) -> OutcomeStyle:
    return _OUTCOME_STYLES_BY_VALUE.get(outcome_value, _FALLBACK_STYLE)


# ---------------------------------------------------------------------------
# Atomic formatters
# ---------------------------------------------------------------------------


def format_score(value: float) -> str:
    """Format a score value to 4 decimal places."""
    return f"{value:.4f}"


def format_delta(
    current: float,
    baseline: float,
    *,
    mode: OutputMode = OutputMode.PLAIN,
) -> str:
    """Format a score delta with sign prefix and optional Rich coloring."""
    delta = current - baseline
    if delta == 0:
        if mode is OutputMode.RICH:
            return "[dim]--[/dim]"
        return "--"
    sign = "+" if delta > 0 else ""
    formatted = f"{sign}{delta:.4f}"
    if mode is OutputMode.RICH:
        color = "green" if delta > 0 else "red"
        return f"[{color}]{formatted}[/{color}]"
    return formatted


def format_cost(cost_usd: float) -> str:
    """Format a USD cost to 4 decimal places."""
    return f"${cost_usd:.4f}"


def format_outcome(
    outcome: Outcome | str,
    *,
    mode: OutputMode = OutputMode.PLAIN,
    abbreviate: bool = True,
) -> str:
    """Format an outcome with optional Rich coloring and abbreviation."""
    value = outcome.value if isinstance(outcome, Outcome) else str(outcome).upper()
    style = _style_for(value)
    label = style.abbreviation if abbreviate else value
    if mode is OutputMode.RICH:
        return f"[{style.rich_color}]{label}[/{style.rich_color}]"
    return label


def format_duration(seconds: float) -> str:
    """Format a duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hr"


# ---------------------------------------------------------------------------
# Composite formatters
# ---------------------------------------------------------------------------


def format_experiment_line(
    record: Any,
    *,
    index: int | None = None,
    max_experiments: int | None = None,
    cumulative_cost: float | None = None,
    budget: float | None = None,
    mode: OutputMode = OutputMode.RICH,
) -> str:
    """Format a single experiment as a one-line progress string.

    Works with both ``ExperimentRecord`` and raw ``dict`` records.
    """
    score = float(_get(record, "score", 0.0))
    baseline = float(_get(record, "baseline_score", 0.0))

    # Experiment index label
    if index is not None:
        if max_experiments is not None and max_experiments > 0:
            exp_label = f"[exp {index}/{max_experiments}]"
        else:
            exp_label = f"[exp {index}]"
    else:
        exp_label = ""

    delta_str = format_delta(score, baseline, mode=mode)
    score_str = (
        f"score: {format_score(baseline)} -> {format_score(score)} ({delta_str})"
    )

    # Cost segment
    cost_str = ""
    if cumulative_cost is not None:
        cost_str = f" | cost: {format_cost(cumulative_cost)}"
        if budget is not None:
            cost_str += f" / {format_cost(budget)} budget"

    parts = [p for p in [exp_label, score_str + cost_str] if p]
    return "  ".join(parts)


def format_progress_status(
    record: Any,
    *,
    best_score: float,
    cumulative_cost: float,
    mode: OutputMode = OutputMode.RICH,
) -> str:
    """Format a progress bar status string after an experiment completes."""
    outcome_value = _outcome_str(record)
    outcome_tag = format_outcome(outcome_value, mode=mode)
    score = float(_get(record, "score", 0.0))
    failure = _get(record, "failure_mode") or ""
    if failure:
        failure = f"  {failure[:60]}"

    return (
        f"{outcome_tag} {format_score(score)}  "
        f"best={format_score(best_score)}  "
        f"{format_cost(cumulative_cost)}{failure}"
    )


def build_run_summary(
    records: list[Any],
    *,
    baseline_score: float | None = None,
    mode: OutputMode = OutputMode.RICH,
) -> str | dict[str, Any]:
    """Build an end-of-run summary from a list of experiment records.

    Returns a formatted string (RICH/PLAIN) or a dict (JSON).
    """
    n = len(records)
    kept = sum(1 for r in records if _outcome_str(r) == "KEPT")
    total_cost = sum(float(_get(r, "cost_usd", 0.0)) for r in records)
    total_duration = sum(float(_get(r, "duration_seconds", 0.0)) for r in records)

    scores = [float(_get(r, "score", 0.0)) for r in records]
    best = max(scores) if scores else 0.0

    if mode is OutputMode.JSON:
        summary: dict[str, Any] = {
            "experiment_count": n,
            "kept_count": kept,
            "kept_rate": kept / max(n, 1),
            "best_score": best,
            "total_cost_usd": total_cost,
            "total_duration_seconds": total_duration,
        }
        if baseline_score is not None:
            summary["baseline_score"] = baseline_score
        return summary

    lines = [
        f"Experiments:  {n}",
        f"Kept:         {kept}",
        f"Best score:   {format_score(best)}",
        f"Total cost:   {format_cost(total_cost)}",
        f"Total time:   {format_duration(total_duration)}",
    ]
    if baseline_score is not None:
        delta = format_delta(best, baseline_score, mode=mode)
        lines.insert(3, f"Improvement:  {delta}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live progress monitor — tails experiments.jsonl during subprocess runs
# ---------------------------------------------------------------------------


class LiveProgressMonitor:
    """Polls an experiments.jsonl file and shows a live Rich progress bar.

    Designed for the benchmark suite runner: start before ``subprocess.run``,
    stop after it returns. Shows a spinner + progress bar between experiments
    so the user sees the subprocess is alive. Experiment results print above
    the progress bar as they arrive.

    The monitor uses a daemon thread that tails the JSONL file by byte
    offset, similar to the dashboard's ``FilePollingBus``.
    """

    def __init__(
        self,
        jsonl_path: Path,
        *,
        run_label: str = "",
        poll_interval: float = 2.0,
        console: Console | None = None,
        max_experiments: int | None = None,
        budget: float | None = None,
    ) -> None:
        self._path = jsonl_path
        self._run_label = run_label
        self._poll_interval = poll_interval
        self._console = console or Console()
        self._max_experiments = max_experiments
        self._budget = budget
        self._records: list[dict[str, Any]] = []
        self._offset: int = 0
        self._cumulative_cost: float = 0.0
        self._best_score: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress: Any = None
        self._task_id: Any = None

    def start(self) -> None:
        """Begin polling in a background daemon thread with a live progress bar."""
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        # Capture initial file size to skip pre-existing records.
        if self._path.exists():
            self._offset = self._path.stat().st_size

        self._progress = Progress(
            SpinnerColumn("dots"),
            TextColumn(f"[bold cyan]{self._run_label}[/]" if self._run_label else ""),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._task_id = self._progress.add_task(
            self._run_label,
            total=self._max_experiments or None,
            status="waiting for first experiment...",
        )
        self._progress.start()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        """Stop polling, read any remaining lines, return all collected records."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Final drain — subprocess may have written after last poll.
        self._read_new_lines()
        if self._progress is not None:
            self._progress.stop()
        return list(self._records)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._read_new_lines()
            self._stop_event.wait(timeout=self._poll_interval)

    def _read_new_lines(self) -> None:
        if not self._path.exists():
            return
        try:
            size = self._path.stat().st_size
            if size <= self._offset:
                return
            with self._path.open("r", encoding="utf-8") as fh:
                fh.seek(self._offset)
                new_data = fh.read()
                self._offset = fh.tell()
        except OSError:
            return

        for line in new_data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._records.append(record)
            self._cumulative_cost += float(record.get("cost_usd", 0.0))
            score = float(record.get("score", 0.0))
            if self._best_score is None:
                self._best_score = score
            else:
                self._best_score = max(self._best_score, score)

            idx = len(self._records)
            exp_line = format_experiment_line(
                record,
                index=idx,
                max_experiments=self._max_experiments,
                cumulative_cost=self._cumulative_cost,
                budget=self._budget,
                mode=OutputMode.RICH,
            )
            status = format_progress_status(
                record,
                best_score=self._best_score,
                cumulative_cost=self._cumulative_cost,
                mode=OutputMode.RICH,
            )
            # Print experiment line above the live progress bar.
            if self._progress is not None:
                self._progress.console.print(f"  {exp_line}")
                self._progress.update(self._task_id, advance=1, status=status)
