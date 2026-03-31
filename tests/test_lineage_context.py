from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from anneal.engine.context import _format_lineage
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.types import ExperimentRecord, Outcome


def _make_record(
    idx: int,
    git_sha: str,
    pre_sha: str,
    score: float,
    outcome: Outcome = Outcome.KEPT,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=f"rec-{idx:04d}",
        target_id="test",
        git_sha=git_sha,
        pre_experiment_sha=pre_sha,
        timestamp=datetime.now(tz=timezone.utc),
        hypothesis=f"hypothesis for experiment {idx}",
        hypothesis_source="agent",
        mutation_diff_summary="",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=0.0,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=10.0,
        tags=[],
        learnings="",
        cost_usd=0.1,
        bootstrap_seed=42,
    )


def _write_records(path: Path, records: list[ExperimentRecord]) -> None:
    experiments_file = path / "experiments.jsonl"
    with open(experiments_file, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# get_lineage tests
# ---------------------------------------------------------------------------


def test_get_lineage_traces_kept_chain(tmp_path: Path) -> None:
    records = [
        _make_record(i, f"sha{i}", f"sha{i - 1}" if i > 0 else "sha-init", score=float(i))
        for i in range(5)
    ]
    _write_records(tmp_path, records)
    store = KnowledgeStore(tmp_path)

    lineage = store.get_lineage("sha4", depth=5)

    assert len(lineage) == 5
    assert [r.git_sha for r in lineage] == ["sha0", "sha1", "sha2", "sha3", "sha4"]


def test_get_lineage_stops_at_depth(tmp_path: Path) -> None:
    records = [
        _make_record(i, f"sha{i}", f"sha{i - 1}" if i > 0 else "sha-init", score=float(i))
        for i in range(10)
    ]
    _write_records(tmp_path, records)
    store = KnowledgeStore(tmp_path)

    lineage = store.get_lineage("sha9", depth=5)

    assert len(lineage) == 5
    assert [r.git_sha for r in lineage] == ["sha5", "sha6", "sha7", "sha8", "sha9"]


def test_get_lineage_skips_discarded(tmp_path: Path) -> None:
    records = [
        _make_record(0, "sha0", "sha-init", score=0.1, outcome=Outcome.KEPT),
        _make_record(1, "sha1", "sha0", score=0.2, outcome=Outcome.DISCARDED),
        _make_record(2, "sha2", "sha0", score=0.3, outcome=Outcome.KEPT),
        _make_record(3, "sha3", "sha2", score=0.4, outcome=Outcome.DISCARDED),
        _make_record(4, "sha4", "sha2", score=0.5, outcome=Outcome.KEPT),
    ]
    _write_records(tmp_path, records)
    store = KnowledgeStore(tmp_path)

    lineage = store.get_lineage("sha4", depth=5)

    assert [r.git_sha for r in lineage] == ["sha0", "sha2", "sha4"]


def test_get_lineage_empty_when_no_kept(tmp_path: Path) -> None:
    records = [
        _make_record(0, "sha0", "sha-init", score=0.1, outcome=Outcome.DISCARDED),
        _make_record(1, "sha1", "sha0", score=0.2, outcome=Outcome.DISCARDED),
    ]
    _write_records(tmp_path, records)
    store = KnowledgeStore(tmp_path)

    lineage = store.get_lineage("sha1", depth=5)

    assert lineage == []


def test_get_lineage_empty_when_sha_not_found(tmp_path: Path) -> None:
    records = [
        _make_record(0, "sha0", "sha-init", score=0.1, outcome=Outcome.KEPT),
    ]
    _write_records(tmp_path, records)
    store = KnowledgeStore(tmp_path)

    lineage = store.get_lineage("nonexistent-sha", depth=5)

    assert lineage == []


# ---------------------------------------------------------------------------
# _format_lineage tests
# ---------------------------------------------------------------------------


def test_format_lineage_includes_header() -> None:
    records = [
        _make_record(i, f"sha{i}", f"sha{i - 1}", score=float(i) * 0.1)
        for i in range(3)
    ]

    result = _format_lineage(records)

    assert result.startswith("# Lineage")


def test_format_lineage_includes_scores_and_hypotheses() -> None:
    records = [
        _make_record(0, "sha0", "sha-init", score=0.1234),
        _make_record(1, "sha1", "sha0", score=0.5678),
    ]

    result = _format_lineage(records)

    assert "0.1234" in result
    assert "0.5678" in result
    assert "hypothesis for experiment 0" in result
    assert "hypothesis for experiment 1" in result


def test_format_lineage_empty_returns_empty_string() -> None:
    result = _format_lineage([])

    assert result == ""
