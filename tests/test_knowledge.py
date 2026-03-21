"""Tests for anneal.engine.knowledge — KnowledgeStore."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from anneal.engine.knowledge import KnowledgeStore, _variance
from anneal.engine.types import ConsolidationRecord, ExperimentRecord, Outcome


def _make_record(
    *,
    idx: int = 0,
    outcome: Outcome = Outcome.KEPT,
    score: float = 0.8,
    baseline_score: float = 0.75,
    hypothesis: str = "improve prompt phrasing",
    tags: list[str] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=f"exp-{idx:04d}",
        target_id="target-1",
        git_sha=f"abc{idx:04d}",
        pre_experiment_sha=f"pre{idx:04d}",
        timestamp=datetime(2026, 1, 1, 0, idx % 60),
        hypothesis=hypothesis,
        hypothesis_source="agent",
        mutation_diff_summary=f"diff {idx}",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=baseline_score,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=1.5,
        tags=tags or ["prompt"],
        learnings=f"learning {idx}",
        cost_usd=0.01,
        bootstrap_seed=42,
    )


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    rec = _make_record(idx=1, score=0.85)
    store.append_record(rec)

    loaded = store.load_records()
    assert len(loaded) == 1
    r = loaded[0]
    assert r.id == "exp-0001"
    assert r.target_id == "target-1"
    assert r.score == 0.85
    assert r.baseline_score == 0.75
    assert r.outcome == Outcome.KEPT
    assert r.hypothesis == "improve prompt phrasing"
    assert r.tags == ["prompt"]


def test_record_count(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    assert store.record_count() == 0

    for i in range(7):
        store.append_record(_make_record(idx=i))

    assert store.record_count() == 7


def test_load_records_limit(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(10):
        store.append_record(_make_record(idx=i))

    last3 = store.load_records(limit=3)
    assert len(last3) == 3
    assert [r.id for r in last3] == ["exp-0007", "exp-0008", "exp-0009"]


def test_validate_and_repair_truncates_corrupt_tail(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    store.append_record(_make_record(idx=0))
    store.append_record(_make_record(idx=1))

    # Append invalid JSON directly
    experiments = tmp_path / "knowledge" / "experiments.jsonl"
    with open(experiments, "a") as f:
        f.write("{broken json\n")

    valid_count = store.validate_and_repair()
    assert valid_count == 2
    assert store.record_count() == 2


def test_should_consolidate_false_before_50(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(49):
        store.append_record(_make_record(idx=i))

    assert store.should_consolidate() is False


def test_should_consolidate_true_at_50(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(50):
        store.append_record(_make_record(idx=i))

    assert store.should_consolidate() is True


def test_consolidate_produces_valid_record(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(50):
        outcome = Outcome.KEPT if i % 3 != 0 else Outcome.DISCARDED
        store.append_record(
            _make_record(idx=i, outcome=outcome, score=0.75 + i * 0.001)
        )

    cr = store.consolidate()
    assert isinstance(cr, ConsolidationRecord)
    assert cr.experiment_range == (0, 50)
    assert cr.total_experiments == 50
    assert cr.kept_count + cr.discarded_count == 50
    assert cr.score_start == 0.75
    assert cr.score_end == 0.75 + 49 * 0.001

    # Persisted to learnings-structured.jsonl
    consolidations = store.load_consolidations()
    assert len(consolidations) == 1
    assert consolidations[0].experiment_range == (0, 50)


def test_regenerate_learnings_creates_file(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(50):
        store.append_record(_make_record(idx=i, score=0.75 + i * 0.001))

    store.consolidate()
    store.regenerate_learnings()

    learnings_file = tmp_path / "knowledge" / "learnings.md"
    assert learnings_file.exists()
    content = learnings_file.read_text()
    assert "# Learnings" in content
    assert "Experiments 0-50" in content


def test_get_context_cold_start(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    # Zero records
    ctx = store.get_context()
    assert "first run" in ctx.lower()

    # <10 records: exploration prompt
    for i in range(5):
        store.append_record(_make_record(idx=i))

    ctx = store.get_context()
    assert "Early exploration" in ctx
    assert "5/10" in ctx


def test_get_context_full(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(15):
        store.append_record(_make_record(idx=i, score=0.75 + i * 0.01))
        store.update_index(_make_record(idx=i, score=0.75 + i * 0.01))

    ctx = store.get_context(current_hypothesis="improve prompt phrasing")
    assert "Recent Experiments" in ctx


def test_raw_records_never_truncated(tmp_path: Path) -> None:
    """Regression: consolidation must not remove raw records from experiments.jsonl."""
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(60):
        store.append_record(_make_record(idx=i))

    assert store.record_count() == 60
    store.consolidate()
    assert store.record_count() == 60

    all_records = store.load_records()
    assert len(all_records) == 60
    assert all_records[0].id == "exp-0000"
    assert all_records[59].id == "exp-0059"


def test_consolidation_does_not_summarize_away_raw_records(tmp_path: Path) -> None:
    """Regression: consolidation appends to learnings-structured.jsonl
    without modifying experiments.jsonl."""
    store = KnowledgeStore(tmp_path / "knowledge")
    experiments_file = tmp_path / "knowledge" / "experiments.jsonl"

    for i in range(50):
        store.append_record(_make_record(idx=i))

    pre_consolidation_content = experiments_file.read_text()
    store.consolidate()
    post_consolidation_content = experiments_file.read_text()

    assert pre_consolidation_content == post_consolidation_content

    consolidations_file = tmp_path / "knowledge" / "learnings-structured.jsonl"
    assert consolidations_file.exists()
    assert len(consolidations_file.read_text().splitlines()) == 1


# ---------------------------------------------------------------------------
# Variance epsilon (Step 1.5)
# ---------------------------------------------------------------------------


def test_variance_near_zero_clamped() -> None:
    """Constant values produce exactly 0.0, not a floating-point residual."""
    assert _variance([0.8, 0.8, 0.8]) == 0.0
    assert _variance([1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0


def test_variance_nonzero_preserved() -> None:
    """Non-constant values produce correct population variance."""
    result = _variance([0.7, 0.8, 0.9])
    # Population variance: ((−0.1)² + 0² + 0.1²) / 3 = 0.02/3 ≈ 0.00667
    expected = sum((x - 0.8) ** 2 for x in [0.7, 0.8, 0.9]) / 3
    assert abs(result - expected) < 1e-10
    assert result > 0


# ---------------------------------------------------------------------------
# Concurrent consolidation safety (Step 1.3)
# ---------------------------------------------------------------------------


def test_concurrent_consolidation_safe(tmp_path: Path) -> None:
    """Two threads calling consolidate_if_due produce at most one consolidation."""
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(55):
        store.append_record(_make_record(idx=i, score=0.75 + i * 0.001))

    assert store.should_consolidate() is True

    results: list[ConsolidationRecord | None] = [None, None]

    def _consolidate(idx: int) -> None:
        results[idx] = store.consolidate_if_due()

    t1 = threading.Thread(target=_consolidate, args=(0,))
    t2 = threading.Thread(target=_consolidate, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one thread produced a consolidation record
    non_none = [r for r in results if r is not None]
    assert len(non_none) == 1
    assert isinstance(non_none[0], ConsolidationRecord)


# ---------------------------------------------------------------------------
# Criterion name tracking through consolidation (Step 2.2)
# ---------------------------------------------------------------------------


def _make_record_with_criterion_names(
    *,
    idx: int = 0,
    criterion_names: list[str],
    raw_scores: list[float],
    outcome: Outcome = Outcome.KEPT,
    score: float = 0.8,
) -> ExperimentRecord:
    rec = _make_record(idx=idx, outcome=outcome, score=score)
    # ExperimentRecord is a dataclass — we can override fields directly
    rec.criterion_names = criterion_names
    rec.raw_scores = raw_scores
    return rec


def test_consolidation_uses_criterion_names_as_keys(tmp_path: Path) -> None:
    """consolidate() uses criterion_names as criterion_variances keys when present."""
    store = KnowledgeStore(tmp_path / "knowledge")
    names = ["clarity", "accuracy"]
    # Vary raw_scores across records so variance is non-zero
    for i in range(50):
        rec = _make_record_with_criterion_names(
            idx=i,
            criterion_names=names,
            raw_scores=[float(i % 2), float((i + 1) % 2)],
            score=0.75 + i * 0.001,
        )
        store.append_record(rec)

    cr = store.consolidate()
    assert "clarity" in cr.criterion_variances
    assert "accuracy" in cr.criterion_variances
    assert "criterion_0" not in cr.criterion_variances
    assert "criterion_1" not in cr.criterion_variances


def test_consolidation_falls_back_to_positional_names_without_criterion_names(
    tmp_path: Path,
) -> None:
    """consolidate() uses criterion_0/criterion_1 keys when criterion_names is absent."""
    store = KnowledgeStore(tmp_path / "knowledge")
    for i in range(50):
        rec = _make_record(idx=i, score=0.75 + i * 0.001)
        rec.raw_scores = [float(i % 2), float((i + 1) % 2)]
        store.append_record(rec)

    cr = store.consolidate()
    assert "criterion_0" in cr.criterion_variances
    assert "criterion_1" in cr.criterion_variances
    assert "clarity" not in cr.criterion_variances


def test_consolidation_criterion_names_variance_values_correct(tmp_path: Path) -> None:
    """Per-criterion variances computed from raw_scores match expected values."""
    store = KnowledgeStore(tmp_path / "knowledge")
    names = ["fluency", "tone"]
    # All records: fluency=1.0 (zero variance), tone alternates 0/1
    for i in range(50):
        rec = _make_record_with_criterion_names(
            idx=i,
            criterion_names=names,
            raw_scores=[1.0, float(i % 2)],
            score=0.75 + i * 0.001,
        )
        store.append_record(rec)

    cr = store.consolidate()
    assert cr.criterion_variances["fluency"] == 0.0
    assert cr.criterion_variances["tone"] > 0.0
