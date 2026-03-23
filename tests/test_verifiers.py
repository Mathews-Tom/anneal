"""Tests for deterministic verification gates (E5)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anneal.engine.eval import run_verifiers
from anneal.engine.types import VerifierCommand


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    return tmp_path


class TestRunVerifiers:
    """Tests for the run_verifiers function."""

    def test_empty_verifiers_returns_empty(self, worktree: Path) -> None:
        result = asyncio.run(run_verifiers(worktree, []))
        assert result == []

    def test_passing_verifier(self, worktree: Path) -> None:
        v = VerifierCommand(name="true_check", run_command="true")
        result = asyncio.run(run_verifiers(worktree, [v]))
        assert len(result) == 1
        assert result[0] == ("true_check", True, "")

    def test_failing_verifier(self, worktree: Path) -> None:
        v = VerifierCommand(name="false_check", run_command="false")
        result = asyncio.run(run_verifiers(worktree, [v]))
        assert len(result) == 1
        name, passed, stderr = result[0]
        assert name == "false_check"
        assert passed is False

    def test_fail_fast_stops_on_first_failure(self, worktree: Path) -> None:
        v1 = VerifierCommand(name="pass", run_command="true")
        v2 = VerifierCommand(name="fail", run_command="false")
        v3 = VerifierCommand(name="never_reached", run_command="true")
        result = asyncio.run(run_verifiers(worktree, [v1, v2, v3]))
        assert len(result) == 2
        assert result[0][0] == "pass"
        assert result[0][1] is True
        assert result[1][0] == "fail"
        assert result[1][1] is False

    def test_timeout_kills_process(self, worktree: Path) -> None:
        v = VerifierCommand(name="slow", run_command="sleep 60", timeout_seconds=1)
        result = asyncio.run(run_verifiers(worktree, [v]))
        assert len(result) == 1
        name, passed, stderr = result[0]
        assert name == "slow"
        assert passed is False
        assert "timed out" in stderr.lower()

    def test_stderr_captured_on_failure(self, worktree: Path) -> None:
        v = VerifierCommand(
            name="stderr_check",
            run_command="echo 'error detail' >&2 && exit 1",
        )
        result = asyncio.run(run_verifiers(worktree, [v]))
        name, passed, stderr = result[0]
        assert passed is False
        assert "error detail" in stderr

    def test_multiple_passing_verifiers(self, worktree: Path) -> None:
        verifiers = [
            VerifierCommand(name=f"check_{i}", run_command="true")
            for i in range(3)
        ]
        result = asyncio.run(run_verifiers(worktree, verifiers))
        assert len(result) == 3
        assert all(passed for _, passed, _ in result)

    def test_verifier_runs_in_worktree_cwd(self, worktree: Path) -> None:
        marker = worktree / "marker.txt"
        marker.write_text("exists")
        v = VerifierCommand(name="cwd_check", run_command="test -f marker.txt")
        result = asyncio.run(run_verifiers(worktree, [v]))
        assert result[0][1] is True


class TestVerifierWarning:
    """Tests for verifier failure warning injection into context."""

    def test_no_warning_without_failures(self) -> None:
        from anneal.engine.context import _build_verifier_warning
        from anneal.engine.types import ExperimentRecord, Outcome
        from datetime import datetime, timezone

        records = [
            ExperimentRecord(
                id=str(i), target_id="t", git_sha="abc", pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc), hypothesis="h",
                hypothesis_source="agent", mutation_diff_summary="", score=1.0,
                score_ci_lower=None, score_ci_upper=None, raw_scores=None,
                baseline_score=0.5, outcome=Outcome.KEPT, failure_mode=None,
                duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
                bootstrap_seed=0,
            )
            for i in range(10)
        ]
        assert _build_verifier_warning(records) == ""

    def test_warning_when_verifier_blocks_above_threshold(self) -> None:
        from anneal.engine.context import _build_verifier_warning
        from anneal.engine.types import ExperimentRecord, Outcome
        from datetime import datetime, timezone

        records = []
        for i in range(10):
            fm = "verifier:typecheck" if i < 7 else None
            outcome = Outcome.BLOCKED if i < 7 else Outcome.KEPT
            records.append(ExperimentRecord(
                id=str(i), target_id="t", git_sha="abc", pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc), hypothesis="h",
                hypothesis_source="agent", mutation_diff_summary="", score=1.0,
                score_ci_lower=None, score_ci_upper=None, raw_scores=None,
                baseline_score=0.5, outcome=outcome, failure_mode=fm,
                duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
                bootstrap_seed=0,
            ))
        warning = _build_verifier_warning(records)
        assert "typecheck" in warning
        assert "7/10" in warning

    def test_no_warning_below_threshold(self) -> None:
        from anneal.engine.context import _build_verifier_warning
        from anneal.engine.types import ExperimentRecord, Outcome
        from datetime import datetime, timezone

        records = []
        for i in range(10):
            fm = "verifier:lint" if i < 5 else None
            outcome = Outcome.BLOCKED if i < 5 else Outcome.KEPT
            records.append(ExperimentRecord(
                id=str(i), target_id="t", git_sha="abc", pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc), hypothesis="h",
                hypothesis_source="agent", mutation_diff_summary="", score=1.0,
                score_ci_lower=None, score_ci_upper=None, raw_scores=None,
                baseline_score=0.5, outcome=outcome, failure_mode=fm,
                duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
                bootstrap_seed=0,
            ))
        # 50% is below 60% threshold
        assert _build_verifier_warning(records) == ""


class TestVerifierConsolidation:
    """Tests for verifier block rate tracking in consolidation."""

    def test_consolidation_tracks_verifier_rates(self, tmp_path: Path) -> None:
        from anneal.engine.knowledge import KnowledgeStore
        from anneal.engine.types import ExperimentRecord, Outcome
        from datetime import datetime, timezone

        ks = KnowledgeStore(tmp_path)
        for i in range(50):
            fm = "verifier:typecheck" if i < 30 else None
            outcome = Outcome.BLOCKED if i < 30 else Outcome.KEPT
            record = ExperimentRecord(
                id=str(i), target_id="t", git_sha="abc", pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc), hypothesis=f"h{i}",
                hypothesis_source="agent", mutation_diff_summary="", score=float(i),
                score_ci_lower=None, score_ci_upper=None, raw_scores=None,
                baseline_score=0.0, outcome=outcome, failure_mode=fm,
                duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
                bootstrap_seed=0,
            )
            ks.append_record(record)

        cr = ks.consolidate()
        assert "typecheck" in cr.verifier_block_rates
        assert cr.verifier_block_rates["typecheck"] == pytest.approx(30 / 50)

    def test_consolidation_empty_verifier_rates_when_no_blocks(self, tmp_path: Path) -> None:
        from anneal.engine.knowledge import KnowledgeStore
        from anneal.engine.types import ExperimentRecord, Outcome
        from datetime import datetime, timezone

        ks = KnowledgeStore(tmp_path)
        for i in range(50):
            record = ExperimentRecord(
                id=str(i), target_id="t", git_sha="abc", pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc), hypothesis=f"h{i}",
                hypothesis_source="agent", mutation_diff_summary="", score=float(i),
                score_ci_lower=None, score_ci_upper=None, raw_scores=None,
                baseline_score=0.0, outcome=Outcome.KEPT, failure_mode=None,
                duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
                bootstrap_seed=0,
            )
            ks.append_record(record)

        cr = ks.consolidate()
        assert cr.verifier_block_rates == {}
