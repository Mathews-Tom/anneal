"""Tests for multi-draft mutation with ensemble selection (E6)."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    ExperimentRecord,
    Outcome,
    VerifierCommand,
)


@pytest.fixture
def git_worktree(tmp_path: Path) -> Path:
    """Create a git repo for draft testing."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    (tmp_path / "artifact.md").write_text("original content\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    return tmp_path


class TestCaptureDiff:
    def test_captures_uncommitted_changes(self, git_worktree: Path) -> None:
        from anneal.engine.environment import GitEnvironment
        git = GitEnvironment()
        (git_worktree / "artifact.md").write_text("modified content\n")
        diff = asyncio.run(git.capture_diff(git_worktree))
        assert "modified content" in diff
        assert "-original content" in diff

    def test_empty_diff_when_clean(self, git_worktree: Path) -> None:
        from anneal.engine.environment import GitEnvironment
        git = GitEnvironment()
        diff = asyncio.run(git.capture_diff(git_worktree))
        assert diff.strip() == ""


class TestApplyDiff:
    def test_applies_diff_cleanly(self, git_worktree: Path) -> None:
        from anneal.engine.environment import GitEnvironment
        git = GitEnvironment()
        (git_worktree / "artifact.md").write_text("modified content\n")
        diff = asyncio.run(git.capture_diff(git_worktree))
        # Reset to original
        asyncio.run(git.reset_hard(git_worktree, "HEAD"))
        assert (git_worktree / "artifact.md").read_text() == "original content\n"
        # Apply the captured diff
        success = asyncio.run(git.apply_diff(git_worktree, diff))
        assert success is True
        assert (git_worktree / "artifact.md").read_text() == "modified content\n"

    def test_empty_diff_returns_false(self, git_worktree: Path) -> None:
        from anneal.engine.environment import GitEnvironment
        git = GitEnvironment()
        success = asyncio.run(git.apply_diff(git_worktree, ""))
        assert success is False

    def test_invalid_diff_returns_false(self, git_worktree: Path) -> None:
        from anneal.engine.environment import GitEnvironment
        git = GitEnvironment()
        bad_diff = (
            "--- a/nonexistent.txt\n"
            "+++ b/nonexistent.txt\n"
            "@@ -1 +1 @@\n"
            "-foo\n"
            "+bar\n"
        )
        success = asyncio.run(git.apply_diff(git_worktree, bad_diff))
        assert success is False


class TestNDraftsDefault:
    """Tests that n_drafts=1 preserves existing behavior."""

    def test_default_n_drafts_is_one(self) -> None:
        config = AgentConfig(mode="api", model="test", evaluator_model="test")
        assert config.n_drafts == 1

    def test_n_drafts_range(self) -> None:
        config = AgentConfig(mode="api", model="test", evaluator_model="test", n_drafts=5)
        assert config.n_drafts == 5

    def test_n_drafts_max(self) -> None:
        config = AgentConfig(mode="api", model="test", evaluator_model="test", n_drafts=10)
        assert config.n_drafts == 10


class TestExperimentRecordDraftFields:
    """Tests for draft tracking fields on ExperimentRecord."""

    def test_defaults(self) -> None:
        from datetime import datetime, timezone
        record = ExperimentRecord(
            id="test", target_id="t", git_sha="abc", pre_experiment_sha="abc",
            timestamp=datetime.now(tz=timezone.utc), hypothesis="h",
            hypothesis_source="agent", mutation_diff_summary="", score=1.0,
            score_ci_lower=None, score_ci_upper=None, raw_scores=None,
            baseline_score=0.5, outcome=Outcome.KEPT, failure_mode=None,
            duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
            bootstrap_seed=0,
        )
        assert record.drafts_generated == 1
        assert record.drafts_survived == 1

    def test_custom_values(self) -> None:
        from datetime import datetime, timezone
        record = ExperimentRecord(
            id="test", target_id="t", git_sha="abc", pre_experiment_sha="abc",
            timestamp=datetime.now(tz=timezone.utc), hypothesis="h",
            hypothesis_source="agent", mutation_diff_summary="", score=1.0,
            score_ci_lower=None, score_ci_upper=None, raw_scores=None,
            baseline_score=0.5, outcome=Outcome.KEPT, failure_mode=None,
            duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
            bootstrap_seed=0, drafts_generated=5, drafts_survived=3,
        )
        assert record.drafts_generated == 5
        assert record.drafts_survived == 3


class TestBudgetSplitting:
    """Tests that per-draft budget is correctly divided."""

    def test_budget_splits_evenly(self) -> None:
        config = AgentConfig(
            mode="api", model="test", evaluator_model="test",
            max_budget_usd=1.0, n_drafts=5,
        )
        per_draft = config.max_budget_usd / config.n_drafts
        assert per_draft == pytest.approx(0.2)

    def test_single_draft_full_budget(self) -> None:
        config = AgentConfig(
            mode="api", model="test", evaluator_model="test",
            max_budget_usd=1.0, n_drafts=1,
        )
        per_draft = config.max_budget_usd / config.n_drafts
        assert per_draft == pytest.approx(1.0)
