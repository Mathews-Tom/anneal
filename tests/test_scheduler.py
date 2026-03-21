"""Tests for anneal.engine.scheduler — stale lock recovery."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from anneal.engine.scheduler import Scheduler
from anneal.engine.types import (
    AgentConfig,
    Direction,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
)


def _make_target(*, target_id: str = "t1", worktree_path: str) -> OptimizationTarget:
    """Build a minimal OptimizationTarget for scheduler tests."""
    agent = AgentConfig(
        mode="api",
        model="gpt-4.1",
        evaluator_model="gpt-4.1-mini",
        max_budget_usd=0.10,
    )
    eval_config = EvalConfig(
        metric_name="accuracy",
        direction=Direction.HIGHER_IS_BETTER,
    )
    return OptimizationTarget(
        id=target_id,
        domain_tier="sandbox",
        artifact_paths=["src/main.py"],
        scope_path="scope.yaml",
        scope_hash="abc123",
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=eval_config,
        agent_config=agent,
        time_budget_seconds=3600,
        loop_interval_seconds=30,
        knowledge_path="knowledge/",
        worktree_path=worktree_path,
        git_branch=f"anneal/{target_id}",
        baseline_score=0.75,
    )


class TestStaleLockRecovery:
    """Verify that tick() detects and removes stale lock files."""

    def test_stale_lock_removed(self, tmp_path: pytest.TempPathFactory) -> None:
        """A lock file older than max_age_seconds is removed and target becomes ready."""
        target = _make_target(worktree_path=str(tmp_path))
        lock_path = tmp_path / ".anneal.lock"
        lock_path.touch()
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        os.utime(lock_path, (old_time, old_time))

        scheduler = Scheduler([target], max_skip_threshold=3)
        ready = asyncio.get_event_loop().run_until_complete(scheduler.tick())

        assert target.id in ready
        # Lock file created by stale recovery was removed; a new one was acquired
        # Release the lock so cleanup works
        scheduler.release_lock(target.id)

    def test_fresh_lock_not_removed(self, tmp_path: pytest.TempPathFactory) -> None:
        """A recently-created lock file persists and the target is skipped."""
        target = _make_target(worktree_path=str(tmp_path))
        lock_path = tmp_path / ".anneal.lock"
        lock_path.touch()
        # mtime is now (fresh) — lock should NOT be removed

        # We need to hold the lock so tick() can't acquire it.
        # Create the lock file and hold a file lock on it externally.
        from filelock import FileLock

        external_lock = FileLock(str(lock_path), timeout=0)
        external_lock.acquire()

        try:
            scheduler = Scheduler([target], max_skip_threshold=3)
            ready = asyncio.get_event_loop().run_until_complete(scheduler.tick())

            assert target.id not in ready
            assert lock_path.exists()
            assert scheduler.get_skip_count(target.id) == 1
        finally:
            external_lock.release()

    def test_stale_lock_resets_skip_count(self, tmp_path: pytest.TempPathFactory) -> None:
        """After removing a stale lock, the skip count resets to 0."""
        target = _make_target(worktree_path=str(tmp_path))
        scheduler = Scheduler([target], max_skip_threshold=10)

        # Artificially set skip count high
        scheduler._skip_counts[target.id] = 7

        # Create a stale lock
        lock_path = tmp_path / ".anneal.lock"
        lock_path.touch()
        old_time = time.time() - 7200
        os.utime(lock_path, (old_time, old_time))

        ready = asyncio.get_event_loop().run_until_complete(scheduler.tick())

        assert target.id in ready
        assert scheduler.get_skip_count(target.id) == 0
        scheduler.release_lock(target.id)
