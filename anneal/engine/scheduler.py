"""Sequential target scheduler with per-worktree file locking."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from filelock import FileLock, Timeout

from anneal.engine.types import OptimizationTarget

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Raised on scheduling failures."""


class Scheduler:
    """Schedules optimization targets sequentially with file-lock concurrency control.

    Each target's worktree gets a `.anneal.lock` file. If the lock is held
    (previous cycle still running), the target is skipped and a counter is
    incremented. When the counter reaches ``max_skip_threshold`` the target
    is marked for HALT.
    """

    def __init__(
        self,
        targets: list[OptimizationTarget],
        max_skip_threshold: int = 10,
    ) -> None:
        self._targets = {t.id: t for t in targets}
        self._max_skip_threshold = max_skip_threshold
        self._skip_counts: dict[str, int] = {t.id: 0 for t in targets}
        self._locks: dict[str, FileLock] = {}
        self._halted: set[str] = set()

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def tick(self) -> list[str]:
        """Check all targets and return IDs of targets that are ready to run.

        For each target:
        1. Try to acquire a file lock on the target's worktree.
        2. If acquired, the target is ready (caller runs the experiment).
        3. If lock is held, increment skip counter and log warning.
        4. If skip counter >= max_skip_threshold, mark target for HALT.

        Returns list of target IDs that are ready to run.
        """
        ready: list[str] = []

        for target_id, target in self._targets.items():
            if target_id in self._halted:
                continue

            lock = self._make_lock(target)

            try:
                await asyncio.to_thread(lock.acquire, timeout=0)
            except Timeout:
                self._skip_counts[target_id] = self._skip_counts.get(target_id, 0) + 1
                count = self._skip_counts[target_id]
                logger.warning(
                    "Target %s locked (skip #%d) — previous cycle still running",
                    target_id,
                    count,
                )
                if count >= self._max_skip_threshold:
                    self._halted.add(target_id)
                    logger.warning(
                        "Target %s marked for HALT: lock held for %d consecutive ticks",
                        target_id,
                        count,
                    )
                continue

            self._locks[target_id] = lock
            ready.append(target_id)

        return ready

    # ------------------------------------------------------------------
    # Lock lifecycle
    # ------------------------------------------------------------------

    def acquire_lock(self, target_id: str) -> FileLock:
        """Acquire the target's lock. Returns the lock for the caller to release."""
        target = self._resolve_target(target_id)
        lock = self._make_lock(target)
        lock.acquire(timeout=0)
        self._locks[target_id] = lock
        return lock

    def release_lock(self, target_id: str) -> None:
        """Release the target's lock."""
        lock = self._locks.pop(target_id, None)
        if lock is None:
            raise SchedulerError(
                f"No active lock for target {target_id!r}"
            )
        lock.release()

    # ------------------------------------------------------------------
    # Skip tracking
    # ------------------------------------------------------------------

    def get_skip_count(self, target_id: str) -> int:
        self._resolve_target(target_id)
        return self._skip_counts.get(target_id, 0)

    def reset_skip_count(self, target_id: str) -> None:
        self._resolve_target(target_id)
        self._skip_counts[target_id] = 0

    def should_halt(self, target_id: str) -> bool:
        self._resolve_target(target_id)
        return target_id in self._halted

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_target(self, target_id: str) -> OptimizationTarget:
        target = self._targets.get(target_id)
        if target is None:
            raise SchedulerError(f"Unknown target {target_id!r}")
        return target

    def _lock_path(self, target: OptimizationTarget) -> Path:
        """Derive the absolute lock path from the target's worktree."""
        path = Path(target.worktree_path).resolve() / ".anneal.lock"
        return path

    def _make_lock(self, target: OptimizationTarget) -> FileLock:
        return FileLock(str(self._lock_path(target)), timeout=0)  # type: ignore[return-value]
