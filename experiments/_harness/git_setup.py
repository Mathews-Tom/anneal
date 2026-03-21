"""Temporary git repo setup for gate experiments that use the production runner.

Creates a real git repo with worktree, scope.yaml, and artifact files so
the ExperimentRunner can operate exactly as it would in production — no
mocks, no stubs.

Usage:
    async with create_experiment_repo(
        source_root=PROJECT_ROOT,
        artifact_paths=["examples/skill-diagram/SKILL.md"],
        scope_path="examples/skill-diagram/scope.yaml",
        eval_criteria_path="examples/skill-diagram/eval_criteria.toml",
        target_id="gate3-test",
    ) as repo:
        # repo.repo_root — git init'd directory
        # repo.worktree_path — checked-out worktree
        # repo.scope_hash — computed from scope.yaml
        runner = ExperimentRunner(..., repo_root=repo.repo_root)
        target.worktree_path = str(repo.worktree_path)
        target.scope_hash = repo.scope_hash
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from anneal.engine.registry import Registry
from anneal.engine.scope import compute_scope_hash
from anneal.engine.types import OptimizationTarget


@dataclass
class ExperimentRepo:
    """A temporary git repo with worktree for gate experiments."""

    repo_root: Path
    worktree_path: Path
    scope_hash: str
    scope_rel_path: str
    _tmpdir: str

    def pre_register_target(self, target: OptimizationTarget) -> None:
        """Register the target in the temp repo's config.toml.

        The production runner calls registry.update_target() on KEPT outcomes,
        which requires the target to already exist in config.toml. This method
        writes the target to config without creating a worktree (already done).
        """
        registry = Registry(self.repo_root)
        registry._targets[target.id] = target
        registry.save()


async def _run(args: list[str], cwd: Path) -> None:
    """Run a command, raise on failure."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {args} failed (rc={proc.returncode}): {stderr.decode()}"
        )


@asynccontextmanager
async def create_experiment_repo(
    source_root: Path,
    artifact_paths: list[str],
    scope_path: str,
    target_id: str,
    eval_criteria_path: str = "",
    extra_files: list[str] | None = None,
) -> AsyncIterator[ExperimentRepo]:
    """Create a temporary git repo with a worktree for experiment execution.

    Copies artifact files, scope.yaml, and eval_criteria.toml from source_root
    into a fresh git repo, commits them, and creates a worktree branch.

    The repo is cleaned up on exit.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"anneal-{target_id}-")
    repo_root = Path(tmpdir) / "repo"
    repo_root.mkdir()

    try:
        # 1. Init git repo
        await _run(["git", "init"], cwd=repo_root)
        await _run(
            ["git", "config", "user.email", "anneal-experiment@test"],
            cwd=repo_root,
        )
        await _run(
            ["git", "config", "user.name", "anneal-experiment"],
            cwd=repo_root,
        )

        # 2. Copy files preserving directory structure
        all_files = list(artifact_paths) + [scope_path]
        if eval_criteria_path:
            all_files.append(eval_criteria_path)
        if extra_files:
            all_files.extend(extra_files)

        for rel_path in all_files:
            src = source_root / rel_path
            dst = repo_root / rel_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # 3. Create .anneal directory with minimal config.toml
        anneal_dir = repo_root / ".anneal"
        anneal_dir.mkdir(parents=True, exist_ok=True)
        (anneal_dir / "config.toml").write_text(
            '[anneal]\nversion = "0.1.0"\n', encoding="utf-8",
        )

        # 4. Initial commit
        await _run(["git", "add", "."], cwd=repo_root)
        await _run(
            ["git", "commit", "-m", "Initial commit for gate experiment"],
            cwd=repo_root,
        )

        # 4. Create worktree
        worktree_path = Path(tmpdir) / "worktree"
        branch = f"anneal/{target_id}"
        await _run(
            ["git", "worktree", "add", str(worktree_path), "-b", branch],
            cwd=repo_root,
        )

        # 5. Compute scope hash
        scope_full = repo_root / scope_path
        scope_hash = compute_scope_hash(scope_full)

        yield ExperimentRepo(
            repo_root=repo_root,
            worktree_path=worktree_path,
            scope_hash=scope_hash,
            scope_rel_path=scope_path,
            _tmpdir=tmpdir,
        )

    finally:
        # Force-close leaked FDs before cleanup (subprocess transports accumulate)
        import gc
        gc.collect()

        # Cleanup: use shutil directly — don't spawn subprocess (FDs may be exhausted)
        shutil.rmtree(tmpdir, ignore_errors=True)
