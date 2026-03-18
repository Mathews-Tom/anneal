"""Tests for anneal.engine.environment — real git operations, no mocks."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
from pathlib import Path

import pytest

from anneal.engine.environment import GitEnvironment, GitError
from anneal.engine.types import WorktreeInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a real git repository with one initial commit."""
    subprocess.run(
        ["git", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    initial_file = tmp_path / "README"
    initial_file.write_text("init\n")
    subprocess.run(
        ["git", "add", "README"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    return tmp_path


@pytest.fixture()
def env() -> GitEnvironment:
    return GitEnvironment()


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_worktree_returns_valid_info(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "t1")

    assert isinstance(info, WorktreeInfo)
    assert info.branch == "anneal/t1"
    assert len(info.head_sha) == 40
    assert info.path.exists()
    assert info.path == (git_repo / ".anneal" / "worktrees" / "t1").resolve()


@pytest.mark.asyncio
async def test_create_worktree_existing_path_raises(
    git_repo: Path, env: GitEnvironment
) -> None:
    await env.create_worktree(git_repo, "dup")

    with pytest.raises(GitError):
        await env.create_worktree(git_repo, "dup")


@pytest.mark.asyncio
async def test_remove_worktree_deletes_directory(
    git_repo: Path, env: GitEnvironment
) -> None:
    await env.create_worktree(git_repo, "rm-me")
    wt_path = git_repo / ".anneal" / "worktrees" / "rm-me"
    assert wt_path.exists()

    await env.remove_worktree(git_repo, "rm-me")
    assert not wt_path.exists()


@pytest.mark.asyncio
async def test_get_worktree_info_returns_correct_data(
    git_repo: Path, env: GitEnvironment
) -> None:
    created = await env.create_worktree(git_repo, "info-check")
    retrieved = await env.get_worktree_info(created.path)

    assert retrieved.branch == "anneal/info-check"
    assert retrieved.head_sha == created.head_sha
    assert len(retrieved.head_sha) == 40


# ---------------------------------------------------------------------------
# Commit / reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_stages_specific_paths_returns_sha(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "commit-test")
    wt = info.path

    (wt / "a.txt").write_text("a\n")
    (wt / "b.txt").write_text("b\n")

    sha = await env.commit(wt, "add a only", ["a.txt"])
    assert len(sha) == 40

    # b.txt should still be untracked
    status = await env.status_porcelain(wt)
    status_by_path = {path: code for code, path in status}
    assert "b.txt" in status_by_path
    assert status_by_path["b.txt"] == "??"


@pytest.mark.asyncio
async def test_commit_creates_new_head(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "new-head")
    wt = info.path
    old_sha = info.head_sha

    (wt / "change.txt").write_text("data\n")
    new_sha = await env.commit(wt, "new file", ["change.txt"])

    assert new_sha != old_sha


@pytest.mark.asyncio
async def test_reset_hard_restores_content(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "reset-test")
    wt = info.path
    original_sha = info.head_sha

    target = wt / "README"
    target.write_text("modified\n")
    await env.commit(wt, "modify", ["README"])

    await env.reset_hard(wt, original_sha)
    assert target.read_text() == "init\n"


@pytest.mark.asyncio
async def test_checkout_paths_restores_specific_file(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "checkout-test")
    wt = info.path

    (wt / "keep.txt").write_text("keep\n")
    await env.commit(wt, "add keep", ["keep.txt"])

    # Modify both committed files
    (wt / "README").write_text("changed\n")
    (wt / "keep.txt").write_text("changed\n")

    # Restore only README
    await env.checkout_paths(wt, ["README"])

    assert (wt / "README").read_text() == "init\n"
    assert (wt / "keep.txt").read_text() == "changed\n"


@pytest.mark.asyncio
async def test_clean_untracked_removes_files(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "clean-test")
    wt = info.path

    (wt / "untracked.txt").write_text("junk\n")
    assert (wt / "untracked.txt").exists()

    await env.clean_untracked(wt)
    assert not (wt / "untracked.txt").exists()


@pytest.mark.asyncio
async def test_rev_parse_returns_correct_sha(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "revparse")
    sha = await env.rev_parse(info.path, "HEAD")
    assert sha == info.head_sha
    assert len(sha) == 40


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_detects_modified(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "status-mod")
    wt = info.path

    (wt / "README").write_text("changed\n")
    status = await env.status_porcelain(wt)

    codes = {path: code for code, path in status}
    assert "README" in codes
    assert "M" in codes["README"]


@pytest.mark.asyncio
async def test_status_detects_untracked(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "status-unt")
    wt = info.path

    (wt / "new.txt").write_text("x\n")
    status = await env.status_porcelain(wt)

    codes = {path: code for code, path in status}
    assert "new.txt" in codes
    assert codes["new.txt"] == "??"


@pytest.mark.asyncio
async def test_status_detects_deleted(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "status-del")
    wt = info.path

    (wt / "README").unlink()
    status = await env.status_porcelain(wt)

    codes = {path: code for code, path in status}
    assert "README" in codes
    assert "D" in codes["README"]


@pytest.mark.asyncio
async def test_status_clean_returns_empty(
    git_repo: Path, env: GitEnvironment
) -> None:
    info = await env.create_worktree(git_repo, "status-clean")
    status = await env.status_porcelain(info.path)
    assert status == []


# ---------------------------------------------------------------------------
# GC configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configure_gc_sets_auto_zero(
    git_repo: Path, env: GitEnvironment
) -> None:
    await env.configure_gc(git_repo)

    result = subprocess.run(
        ["git", "config", "gc.auto"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0"


# ---------------------------------------------------------------------------
# Index lock cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_index_lock_no_lock_returns_false(
    git_repo: Path, env: GitEnvironment
) -> None:
    result = await env.cleanup_index_lock(git_repo)
    assert result is False


@pytest.mark.asyncio
async def test_cleanup_index_lock_removes_existing_lock(
    git_repo: Path, env: GitEnvironment
) -> None:
    lock = git_repo / ".git" / "index.lock"
    lock.write_text("")

    result = await env.cleanup_index_lock(git_repo)
    assert result is True
    assert not lock.exists()


# ---------------------------------------------------------------------------
# Design invariant: no stash usage
# ---------------------------------------------------------------------------


def test_stash_never_called() -> None:
    """The word 'stash' must not appear in environment.py as a git command."""
    source = inspect.getsource(GitEnvironment)
    assert "stash" not in source.lower()
