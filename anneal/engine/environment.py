"""Git environment module — worktree management, commits, resets, and gc configuration.

All git operations are async (asyncio.create_subprocess_exec). git stash is
never used; the only snapshot/restore mechanism is commit / reset --hard.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from anneal.engine.types import WorktreeInfo

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Raised on any git command failure."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        cmd_str = " ".join(command)
        super().__init__(
            f"git command failed (rc={returncode}): {cmd_str}\nstderr: {stderr}"
        )


class GitEnvironment:
    """All git operations for the Anneal engine."""

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_git(args: list[str], cwd: Path) -> str:
        """Run a git command, capture stdout, raise GitError on non-zero exit."""
        cmd = ["git", *args]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd.resolve()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()

        if proc.returncode is None or proc.returncode != 0:
            raise GitError(cmd, proc.returncode or -1, stderr)

        return stdout

    # ------------------------------------------------------------------
    # Worktree management
    # ------------------------------------------------------------------

    async def create_worktree(
        self, repo_root: Path, target_id: str
    ) -> WorktreeInfo:
        """Create ``.anneal/worktrees/<target-id>`` with branch ``anneal/<target-id>``.

        If the branch already exists (e.g., from a previous registration),
        checks it out instead of creating a new one. Prunes stale worktree
        references before attempting creation.

        Raises ``GitError`` if the worktree path already exists on disk.
        """
        worktree_path = repo_root / ".anneal" / "worktrees" / target_id
        branch = f"anneal/{target_id}"

        if worktree_path.exists():
            raise GitError(
                ["worktree", "add", str(worktree_path)],
                1,
                f"Worktree path already exists: {worktree_path}",
            )

        # Prune stale worktree references (e.g., after manual directory removal)
        await self._run_git(["worktree", "prune"], cwd=repo_root)

        # Check if branch already exists
        try:
            await self._run_git(
                ["rev-parse", "--verify", f"refs/heads/{branch}"],
                cwd=repo_root,
            )
            branch_exists = True
        except GitError:
            branch_exists = False

        if branch_exists:
            # Check out existing branch
            await self._run_git(
                ["worktree", "add", str(worktree_path.resolve()), branch],
                cwd=repo_root,
            )
        else:
            # Create new branch
            await self._run_git(
                ["worktree", "add", str(worktree_path.resolve()), "-b", branch],
                cwd=repo_root,
            )

        head_sha = await self.rev_parse(worktree_path, "HEAD")

        return WorktreeInfo(
            path=worktree_path.resolve(),
            branch=branch,
            head_sha=head_sha.strip(),
        )

    async def remove_worktree(
        self, repo_root: Path, target_id: str
    ) -> None:
        """Remove worktree but preserve ``.anneal/targets/<target-id>/`` experiment data."""
        worktree_path = repo_root / ".anneal" / "worktrees" / target_id

        await self._run_git(
            ["worktree", "remove", str(worktree_path.resolve()), "--force"],
            cwd=repo_root,
        )

    async def get_worktree_info(self, worktree_path: Path) -> WorktreeInfo:
        """Return current HEAD sha, branch, and path for a worktree."""
        head_sha = await self.rev_parse(worktree_path, "HEAD")

        branch_output = await self._run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=worktree_path,
        )

        return WorktreeInfo(
            path=worktree_path.resolve(),
            branch=branch_output.strip(),
            head_sha=head_sha.strip(),
        )

    # ------------------------------------------------------------------
    # Commit / reset operations
    # ------------------------------------------------------------------

    async def commit(
        self, worktree_path: Path, message: str, paths: list[str]
    ) -> str:
        """Stage specific paths, commit, return the new sha.

        Raises ``GitError`` on empty commit or staging failure.
        """
        absolute_paths = [str((worktree_path / p).resolve()) for p in paths]
        await self._run_git(["add", "--", *absolute_paths], cwd=worktree_path)

        await self._run_git(
            ["commit", "-m", message, "--allow-empty-message"],
            cwd=worktree_path,
        )

        sha = await self.rev_parse(worktree_path, "HEAD")
        return sha.strip()

    async def reset_hard(self, worktree_path: Path, sha: str) -> None:
        """``git reset --hard <sha>``."""
        await self._run_git(["reset", "--hard", sha], cwd=worktree_path)

    async def checkout_paths(
        self, worktree_path: Path, paths: list[str]
    ) -> None:
        """``git checkout -- <paths>`` for selective reset."""
        absolute_paths = [str((worktree_path / p).resolve()) for p in paths]
        await self._run_git(
            ["checkout", "--", *absolute_paths], cwd=worktree_path
        )

    async def clean_untracked(self, worktree_path: Path) -> None:
        """``git clean -fd`` in worktree."""
        await self._run_git(["clean", "-fd"], cwd=worktree_path)

    async def rev_parse(self, worktree_path: Path, ref: str) -> str:
        """Return the sha for a ref (HEAD, branch name, etc.)."""
        output = await self._run_git(
            ["rev-parse", ref], cwd=worktree_path
        )
        return output.strip()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status_porcelain(
        self, worktree_path: Path
    ) -> list[tuple[str, str]]:
        """Return list of ``(status_code, file_path)`` from ``git status --porcelain``.

        Parses all status codes: M, ??, D, R, A, etc.
        """
        output = await self._run_git(
            ["status", "--porcelain"], cwd=worktree_path
        )

        results: list[tuple[str, str]] = []
        for line in output.splitlines():
            if not line:
                continue
            # Porcelain format: XY filename
            # First two characters are the status code, then a space, then path.
            # Renames show as "R  old -> new".
            status_code = line[:2].strip()
            file_path = line[3:]

            # Handle renames: "old -> new"
            if status_code.startswith("R") and " -> " in file_path:
                _, new_path = file_path.rsplit(" -> ", 1)
                file_path = new_path

            results.append((status_code, file_path))

        return results

    # ------------------------------------------------------------------
    # GC configuration
    # ------------------------------------------------------------------

    async def configure_gc(self, repo_root: Path) -> None:
        """Disable auto-gc and set reflog expiry to ``never`` for unreachable commits."""
        await self._run_git(
            ["config", "gc.auto", "0"], cwd=repo_root
        )
        await self._run_git(
            ["config", "gc.reflogExpire", "never"], cwd=repo_root
        )
        await self._run_git(
            ["config", "gc.reflogExpireUnreachable", "never"], cwd=repo_root
        )

    # ------------------------------------------------------------------
    # Index lock cleanup
    # ------------------------------------------------------------------

    async def cleanup_index_lock(self, worktree_path: Path) -> bool:
        """Remove ``.git/index.lock`` if it exists (for KILLED recovery).

        Returns whether a lock file was found and removed.
        """
        # In a worktree, the git dir reference is stored in a .git file
        # pointing to the main repo's worktree-specific directory.
        git_path = worktree_path / ".git"

        if git_path.is_file():
            # Worktree: .git is a file containing "gitdir: <path>"
            content = git_path.read_text().strip()
            if content.startswith("gitdir: "):
                git_dir = Path(content[len("gitdir: "):])
                if not git_dir.is_absolute():
                    git_dir = (worktree_path / git_dir).resolve()
            else:
                git_dir = git_path
        elif git_path.is_dir():
            git_dir = git_path
        else:
            return False

        lock_file = git_dir / "index.lock"
        if lock_file.exists():
            lock_file.unlink()
            logger.warning("Removed stale index.lock at %s", lock_file)
            return True

        return False
