"""Git environment module — worktree management, commits, resets, and gc configuration.

All git operations are async (asyncio.create_subprocess_exec). git stash is
never used; the only snapshot/restore mechanism is commit / reset --hard.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
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

    async def stage_untracked_artifacts(
        self,
        repo_root: Path,
        worktree_path: Path,
        artifact_paths: list[str],
        scope_path: str,
    ) -> list[str]:
        """Copy untracked artifacts from repo working directory into worktree.

        For each path in ``artifact_paths`` plus scope-adjacent files
        (scope.yaml, eval_criteria.toml), copies files that exist in
        ``repo_root`` but are missing from ``worktree_path``.

        Returns list of relative paths that were staged and committed.
        """
        staged: list[str] = []

        # Collect all paths to check: artifacts + scope-adjacent files
        scope_parent = str(Path(scope_path).parent)
        criteria_rel = str(Path(scope_parent) / "eval_criteria.toml")
        paths_to_check = list(artifact_paths)
        for adjacent in [scope_path, criteria_rel]:
            if adjacent not in paths_to_check:
                paths_to_check.append(adjacent)

        for rel_path in paths_to_check:
            src = repo_root / rel_path
            dst = worktree_path / rel_path
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                staged.append(rel_path)

        if not staged:
            return staged

        # Stage and commit on the anneal branch
        await self._run_git(
            ["add", "--", *staged],
            cwd=worktree_path,
        )
        await self._run_git(
            ["commit", "-m", "anneal: stage untracked artifacts"],
            cwd=worktree_path,
        )
        logger.info("Staged %d untracked file(s) in worktree: %s", len(staged), staged)
        return staged

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

    async def checkout(self, worktree: Path, sha: str) -> None:
        """Checkout a specific commit in the worktree.

        Used by tree search to explore non-HEAD ancestors.
        Wraps reset_hard — the worktree moves to the specified SHA.
        """
        await self.reset_hard(worktree, sha)

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

    async def fsck(self, worktree: Path) -> bool:
        """Run git fsck. Returns True if clean."""
        proc = await asyncio.create_subprocess_exec(
            "git", "fsck", "--no-full", "--no-dangling",
            cwd=str(worktree),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        return proc.returncode == 0

    # ------------------------------------------------------------------
    # Diff capture / apply (multi-draft support)
    # ------------------------------------------------------------------

    async def capture_diff(self, worktree: Path) -> str:
        """Capture the current uncommitted changes as a unified diff.

        Returns the diff text. Used for multi-draft isolation:
        capture each draft's changes before resetting.
        """
        return await self._run_git(["diff", "HEAD"], worktree)

    async def apply_diff(self, worktree: Path, diff_text: str) -> bool:
        """Apply a previously captured diff to the worktree.

        Returns True if applied cleanly, False on failure.
        """
        if not diff_text.strip():
            return False
        proc = await asyncio.create_subprocess_exec(
            "git", "apply", "--allow-empty",
            cwd=str(worktree.resolve()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate(input=diff_text.encode())
        if proc.returncode != 0:
            logger.warning(
                "Failed to apply diff: %s",
                stderr_bytes.decode(errors="replace").strip()[:200],
            )
            return False
        return True

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


# ---------------------------------------------------------------------------
# File-based backup for in-place mode (no git worktree)
# ---------------------------------------------------------------------------


class FileBackupEnvironment:
    """File-based backup/restore for in-place optimization without git worktree.

    Stores timestamped copies of artifact files in
    ``.anneal/backups/<target_id>/<backup_id>/``. Each backup mirrors the
    relative directory structure of the artifact paths.
    """

    def __init__(self, backup_dir: Path) -> None:
        self._backup_dir = backup_dir
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    async def backup(
        self, artifact_paths: list[str], base_dir: Path
    ) -> str:
        """Copy artifact files to a timestamped backup directory.

        Returns the backup_id (usable with restore/cleanup).
        """
        import time as _time

        backup_id = f"{_time.time_ns()}"
        dest = self._backup_dir / backup_id

        for rel_path in artifact_paths:
            src = base_dir / rel_path
            if src.is_file():
                target = dest / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)

        logger.info("Backed up %d artifact(s) to %s", len(artifact_paths), dest)
        return backup_id

    async def restore(
        self, backup_id: str, artifact_paths: list[str], base_dir: Path
    ) -> None:
        """Restore artifact files from a previous backup."""
        src_dir = self._backup_dir / backup_id
        if not src_dir.exists():
            raise FileNotFoundError(f"Backup not found: {src_dir}")

        for rel_path in artifact_paths:
            src = src_dir / rel_path
            dst = base_dir / rel_path
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        logger.info("Restored %d artifact(s) from backup %s", len(artifact_paths), backup_id)

    async def cleanup(self, backup_id: str) -> None:
        """Remove a backup directory after a successful keep."""
        target = self._backup_dir / backup_id
        if target.exists():
            shutil.rmtree(target)
            logger.debug("Cleaned up backup %s", backup_id)
