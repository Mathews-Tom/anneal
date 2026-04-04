"""Target registry: manages config.toml persistence, target lifecycle,
worktree creation, and project initialization.

The registry is the authoritative store for optimization target configuration.
It owns config.toml and mediates all target registration and deregistration.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

import tomli_w

from anneal.engine.environment import GitEnvironment
from anneal.engine.scope import (
    ScopeError,
    compute_scope_hash,
    load_scope,
    validate_scope,
)
from anneal.engine.types import (
    OptimizationTarget,
)

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Raised on registry operations failure."""


# ---------------------------------------------------------------------------
# TOML serialization/deserialization via Pydantic + tomli-w
# ---------------------------------------------------------------------------


def _serialize_target_toml(target: OptimizationTarget) -> str:
    """Serialize an OptimizationTarget to a TOML string."""
    data = target.model_dump(mode="json", exclude_none=True)
    doc = {"targets": {target.id: data}}
    return tomli_w.dumps(doc)


def _parse_target(data: dict[str, object]) -> OptimizationTarget:
    """Parse an OptimizationTarget from a TOML dict."""
    return OptimizationTarget.model_validate(data)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CONFIG_VERSION = "0.1.0"


class Registry:
    """Target registry backed by config.toml."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._targets: dict[str, OptimizationTarget] = {}
        self._git = GitEnvironment()
        self.load()

    @property
    def config_path(self) -> Path:
        return self._repo_root / ".anneal" / "config.toml"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load config.toml from disk. No-op if file doesn't exist."""
        if not self.config_path.exists():
            return

        raw = self.config_path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))

        targets_data = data.get("targets", {})
        self._targets = {}
        for tid, tdata in targets_data.items():
            self._targets[tid] = _parse_target(tdata)

    def save(self) -> None:
        """Persist current state to config.toml."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        targets_dict = {}
        for target in self._targets.values():
            targets_dict[target.id] = target.model_dump(mode="json", exclude_none=True)

        doc: dict[str, object] = {
            "anneal": {"version": _CONFIG_VERSION},
            "targets": targets_dict,
        }
        self.config_path.write_bytes(tomli_w.dumps(doc).encode("utf-8"))

    # ------------------------------------------------------------------
    # Target lifecycle
    # ------------------------------------------------------------------

    async def register_target(self, target: OptimizationTarget) -> list[str]:
        """Register a new target: validate scope, create worktree, store config.

        Returns list of paths that were auto-staged from untracked files.
        Raises RegistryError if validation fails or target ID already exists.
        """
        if target.id in self._targets:
            raise RegistryError(f"Target already registered: {target.id}")

        # Resolve scope path relative to repo root
        scope_path = self._repo_root / target.scope_path
        try:
            scope = load_scope(scope_path)
        except ScopeError as exc:
            raise RegistryError(f"Scope load failed for {target.id}: {exc}") from exc

        # Validate scope against invariants
        sibling_targets = list(self._targets.values())
        errors = validate_scope(
            scope, target.eval_mode, sibling_targets, in_place=target.in_place
        )
        if errors:
            raise RegistryError(
                f"Scope validation failed for {target.id}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Validate eval environment variables (fail fast, not mid-experiment)
        if target.eval_environment and target.eval_environment.env_vars:
            missing = [
                v for v in target.eval_environment.env_vars if v not in os.environ
            ]
            if missing:
                raise RegistryError(
                    f"Eval environment requires env vars not set: {', '.join(missing)}"
                )

        # Compute and store scope hash
        target.scope_hash = compute_scope_hash(scope_path)

        staged: list[str] = []

        if target.in_place:
            # In-place mode: no worktree, artifacts live in repo root
            target.worktree_path = str(self._repo_root)
            target.git_branch = ""
            backup_dir = self._repo_root / ".anneal" / "backups" / target.id
            backup_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Standard mode: create git worktree
            worktree_info = await self._git.create_worktree(self._repo_root, target.id)
            target.worktree_path = str(worktree_info.path.relative_to(self._repo_root))
            target.git_branch = worktree_info.branch

            # Auto-stage untracked artifacts into worktree
            worktree_abs = worktree_info.path
            staged = await self._git.stage_untracked_artifacts(
                self._repo_root,
                worktree_abs,
                target.artifact_paths,
                target.scope_path,
            )
            if staged:
                logger.info(
                    "Staged %d untracked artifact(s) on branch %s: %s",
                    len(staged),
                    target.git_branch,
                    staged,
                )

            # Warn if artifacts are still missing after staging attempt
            still_missing = [
                p for p in target.artifact_paths if not (worktree_abs / p).exists()
            ]
            if still_missing:
                logger.warning(
                    "Artifacts still missing from worktree for %s after staging: %s. "
                    "These files do not exist in the repo working directory either.",
                    target.id,
                    still_missing,
                )

        # Configure gc to preserve experiment history
        await self._git.configure_gc(self._repo_root)

        # Clear stale loop state from prior runs so the new registration
        # starts with a fresh experiment counter (preserves crash recovery
        # semantics — only registration resets, not run restarts).
        knowledge_dir = self._repo_root / target.knowledge_path
        loop_state_file = knowledge_dir / ".loop-state.json"
        if loop_state_file.exists():
            loop_state_file.unlink()
            logger.info("Cleared stale loop state for %s", target.id)

        # Store and persist
        self._targets[target.id] = target
        self.save()

        logger.info(
            "Registered target %s (worktree: %s)", target.id, target.worktree_path
        )
        return staged

    async def deregister_target(self, target_id: str) -> None:
        """Remove target: remove worktree, remove from config, preserve experiment history.

        Raises RegistryError if target doesn't exist.
        """
        if target_id not in self._targets:
            raise RegistryError(f"Target not found: {target_id}")

        # Remove worktree (experiment history in .anneal/targets/<id>/ is preserved)
        await self._git.remove_worktree(self._repo_root, target_id)

        del self._targets[target_id]
        self.save()

        logger.info("Deregistered target %s", target_id)

    def get_target(self, target_id: str) -> OptimizationTarget:
        """Get a target by ID. Raises RegistryError if not found."""
        if target_id not in self._targets:
            raise RegistryError(f"Target not found: {target_id}")
        return self._targets[target_id]

    def all_targets(self) -> list[OptimizationTarget]:
        """Return all registered targets."""
        return list(self._targets.values())

    def update_target(self, target: OptimizationTarget) -> None:
        """Update a target's config (e.g., after baseline score change). Persists to disk."""
        if target.id not in self._targets:
            raise RegistryError(f"Target not found: {target.id}")
        self._targets[target.id] = target
        self.save()


# ---------------------------------------------------------------------------
# Project initialization
# ---------------------------------------------------------------------------


async def init_project(repo_root: Path) -> None:
    """One-time project setup.

    1. Verify repo_root is a git repository (has .git)
    2. Create .anneal/ directory structure (config, targets, templates, worktrees)
    3. Add '.anneal/' to .gitignore if not already present
    4. Raise RegistryError if not a git repo or already initialized
    """
    repo_root = repo_root.resolve()

    # Verify git repository
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        raise RegistryError(f"Not a git repository: {repo_root}")

    # Check not already initialized
    anneal_dir = repo_root / ".anneal"
    config_path = anneal_dir / "config.toml"
    if config_path.exists():
        raise RegistryError(f"Project already initialized: {config_path} exists")

    # Create directory structure under .anneal/
    (anneal_dir / "targets").mkdir(parents=True, exist_ok=True)
    (anneal_dir / "templates").mkdir(parents=True, exist_ok=True)
    (anneal_dir / "worktrees").mkdir(parents=True, exist_ok=True)

    # Write default config.toml
    config_path.write_text(
        f'[anneal]\nversion = "{_CONFIG_VERSION}"\n',
        encoding="utf-8",
    )

    # Ensure .anneal/ is in .gitignore
    gitignore_path = repo_root / ".gitignore"
    ignore_entry = ".anneal/"

    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
        if ignore_entry not in existing.splitlines():
            with gitignore_path.open("a", encoding="utf-8") as f:
                if not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{ignore_entry}\n")
    else:
        gitignore_path.write_text(f"{ignore_entry}\n", encoding="utf-8")

    logger.info("Initialized anneal project at %s", repo_root)
