"""Scope enforcer: parses scope.yaml, validates schemas, checks runtime
integrity, and performs post-write validation with selective reset.

The scope enforcer does NOT intercept file writes. The agent writes freely
to the worktree. After the agent exits and before execution begins, the
scope enforcer inspects all changes and validates every changed or new file
against the editable list.
"""

from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path

import yaml

from anneal.engine.types import (
    EvalMode,
    OptimizationTarget,
    ScopeConfig,
    ScopeViolationResult,
)


class ScopeError(Exception):
    """Raised on scope.yaml parse errors or validation failures."""


# ---------------------------------------------------------------------------
# Required immutable entries
# ---------------------------------------------------------------------------

_REQUIRED_IMMUTABLE: list[str] = [
    "scope.yaml",
    "metrics.yaml",
]

_STOCHASTIC_REQUIRED_IMMUTABLE: list[str] = [
    "eval_criteria.toml",
]


# ---------------------------------------------------------------------------
# scope.yaml parsing
# ---------------------------------------------------------------------------


def load_scope(scope_path: Path) -> ScopeConfig:
    """Parse a scope.yaml file into a ScopeConfig dataclass.

    Raises ScopeError on parse failure or missing required fields.
    """
    try:
        raw = scope_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScopeError(f"Cannot read scope file {scope_path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ScopeError(f"YAML parse error in {scope_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ScopeError(
            f"scope.yaml must be a YAML mapping, got {type(data).__name__}"
        )

    # Validate required fields
    for field in ("editable", "immutable"):
        if field not in data:
            raise ScopeError(f"scope.yaml missing required field: '{field}'")
        value = data[field]
        if not isinstance(value, list) or len(value) == 0:
            raise ScopeError(
                f"scope.yaml field '{field}' must be a non-empty list"
            )

    return ScopeConfig(
        editable=data["editable"],
        immutable=data["immutable"],
        watch=data.get("watch", []),
        allowed_deletions=data.get("allowed_deletions", []),
        constraints=data.get("constraints", []),
    )


# ---------------------------------------------------------------------------
# Schema validation (at registration time)
# ---------------------------------------------------------------------------


def validate_scope(
    scope: ScopeConfig,
    eval_mode: EvalMode,
    sibling_targets: list[OptimizationTarget] | None = None,
) -> list[str]:
    """Validate a ScopeConfig against required invariants.

    Returns a list of error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Required immutable entries
    for required in _REQUIRED_IMMUTABLE:
        if required not in scope.immutable:
            errors.append(
                f"scope.yaml must declare '{required}' as immutable"
            )

    # Stochastic-specific requirements
    if eval_mode is EvalMode.STOCHASTIC:
        for required in _STOCHASTIC_REQUIRED_IMMUTABLE:
            if required not in scope.immutable:
                errors.append(
                    f"Stochastic targets must declare '{required}' as immutable"
                )

    # No overlap between editable and immutable
    overlap = set(scope.editable) & set(scope.immutable)
    if overlap:
        errors.append(
            f"Files cannot be both editable and immutable: {overlap}"
        )

    # Cross-target protection
    if sibling_targets:
        for sibling in sibling_targets:
            protected_files = [
                sibling.scope_path,
                f"targets/{sibling.id}/program.md",
                f"targets/{sibling.id}/eval_criteria.toml",
            ]
            for config_file in protected_files:
                if config_file in scope.editable:
                    errors.append(
                        f"Cannot declare sibling target config as editable: {config_file}"
                    )

    return errors


# ---------------------------------------------------------------------------
# Hash computation and verification
# ---------------------------------------------------------------------------


def compute_scope_hash(scope_path: Path) -> str:
    """Compute SHA-256 hash of the file contents, returned as hex string."""
    content = scope_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def verify_scope_hash(scope_path: Path, expected_hash: str) -> bool:
    """Return True if the current hash matches the expected hash."""
    return compute_scope_hash(scope_path) == expected_hash


# ---------------------------------------------------------------------------
# Path matching helper
# ---------------------------------------------------------------------------


def _is_path_editable(file_path: str, editable: list[str]) -> bool:
    """Check if a file path is permitted by the editable list.

    Matching rules:
    - Exact match: "SKILL.md" matches "SKILL.md"
    - Directory prefix: "src/components/" matches "src/components/Button.tsx"
      (editable entry ends with "/")
    - Glob pattern: fnmatch for entries containing "*" or "?"
    """
    for pattern in editable:
        # Directory prefix match
        if pattern.endswith("/") and file_path.startswith(pattern):
            return True
        # Glob pattern match
        if "*" in pattern or "?" in pattern:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        # Exact match
        if file_path == pattern:
            return True
    return False


# ---------------------------------------------------------------------------
# Post-write validation (runtime, after agent writes)
# ---------------------------------------------------------------------------


async def enforce_scope(
    worktree_path: Path,
    scope: ScopeConfig,
    git_status: list[tuple[str, str]],
) -> ScopeViolationResult:
    """Validate all changed files against the scope's editable list.

    Args:
        worktree_path: Root of the git worktree.
        scope: Parsed ScopeConfig.
        git_status: Pre-parsed output of ``git status --porcelain`` as
            (status_code, path) tuples.

    Returns:
        ScopeViolationResult with violated/valid paths and all_blocked flag.

    This function does NOT perform git operations (reset/checkout). That is
    the runner's responsibility using the environment module.
    """
    violated: list[str] = []
    valid: list[str] = []

    for status_code, file_path in git_status:
        status_code = status_code.strip()

        if status_code in ("M", "A", "??"):
            if _is_path_editable(file_path, scope.editable):
                valid.append(file_path)
            else:
                violated.append(file_path)

        elif status_code == "D":
            if (
                _is_path_editable(file_path, scope.editable)
                and _is_path_editable(file_path, scope.allowed_deletions)
            ):
                valid.append(file_path)
            else:
                violated.append(file_path)

        elif status_code.startswith("R"):
            # Rename: status line contains "old_path -> new_path"
            # The file_path from porcelain parsing may be "old -> new"
            parts = file_path.split(" -> ")
            if len(parts) == 2:
                source, destination = parts[0].strip(), parts[1].strip()
            else:
                # Fallback: treat entire path as violated
                violated.append(file_path)
                continue

            source_ok = (
                _is_path_editable(source, scope.editable)
                and _is_path_editable(source, scope.allowed_deletions)
            )
            dest_ok = _is_path_editable(destination, scope.editable)

            if source_ok and dest_ok:
                valid.append(file_path)
            else:
                violated.append(file_path)

        else:
            # Unknown status code: treat as violation
            violated.append(file_path)

    has_violations = len(violated) > 0
    all_blocked = has_violations and len(valid) == 0

    return ScopeViolationResult(
        has_violations=has_violations,
        violated_paths=violated,
        valid_paths=valid,
        all_blocked=all_blocked,
    )
