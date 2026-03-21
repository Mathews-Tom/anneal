"""Scope generator — produce scope.yaml from codebase scan and problem intent."""

from __future__ import annotations

from pathlib import Path

import yaml

from anneal.suggest.scanner import scan_related_files, scan_test_files
from anneal.suggest.types import ProblemIntent, ScopeResult


def generate_scope(
    repo_root: Path,
    artifact_paths: list[str],
    intent: ProblemIntent,
) -> ScopeResult:
    """Generate scope.yaml content from artifact analysis.

    Editable: artifact files + closely related source files.
    Immutable: test files, scope.yaml, eval criteria, package manifests.
    """
    # Always editable: the artifacts themselves
    editable = list(artifact_paths)

    # Discover related source files (same package, imports)
    related = scan_related_files(repo_root, artifact_paths)

    # For code domain, include closely related source files as editable
    # For prompt/config/document, keep editable tight (artifact only)
    if intent.domain.value == "code":
        # Add related .py/.ts/.js files (not tests, configs, or package files)
        code_extensions = {".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs"}
        for rel in related:
            p = Path(rel)
            if p.suffix in code_extensions and not _is_test_or_config(rel):
                editable.append(rel)

    # Immutable: tests, scope, eval criteria, package manifests
    immutable = ["scope.yaml", "metrics.yaml"]

    test_files = scan_test_files(repo_root, artifact_paths)
    for tf in test_files:
        if tf not in immutable:
            immutable.append(tf)

    # Package manifests
    manifests = [
        "package.json", "package-lock.json", "pnpm-lock.yaml",
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
        "Cargo.toml", "go.mod", "go.sum",
    ]
    for manifest in manifests:
        if (repo_root / manifest).exists() and manifest not in immutable:
            immutable.append(manifest)

    # Eval criteria (stochastic)
    if intent.eval_mode == "stochastic":
        immutable.append("eval_criteria.toml")

    # Deduplicate and sort
    editable = sorted(set(editable))
    immutable = sorted(set(immutable))

    # Build YAML content
    scope_data = {
        "editable": editable,
        "immutable": immutable,
    }
    scope_yaml = yaml.dump(scope_data, default_flow_style=False, sort_keys=False)

    return ScopeResult(
        editable=editable,
        immutable=immutable,
        scope_yaml_content=scope_yaml,
    )


def _is_test_or_config(rel_path: str) -> bool:
    """Check if a path looks like a test file or config file."""
    lower = rel_path.lower()
    test_indicators = ["test_", "_test.", ".test.", ".spec.", "/tests/", "/test/", "__tests__"]
    config_indicators = [
        ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".env",
        "config", "setup.", "manifest", "lock",
    ]
    return (
        any(ind in lower for ind in test_indicators)
        or any(ind in lower for ind in config_indicators)
    )
