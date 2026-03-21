"""Codebase scanner — discover related files, tests, and entry points."""

from __future__ import annotations

import ast
from pathlib import Path


def scan_related_files(
    repo_root: Path,
    artifact_paths: list[str],
    max_depth: int = 3,
) -> list[str]:
    """Discover files related to the given artifacts.

    Walks the import graph (Python) or reference graph (other languages)
    to find closely related files. Returns relative paths from repo_root.
    """
    related: set[str] = set()

    for rel_path in artifact_paths:
        full_path = repo_root / rel_path
        if not full_path.exists():
            continue

        # Python import scanning
        if full_path.suffix == ".py":
            imports = _extract_python_imports(full_path, repo_root)
            related.update(imports)

        # Sibling files in the same directory
        for sibling in full_path.parent.iterdir():
            if sibling.is_file() and sibling.name != full_path.name:
                rel = str(sibling.relative_to(repo_root))
                related.add(rel)

    # Remove the artifacts themselves from related
    related -= set(artifact_paths)
    return sorted(related)


def scan_test_files(
    repo_root: Path,
    artifact_paths: list[str],
) -> list[str]:
    """Find test files that likely test the given artifacts."""
    test_dirs = ["tests", "test", "spec", "__tests__"]
    test_patterns = ["test_*.py", "*_test.py", "*.test.ts", "*.test.js", "*.spec.ts", "*.spec.js"]
    test_files: set[str] = set()

    # Find test directories
    for test_dir in test_dirs:
        td = repo_root / test_dir
        if td.is_dir():
            for pattern in test_patterns:
                for f in td.rglob(pattern):
                    test_files.add(str(f.relative_to(repo_root)))

    # Find test files alongside artifacts (same directory)
    for rel_path in artifact_paths:
        full = repo_root / rel_path
        stem = full.stem
        for pattern in [f"test_{stem}.py", f"{stem}_test.py", f"{stem}.test.ts", f"{stem}.test.js"]:
            candidate = full.parent / pattern
            if candidate.exists():
                test_files.add(str(candidate.relative_to(repo_root)))

    return sorted(test_files)


def _extract_python_imports(file_path: Path, repo_root: Path) -> set[str]:
    """Extract local (relative) Python imports and resolve to file paths."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_python_module(alias.name, repo_root)
                if resolved:
                    imports.add(resolved)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                resolved = _resolve_python_module(node.module, repo_root)
                if resolved:
                    imports.add(resolved)

    return imports


def _resolve_python_module(module_name: str, repo_root: Path) -> str | None:
    """Resolve a Python module name to a relative file path if it exists locally."""
    # Convert module.path to file path
    parts = module_name.split(".")
    candidates = [
        Path(*parts).with_suffix(".py"),
        Path(*parts) / "__init__.py",
    ]
    for candidate in candidates:
        full = repo_root / candidate
        if full.exists():
            return str(candidate)
    return None
