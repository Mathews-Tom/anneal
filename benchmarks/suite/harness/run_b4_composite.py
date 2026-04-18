from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ARTIFACT_PATH = Path("benchmarks/suite/artifacts/B4_api_endpoint.py")
LINT_VIOLATION_SCALE = 20


# ---------------------------------------------------------------------------
# Sub-score: correctness (hidden test suite)
# ---------------------------------------------------------------------------


def _load_module() -> object | None:
    """Load the B4 artifact as a fresh module (bypass sys.modules cache)."""
    mod_name = "B4_api_endpoint"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, ARTIFACT_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    artifacts_dir = str(ARTIFACT_PATH.parent.resolve())
    if artifacts_dir not in sys.path:
        sys.path.insert(0, artifacts_dir)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run_tests(module: object) -> tuple[int, int]:
    """Run hidden correctness suite.  Returns (passed, total).

    Individual test identities are NOT printed — the optimizer sees only
    the aggregate pass count to preserve signal quality.
    """
    results: list[bool] = []

    def check(condition: bool) -> None:
        results.append(condition)

    def reset() -> None:
        """Reset module-level state for test isolation."""
        try:
            module.db = {}  # type: ignore[attr-defined]
            module._next_id = 1  # type: ignore[attr-defined]
        except AttributeError:
            pass

    # ── Scenario 1: Basic CRUD flow ──────────────────────────────────────
    reset()

    # T01: Create valid user
    try:
        res, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "Alice", "email": "alice@example.com", "age": 30},
        )
        check(
            status == 201
            and res.get("id") == 1
            and res.get("name") == "Alice"
            and res.get("email") == "alice@example.com"
            and res.get("age") == 30
            and res.get("active") is True
        )
    except Exception:
        check(False)

    # T02: Create second user (monotonic IDs)
    try:
        res, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "Bob", "email": "bob@example.com", "age": 25},
        )
        check(status == 201 and res.get("id") == 2)
    except Exception:
        check(False)

    # T03: Get existing user
    try:
        res, status = module.get_user(1)  # type: ignore[attr-defined]
        check(status == 200 and res.get("name") == "Alice")
    except Exception:
        check(False)

    # T04: Update user fields
    try:
        res, status = module.update_user(  # type: ignore[attr-defined]
            1,
            {"age": 31, "name": "Alice Updated"},
        )
        check(
            status == 200
            and res.get("age") == 31
            and res.get("name") == "Alice Updated"
        )
    except Exception:
        check(False)

    # T05: List all users — count check
    try:
        res, status = module.list_users({})  # type: ignore[attr-defined]
        check(status == 200 and len(res) == 2)
    except Exception:
        check(False)

    # T06: List with active filter after deactivation
    try:
        module.update_user(2, {"active": False})  # type: ignore[attr-defined]
        res, status = module.list_users({"active": True})  # type: ignore[attr-defined]
        check(status == 200 and len(res) == 1)
    except Exception:
        check(False)

    # T07: Delete user
    try:
        res, status = module.delete_user(2)  # type: ignore[attr-defined]
        check(status == 200)
    except Exception:
        check(False)

    # T08: Get deleted user → 404
    try:
        res, status = module.get_user(2)  # type: ignore[attr-defined]
        check(status == 404)
    except Exception:
        check(False)

    # ── Scenario 2: Standard error paths ─────────────────────────────────
    reset()

    # T09: Get non-existent → 404
    try:
        _, status = module.get_user(999)  # type: ignore[attr-defined]
        check(status == 404)
    except Exception:
        check(False)

    # T10: Delete non-existent → 404
    try:
        _, status = module.delete_user(999)  # type: ignore[attr-defined]
        check(status == 404)
    except Exception:
        check(False)

    # T11: Create with invalid email → 400
    try:
        _, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "X", "email": "notanemail"},
        )
        check(status == 400)
    except Exception:
        check(False)

    # T12: Update non-existent → 404
    try:
        _, status = module.update_user(  # type: ignore[attr-defined]
            999,
            {"name": "Ghost"},
        )
        check(status == 404)
    except Exception:
        check(False)

    # ── Scenario 3: Input validation (baseline fails) ────────────────────
    reset()

    # T13: Missing required field → 400 (baseline: bare except → 500)
    try:
        _, status = module.create_user(  # type: ignore[attr-defined]
            {"email": "x@example.com"},
        )
        check(status == 400)
    except Exception:
        check(False)

    # T14: Empty name → 400 (baseline: allows empty string)
    try:
        _, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "", "email": "y@example.com"},
        )
        check(status == 400)
    except Exception:
        check(False)

    # T15: Whitespace-only name → 400
    try:
        _, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "   ", "email": "z@example.com"},
        )
        check(status == 400)
    except Exception:
        check(False)

    # T16: Negative age → 400
    try:
        _, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "Test", "email": "t@example.com", "age": -5},
        )
        check(status == 400)
    except Exception:
        check(False)

    # ── Scenario 4: Data integrity (baseline fails) ──────────────────────
    reset()

    # T17: Duplicate email → reject (400 or 409)
    try:
        module.create_user(  # type: ignore[attr-defined]
            {"name": "Alice", "email": "alice@example.com"},
        )
        _, status = module.create_user(  # type: ignore[attr-defined]
            {"name": "Alice2", "email": "alice@example.com"},
        )
        check(status in (400, 409))
    except Exception:
        check(False)

    # T18: Update email to another user's email → reject (400 or 409)
    try:
        reset()
        module.create_user(  # type: ignore[attr-defined]
            {"name": "Alice", "email": "alice@example.com"},
        )
        module.create_user(  # type: ignore[attr-defined]
            {"name": "Bob", "email": "bob@example.com"},
        )
        _, status = module.update_user(  # type: ignore[attr-defined]
            2,
            {"email": "alice@example.com"},
        )
        check(status in (400, 409))
    except Exception:
        check(False)

    # ── Scenario 5: Response consistency (baseline fails) ────────────────
    reset()

    # T19: Update response includes created_at
    try:
        module.create_user(  # type: ignore[attr-defined]
            {"name": "Alice", "email": "alice@example.com", "age": 30},
        )
        res, status = module.update_user(  # type: ignore[attr-defined]
            1,
            {"age": 31},
        )
        check(status == 200 and "created_at" in res)
    except Exception:
        check(False)

    # T20: List response includes created_at for each user
    try:
        res, status = module.list_users({})  # type: ignore[attr-defined]
        check(status == 200 and len(res) > 0 and all("created_at" in u for u in res))
    except Exception:
        check(False)

    return sum(results), len(results)


def check_correctness() -> tuple[float, int, int]:
    """Run hidden test suite.  Returns (score, passed, total)."""
    module = _load_module()
    if module is None:
        return 0.0, 0, 0
    try:
        passed, total = _run_tests(module)
    except Exception:
        return 0.0, 0, 0
    score = passed / total if total > 0 else 0.0
    return score, passed, total


# ---------------------------------------------------------------------------
# Sub-score: lint (ruff)
# ---------------------------------------------------------------------------


def check_lint() -> tuple[int, float]:
    """Return (violation_count, score).  20 violations → score 0."""
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                str(ARTIFACT_PATH),
                "--output-format=json",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        violations_data = json.loads(proc.stdout) if proc.stdout.strip() else []
        violations = len(violations_data)
        score = max(0.0, 1.0 - (violations / LINT_VIOLATION_SCALE))
        return violations, score
    except Exception:
        return 0, 0.0


# ---------------------------------------------------------------------------
# Sub-score: type coverage (AST inspection)
# ---------------------------------------------------------------------------


def check_type_coverage() -> float:
    """Fraction of function params + returns that carry type annotations."""
    try:
        source = ARTIFACT_PATH.read_text()
        tree = ast.parse(source)

        total_functions = 0
        annotated_returns = 0
        total_params = 0
        annotated_params = 0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            total_functions += 1
            if node.returns is not None:
                annotated_returns += 1

            args = node.args
            all_args = (
                args.posonlyargs
                + args.args
                + args.kwonlyargs
                + ([args.vararg] if args.vararg else [])
                + ([args.kwarg] if args.kwarg else [])
            )
            non_self_args = [a for a in all_args if a.arg != "self"]
            total_params += len(non_self_args)
            annotated_params += sum(
                1 for a in non_self_args if a.annotation is not None
            )

        if total_functions == 0:
            return 0.0

        return_coverage = annotated_returns / total_functions
        param_coverage = annotated_params / total_params if total_params > 0 else 1.0
        return (return_coverage + param_coverage) / 2.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def main() -> None:
    correctness, passed, total = check_correctness()
    lint_violations, lint_score = check_lint()
    type_coverage = check_type_coverage()

    composite = 0.5 * correctness + 0.3 * lint_score + 0.2 * type_coverage

    print(f"correctness: {correctness:.4f} ({passed}/{total})")
    print(f"lint_violations: {lint_violations}")
    print(f"lint_score: {lint_score:.2f}")
    print(f"type_coverage: {type_coverage:.2f}")
    print(f"composite: {composite:.3f}")


if __name__ == "__main__":
    main()
