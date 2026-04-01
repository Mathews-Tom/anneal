# Contributing to anneal

## Development Setup

```bash
uv sync --dev
uv run pytest tests/ -x -q
```

Python 3.12 or 3.13 is required.

## Architecture Overview

See [docs/architecture.md](docs/architecture.md) for the full module map and component relationships.

Key modules:

- `anneal/engine/runner.py` — experiment loop, worktree lifecycle, scope enforcement
- `anneal/engine/eval.py` — deterministic and stochastic evaluators, bootstrap CI
- `anneal/engine/agent.py` — Claude Code subprocess invocation with tool restrictions
- `anneal/engine/safety.py` — pre-experiment cost estimation and budget enforcement
- `anneal/engine/scope.py` — scope.yaml parsing, path validation, immutable file hashing
- `anneal/engine/search.py` — search strategies (greedy, SA, Thompson sampling, Bayesian)
- `anneal/engine/types.py` — all shared Pydantic types and dataclasses

## Where to Start

Good entry points for new contributors:

- Issues labeled `good-first-issue`
- Eval criteria templates in `anneal/templates/` (low risk, high impact)
- Documentation improvements in `docs/`
- Provider adapter support (new LLM backends via the OpenAI-compatible client)
- New search strategies implementing the `SearchStrategy` protocol

## Code Style

- Type hints on every function and class — Python 3.12+ built-in generics (`list[str]`, `dict[str, int]`), `|` unions, no `Optional`
- `from __future__ import annotations` at the top of every module
- Pydantic v2 for all data models
- No global mutable state, no mutable default arguments
- `pytest` for tests; `pytest-asyncio` for async tests (mode is `auto`)
- Raise specific exceptions with descriptive messages — no bare `except`, no silent failures

## Testing

Run the full test suite:

```bash
uv run pytest tests/ -x -q
```

Run with coverage:

```bash
uv run pytest tests/ --cov=anneal --cov-report=term-missing
```

Run a specific test file:

```bash
uv run pytest tests/test_scope.py -x -q
```

Coverage thresholds:

- Overall: 80% minimum
- New or modified code: 90% minimum
- Critical paths (scope enforcement, cost estimation, eval correctness): 95% minimum

All new features require tests. Follow the Arrange-Act-Assert pattern. Test names use the convention `test_<unit>_<scenario>_<expected_outcome>`.

## CI

CI runs on every push and pull request via `.github/workflows/ci.yml`:

- Tests on Python 3.12 and 3.13
- Coverage report uploaded as an artifact
- Benchmarks run after tests pass

All CI checks must pass before a PR is merged.

## PR Process

1. Fork the repository and create a branch: `git checkout -b feat/my-feature`
2. Implement the change with tests
3. Verify locally: `uv run pytest tests/ -x -q`
4. Open a pull request against `main`
5. All tests must pass in CI
6. New features require tests — PRs without tests for new behavior will not be merged
7. Breaking changes (modified public APIs, removed CLI flags, schema changes) require discussion in a GitHub issue before implementation
8. Use conventional commits: `type(scope): description`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`
   - Example: `feat(eval): add composite scoring with constraint thresholds`

## License

Apache-2.0. Contributions are accepted under the same license.
