# Changelog

## [0.2.0] - 2026-03-24

### Features

- **Tree Search**: UCB tree search with pruning and persistence for systematic prompt-space exploration
- **Policy Agent**: Continuous instruction rewriting agent that adapts optimization strategy mid-run
- **Failure Taxonomy**: LLM-powered failure classification with blind spot detection and distribution tracking
- **Multi-Draft Generation**: Parallel draft generation with diff capture/apply for broader search
- **Verification Layer**: Verifier gates that validate candidate prompts before acceptance
- **Random Restarts**: SA-temperature-linked restart mechanism to escape local optima
- **MCP Server**: Scaffold with status, history, and list tools for external integrations
- **Eval Environment**: Config with env var validation, lifecycle management, retry and flake detection
- **Loop Persistence**: State persistence across process restarts with auto-resume
- **Knowledge Auto-Enable**: Automatic knowledge context activation after 20 KEPT experiments
- **Dry-Run Cost Preview**: `--dry-run` flag for cost estimation before committing to a run
- **Tiktoken Counting**: Replace character heuristic with tiktoken for accurate token counting
- **CI Status JSON**: Enriched `anneal status --json` output for CI/CD consumption

### CLI

- `--verifier` and `--restart-probability` flags
- `--failure-categories` and `--n-drafts` flags
- Policy agent and UCB tree search flags

### Fixes

- Derive agent_model from target in early record creation
- Create parent directory before saving loop state
- Catch API timeout and connection errors in stochastic eval
- Remove module-level log handler that caused duplicate notifications
- Replace deprecated `asyncio.get_event_loop()` with `asyncio.run()`
- Fix flaky learning pool tests with explicit seeds and structural checks

### Refactoring

- Migrate type system from dataclasses to Pydantic BaseModel
- Replace hand-rolled TOML serialization with Pydantic model_dump + tomli-w
- Extract pipeline stages from `run_one` into discrete functions
- Deduplicate 7 exit paths via `_make_early_record` extraction
- Simplify knowledge store boolean logic and naming
- Remove dead code, backward-compat aliases, and unused imports

### Build & CI

- Add GitHub Actions workflow for tests and benchmarks
- Add pydantic, tomli-w, tiktoken dependencies
- Move aiohttp to optional dashboard dependency group
- Add build-system to make package installable

### Documentation

- Update README, system design, and overview for research-driven enhancements
- Add archetype recipes for common optimization targets
- Add case study results from live experiment runs
- Add standalone Installation section to README

## [0.1.1] - 2025-12-01

Initial patch release.

## [0.1.0] - 2025-11-15

Initial release.
