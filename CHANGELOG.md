# Changelog

## [0.4.0] - 2026-04-01

### Features

- **`anneal validate` command**: Run a single eval pass on the current artifact to verify eval setup before committing to a full run. Reports baseline score, bootstrap CI, per-criterion breakdown, and health assessment
- **`anneal suggest` positional form**: Accepts a natural-language description as a positional argument alongside the existing `--problem` flag
- **Interactive cost confirmation**: Prompts when estimated cost exceeds a configurable threshold ($5.00 default). Skip with `--yes` / `-y` for CI/automation. Threshold configurable via `.anneal/config.toml`
- **Per-experiment cost display**: Each experiment prints a running cost line: `[exp N/M] score: X -> Y (+delta) | cost: $A / $B budget`
- **`--template` flag on `anneal register`**: Load eval criteria and defaults from named templates. Explicit CLI args override template values. List available templates with `anneal templates`
- **HybridSearch default**: New default search strategy — greedy for the first 10 experiments, then simulated annealing. Override with `--search greedy` or any explicit strategy flag
- **Typed evaluator protocol**: `Evaluator` protocol and `EvalResult` dataclass in `anneal/eval_protocol.py` for plugin-based evaluation via Python callables. `load_evaluator("module.py:callable")` supports both file paths and dotted module specs
- **Eval criteria templates**: Five TOML templates shipped in `anneal/templates/` — prompt-quality, code-readability, test-coverage, api-latency, documentation

### Build

- Move matplotlib from core to `[dashboard]` optional dependency group (zero imports in anneal package)
- Add Python 3.12/3.13 classifiers, license, and OS classifiers to package metadata
- Document tiktoken dependency trade-off inline

### Documentation

- **README restructured** for adopters: badges, pitch, quick start, provider table, eval modes (collapsible), use-case tables with cost estimates, results section with case study
- **Provider support table**: Surfaces existing multi-provider support (Anthropic, OpenAI, Google, Ollama, LM Studio, any OpenAI-compatible)
- **Results section**: Code-golf case study (93.7% reduction, 3592 to 228 bytes) with score trajectory SVG
- **CONTRIBUTING.md**: Dev setup, module map, code style, testing, PR process
- **SECURITY.md**: Threat model and isolation boundaries verified against code
- **Dockerfile + .dockerignore**: python:3.12-slim, non-root user, uv install
- **4 ADRs**: Git worktrees for isolation, Wilcoxon over t-test, artifact-eval-agent triplet, scope enforcement design
- **Complexity tiers guide**: Three-tier search strategy documentation (simple/intermediate/advanced)
- **Composite scoring docs**: Guard-rail constraints for preventing Goodhart's Law failures in eval-guide.md

### Fixes

- Correct license from MIT to Apache-2.0 in README
- Regenerate infographic PNG at 2x retina resolution

## [0.3.0] - 2026-04-01

### Features

- **Research Operator**: External knowledge injection triggered on optimization plateaus. Queries an LLM for technique suggestions and injects them as advisory context hints. Plateau-gated, budget-capped, and self-disabling after consecutive failures
- **Island-Based Population Search**: Multiple independent population islands with round-robin experiment assignment and periodic best-candidate migration. `island_count=1` (default) preserves standard population search behavior
- **Strategy Manifest**: Structured YAML strategy with named components and component-level evolution. Weakest components are rewritten independently instead of rewriting the entire strategy
- **Two-Phase Mutation**: Structured diagnosis step identifies weakest criteria and root cause before generating a targeted fix. Diagnosis model can be cheaper than the mutation model
- **Episodic Memory**: Structured `Lesson` objects extracted from each experiment capturing changes, improvements/regressions, and transferable insights
- **Eval Consistency Monitoring**: Per-criterion score variance tracking over sliding windows with periodic consistency reports for evaluator drift detection
- **Adaptive Sample Sizing**: Dynamically extends or early-stops stochastic evaluation based on observed effect size, reducing eval cost for clear outcomes
- **Context Compression**: Three compression modes (none, moderate, aggressive) trade history detail for token budget. Aggressive mode deduplicates per-criterion trends
- **Lineage Tracking**: Traces the chain of accepted mutations leading to the current best artifact, providing causal context for the agent
- **Dual-Agent Selector**: Thompson Sampling meta-strategy that adaptively selects between exploration and exploitation agent configurations
- **Pareto Front Persistence**: Pareto front saved to JSON for dashboard visualization after each experiment
- **Component Evolution**: Strategy manifest components track streak-without-improvement and trigger targeted rewriting when stalled

### CLI

- `--island-count` and `--migration-interval` flags for island-based population search
- `--research-model` and `--research-budget` flags for research operator configuration

### Refactoring

- Extract `ExperimentContext` to reduce pipeline parameter lists across runner methods
- Deduplicate hypothesis loading with shared loader function and use numpy for variance calculation
- Deduplicate criterion scoring logic in stochastic evaluator
- Wire Holm-Bonferroni correction into all search strategies
- Integrate EvalCache into stochastic evaluation pipeline
- Add search_strategy config field and strategy factory method

### Fixes

- Resolve unawaited coroutine warnings in eval tests
- Resolve CI failures from optional dashboard dependency and serialization order
- Reset consecutive failure counter on `anneal resume`
- Detect Claude Code error responses and scale eval budgets correctly
- Use temperature 1.0 in taxonomy classifier and judgment for gpt-5 compatibility
- Add gpt-5.4-mini pricing and deduplicate cost warnings
- Fail fast on missing artifact files during registration

### Documentation

- Add CI integration guide with GitHub Actions workflow
- Improve pricing discoverability and unknown model guidance
- Document auto-staging and in-place optimization mode
- Update architecture module map and experiment loop description
- Update features list with all new capabilities

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
