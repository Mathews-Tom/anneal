# anneal

Autonomous optimization for any measurable artifact.

<video src="https://github.com/user-attachments/assets/cb0cd879-444e-444b-8b69-32623ff9af76" width="100%" autoplay loop muted playsinline></video>

Define _what_ to improve, _how to measure_ improvement, and _what's allowed to change_. An LLM agent handles the rest — generating hypotheses, running experiments, keeping winners, discarding losers, compounding learnings. Overnight. Unattended.

```text
(Artifact, Eval, Agent) → continuous improvement
```

## How It Works

1. **Register** a target: artifact files, evaluation command or criteria, scope constraints
2. **Run** the optimization loop: agent mutates → eval scores → keep or revert → learn → repeat
3. **Review** results: experiment history, score trajectories, cost tracking

Every experiment is git-committed. Every mutation is scope-enforced. Every decision is logged.

```bash
anneal init
anneal register \
  --name api-perf \
  --artifact src/api/handler.py \
  --eval-mode deterministic \
  --run-cmd "wrk -t4 -c100 -d10s http://localhost:8080/api | grep Latency" \
  --parse-cmd "awk '{print \$2}'" \
  --direction minimize \
  --scope scope.yaml

anneal run --target api-perf --experiments 50
anneal dashboard
```

## Two Evaluation Modes

**Deterministic** — a shell command produces a number. Run code, parse output, compare.

```text
eval: pytest --cov=src | grep TOTAL | awk '{print $4}' → 72.3
```

**Stochastic** — an LLM judges N samples against K binary criteria. Supports two comparison modes:

- **Majority voting** (default) — 3 votes per judgment, binary YES/NO, Wilcoxon signed-rank test for comparison (with Cohen's d fallback for n < 6 paired samples)
- **Bradley-Terry** — Bayesian Beta estimation with Laplace smoothing, calibrated uncertainty, early stopping when the 95% CI clears the 0.5 threshold (saves 1–8 API calls per criterion per sample)

Bootstrap CI provides variance estimates. Position debiasing splits votes between forward and reverse criterion orderings when votes ≥ 2.

```text
criteria:
  - "Is the text scannable?" (YES/NO)
  - "Are all claims cited?" (YES/NO)
samples: 10 test prompts × 4 criteria → score with confidence interval
```

## Where Anneal Works

The system works when: (1) the artifact is a text file in a git repo, (2) quality is measurable as a scalar (or per-criterion vector for Pareto optimization), and (3) the feedback loop completes in under ~10 minutes.

| Use Case                             | Eval Mode     | Feedback Speed   | Verdict         |
| ------------------------------------ | ------------- | ---------------- | --------------- |
| **Prompt / SKILL.md optimization**   | stochastic    | 1–3 min          | Works perfectly |
| **API response time**                | deterministic | 2–5 min          | Works well      |
| **Bundle size reduction**            | deterministic | 1–3 min          | Works well      |
| **ML training config**               | deterministic | 5–15 min (proxy) | Works well      |
| **RAG retrieval prompts**            | deterministic | 1–3 min          | Works well      |
| **Documentation quality**            | stochastic    | 2–5 min          | Works well      |
| **Multi-agent system prompts**       | stochastic    | 2–5 min          | Works well      |
| **Guardrail / safety filter tuning** | deterministic | 1–2 min          | Works well      |
| **Config tuning** (build, infra, db) | deterministic | 1–10 min         | Works well      |
| **Eval rubric calibration**          | deterministic | 1–2 min          | Works well      |
| **Test coverage improvement**        | deterministic | 2–5 min          | Works well      |
| **Data preprocessing pipeline**      | deterministic | 5–15 min         | Works well      |

## Where Anneal Does NOT Work

| Use Case                                   | Reason                                                                       |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| **Non-git projects / binary artifacts**    | Git worktrees + text diffs are the mutation mechanism.                       |
| **Live system tuning (real-time metrics)** | Evaluates at experiment end, not continuously.                               |
| **Database schema migrations**             | Multi-step stateful operations, not file edits.                              |
| **Cross-service distributed optimization** | Targets are scoped to one repo, one worktree.                                |
| **Embedding model selection**              | Re-embedding a corpus isn't a file edit. One-shot comparison, not iterative. |
| **Inter-agent protocol changes**           | Requires coordinated edits across multiple files simultaneously.             |

## Architecture

```text
anneal/engine/
  runner.py            # Experiment state machine (mutate → eval → decide → log)
  eval.py              # Deterministic + stochastic eval, Bradley-Terry, position debiasing
  eval_cache.py        # Content-hash LRU cache for eval results
  search.py            # Greedy, simulated annealing (adaptive), population (crossover), Pareto
  bayesian.py          # GP surrogate model for mutation ranking (optional scikit-learn)
  strategy_selector.py # Thompson Sampling meta-strategy over search algorithms
  archive.py           # MAP-Elites quality-diversity archive
  agent.py             # LLM mutation (Claude Code subprocess or API mode)
  scope.py             # Editable/immutable enforcement with path traversal protection
  knowledge.py         # JSONL experiment store, TF-IDF/embedding retrieval, sliding window drift
  learning_pool.py     # Cross-condition/target/project knowledge transfer with domain filtering
  context.py           # Token budget assembly with per-criterion feedback formatting
  environment.py       # Git worktree management with fsck integrity checks
  safety.py            # Budget caps, failure limits, disk checks, process time-boxing
  client.py            # Multi-provider LLM client with configurable pricing (TOML overlay)
  scheduler.py         # Sequential target scheduler with stale lock recovery
  taxonomy.py          # Failure classification: LLM-based categorization, distribution, blind spots
  tree_search.py       # UCB tree search: backtracking, pruning, persistence, history bootstrap
  policy_agent.py      # Policy agent: continuous instruction rewriting, reward tracking
  registry.py          # Target configuration (config.toml persistence)
  dashboard.py         # File-based SSE live dashboard
```

## Key Features

### Core

- **Scope enforcement** — declare what the agent can and cannot modify. Path traversal attempts and absolute paths are rejected. Violations are reverted automatically.
- **Knowledge compounding** — experiment history + consolidated learnings + cross-condition insights available for agent context. Per-criterion feedback (PASS/FAIL per criterion) helps the agent target specific weaknesses.
- **Cost control** — per-experiment and daily budget caps. Pricing loaded from `~/.anneal/pricing.toml` with hardcoded defaults. Local models tracked at $0.
- **Safety** — process group time-boxing (SIGKILL), consecutive failure halting, disk space checks, JSONL corruption recovery, git fsck integrity checks after kill recovery.
- **Graceful shutdown** — `anneal stop --target <id>` writes a stop file; the runner exits cleanly after the current experiment.
- **Verification gates** — binary pass/fail commands that run after scope enforcement, before eval. Discard mutations that fail structural checks without spending eval budget. Stderr captured for diagnosis.
- **Failure taxonomy** — LLM-based classification of failed experiments into structured categories (output_format, logic_error, regression, etc.) with blind spot detection for unattributed failure modes.
- **Multi-draft mutation** — generate N candidate mutations per cycle with temperature variation. Per-draft verifier pruning selects the best survivor. Budget split evenly across drafts.
- **Random restart** — probabilistic fresh-start experiments that escape local optima. SA temperature-linked decay reduces restart probability as search converges.
- **Policy agent** — continuous meta-optimizer that rewrites mutation instructions between experiments based on failure patterns. Complements the plateau-triggered program.md rewriting at a faster cadence (~$0.001/call).

### Statistical Rigor

- **Wilcoxon signed-rank test** for stochastic comparison with minimum sample size guard (n ≥ 6); falls back to Cohen's d effect-size threshold for small samples.
- **Holm-Bonferroni correction** adjusts acceptance threshold across the consolidation window, reducing false positive rate by ~86% on null distributions.
- **Bootstrap confidence intervals** with deterministic seeding (float precision normalized for cross-platform reproducibility).
- **Held-out evaluation** with two-tier divergence detection: 10% warning, 25% critical (possible evaluator compromise).
- **Sliding window drift detection** compares first-half vs second-half variance within consolidation windows to catch temporal drift.

### Search Strategies

- **Greedy** (default) — accept only strict improvements, verified by statistical test.
- **Simulated annealing** — adaptive cooling with reheat when acceptance drops below target rate. Escapes local optima early, converges to greedy behavior over time.
- **Population-based** — tournament selection with LLM-guided crossover. Top candidates' hypotheses are combined into crossover prompts.
- **Pareto** — multi-objective search over per-criterion score vectors. Maintains a Pareto front; non-dominated trade-off solutions are preserved.
- **Thompson Sampling** — contextual bandit meta-strategy that adaptively selects between search algorithms based on observed reward.
- **Bayesian surrogate** — Gaussian Process model predicts mutation quality from experiment history. Expected Improvement acquisition balances exploration and exploitation. Requires optional `scikit-learn` dependency.
- **UCB tree search** — maps experiment history to a tree of git commits. Selects the most promising ancestor to branch from via UCB1 (balancing exploitation and exploration). Supports subtree pruning and crash recovery via JSON persistence.

### Evaluation Intelligence

- **Per-criterion structured feedback** — agents see which criteria passed/failed, not just the aggregate score.
- **Position debiasing** — when votes ≥ 2, splits between forward and reverse criterion orderings to cancel LLM judge position bias at zero additional API cost.
- **Bradley-Terry comparison** — calibrated Bayesian strength estimation with early stopping, replacing binary majority voting when configured.
- **Eval result caching** — content-hash LRU cache avoids re-evaluating identical artifact content + criteria combinations.
- **Multi-fidelity pipeline** — cheap deterministic stages filter out bad mutations before expensive stochastic evaluation. Constraint pre-checks also run before eval.

### Knowledge System

- **TF-IDF similarity retrieval** — IDF-weighted cosine similarity replaces word-level Jaccard for hypothesis matching.
- **Embedding-based retrieval** (optional) — sentence-transformer embeddings with lazy loading and TF-IDF fallback when `sentence-transformers` is not installed.
- **Domain-aware learning transfer** — cross-domain learnings are penalized by a configurable factor, preventing negative transfer between unrelated optimization targets.
- **Criterion delta exposure** — learning summaries show per-criterion improvements/regressions, not just aggregate score deltas.
- **MAP-Elites archive** — quality-diversity archive maintaining best solutions per behavioral region, enabling warm-starting and trade-off exploration.

### Operations

- **Meta-optimization** — two complementary timescales: (1) policy agent rewrites mutation instructions every N experiments (continuous, ~$0.001/call), (2) plateau-triggered program.md rewriting when M consecutive experiments fail (episodic).
- **Stale lock recovery** — scheduler detects and removes lock files older than 1 hour from crashed runners.
- **Concurrent consolidation safety** — check-and-act consolidation is atomic under FileLock.
- **Live dashboard** — `anneal dashboard` reads from `.anneal/` directory. No coupling to the runner process.

## Installation

```bash
# From PyPI
uv tool install anneal-cli

# With ML extras (Bayesian surrogate, optional)
uv tool install anneal-cli --with scikit-learn

# Or with pip
pip install anneal-cli
```

Requires Python 3.12+. The `anneal` command is available globally after installation.

## Quick Start

```bash
# Initialize in a git repo
anneal init

# Register a deterministic target
anneal register \
  --name my-target \
  --artifact path/to/file.py \
  --eval-mode deterministic \
  --run-cmd "python benchmark.py" \
  --parse-cmd "grep 'score' | awk '{print \$2}'" \
  --direction maximize \
  --scope scope.yaml

# Register with verification gates and restart
anneal register \
  --name my-target \
  --artifact path/to/file.py \
  --eval-mode deterministic \
  --run-cmd "python benchmark.py" \
  --parse-cmd "grep 'score' | awk '{print \$2}'" \
  --direction maximize \
  --scope scope.yaml \
  --verifier "typecheck:python -m mypy path/to/file.py" \
  --verifier "lint:ruff check path/to/file.py" \
  --restart-probability 0.05 \
  --n-drafts 3 \
  --policy-model gpt-4.1-mini

# Run 20 experiments
anneal run --target my-target --experiments 20

# Stop gracefully
anneal stop --target my-target

# Monitor
anneal status --target my-target
anneal history --target my-target
anneal dashboard --open
```

## Testing

```bash
# Run all tests (492 tests)
uv run pytest tests/ -x -q

# Run with coverage
uv run pytest tests/ --cov=anneal --cov-report=term-missing

# Run e2e tests only
uv run pytest tests/test_e2e.py -v

# Run validation benchmarks
uv run python benchmarks/bench_false_positives.py
uv run python benchmarks/bench_sa_convergence.py
uv run python benchmarks/bench_retrieval_precision.py
```

## Project Status

492 tests passing. 3 validation benchmarks passing.

### Complete

- Core engine (git worktrees, scope enforcement, registry, agent invoker, eval engine, runner state machine)
- Production hardening (safety layer, knowledge store, learning pool, notifications, JSONL recovery)
- Multi-target orchestration, context budget assembly, rate limiting, background daemon
- Search strategies: greedy, simulated annealing (adaptive), population (crossover), Pareto, Thompson Sampling, Bayesian surrogate
- Statistical rigor: Wilcoxon guard, Holm-Bonferroni correction, criterion name tracking, divergence thresholds
- Evaluation intelligence: per-criterion feedback, position debiasing, Bradley-Terry comparison, eval caching, multi-fidelity pipeline
- Knowledge upgrades: TF-IDF retrieval, embedding-based retrieval (optional), domain-aware transfer, criterion delta summaries
- Operational hardening: `anneal stop`, git fsck, constraint pre-check ordering, pricing externalization
- Quality-diversity archive (MAP-Elites), stale lock recovery, concurrent consolidation safety
- File-based live dashboard, deployment-tier approval gates, meta-optimization
- End-to-end test suite, validation benchmark suite
- Research-driven enhancements: verification gates, failure taxonomy, multi-draft mutation, random restart, UCB tree search, policy agent

### Planned

- Adaptive draft count — auto-adjust `n_drafts` based on per-draft survival rate
- Population immigration — restart mutations enter population search via tournament selection
- Cross-enhancement runner integration tests for multi-draft + tree search + policy agent running simultaneously
