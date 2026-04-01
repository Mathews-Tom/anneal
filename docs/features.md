# Features

## Core

- **Scope enforcement** — declare what the agent can and cannot modify. Path traversal attempts and absolute paths are rejected. Violations are reverted automatically.
- **Auto-staging** — untracked artifact files are automatically copied into the worktree and committed on the anneal branch during registration. No manual `git add` required.
- **In-place mode** — `--in-place` flag skips worktree creation for local-only artifacts. Uses file backup for rollback instead of git. Useful for skills, configs, and files not in version control.
- **Knowledge compounding** — experiment history + consolidated learnings + cross-condition insights available for agent context. Per-criterion feedback (PASS/FAIL per criterion) helps the agent target specific weaknesses.
- **Cost control** — per-experiment and daily budget caps. Pricing loaded from `~/.anneal/pricing.toml` with hardcoded defaults. Local models tracked at $0. See [Pricing Configuration](#pricing-configuration) below.
- **Safety** — process group time-boxing (SIGKILL), consecutive failure halting, disk space checks, JSONL corruption recovery, git fsck integrity checks after kill recovery.
- **Graceful shutdown** — `anneal stop --target <id>` writes a stop file; the runner exits cleanly after the current experiment.
- **Verification gates** — binary pass/fail commands that run after scope enforcement, before eval. Discard mutations that fail structural checks without spending eval budget.
- **Failure taxonomy** — LLM-based classification of failed experiments into structured categories with blind spot detection.
- **Multi-draft mutation** — generate N candidate mutations per cycle with temperature variation. Per-draft verifier pruning selects the best survivor.
- **Random restart** — probabilistic fresh-start experiments that escape local optima. SA temperature-linked decay reduces restart probability as search converges.
- **Policy agent** — continuous meta-optimizer that rewrites mutation instructions between experiments based on failure patterns.

## Statistical Rigor

- **Wilcoxon signed-rank test** for stochastic comparison with minimum sample size guard (n ≥ 6); falls back to Cohen's d effect-size threshold for small samples.
- **Holm-Bonferroni correction** adjusts acceptance threshold across the consolidation window, reducing false positive rate by ~86% on null distributions.
- **Bootstrap confidence intervals** with deterministic seeding (float precision normalized for cross-platform reproducibility).
- **Held-out evaluation** with two-tier divergence detection: 10% warning, 25% critical.
- **Sliding window drift detection** compares first-half vs second-half variance within consolidation windows to catch temporal drift.

## Search Strategies

- **Greedy** (default) — accept only strict improvements, verified by statistical test.
- **Simulated annealing** — adaptive cooling with reheat when acceptance drops below target rate. Escapes local optima early, converges to greedy behavior over time.
- **Population-based** — tournament selection with LLM-guided crossover. Top candidates' hypotheses are combined into crossover prompts.
- **Island-based population** — wraps multiple independent population islands with round-robin experiment assignment. Periodic migration copies each island's best candidate to all other islands, balancing solution diversity with cross-pollination. `island_count=1` (default) falls back to standard population search.
- **Pareto** — multi-objective search over per-criterion score vectors. Maintains a Pareto front; non-dominated trade-off solutions are preserved.
- **Thompson Sampling** — contextual bandit meta-strategy that adaptively selects between search algorithms based on observed reward.
- **Bayesian surrogate** — Gaussian Process model predicts mutation quality from experiment history. Expected Improvement acquisition balances exploration and exploitation. Requires optional `scikit-learn`.
- **UCB tree search** — maps experiment history to a tree of git commits. Selects the most promising ancestor to branch from via UCB1. Supports subtree pruning and crash recovery.

## Evaluation Intelligence

- **Per-criterion structured feedback** — agents see which criteria passed/failed, not just the aggregate score.
- **Position debiasing** — when votes ≥ 2, splits between forward and reverse criterion orderings to cancel LLM judge position bias.
- **Bradley-Terry comparison** — calibrated Bayesian strength estimation with early stopping.
- **Eval result caching** — content-hash LRU cache avoids re-evaluating identical artifact content.
- **Multi-fidelity pipeline** — cheap deterministic stages filter out bad mutations before expensive stochastic evaluation.
- **Eval consistency monitoring** — tracks per-criterion score variance over sliding windows. Generates consistency reports at configurable intervals to detect evaluator drift.

## Knowledge System

- **TF-IDF similarity retrieval** — IDF-weighted cosine similarity for hypothesis matching.
- **Embedding-based retrieval** (optional) — sentence-transformer embeddings with lazy loading and TF-IDF fallback.
- **Domain-aware learning transfer** — cross-domain learnings are penalized by a configurable factor, preventing negative transfer.
- **Criterion delta exposure** — learning summaries show per-criterion improvements/regressions, not just aggregate score deltas.
- **MAP-Elites archive** — quality-diversity archive maintaining best solutions per behavioral region.
- **Episodic memory** — structured `Lesson` objects extracted from each experiment capturing what changed, what improved/regressed, and transferable insights. Stored in experiment records for cross-experiment learning.
- **Lineage tracking** — traces the chain of accepted mutations leading to the current best artifact, providing causal context for the agent.

## Operations

- **Meta-optimization** — two complementary timescales: (1) policy agent rewrites mutation instructions every N experiments (continuous), (2) plateau-triggered program.md rewriting (episodic).
- **Research operator** — external knowledge injection triggered on optimization plateaus. Queries an LLM for technique suggestions and injects them as advisory context hints into mutation prompts. Plateau-gated, budget-capped, and self-disabling after consecutive failures.
- **Strategy manifest** — structured YAML strategy with named components. Component-level evolution rewrites the weakest component when progress stalls, instead of rewriting the entire strategy.
- **Two-phase mutation** — structured diagnosis step identifies weakest criteria and root cause before generating a targeted mutation. Diagnosis model can be cheaper than the mutation model.
- **Context compression** — three modes (none, moderate, aggressive) that trade history detail for token budget. Aggressive mode deduplicates per-criterion trends across experiments.
- **Adaptive sample sizing** — dynamically extends or early-stops stochastic evaluation based on observed effect size. Reduces eval cost for clear winners and losers.
- **Stale lock recovery** — scheduler detects and removes lock files older than 1 hour from crashed runners.
- **Concurrent consolidation safety** — check-and-act consolidation is atomic under FileLock.
- **Live dashboard** — `anneal dashboard` reads from `.anneal/` directory. No coupling to the runner process.

## Pricing Configuration

Anneal ships with hardcoded pricing for common models (GPT-4.1, GPT-5, Gemini 2.5, Claude Sonnet/Opus/Haiku). For new or custom models, create `~/.anneal/pricing.toml`:

```toml
# Prices in USD per million tokens (MTok)
[models."gpt-5.4-mini"]
input = 1.0
output = 4.0

[models."my-custom-model"]
input = 0.5
output = 2.0
```

User overrides merge with hardcoded defaults — you only need to add models that aren't built in. Local models (`ollama/*`, `lmstudio/*`, `local/*`) are always tracked at $0 regardless of pricing config.

If a model has no pricing data, anneal uses a conservative default ($2.00/$8.00 per MTok) and logs a warning with instructions to add the model to `pricing.toml`.
