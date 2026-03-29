# Features

## Core

- **Scope enforcement** — declare what the agent can and cannot modify. Path traversal attempts and absolute paths are rejected. Violations are reverted automatically.
- **Auto-staging** — untracked artifact files are automatically copied into the worktree and committed on the anneal branch during registration. No manual `git add` required.
- **In-place mode** — `--in-place` flag skips worktree creation for local-only artifacts. Uses file backup for rollback instead of git. Useful for skills, configs, and files not in version control.
- **Knowledge compounding** — experiment history + consolidated learnings + cross-condition insights available for agent context. Per-criterion feedback (PASS/FAIL per criterion) helps the agent target specific weaknesses.
- **Cost control** — per-experiment and daily budget caps. Pricing loaded from `~/.anneal/pricing.toml` with hardcoded defaults. Local models tracked at $0.
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

## Knowledge System

- **TF-IDF similarity retrieval** — IDF-weighted cosine similarity for hypothesis matching.
- **Embedding-based retrieval** (optional) — sentence-transformer embeddings with lazy loading and TF-IDF fallback.
- **Domain-aware learning transfer** — cross-domain learnings are penalized by a configurable factor, preventing negative transfer.
- **Criterion delta exposure** — learning summaries show per-criterion improvements/regressions, not just aggregate score deltas.
- **MAP-Elites archive** — quality-diversity archive maintaining best solutions per behavioral region.

## Operations

- **Meta-optimization** — two complementary timescales: (1) policy agent rewrites mutation instructions every N experiments (continuous), (2) plateau-triggered program.md rewriting (episodic).
- **Stale lock recovery** — scheduler detects and removes lock files older than 1 hour from crashed runners.
- **Concurrent consolidation safety** — check-and-act consolidation is atomic under FileLock.
- **Live dashboard** — `anneal dashboard` reads from `.anneal/` directory. No coupling to the runner process.
