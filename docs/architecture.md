# Architecture

## Module Map

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

## The Experiment Loop

```text
Load context (artifact + history + learnings)
  → Generate hypothesis + mutation
  → Checkpoint (git commit)
  → Execute (time-boxed)
  → Evaluate (score output)
  → Keep or discard (statistical test)
  → Log learnings
  → Consolidate (periodic)
  → Repeat
```

Every experiment is git-committed. Every mutation is scope-enforced. Every decision is logged.

## Design Principles

1. **Artifact-agnostic** — operates on the (Artifact, Eval, Agent) triplet regardless of domain.
2. **Eval is king** — eval quality determines improvement quality. Bad eval → Goodhart's law.
3. **Immutable evaluation boundary** — eval function, eval data, and eval criteria are always outside the agent's mutation scope.
4. **Git as the journal** — every mutation is a commit. Every experiment is recoverable.
5. **Knowledge compounds** — structured experiment records alongside narrative summaries. Records are authoritative.
6. **Time-boxing is non-negotiable** — every experiment has a hard wall-clock limit.
7. **Fail fast, fail loud** — crashed experiments are logged. Terminal states produce notifications, not silence.
8. **Isolation by default** — each target runs in its own git worktree.
9. **Least privilege** — the agent receives only the tools it needs for mutation.
