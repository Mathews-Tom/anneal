# Architecture

## Module Map

```text
anneal/engine/
  runner.py            # Experiment state machine (mutate → eval → decide → log)
  eval.py              # Deterministic + stochastic eval, Bradley-Terry, position debiasing
  eval_cache.py        # Content-hash LRU cache for eval results with consistency monitoring
  search.py            # Greedy, SA, population, island population, Pareto search strategies
  bayesian.py          # GP surrogate model for mutation ranking (optional scikit-learn)
  strategy_selector.py # Thompson Sampling meta-strategy over search algorithms
  strategy.py          # Strategy manifest: structured YAML components with evolution
  archive.py           # MAP-Elites quality-diversity archive
  agent.py             # LLM mutation (Claude Code subprocess or API mode)
  research.py          # Research operator: external knowledge injection on plateau
  scope.py             # Editable/immutable enforcement with path traversal protection
  knowledge.py         # JSONL experiment store, TF-IDF/embedding retrieval, episodic memory
  learning_pool.py     # Cross-condition/target/project knowledge transfer with domain filtering
  context.py           # Token budget assembly with compression modes and research hints
  environment.py       # Git worktree management with fsck integrity checks
  safety.py            # Budget caps, failure limits, disk checks, process time-boxing
  client.py            # Multi-provider LLM client with configurable pricing (TOML overlay)
  scheduler.py         # Sequential target scheduler with stale lock recovery
  taxonomy.py          # Failure classification: LLM-based categorization, distribution, blind spots
  tree_search.py       # UCB tree search: backtracking, pruning, persistence, history bootstrap
  policy_agent.py      # Policy agent: continuous instruction rewriting, reward tracking
  registry.py          # Target configuration (config.toml persistence)
  dashboard.py         # File-based SSE live dashboard
  notifications.py     # Webhook notification hooks with retry
  compare.py           # Statistical comparison utilities
  rate_limiter.py      # API rate limiting
  daemon.py            # Daemon management
```

## The Experiment Loop

```text
Assemble context (artifact + history + learnings + research hints)
  → Select search node (greedy / SA / population island / UCB tree)
  → Generate hypothesis + mutation (optional: two-phase diagnosis, multi-draft)
  → Scope enforcement + verification gates
  → Evaluate (deterministic command or stochastic LLM judge)
  → Statistical decision (Wilcoxon / effect-size / Pareto dominance)
  → Keep winner or revert (git commit / rollback)
  → Extract lesson + update knowledge
  → Adapt strategy (policy rewrite / component evolution / research on plateau)
  → Island migration (if due)
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
