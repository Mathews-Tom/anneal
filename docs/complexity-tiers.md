# Complexity Tiers

Anneal exposes six search strategies, two eval modes, and several optional components (Bayesian surrogate, MAP-Elites archive, meta-optimizer, policy agent). This document groups them into three tiers to help users choose the right configuration for their use case.

Tier 1 is the default and is production-ready for most use cases. Tiers 2 and 3 require intentional configuration and are appropriate for users who understand the trade-offs.

---

## Tier 1: Simple (default)

**Search strategy:** `greedy`
**Eval mode:** deterministic
**Optional components:** none

When no `population_config` is set, the runner uses `GreedySearch`. A challenger is accepted only if it strictly beats the baseline score in the configured direction. For deterministic evals, this is a direct numerical comparison with an optional `min_improvement_threshold`.

**Configuration (minimal):**
```toml
[targets.my-target.population_config]
# omit entirely — defaults to greedy
```

**Cost model:** one eval call per experiment. No statistical sampling overhead.

**Suitable for:**
- First-time users learning the workflow.
- Deterministic optimization targets: file size (`wc -c`), test pass rate (`pytest --co -q`), response latency, line count.
- Targets where the eval function is noise-free and a single score reliably ranks candidates.

**Not suitable for:**
- Stochastic targets where a single eval run produces noisy scores.
- Targets with many local optima where greedy acceptance gets stuck.

---

## Tier 2: Intermediate

**Search strategies:** `simulated_annealing`, `population`, `pareto`
**Eval mode:** deterministic or stochastic
**Optional components:** budget caps, cost tracking, Holm-Bonferroni correction

### Simulated Annealing (`simulated_annealing`)

Accepts regressions with probability `exp(delta / temperature)`. Temperature decreases after each experiment (`cooling_rate = 0.95` by default), with adaptive reheating if the acceptance ratio drops too low. This allows escape from local optima early in the run, converging to greedy behavior as temperature approaches `min_temperature`.

### Population Search (`population`)

Maintains a population of candidates (default size 4). Each experiment is evaluated against the current baseline; the population is pruned by tournament selection. Supports crossover: the agent receives the hypotheses of the two best-scoring candidates when generating a new mutation, combining ideas from the population's top performers.

### Island Population Search (`population` with `island_count > 1`)

Runs multiple independent `PopulationSearch` instances ("islands") in round-robin. Every `migration_interval` experiments, the best candidate from each island migrates to all other islands. Increases diversity at the cost of slower convergence on any single island.

### Pareto Search (`pareto`)

Accepts any candidate that is non-dominated in the per-criterion score space (Pareto front). Suitable for multi-objective optimization where no single score can summarize quality across all criteria. Requires stochastic eval with multiple `criteria`.

### Stochastic eval with Wilcoxon comparison

When `eval_mode = "stochastic"`, each eval run produces `sample_count` per-criterion scores. Comparison between challenger and baseline uses the Wilcoxon signed-rank test on paired differences (see [ADR-002](decisions/002-wilcoxon-over-t-test.md)). Requires at least 6 paired samples for the test; falls back to Cohen's d effect size below that threshold.

**Configuration example:**
```toml
[targets.my-target.population_config]
search_strategy = "simulated_annealing"

[targets.my-target.budget_cap]
max_usd_per_day = 10.0
```

**Suitable for:**
- Prompt optimization with LLM-judged criteria.
- Targets with noisy or stochastic evaluation.
- Targets where greedy search has reached a score plateau.
- Multi-objective targets with per-criterion scoring.

---

## Tier 3: Advanced

**Search strategies:** UCB tree search
**Optional components:** Thompson Sampling strategy selector, Bayesian GP surrogate, MAP-Elites archive, policy agent (meta-optimizer)

### UCB Tree Search

`UCBTreeSearch` (`anneal/engine/tree_search.py`) maintains a tree of mutations and selects the next experiment using Upper Confidence Bound (UCB) scoring. Balances exploitation of high-scoring branches against exploration of undersampled areas of the mutation space. Activated when `UCBTreeSearch` is passed directly to the runner (not exposed through `population_config` in the current default path).

### Thompson Sampling Strategy Selector

`StrategySelector` (`anneal/engine/strategy_selector.py`) maintains a Beta-Binomial arm per search strategy. At each experiment, it samples from each arm's posterior and selects the strategy with the highest sample. Arms are updated with reward signals after each experiment result. This is a meta-level selector that chooses between strategies like `greedy`, `simulated_annealing`, and `population` based on observed performance.

`AgentSelector` uses the same Thompson Sampling mechanism to choose between a primary mutation agent and an exploration agent, with a bias toward exploration in the first 20% of experiments and toward the primary agent in the final 20%.

### Bayesian Gaussian Process Surrogate

`SurrogateModel` (`anneal/engine/bayesian.py`) uses a Gaussian Process with a Matérn 5/2 kernel to predict mutation scores from feature vectors extracted from experiment history. Requires `scikit-learn` (optional dependency, `[ml]` extra). The model is not fitted until at least 10 observations are available. Once fitted, it can guide hypothesis generation toward regions of the feature space predicted to score well.

### MAP-Elites Archive

`MapElitesArchive` (`anneal/engine/archive.py`) maintains a quality-diversity grid indexed by discretized per-criterion scores. Each cell holds the highest-fitness solution that maps to that behavioral region. Enables diverse solution discovery: solutions that score well on different subsets of criteria are preserved even if their total score is not the global maximum. Useful when the goal is a library of diverse high-quality solutions rather than a single optimum.

### Policy Agent (meta-optimizer)

`PolicyAgent` (`anneal/engine/policy_agent.py`) rewrites the mutation instructions (`program.md`) every `rewrite_interval` experiments based on the recent outcome history. Enabled via `PolicyConfig`. Acts as a continuous meta-optimizer: if mutations targeting a particular criterion keep failing, the policy agent updates the strategy document to redirect effort. This adds an additional LLM call overhead of `max_budget_usd` per rewrite cycle.

**Configuration example:**
```toml
[targets.my-target.policy_config]
enabled = true
model = "claude-sonnet-4-5"
max_budget_usd = 0.02
lookback_window = 10
rewrite_interval = 3
```

**Suitable for:**
- Large experiment budgets (50+ experiments) where meta-learning pays off.
- Research applications exploring the diversity of optimal solutions.
- Targets where the right mutation strategy is not known in advance.
- Users who want the system to self-improve its own search heuristics.

**Not suitable for:**
- Small budgets (< 20 experiments): Bayesian surrogate and Thompson Sampling need sufficient history before they outperform simpler strategies.
- Time-sensitive runs: the GP fit and policy rewrite add latency per experiment cycle.

---

## Summary

| Feature | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Search | Greedy | SA / Population / Pareto | UCB tree |
| Strategy selection | Fixed | Fixed | Thompson Sampling |
| Score prediction | None | None | Bayesian GP surrogate |
| Solution diversity | None | None | MAP-Elites archive |
| Mutation strategy adaptation | None | None | Policy agent |
| Eval mode | Deterministic | Deterministic or stochastic | Deterministic or stochastic |
| Statistical comparison | Direct threshold | Wilcoxon / effect size | Wilcoxon / effect size |
| scikit-learn required | No | No | Optional (GP surrogate only) |
| Minimum useful budget | 1 experiment | 10 experiments | 20 experiments |
