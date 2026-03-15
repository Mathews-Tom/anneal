# Anneal

### What if your systems got better while you slept?

---

## The Idea in 30 Seconds

You define _what_ to improve and _how to measure_ improvement. An AI agent does the rest — generating hypotheses, running experiments, keeping winners, discarding losers, and compounding learnings. Overnight. Unattended. Across any domain.

This isn't hyperparameter tuning. It's an autonomous research loop that works on code, prompts, email copy, configs, website performance — anything with a measurable outcome.

## Why Now

Three things converged in March 2026:

**The pattern was proven.** Andrej Karpathy released `autoresearch` — 630 lines of Python that let an AI agent run 100+ ML experiments overnight on a single GPU. In one run, the agent found improvements that Karpathy himself had missed over two decades. 32k GitHub stars in 9 days.

**The pattern was generalized.** Within a week, builders demonstrated the same loop on cold email optimization (reply rate: 1.5% → 3.4%), Claude Code skill prompts (32/40 → 39/40 eval score), and website performance (Lighthouse: 1100ms → 67ms). Shopify's CEO ran it overnight and produced a 0.8B model that outperformed his previous 1.6B.

**The gap was identified.** Autoresearch works brilliantly for ML training: one file, one metric, one GPU. But most optimization problems aren't ML training. They involve multiple files, stochastic outputs, external APIs, and metrics that take hours to compute. Nobody has built the production-grade engine that handles the general case.

## What We're Building

A domain-agnostic autonomous optimization framework. The core abstraction:

> (Artifact, Eval, Agent)

**Artifact** — the thing being improved (code, prompt, config, content).
**Eval** — how you measure quality (scalar metric, or N-sample binary criteria matrix).
**Agent** — the LLM that generates informed mutations, drawing on accumulated experiment history.

If you can define these three for your domain, the engine optimizes it.

### What Makes It Different

|              | Autoresearch                          | This Engine                                            |
| ------------ | ------------------------------------- | ------------------------------------------------------ |
| **Domains**  | ML training only                      | Anything measurable                                    |
| **Eval**     | One number from a log file            | Deterministic metrics + stochastic N×K binary criteria |
| **Memory**   | Zero — each experiment is independent | Structured experiment DAG with semantic retrieval      |
| **Hardware** | NVIDIA GPU required                   | Runs anywhere                                          |
| **Targets**  | One at a time                         | Multi-target registry, independent schedules           |
| **Safety**   | git reset                             | Budget caps, failure limits, immutable eval boundary   |

## Where It Applies

### Fast feedback loops (minutes)

Prompt and skill optimization — generate N samples, score against binary criteria, keep or discard. $0.20/experiment. 50 experiments to convergence ≈ $10.

Website performance — Lighthouse as the metric. Code splitting, lazy loading, image optimization discovered autonomously.

Test coverage, API latency, build times — any deterministic metric on code.

### Slow feedback loops (hours to days)

Cold email copy — reply rate via CRM API. 4-hour loops, 6 experiments/day.

Landing pages, ad creatives — conversion rate, CTR. Deploy variants, wait, measure.

Support routing, chatbot scripts — resolution time, CSAT.

### Embeddable (as a product feature)

The "optimize button" — embed the loop inside an existing SaaS product. Users press optimize, the engine runs a mini research loop on their specific configuration, and surfaces the winner for approval.

## Early Results From the Pattern

These are from the broader autoresearch ecosystem — the pattern this engine productionizes:

| Who                  | What                   | Result                                             |
| -------------------- | ---------------------- | -------------------------------------------------- |
| Karpathy             | ML training (nanochat) | val_bpb 0.9979 → 0.9697 in 126 experiments         |
| Karpathy (2-day run) | depth-12 model         | ~20 additive improvements, Time-to-GPT-2 down 11%  |
| Tobi Lütke (Shopify) | QMD query expansion    | 0.8B model beat 1.6B after 37 experiments, 8 hours |
| Nick Saraev          | Claude Code skills     | 32/40 → 39/40 eval score                           |
| Nick Saraev          | Cold email copy        | Reply rate improvement via Instantly API           |
| Nick Saraev          | Website performance    | Lighthouse 1100ms → 67ms load time                 |

## The Hard Problems We Solve

**Metric synthesis** — most domains don't have a single clean number. We provide composite metrics with weighted aggregation, Pareto tracking, and lexicographic ordering.

**Stochastic evaluation** — LLM-generated outputs vary per run. The N×K binary criteria matrix (generate N samples, score each against K yes/no questions, aggregate) gives stable signal from noisy outputs.

**Experiment memory** — the agent reads what was tried before, what worked, and what failed. Prevents re-exploring dead ends, enables building on successful patterns. Learnings consolidate every 50 experiments to keep context manageable.

**Safety** — immutable eval boundary (the agent cannot modify its own scoring), budget caps, failure limits, time-boxing, regression guards. The `scope.yaml` declares what's mutable and what's not. The eval is always immutable.

## The Risk That Kills Most Attempts

**Goodhart's Law.** Give an agent a metric and freedom to optimize, and it will find ways to improve the metric that don't improve the system. Test coverage goes up via `assert True`. Lighthouse scores improve by removing features. Prompts parrot eval keywords.

Autoresearch sidesteps this because val_bpb is computed by immutable code on a held-out dataset. Our engine preserves the same property: the eval is always outside the agent's mutation scope. The `immutable` section of scope.yaml is the integrity constraint that makes the difference between a useful engine and a metric-gaming machine.

## What We Need to Validate

1. **Does the stochastic eval framework produce stable rankings?** The binary criteria approach needs validation across more domains — how many samples are sufficient, how consistent are evaluator models across runs.

2. **Does experiment memory improve search efficiency?** The hypothesis: agents with access to past experiment history converge faster than memoryless agents. Testable with A/B comparison.

3. **Where does the pattern break?** Domains with no computable metric, very long feedback loops (weeks), or high cost per experiment may not be viable.

4. **What's the right business model?** Open-source engine with managed hosting? SaaS with embedded optimization? Consulting + engine licensing? Research-as-a-service retainer?

## Build Plan

**MVP** (covers the two highest-value use cases: code optimization + skill/prompt optimization):

1. Metric protocol + composite evaluator
2. Scope enforcement (immutable eval boundary)
3. Core greedy loop + stochastic eval framework
4. Experiment memory with semantic retrieval

**Post-MVP**:

5. Environment abstraction beyond git (Docker, databases)
6. Pluggable search strategies (simulated annealing, population-based)
7. Dashboard + alerting
8. Multi-target orchestration

## Why This Matters Beyond the Tool

The autoresearch pattern is a category definition. Karpathy identified it: the researcher's job shifts from running experiments to designing the search. The human writes `program.md` — the research strategy — and the agent executes.

This engine is the infrastructure that makes that shift practical for every domain, not just ML training. The accumulated research data — what works, what doesn't, what patterns emerge — becomes a durable asset that transfers across model generations.

When Opus 5.0 arrives, you hand it the research log from its predecessors. It picks up where they left off, but with better reasoning. The knowledge compounds independently of any specific model.

---

_Interested? Let's talk about which domain you'd point this at first._
