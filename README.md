# anneal

Autonomous optimization for any measurable artifact.

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

**Stochastic** — an LLM judges N samples against K binary criteria. Bootstrap CI ensures statistical rigor.

```text
criteria:
  - "Is the text scannable?" (YES/NO)
  - "Are all claims cited?" (YES/NO)
samples: 10 test prompts × 4 criteria → score with confidence interval
```

## Where Anneal Works

The system works when: (1) the artifact is a text file in a git repo, (2) quality is measurable as a scalar, and (3) the feedback loop completes in under ~10 minutes.

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
| **Multi-objective without composition**    | Single scalar metric. Compose objectives or use constraints.                 |
| **Non-git projects / binary artifacts**    | Git worktrees + text diffs are the mutation mechanism.                       |
| **Live system tuning (real-time metrics)** | Evaluates at experiment end, not continuously.                               |
| **Database schema migrations**             | Multi-step stateful operations, not file edits.                              |
| **Cross-service distributed optimization** | Targets are scoped to one repo, one worktree.                                |
| **Embedding model selection**              | Re-embedding a corpus isn't a file edit. One-shot comparison, not iterative. |
| **Inter-agent protocol changes**           | Requires coordinated edits across multiple files simultaneously.             |

See [docs/use-cases.md](docs/use-cases.md) for detailed analysis of each scenario.

## Architecture

```text
anneal/engine/
  agent.py          # LLM mutation (Claude Code subprocess or API mode)
  eval.py           # Deterministic + stochastic evaluation with bootstrap CI
  runner.py         # Experiment state machine (mutate → eval → decide → log)
  scope.py          # Editable/immutable enforcement via git status
  knowledge.py      # JSONL experiment store, consolidation, similarity retrieval
  learning_pool.py  # Cross-condition, cross-target, cross-project knowledge transfer
  search.py         # Greedy, simulated annealing, population-based search
  safety.py         # Budget caps, failure limits, disk checks, process time-boxing
  dashboard.py      # File-based SSE live dashboard
  environment.py    # Git worktree management
  registry.py       # Target configuration (config.toml persistence)
  context.py        # Token budget assembly with priority-ordered truncation
```

## Key Features

- **Scope enforcement** — declare what the agent can and cannot modify. Violations are reverted automatically.
- **Knowledge compounding** — experiment history + consolidated learnings + cross-condition insights feed into each hypothesis.
- **Statistical rigor** — Wilcoxon signed-rank tests for stochastic eval. Bootstrap confidence intervals. Held-out evaluation for overfitting detection.
- **Cost control** — per-experiment and daily budget caps. Cost tracked per invocation.
- **Safety** — process group time-boxing (SIGKILL), consecutive failure halting, disk space checks, JSONL corruption recovery.
- **Search strategies** — greedy (default), simulated annealing (escape local optima), population-based (tournament selection).
- **Meta-optimization** — on plateau, the agent revises its own optimization strategy (program.md).
- **Live dashboard** — `anneal dashboard` reads from `.anneal/` directory. No coupling to the runner process.

## Quick Start

```bash
# Install
uv pip install -e .

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

# Run 20 experiments
anneal run --target my-target --experiments 20

# Monitor
anneal status --target my-target
anneal history --target my-target
anneal dashboard --open
```

## Project Status

262 tests passing.

### Complete

- Core engine (git worktrees, scope enforcement, registry, agent invoker, eval engine, runner state machine)
- Production hardening (safety layer, knowledge store, learning pool, notifications, JSONL recovery)
- Multi-target orchestration, context budget assembly, rate limiting, background daemon
- Simulated annealing, population-based search, meta-optimization, held-out evaluation
- Evaluator drift monitoring, confidence decay, cross-project learning pool
- File-based live dashboard, deployment-tier approval gates
- Repeatable validation gate experiments (`experiments/gate1-guided-search/` through `gate4-multi-domain-stress/`)

### In progress

- Validation gate execution (Gate 1 passed, Gates 2–4 running)

### Planned

- Experiment scaffolding — `anneal suggest` generates experiment configs from natural-language problem descriptions
- Semantic retrieval — embedding-based similarity for knowledge store and learning pool (currently Jaccard word-level)
