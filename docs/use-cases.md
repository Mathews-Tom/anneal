# Anneal — Use Cases

Anneal optimizes any text artifact in a git repo that can be scored with a scalar metric. This document catalogs where the system works well, where it works with caveats, and where it does not work — with specific reasoning for each.

## The Rule

Anneal works when three conditions hold:

1. **The artifact is a text file in a git repo.** Binary files, databases, and non-file resources are out of scope.
2. **Quality is measurable as a single number.** Either a shell command that outputs a float, or LLM-evaluated binary criteria aggregated into a score.
3. **The feedback loop completes in under ~10 minutes.** Longer loops are possible but yield fewer experiments per dollar.

---

## Code Optimization

### API Response Time — WORKS WELL

**Artifact:** `src/api/handler.py`, `src/api/middleware.py`
**Eval mode:** Deterministic
**Eval command:** `wrk -t4 -c100 -d10s http://localhost:8080/api | grep 'Latency' | awk '{print $2}'`
**Direction:** Minimize
**Scope:** Source files editable. Tests, config, and eval scripts immutable.

The agent tries algorithmic improvements, caching, query optimization, middleware reordering. Each mutation is benchmarked against a fixed workload. The deterministic eval produces a millisecond value. Scope enforcement prevents the agent from modifying the benchmark script to game the metric.

### Bundle Size Reduction — WORKS WELL

**Artifact:** `src/index.ts`, `src/utils/*.ts`
**Eval mode:** Deterministic
**Eval command:** `npm run build && du -sb dist/ | cut -f1`
**Direction:** Minimize

The agent targets tree-shaking, import elimination, lazy loading, and dead code removal. The eval is fast (~30 seconds for build + measure). Scope locks `package.json`, `tsconfig.json`, and test files — the agent optimizes application code, not build configuration. Use `--constraint "test_pass_rate>=1.0"` with a secondary eval to ensure mutations don't break functionality.

### Test Coverage Improvement — WORKS WELL

**Artifact:** `tests/*.py`
**Eval mode:** Deterministic
**Eval command:** `pytest --cov=src --cov-report=term | grep TOTAL | awk '{print $4}'`
**Direction:** Maximize
**Scope:** Test files editable. Source code immutable.

Inverting the usual pattern: the agent writes and improves tests, not source code. Source files are locked as immutable, so the agent can't game coverage by deleting source code. This is a safe, high-value use case — tests are additive, low-risk, and directly measurable.

---

## Machine Learning & Deep Learning

### Training Configuration — WORKS WELL

**Artifact:** `config.yaml` (learning rate, batch size, scheduler params, augmentation pipeline)
**Eval mode:** Deterministic
**Eval command:** `python train.py --epochs 5 && python eval.py | grep val_loss`
**Direction:** Minimize (val_loss) or Maximize (accuracy)

This is Karpathy's autoresearch pattern. Config files are text, the eval produces a scalar, and scope locks model code and data as immutable. The agent mutates hyperparameters: learning rate schedules, batch sizes, optimizer choices, augmentation parameters.

**Critical:** Use a short proxy evaluation (5 epochs, data subset, smaller model variant) to keep the feedback loop under 10–15 minutes. Full training runs (hours/days) are too slow for iterative optimization. Validate the best config on a full run at milestones.

### Data Preprocessing Pipeline — WORKS WELL

**Artifact:** `preprocess.py` or `transforms.yaml`
**Eval mode:** Deterministic
**Eval command:** `python train.py --epochs 3 && python eval.py --metric f1`
**Direction:** Maximize

Preprocessing changes (normalization strategies, augmentation parameters, feature engineering, filtering thresholds) are high-leverage and fast to evaluate with a short training proxy. The search space is rich but mutations are safe — bad preprocessing produces bad scores, not crashes.

### Model Architecture Code — WORKS WITH CAVEATS

**Artifact:** `model.py`
**Eval mode:** Deterministic
**Eval command:** Quick train + eval on data subset

The agent can modify layer dimensions, add/remove layers, change activation functions, adjust attention mechanisms. Scope keeps data loading and training loop immutable.

**Caveat:** Many mutations produce invalid code — shape mismatches, dimension errors, import failures. Expect a ~60% crash rate. Anneal handles this (CRASHED outcome, consecutive failure counter), but experiment throughput is low. Simulated annealing (`--search annealing`) helps escape local optima once valid mutations are found.

### Loss Function Coefficients — WORKS WELL

**Artifact:** `config.yaml` or `losses.py` (loss weights, margin values, temperature parameters)
**Eval mode:** Deterministic

Pure config optimization. The agent adjusts `alpha`, `beta`, `temperature`, `margin` values. Fast to evaluate, low crash rate.

### Training Loop Structure — PARTIAL

**Artifact:** `train.py` (optimizer setup, gradient accumulation, learning rate scheduling)

**Why it's risky:** Mutations to training loops can cause silent numerical instability (NaN gradients, mode collapse, divergence) that only manifests after many epochs. A 5-epoch proxy evaluation might not catch it. The agent can produce code that is syntactically valid and runs without errors but produces a model that is numerically broken.

**Verdict:** Works for optimizer hyperparameters. Risky for structural changes to the gradient computation pipeline.

---

## Multi-Agent Systems

### Orchestration Prompts — WORKS WELL

**Artifact:** `prompts/orchestrator.md`, `prompts/researcher.md`, `prompts/synthesizer.md`
**Eval mode:** Stochastic
**Criteria:** "Does the research output cite specific sources?", "Is the synthesis under 500 words?", "Does the answer address the original question?"
**Test prompts:** Representative queries the system should handle

Each agent's prompt is a text file. The orchestrator's routing instructions, the researcher's search strategy description, the synthesizer's formatting guidelines — all are prompt-driven and LLM-evaluable. Scope locks the framework code (agent class, tool definitions, API calls) as immutable while prompts are editable.

### Tool Definitions / Schemas — WORKS WELL

**Artifact:** `tools/search.json`, `tools/calculator.json`
**Eval mode:** Stochastic
**Criteria:** "Did the agent select the correct tool?", "Were the parameters well-formed?", "Did the tool call return useful results?"

Better tool descriptions lead to better tool selection. The agent optimizes description wording, parameter explanations, and example values. Fast feedback — run N test queries, score tool selection accuracy.

### Agent Routing Logic (Config-Based) — WORKS WELL

**Artifact:** `config/routing.yaml` (routing rules, thresholds, model assignments)
**Eval mode:** Deterministic
**Eval command:** Run labeled test set, count correct routes

When routing is configuration-driven (rules, thresholds, regex patterns), anneal optimizes the parameters directly. Fast eval, clear metric.

### Agent Routing Logic (Code-Based) — WORKS WITH CAVEATS

**Artifact:** `router.py`

Code mutations to routing logic can break the entire pipeline. High crash rate. The feedback loop includes spinning up the full multi-agent system per experiment.

**Caveat:** If the multi-agent system takes 3 minutes per query and stochastic eval requires 10 test queries, that's 30 minutes per experiment. At 50 experiments, that's 25 hours. Use a fast subset — 3 representative queries, not 10.

### Inter-Agent Communication Protocols — DOES NOT WORK

**Problem:** Optimize how agents pass context, manage shared memory, or handle handoffs.

Protocol changes require synchronized edits across multiple files (agent A's output format + agent B's input parser + shared memory schema). Anneal's scope enforcement scopes mutations to declared editable files, but changing a protocol requires coordinated edits. A mutation that changes agent A's output format without updating agent B's parser crashes the system — and the agent has no mechanism to make atomic multi-file protocol changes.

---

## Single Agent Systems

### System Prompt Optimization — WORKS PERFECTLY

**Artifact:** `system_prompt.md`
**Eval mode:** Stochastic
**Criteria:** Domain-specific quality criteria
**Test prompts:** Representative user queries

This is the canonical anneal use case. The system prompt is a single text artifact. The eval runs the agent on test queries and judges output quality. Scope keeps code, tools, and API configs immutable. The entire optimization surface is the prompt text.

**Example (AI travel agency):** Criteria: "Did it ask for travel dates?", "Did it present real destinations?", "Was the tone professional but friendly?". Test prompts: "Book a flight from NYC to Tokyo", "Plan a 5-day itinerary in Portugal", "Find cheap hotels in Barcelona for next weekend".

### Few-Shot Example Curation — WORKS WELL

**Artifact:** `examples.jsonl` or `few_shot_examples.md`
**Eval mode:** Stochastic

The agent optimizes which examples to include, how to format them, and in what order. Few-shot examples are text, easily mutated (reorder, add, remove, rephrase), and directly measurable via output quality on test queries.

### Tool Configuration — WORKS WELL

**Artifact:** `tools_config.yaml`
**Eval mode:** Deterministic or stochastic

Which APIs to call, rate limits, retry logic, prompt templates per tool, parameter defaults. Config-driven systems are ideal for anneal.

### Conversation State Machine — PARTIAL

**Problem:** Optimize a multi-turn conversation flow (greeting → intent detection → fulfillment → confirmation).

State machines involve code logic, not just configuration. Mutations can create unreachable states, infinite loops, or broken transitions. The eval requires multi-turn simulation — slow and expensive (each "sample" is a full conversation).

**Verdict:** Works if the state machine is config-driven (YAML/JSON transitions). Does not work if it's imperative code logic.

---

## RAG Systems

### Retrieval Query Prompt — WORKS PERFECTLY

**Artifact:** `prompts/retrieval_query.md`
**Eval mode:** Deterministic
**Eval command:** `python evaluate_retrieval.py --test-set golden_qa.jsonl | grep mrr`
**Direction:** Maximize

The retrieval query template ("Given the user question: {query}, generate a search query that...") is a text artifact. Better templates produce better search queries, which improve retrieval precision. Fast eval: run N test queries, measure MRR or recall@k.

### Synthesis / Answer Prompt — WORKS WELL

**Artifact:** `prompts/synthesis.md`
**Eval mode:** Stochastic
**Criteria:** "Is the answer grounded in the retrieved documents?", "Does it address the question directly?", "Are sources cited?"

The synthesis prompt controls how retrieved context is transformed into an answer. LLM-evaluable quality criteria. Fast feedback — retrieve + synthesize + judge per test query.

### Chunking Strategy Configuration — WORKS WELL

**Artifact:** `config/chunking.yaml` (chunk size, overlap, separator patterns, metadata rules)
**Eval mode:** Deterministic
**Eval command:** Re-index test corpus, run test queries, measure MRR

Chunking parameters are config values. The eval re-indexes a corpus subset with the new config and measures retrieval quality.

**Caveat:** Re-indexing can be slow. If the corpus is 100K documents, re-chunking + re-embedding takes 20+ minutes. Use a representative subset (1K docs) for fast iteration. Validate on the full corpus at milestones.

### Reranking Configuration — WORKS WELL

**Artifact:** `config/retrieval.yaml` (reranker model name, top-k, score threshold, fusion weights)
**Eval mode:** Deterministic

Pure config optimization. The agent tweaks `top_k: 5 → 10`, `rerank_threshold: 0.7 → 0.6`, `fusion_weight_bm25: 0.3 → 0.5`. Each mutation is fast to evaluate against a test query set.

### Embedding Model Selection — DOES NOT WORK

Switching embedding models requires re-embedding the entire corpus — a heavyweight operation that can't be expressed as a file edit. The "artifact" would be a model name in a config, but the eval requires a full re-index cycle. This is a one-shot comparison task (benchmark N models), not iterative optimization.

### Knowledge Base Content — WORKS WITH CAVEATS

**Artifact:** Markdown source documents, FAQ entries
**Eval mode:** Stochastic
**Criteria:** "Does the RAG system correctly answer this question using the knowledge base?"

If the knowledge base is markdown files in git, anneal can optimize their structure, wording, and organization for better retrievability.

**Caveat:** The search space is enormous. Most mutations to document text won't affect retrieval. Better suited for targeted optimization: a specific FAQ section where retrieval consistently fails, not the entire knowledge base.

---

## Other AI Systems

### Evaluation Rubric Optimization — WORKS WELL

**Artifact:** `eval_criteria.toml` or `rubric.md`
**Eval mode:** Deterministic
**Eval command:** Measure agreement between LLM evaluator and human labels (Cohen's kappa, accuracy)

You have a labeled test set with human judgments. Anneal optimizes the rubric wording until the LLM evaluator's scores align with human scores. Scope locks the test set as immutable. Fast feedback, clear metric.

### Guardrail / Safety Filter Tuning — WORKS WELL

**Artifact:** `guardrails/config.yaml` (blocked patterns, sensitivity thresholds, bypass rules)
**Eval mode:** Deterministic
**Eval command:** Compute false positive rate + false negative rate on labeled test set
**Direction:** Minimize false positives
**Constraints:** `--constraint "false_negative_rate<=0.02"`

Config-driven, fast eval, clear metric. The agent tweaks thresholds to minimize false positives without increasing false negatives. The constraint mechanism ensures safety is never degraded.

### LLM Router / Model Selection — WORKS WELL

**Artifact:** `router_config.yaml` (query → model mapping, cost/quality thresholds)
**Eval mode:** Deterministic
**Eval command:** Run test queries through router, measure composite quality + cost

Routing rules are config. Eval runs test queries, measures average quality and total cost. The agent optimizes routing thresholds to maximize quality within a cost budget.

---

## Non-AI Use Cases

### Configuration Tuning (Build, Infra, DB) — WORKS WELL

**Artifact:** `nginx.conf`, `webpack.config.js`, `postgresql.conf`
**Eval mode:** Deterministic

Any config file where you can measure the effect of parameter changes. Build time, request throughput, query latency, cache hit rates. The agent mutates config values while scope locks application code.

### Email / Marketing Copy — WORKS WITH CAVEATS

**Artifact:** `templates/welcome_email.html`
**Eval mode:** Stochastic (LLM-evaluated) or Deterministic (A/B test metrics from external API)
**Domain tier:** Deployment (approval-gated)

Stochastic eval judges clarity, scannability, call-to-action strength. Deployment tier means the agent proposes changes but a human approves before they're applied.

**Caveat:** Real engagement metrics (open rate, click rate) have multi-day feedback loops — too slow for anneal's experiment cycle. Use LLM-evaluated proxy metrics for fast iteration, then validate winners with real A/B tests.

---

## Summary

| Category         | Use Case                      | Verdict                      |
| ---------------- | ----------------------------- | ---------------------------- |
| **Code**         | API performance               | Works well                   |
| **Code**         | Bundle size                   | Works well                   |
| **Code**         | Test coverage                 | Works well                   |
| **ML/DL**        | Training config (hyperparams) | Works well                   |
| **ML/DL**        | Data preprocessing            | Works well                   |
| **ML/DL**        | Model architecture            | Partial (high crash rate)    |
| **ML/DL**        | Training loop structure       | Partial (silent instability) |
| **Multi-Agent**  | Orchestration prompts         | Works well                   |
| **Multi-Agent**  | Tool definitions              | Works well                   |
| **Multi-Agent**  | Routing (config)              | Works well                   |
| **Multi-Agent**  | Routing (code)                | Partial (slow feedback)      |
| **Multi-Agent**  | Inter-agent protocols         | Does not work                |
| **Single Agent** | System prompt                 | Works perfectly              |
| **Single Agent** | Few-shot examples             | Works well                   |
| **Single Agent** | Conversation state machine    | Partial                      |
| **RAG**          | Retrieval prompts             | Works perfectly              |
| **RAG**          | Synthesis prompts             | Works well                   |
| **RAG**          | Chunking config               | Works well                   |
| **RAG**          | Reranking config              | Works well                   |
| **RAG**          | Embedding model selection     | Does not work                |
| **RAG**          | Knowledge base content        | Partial                      |
| **AI Systems**   | Eval rubric calibration       | Works well                   |
| **AI Systems**   | Guardrail tuning              | Works well                   |
| **AI Systems**   | LLM router config             | Works well                   |
| **General**      | Config tuning (build/infra)   | Works well                   |
| **General**      | Email / marketing copy        | Partial (proxy metrics)      |
| **General**      | Multi-objective optimization  | Does not work natively       |
| **General**      | Non-git / binary artifacts    | Does not work                |
| **General**      | Live system tuning            | Does not work                |
| **General**      | Database schema migrations    | Does not work                |
| **General**      | Cross-service optimization    | Does not work                |
