# Token Optimization Analysis: OpenWolf Techniques Applied to Anneal

**Date:** 2026-03-29
**Sources:**
- Blog: "Claude Code used 25M tokens on my project — I got it down to 425K with 6 hook scripts" (dev.to/cytostack)
- Repository: [cytostack/openwolf](https://github.com/cytostack/openwolf)

**Context:** Anneal runs autonomous optimization loops where an LLM agent mutates artifacts, evaluates them, and keeps winners. Each experiment involves one or more LLM calls. Token efficiency directly affects experiments-per-dollar. The blog claims 58x reduction through hook scripts — this analysis assesses which techniques transfer to anneal's batch optimization architecture.

---

## Architecture Comparison

| Dimension | OpenWolf (Claude Code hooks) | Anneal |
|-----------|------------------------------|--------|
| LLM interaction model | Interactive agent freely reading/writing files | Programmatic API calls with assembled prompts |
| Token waste source | Redundant file reads, re-discovered context | Prompt content bloat, redundant eval calls, knowledge overhead |
| Context control | Indirect (hints via stderr warnings) | Direct (`ContextBudget` with priority-ordered slots) |
| Call sites | Single agent session | 5+ distinct call sites (mutation, eval gen, eval score, policy, taxonomy) |

OpenWolf's wins come from steering an interactive agent away from wasteful file reads. Anneal already controls prompt assembly programmatically. The applicable techniques are the *principles*, not the hook implementations.

---

## Technique-by-Technique Assessment

### 1. Project Index (anatomy.md) — File Summaries with Token Estimates

**Source:** OpenWolf scans every project file, extracts a one-line description and token estimate. Claude sees the summary before reading and can skip full reads.

**Does anneal already do this?** Partially. `context.py` reads artifact files in full and includes them as required context slots. Watch files from `scope.yaml` are included as optional context. There is no summary/description layer — every included file is included in full.

**Applicability:** Medium. Anneal's artifact files MUST be included in full (the agent needs to edit them). However:
- **Watch files** could benefit from summarization. If a watch file is 2000 tokens but the agent only needs to know "this is the test harness that runs X", a 50-token summary saves 1950 tokens per experiment.
- **Artifact token warnings** already exist (line 384-390 of `context.py` warns when artifact > 60% of budget), but there's no mechanism to *act* on this warning.

**Estimated impact:** Medium — 500–2000 tokens/experiment for projects with large watch files.

**Implementation sketch:**
- Add an optional `watch_summaries` field to `scope.yaml` allowing users to provide one-line descriptions of watch files.
- In `build_target_context()`, when a watch file exceeds a threshold (e.g., 500 tokens), use the summary instead of full content, with a note: "Full file available at {path}".
- Alternative: auto-generate summaries at registration time using an LLM call (one-time cost, amortized over all experiments).

**Conflicts with anneal's design:** None for watch files. Summarizing artifacts would break the mutation loop — the agent needs full content to produce valid diffs.

---

### 2. Learning Memory (cerebrum.md) — Persistent Coding Preferences and Do-Not-Repeat List

**Source:** OpenWolf maintains a structured memory of user preferences, project conventions, and a dated list of mistakes. Pre-write hooks check new code against the Do-Not-Repeat list.

**Does anneal already do this?** Yes — via two mechanisms:
1. **Knowledge store** (`knowledge.py`): records every experiment outcome, retrieves similar past experiments via TF-IDF/embeddings, and consolidates learnings every 50 experiments into `learnings.md`.
2. **Failure taxonomy** (`taxonomy.py`): classifies failed experiments into 8 named categories and surfaces failure distribution in context.

**Applicability:** Low-Medium. Anneal's knowledge system is more sophisticated than cerebrum.md — it does retrieval, consolidation, and failure-pattern detection. However:
- The knowledge store uses full experiment records (hypothesis, diff summary, scores, tags, learnings). Each record is verbose. A **compressed summary format** for history inclusion could save tokens while preserving signal.
- The `_format_experiment_record()` function (context.py:206-227) outputs ~150-300 tokens per record. With 5 recent + K retrieved records, this is 1000-3000 tokens. A terser format (50 tokens/record) would save 500-1500 tokens.

**Estimated impact:** Low — 500–1500 tokens/experiment from terser history formatting.

**Implementation sketch:**
- Add a `compact` mode to `_format_experiment_record()` that emits: `"#{id} {outcome} {score:.3f} (Δ{delta:+.3f}) [{tags}] {one_line_learning}"` — ~50 tokens vs ~200 tokens.
- Use compact format for retrieved (non-recent) history. Keep full format for the last 2-3 experiments where detail matters most.

**Conflicts:** None. The consolidated `learnings.md` already does a form of this.

---

### 3. Bug Fix Memory (buglog.json) — Error Pattern Cache

**Source:** OpenWolf logs every bug fix with error message, root cause, solution, and tags. Before debugging, checks if the error already exists and serves the cached solution.

**Does anneal already do this?** Yes, more thoroughly:
- Experiment records include `failure_mode`, `learnings`, and `mutation_diff_summary`.
- `KnowledgeStore.retrieve()` does TF-IDF and optional embedding-based similarity search across all past experiments.
- `FailureTaxonomy.classify()` categorizes failures for pattern detection.
- Consolidation produces `learnings-structured.jsonl` with failure pattern aggregates.

**Applicability:** Very Low. Anneal's knowledge retrieval already serves this purpose. The failure taxonomy classifier costs tokens on its own (one LLM call per failed experiment) — reducing *its* token usage would be more impactful than adding a separate bug memory.

**Estimated impact:** Negligible. Already covered.

**Implementation sketch:** N/A — existing system is sufficient.

**Potential improvement on existing system:** The taxonomy classifier (`taxonomy.py`) makes an LLM call for *every* failed/blocked experiment. A heuristic pre-classifier that catches obvious patterns (verifier failures, empty diffs, identical score) without an LLM call could eliminate 30-50% of taxonomy calls.

**Estimated impact of heuristic pre-classifier:** Medium — saves one LLM call per easily-classifiable failure. At ~200-500 tokens per taxonomy call across 50-70% of experiments that fail, this is significant over hundreds of experiments.

---

### 4. Repeated Read Detection — Preventing Duplicate File Reads

**Source:** OpenWolf tracks files read per session and warns Claude before re-reading a file it already has in context. The blog claims 71% of file reads were duplicates in unoptimized sessions.

**Does anneal already do this?** Not applicable in the same way. Anneal doesn't have an agent that freely reads files — it assembles context programmatically. Each experiment gets a fresh prompt. The "duplicate read" problem doesn't exist because the system controls exactly what's included.

**Applicability:** Very Low for direct reads. However, the *principle* of avoiding redundant content across calls maps to:
- **Eval prompt deduplication**: In stochastic eval, the same `artifact_content` is sent to every generation call (one per test prompt). If there are 10 test prompts and the artifact is 5000 tokens, that's 50,000 tokens of redundant artifact content across the batch.

**Estimated impact:** Low for current architecture (no redundant reads). High if eval prompt deduplication is addressed (see Technique 9 below).

---

### 5. Session Tracking and Token Ledger — Cost Visibility

**Source:** OpenWolf maintains a per-session and lifetime token ledger, tracking estimated tokens consumed by every read/write operation.

**Does anneal already do this?** Yes, comprehensively:
- `compute_cost()` in `client.py` tracks cost per LLM call using actual API response token counts.
- `ExperimentRecord.cost_usd` accumulates per-experiment cost.
- `RunLoopState.cumulative_cost_usd` persists total spend across restarts.
- `safety.py` enforces budget caps using pre-experiment cost estimation.
- `ContextBudget.summary()` reports per-slot token allocation.

**Applicability:** Very Low. Anneal's cost tracking is already more granular than OpenWolf's character-ratio estimation. Anneal uses `tiktoken` for token counting and actual API response usage for cost, while OpenWolf uses `text.length / 3.75` approximations.

**Estimated impact:** Negligible. Already covered.

---

### 6. Action Memory (memory.md) — Session Activity Log

**Source:** OpenWolf appends every read/write action to a chronological log, consolidated weekly.

**Does anneal already do this?** Yes. The JSONL experiment records serve as a detailed action log. `KnowledgeStore` manages retrieval and consolidation.

**Applicability:** Very Low. Already covered.

---

### 7. Protocol Enforcement (OPENWOLF.md) — Agent Behavior Constraints

**Source:** OpenWolf installs instruction files that tell Claude to check anatomy before reading, never re-read files, log bugs, etc.

**Does anneal already do this?** Partially. The `program.md` system prompt and policy instructions guide the mutation agent's behavior. However:
- When using `claude_code` mode, the subprocess invocation (`claude -p ...`) doesn't inject OpenWolf-style efficiency constraints.
- In `api` mode, the single chat completion call makes this irrelevant.

**Applicability:** Low-Medium for `claude_code` mode only. If anneal's Claude Code subprocess does excessive file reads within a single invocation, adding efficiency constraints to the prompt could reduce per-invocation token usage.

**Estimated impact:** Low — the `--max-budget-usd` flag already caps per-invocation spend. Protocol enforcement would reduce waste within that cap.

---

## Novel Techniques Derived from OpenWolf Principles

### 8. Graduated Context Depth (from anatomy.md principle)

**Principle:** Don't include full content when a summary suffices.

**Applied to anneal:** The context budget currently has a binary choice — include a slot or don't. A **graduated depth** system would:
1. Include recent history (last 3) at full detail.
2. Include retrieved history at compact detail (one-line per record).
3. Include knowledge context as bullet-point summaries.
4. Include watch files as summaries unless they fit within remaining budget.

**Estimated impact:** High — could reclaim 2000-5000 tokens per experiment, allowing either smaller context windows (cheaper models) or more knowledge slots.

**Implementation sketch:**
- Modify `build_target_context()` to support multi-resolution history:
  - `_format_experiment_record(record, compact=False)` for recent
  - `_format_experiment_record(record, compact=True)` for retrieved
- Add `ScopeConfig.watch_summaries: dict[str, str]` for pre-computed watch file descriptions.

---

### 9. Eval Batch Deduplication (from repeated-read principle)

**Principle:** Don't send the same content multiple times.

**Applied to anneal:** In stochastic evaluation, `_generate_sample()` sends the full artifact content with every test prompt. With N test prompts, the artifact is transmitted N times.

**Current behavior** (eval.py):
```
For each test_prompt in prompts:
    prompt = template.format(test_prompt=test_prompt, artifact_content=content)
    response = chat.completions.create(messages=[...prompt...])
```

**Optimization:** Use a shared system message containing the artifact content, and vary only the user message per test prompt. This is architecturally simple with the OpenAI API:
```
system: "You are evaluating the following artifact:\n{artifact_content}"
user: "{test_prompt_1}"  # call 1
user: "{test_prompt_2}"  # call 2
```

With prompt caching (supported by OpenAI, Anthropic, and Google), the shared prefix (artifact content) is cached after the first call, and subsequent calls only pay for the delta.

**Estimated impact:** High — for a 5000-token artifact with 10 test prompts, this saves ~45,000 input tokens per evaluation. Over hundreds of experiments, this is the single largest token optimization available.

**Implementation sketch:**
- Refactor `StochasticEvaluator._generate_sample()` to split context into `system` (artifact + evaluation instructions) and `user` (test prompt).
- The OpenAI API automatically caches matching prefixes. No explicit cache management needed.
- Same optimization applies to `_score_criterion_once()` — the evaluated output could be in the system message with criteria varying in user messages.

---

### 10. Heuristic Failure Pre-Classification (from buglog principle)

**Principle:** Check cheap local patterns before making an expensive LLM call.

**Applied to anneal:** `taxonomy.classify()` makes an LLM call for every failed experiment. Many failures have deterministic signatures:

| Pattern | Classification | Detection |
|---------|---------------|-----------|
| Empty git diff | `no_change` | `len(diff) == 0` |
| Verifier exit code > 0 | `verifier_failure` | Already in `failure_mode` field |
| Score == baseline exactly | `no_improvement` | `score == baseline_score` |
| Agent timeout | `timeout` | `AgentTimeoutError` caught |
| Scope violation reverted all changes | `scope_violation` | Scope enforcement logs |

**Estimated impact:** Medium — eliminates 30-50% of taxonomy LLM calls. At ~500 tokens per call, saves 150-250 tokens × (0.3-0.5) × total_experiments.

**Implementation sketch:**
- Add `_heuristic_classify(record: ExperimentRecord) -> str | None` in `taxonomy.py`.
- Call it before the LLM classifier. If it returns a category, skip the LLM call.

---

### 11. Persistent Eval Cache (from token-ledger principle)

**Principle:** Persist expensive computations across sessions.

**Applied to anneal:** `EvalCache` is in-memory only (lost on restart). For long-running optimization targets that restart frequently (crashes, budget pauses, daemon restarts), re-evaluating previously-seen artifact states wastes tokens.

**Estimated impact:** Medium — depends on restart frequency. Each cache miss costs one full evaluation (500-5000 tokens for stochastic eval). A target that restarts 10 times across its lifetime with 5% cache-eligible artifacts saves 25-250K tokens total.

**Implementation sketch:**
- Serialize `EvalCache` to `.anneal/targets/{id}/eval-cache.json` on shutdown.
- Load on startup. Use the existing content-hash keys — they're deterministic.
- Add a max-age eviction (e.g., 7 days) to prevent stale cache from affecting results if eval criteria change.

---

## Techniques That Conflict with Anneal's Design

### Artifact Summarization
Summarizing artifact content (as anatomy.md does for files) would break the mutation loop. The agent needs full artifact content to produce valid, applicable edits.

### Context Truncation of Recent History
Aggressive truncation of recent experiment history could break the agent's ability to learn from immediately preceding failures. The last 3-5 records should remain at full detail.

### Pre-write Blocking Hooks
OpenWolf's cerebrum enforcement hooks warn but never block. In anneal, the agent's output IS the experiment — blocking mutations would prevent exploration. Anneal handles bad mutations post-hoc via verifiers and score comparison.

---

## Top 3 Highest-Impact Recommendations

### 1. Eval Batch Prompt Caching (Technique 9)
**Estimated savings:** 30-50% reduction in stochastic eval input tokens.
**Effort:** Small refactor of `eval.py` to split system/user messages.
**Next steps:**
1. Refactor `_generate_sample()` to use `system` message for artifact content + eval instructions.
2. Refactor `_score_criterion_once()` to use `system` message for the evaluated output.
3. Benchmark token usage before/after on a stochastic eval target.
4. Verify prompt caching activation via API response headers (`x-cache` or usage breakdown showing cached tokens).

### 2. Graduated Context Depth (Technique 8)
**Estimated savings:** 2000-5000 tokens per experiment on knowledge-heavy targets.
**Effort:** Medium refactor of `context.py` and `_format_experiment_record()`.
**Next steps:**
1. Add `compact` parameter to `_format_experiment_record()`.
2. Modify `build_target_context()` to use compact format for retrieved (non-recent) history.
3. Add `watch_summaries` support to `scope.yaml` schema and `build_target_context()`.
4. Benchmark context assembly token counts before/after.

### 3. Heuristic Failure Pre-Classification (Technique 10)
**Estimated savings:** Eliminates 30-50% of taxonomy LLM calls (~150-500 tokens each).
**Effort:** Small addition to `taxonomy.py`.
**Next steps:**
1. Implement `_heuristic_classify()` with the 5 deterministic patterns listed above.
2. Add a counter to track heuristic vs LLM classification rates.
3. Validate that heuristic classifications match what the LLM would have produced (run both in parallel for 100 experiments, compare).

---

## Summary Table

| # | Technique | Source | Anneal Status | Applicability | Impact | Conflicts |
|---|-----------|--------|---------------|---------------|--------|-----------|
| 1 | Project Index (file summaries) | anatomy.md | Partial (no watch summaries) | Medium | Medium | None for watch files |
| 2 | Learning Memory (Do-Not-Repeat) | cerebrum.md | Covered (knowledge store) | Low-Medium | Low | None |
| 3 | Bug Fix Cache | buglog.json | Covered (knowledge + taxonomy) | Very Low | Negligible | None |
| 4 | Repeated Read Detection | pre-read hook | N/A (programmatic context) | Very Low | Negligible | None |
| 5 | Token Ledger | session tracking | Covered (cost tracking) | Very Low | Negligible | None |
| 6 | Action Memory | memory.md | Covered (experiment records) | Very Low | Negligible | None |
| 7 | Protocol Enforcement | OPENWOLF.md | Partial (claude_code mode) | Low-Medium | Low | None |
| 8 | **Graduated Context Depth** | Derived | Not implemented | **High** | **High** | None |
| 9 | **Eval Batch Prompt Caching** | Derived | Not implemented | **High** | **High** | None |
| 10 | **Heuristic Failure Pre-Classification** | Derived | Not implemented | **Medium** | **Medium** | None |
| 11 | Persistent Eval Cache | Derived | Partial (in-memory only) | Medium | Medium | None |

**Bottom line:** Anneal already implements the *spirit* of most OpenWolf techniques through its knowledge store, failure taxonomy, and cost tracking. The highest-value new optimizations are derived principles — eval prompt caching, graduated context depth, and heuristic pre-classification — not direct ports of OpenWolf's hook scripts.
