# Writing Binary Evaluation Criteria

Reference for writing eval criteria in anneal. Every criterion is a yes/no question scored by an LLM judge. This document covers why, how, and the mistakes to avoid.

---

## The Golden Rule: Binary Only

Every eval criterion must be a yes/no question. No 1-5 scales. No "rate the quality." No vibes.

Why: scales compound variability. Ask three judges to rate something 1-5 and you get 3, 4, 5. Ask them "does it meet X?" and you get YES, YES, YES. Binary questions produce consistent answers across judges and across runs. When anneal compares two versions of a skill, it needs signal, not noise. A 10-sample eval with 4 binary criteria gives 40 data points that cluster tightly. The same eval with 4 scaled criteria gives 40 data points spread across a range that obscures real differences.

Scales also create a second problem: anchoring. The first criterion scored influences the second. Binary criteria are independent -- each is evaluated in isolation with no frame of reference beyond pass/fail.

---

## Good vs Bad Evals by Domain

### Text/Copy (newsletters, tweets, emails, landing pages)

**Bad:**
```toml
[[criteria]]
name = "quality"
question = "Is this well-written? Answer only YES or NO."
```
"Well-written" is subjective. Two judges will disagree.

**Good:**
```toml
[[criteria]]
name = "single_cta"
question = "Does the email contain exactly ONE call-to-action? Answer only YES or NO."

[[criteria]]
name = "no_jargon"
question = "Is the text free of industry jargon and acronyms that a general audience would not know? Answer only YES or NO."

[[criteria]]
name = "opens_with_hook"
question = "Does the first sentence present a specific problem, question, or surprising fact (not a greeting or introduction)? Answer only YES or NO."

[[criteria]]
name = "under_word_limit"
question = "Is the entire text under 200 words? Answer only YES or NO."
```

### Visual/Design (diagrams, images, slides)

**Bad:**
```toml
[[criteria]]
name = "looks_good"
question = "Does the diagram look professional? Answer only YES or NO."
```
"Professional" means different things to different judges.

**Good:**
```toml
[[criteria]]
name = "text_legibility"
question = "Is ALL text in the diagram clearly legible at 100% zoom? Answer only YES or NO."

[[criteria]]
name = "pastel_colors"
question = "Does the diagram use ONLY pastel or muted colors? Answer only YES or NO."

[[criteria]]
name = "linear_layout"
question = "Does the diagram follow a strictly linear layout with no crossing arrows? Answer only YES or NO."

[[criteria]]
name = "no_ordinals"
question = "Does the diagram contain NO ordinal labels (1., 2., Step 1, Phase 2, etc.)? Answer only YES or NO."
```

### Code/Technical (code generation, configs, scripts)

**Bad:**
```toml
[[criteria]]
name = "code_quality"
question = "Is the code clean and maintainable? Answer only YES or NO."
```

**Good:**
```toml
[[criteria]]
name = "no_comments"
question = "Does the code contain ZERO inline comments? Answer only YES or NO."

[[criteria]]
name = "type_annotated"
question = "Does every function have complete type annotations on all parameters and the return type? Answer only YES or NO."

[[criteria]]
name = "single_responsibility"
question = "Does each function do exactly one thing (no 'and' needed to describe it)? Answer only YES or NO."

[[criteria]]
name = "no_hardcoded_strings"
question = "Are all user-facing strings extracted into variables or constants (none inline in function bodies)? Answer only YES or NO."
```

### Document (proposals, reports, decks)

**Bad:**
```toml
[[criteria]]
name = "persuasive"
question = "Is the proposal persuasive? Answer only YES or NO."
```

**Good:**
```toml
[[criteria]]
name = "has_executive_summary"
question = "Does the document begin with a summary of 3 sentences or fewer that states the problem, proposed solution, and expected outcome? Answer only YES or NO."

[[criteria]]
name = "quantified_claims"
question = "Does every claim about impact or improvement include a specific number, percentage, or measurable target? Answer only YES or NO."

[[criteria]]
name = "no_passive_voice"
question = "Is the document written entirely in active voice with no passive constructions? Answer only YES or NO."
```

### Configuration Optimization (build configs, infra configs, DB configs)

**Bad:**
```toml
[[criteria]]
name = "optimized"
question = "Is the configuration optimized? Answer only YES or NO."
```

**Good:**
```toml
[[criteria]]
name = "no_defaults"
question = "Has every parameter been explicitly set (no reliance on implicit defaults)? Answer only YES or NO."

[[criteria]]
name = "has_resource_limits"
question = "Does the config define explicit memory and CPU limits for every service? Answer only YES or NO."

[[criteria]]
name = "no_deprecated_flags"
question = "Are all flags and parameters current (none marked deprecated in the tool's documentation)? Answer only YES or NO."
```

Note: configuration optimization often pairs stochastic criteria with deterministic constraints (build time, bundle size, response latency). Use `[[constraint_commands]]` for the measurable parts and binary criteria for structural properties.

### RAG Systems (retrieval prompts, synthesis prompts, chunking)

**Bad:**
```toml
[[criteria]]
name = "accurate"
question = "Is the response accurate? Answer only YES or NO."
```

**Good:**
```toml
[[criteria]]
name = "cites_sources"
question = "Does the response reference at least one specific source document by name or identifier? Answer only YES or NO."

[[criteria]]
name = "no_fabrication"
question = "Does every factual claim in the response appear verbatim or paraphrased in the provided context chunks? Answer only YES or NO."

[[criteria]]
name = "answers_question"
question = "Does the response directly answer the user's question in the first paragraph? Answer only YES or NO."

[[criteria]]
name = "admits_gaps"
question = "If the context chunks do not contain enough information to fully answer the question, does the response explicitly state what is missing? Answer only YES or NO."
```

---

## Common Mistakes

### Too Many Criteria

Keep criteria at 3-6 per eval. Beyond 6, the skill starts gaming individual criteria at the expense of overall quality. Each criterion is a constraint the optimization must satisfy simultaneously. More constraints mean the search space narrows to outputs that technically pass each check but feel mechanical and over-fitted.

If you need more than 6 criteria, you are probably evaluating multiple distinct qualities. Split them across separate eval configs or collapse overlapping ones.

### Too Narrow/Rigid

```toml
# Too rigid -- prescribes format instead of testing quality
name = "bullet_format"
question = "Does the response use exactly 5 bullet points, each starting with a verb? Answer only YES or NO."
```

This forces a specific structure and prevents the skill from discovering better formats. Test the property you care about, not the implementation.

```toml
# Tests the underlying quality
name = "scannable"
question = "Can the key points be identified by reading only the first line of each paragraph or list item? Answer only YES or NO."
```

### Overlapping Criteria

```toml
# These overlap -- "concise" and "no filler" measure the same thing
name = "concise"
question = "Is the response under 100 words? Answer only YES or NO."

name = "no_filler"
question = "Is the response free of filler phrases and unnecessary words? Answer only YES or NO."
```

Overlapping criteria double-penalize a single failure mode. If the output is verbose, it fails both. This distorts the score -- one failure looks like two. Each criterion should test an independent property.

### Unmeasurable by an Agent

```toml
# Requires running the code or checking external state
name = "compiles"
question = "Does this code compile without errors? Answer only YES or NO."
```

An LLM judge cannot execute code. It will guess. Use deterministic eval (`run_command` + `parse_command`) for anything that requires execution, network access, or external state. Binary criteria are for properties an LLM can assess by reading the output.

```toml
# Requires domain expertise the judge may not have
name = "medically_accurate"
question = "Are all medical claims in this text factually correct? Answer only YES or NO."
```

The judge will hallucinate confidence. Only use criteria the evaluator model can reliably assess from the text alone.

---

## The 3-Question Litmus Test

Run every criterion through these three checks before shipping it.

### 1. Consistency

> Could two different agents score the same output and agree?

If the question contains subjective terms (good, clean, professional, appropriate, effective), two agents will diverge. Rewrite until the question has a deterministic answer for any given output.

**Fails:** "Is the tone appropriate for the audience?"
**Passes:** "Is the text written in second person (addressing the reader as 'you')?"

### 2. Gaming Resistance

> Could a skill game this eval without actually improving?

If a criterion can be trivially satisfied with a degenerate output, it is gameable. "Is the response under 50 words?" -- the skill can return a single sentence and pass. Pair length constraints with content requirements, or rethink the criterion.

**Fails:** "Does the code have fewer than 20 lines?" (skill outputs one-liners)
**Passes:** "Does every function fit on a single screen (under 40 lines) while handling all specified edge cases?"

### 3. Relevance

> Does this eval test something the user actually cares about?

Proxy metrics drift from actual goals. "Does the email contain exactly 3 paragraphs?" -- the user cares about conversion, not paragraph count. Every criterion should trace back to a user-visible outcome.

**Fails:** "Does the config file use alphabetical ordering for keys?"
**Passes:** "Does the config file group related settings under clearly labeled sections?"

---

## Template

Copy this for each criterion in your `eval_criteria.toml`:

```toml
[[criteria]]
name = ""          # snake_case identifier, unique within the file
question = ""      # YES/NO question. End with "Answer only YES or NO."
```

Rules for the question field:
- State what you want to be true, then ask if it is
- Use ALL CAPS for critical qualifiers: "ALL text", "NO exceptions", "ONLY pastel colors", "ZERO inline comments"
- End every question with "Answer only YES or NO."
- One property per question -- no "and" clauses
- Reference specific, observable properties, not subjective impressions

Full example for a newsletter skill:

```toml
[meta]
sample_count = 10
confidence_level = 0.95

[generation]
prompt_template = "Using the current SKILL.md, generate a newsletter for: '{test_prompt}'"
output_format = "markdown"

[[criteria]]
name = "opens_with_hook"
question = "Does the newsletter open with a specific question, statistic, or provocative statement (not a greeting)? Answer only YES or NO."

[[criteria]]
name = "single_cta"
question = "Does the newsletter contain exactly ONE call-to-action? Answer only YES or NO."

[[criteria]]
name = "under_500_words"
question = "Is the entire newsletter under 500 words? Answer only YES or NO."

[[criteria]]
name = "no_jargon"
question = "Is the newsletter free of unexplained acronyms and technical jargon? Answer only YES or NO."

[[test_prompts]]
prompt = "Weekly update for a developer tools startup"

[[test_prompts]]
prompt = "Monthly digest for an open source project with 500 contributors"

[[test_prompts]]
prompt = "Product launch announcement for a B2B SaaS platform"
```

---

## Integration with Anneal

### How Criteria Map to eval_criteria.toml

The `eval_criteria.toml` file is loaded by anneal's scope loader and parsed into `StochasticEval` config. Each `[[criteria]]` entry becomes a `BinaryCriterion(name, question)`. The `[meta]` section maps to `sample_count` and `confidence_level`. The `[[test_prompts]]` entries become the prompt set used for generation.

### Scoring: N Samples x K Criteria

For each evaluation cycle, anneal:

1. Generates **N** samples (one per test prompt) using the skill being optimized
2. Scores each sample against **K** binary criteria independently
3. Each (sample, criterion) pair produces a 0 or 1
4. The per-sample score is the sum of passed criteria (range: 0 to K)
5. The aggregate score is the mean across all N samples

A skill with 4 criteria and 10 test prompts produces 40 binary judgments per evaluation. The aggregate score ranges from 0.0 to 4.0.

### Majority Voting Reduces Noise

Each binary judgment is made **3 times** by default (`judgment_votes = 3`). The majority answer wins. This eliminates the primary source of noise in LLM-as-judge evaluation: non-deterministic API responses.

Without voting, the same (sample, criterion) pair can flip between YES and NO across runs due to sampling temperature in the judge model (even at temperature=0, API responses are not perfectly deterministic). With 3 votes, a judgment only flips if 2 of 3 calls disagree with the "true" answer -- substantially less likely.

The vote count is configurable per eval config:

```toml
[meta]
judgment_votes = 3   # default; use 5 for high-stakes evals, 1 for fast iteration
```

### Criterion Order Randomization

Anneal randomizes the order in which criteria are evaluated for each sample. This prevents anchoring bias where the judge's answer to criterion 1 influences its answer to criterion 2.

### Confidence Intervals

The N per-sample scores feed into a bootstrap confidence interval (10,000 resamples by default). Anneal uses CI overlap to determine whether an improvement is statistically significant. A mutation is KEPT only when the new score's CI lower bound exceeds the baseline score's CI upper bound (or meets the `min_improvement_threshold`).

### Constraints

For criteria that must always pass (not just improve on average), use `min_criterion_scores`:

```toml
[meta]
sample_count = 10

[min_criterion_scores]
no_fabrication = 0.9    # must pass on at least 90% of samples
cites_sources = 0.8     # must pass on at least 80% of samples
```

A mutation that improves the aggregate score but violates a constraint is BLOCKED.

---

## Composite Scoring

Single-metric optimization reliably produces degenerate solutions. A coverage
eval will generate trivially passing assertions. A latency eval will delete
features. A readability eval will add docstrings to otherwise incomprehensible
code. This is Goodhart's Law: when a measure becomes a target, it ceases to be
a good measure.

The fix is composite scoring — a primary metric the optimizer maximizes, plus
guard-rail constraints that block solutions that game the primary metric at the
expense of other quality dimensions.

### How constraints differ from primary criteria

The primary metric (stochastic score or deterministic run-cmd output) drives
the optimization. Constraints are binary pass/fail gates applied after
evaluation. A mutation is KEPT only if it improves the primary score AND passes
all constraints.

Key distinction: the optimizer never sees constraint values as part of the
objective function. It cannot trade off a constraint violation against a score
improvement. Constraints are hard walls, not soft costs.

### Adding constraints to a deterministic target

Register with `--constraint-cmd` flags alongside the primary eval:

```bash
anneal register \
  --name my-handler \
  --artifact src/handler.py \
  --eval-mode deterministic \
  --direction minimize \
  --run-cmd "ab -n 1000 -c 50 http://localhost:8080/api/handler" \
  --parse-cmd "grep 'Time per request' | head -1 | awk '{print \$4}'" \
  --constraint-cmd "radon cc src/handler.py -a -nc | tail -1 | awk '{print \$NF}'" \
  --constraint-parse "cat" \
  --constraint-threshold 10 \
  --constraint-direction lower_is_better \
  --constraint-name "cyclomatic_complexity" \
  --constraint-cmd "wc -l < src/handler.py" \
  --constraint-parse "cat" \
  --constraint-threshold 200 \
  --constraint-direction lower_is_better \
  --constraint-name "line_count"
```

The agent can reduce latency in any way that keeps cyclomatic complexity below
10 and line count below 200. An obfuscated-but-fast solution that pushes
complexity to 25 is BLOCKED.

### Adding floor constraints to a stochastic target

For stochastic evals, use `min_criterion_scores` to floor individual criteria
while the aggregate score is still optimized:

```toml
[meta]
sample_count = 10

[min_criterion_scores]
no_fabrication = 0.9    # must pass on 90%+ of samples regardless of aggregate
cites_sources = 0.8     # must pass on 80%+ of samples regardless of aggregate
```

The optimizer maximizes the aggregate (all criteria averaged), but a mutation
that scores well overall while letting `no_fabrication` drop to 0.7 is BLOCKED.

### Composite setups by domain

**Code optimization (latency/size target)**

Primary metric: latency or byte count (minimize)
Guard rails:
- Cyclomatic complexity (radon cc) — prevents obfuscated solutions
- Test pass count (pytest) — prevents feature deletion
- Line count (wc -l) — prevents single-line hacks

```bash
# radon must be installed: uv add radon
--constraint-cmd "radon cc src/ -a -nc | tail -1 | awk '{print \$NF}'" \
--constraint-parse "cat" --constraint-threshold 10 --constraint-direction lower_is_better \
--constraint-name "cyclomatic_complexity"
```

**Prompt improvement (stochastic target)**

Primary metric: aggregate criteria score (maximize)
Guard rails via `min_criterion_scores`:
- `no_fabrication = 0.9` — prevents confident but wrong answers
- `on_topic = 0.85` — prevents score gaming via unrelated high-quality content

The optimizer cannot sacrifice factual accuracy to improve scannability.

**Test coverage (deterministic target)**

Primary metric: coverage percentage (maximize)
Guard rails:
- Test pass count — prevents adding tests that pass trivially via skip/xfail
- Assertion count (grep assert) — catches tests with no assertions

```bash
--constraint-cmd "pytest tests/ -q 2>&1 | tail -1 | awk '{print \$1}'" \
--constraint-parse "cat" --constraint-threshold 1 --constraint-direction higher_is_better \
--constraint-name "passing_tests"
```

### Best practices for criterion design

**Measurable.** Every criterion must produce a numeric value (deterministic) or
a YES/NO answer (stochastic). "Better code" is not measurable. "Cyclomatic
complexity below 10" is measurable.

**Independent.** Each criterion should catch a distinct failure mode. Overlapping
constraints double-penalize a single issue and obscure what actually changed.
If two constraints always pass or fail together, collapse them into one.

**Covering different quality dimensions.** A good composite setup spans at
least three orthogonal axes: primary performance metric, structural integrity
(complexity, size, coupling), and behavioral correctness (tests pass, output
matches spec). Solutions that improve one axis while degrading another are
structurally different from genuine improvements.

**Calibrated thresholds.** Set constraint thresholds at the current baseline
plus a small margin, not at an aspirational target. A constraint set to 20%
below baseline will block almost every mutation. A constraint at 110% of
baseline gives the optimizer room to work while preventing catastrophic
regressions.
