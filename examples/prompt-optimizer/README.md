# Example: Prompt Optimization

Improve an LLM system prompt using AI-judged evaluation.

## What it does

Anneal mutates `system_prompt.md` (an article summarizer prompt), generates summaries from 5 test articles, and scores each summary against 4 binary criteria. Mutations that don't show statistically significant improvement are discarded.

**Eval mode:** Stochastic — 5 samples × 4 criteria × 3 votes per judgment.

## Run it

```bash
anneal init
anneal register \
  --name prompt-optimizer \
  --artifact examples/prompt-optimizer/system_prompt.md \
  --eval-mode stochastic \
  --criteria examples/prompt-optimizer/eval_criteria.toml \
  --direction maximize \
  --scope examples/prompt-optimizer/scope.yaml

anneal run --target prompt-optimizer --experiments 10
```

## What to expect

- ~$0.40–0.60 per experiment (LLM judge calls dominate cost)
- ~2–3 minutes per experiment
- The agent will try restructuring instructions, adding constraints, changing tone guidance
- Statistical testing ensures only reliably better prompts are kept

## Files

| File | Purpose |
|------|---------|
| `system_prompt.md` | The artifact being optimized (editable) |
| `eval_criteria.toml` | 4 binary criteria + 5 test articles (immutable) |
| `scope.yaml` | Declares what the agent can and cannot modify |
