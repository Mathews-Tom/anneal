# Skill Diagram Example

Stochastic optimization target: improve a SKILL.md prompt for generating architecture diagrams.

## Metric

Binary criteria judging (maximize). 10 test prompts × 4 criteria × 3 votes with position
debiasing. Criteria: text legibility, pastel colors, linear layout, no ordinal labels.

## Usage

```bash
anneal init
anneal register \
  --name skill-diagram \
  --artifact examples/skill-diagram/SKILL.md \
  --eval-mode stochastic \
  --criteria examples/skill-diagram/eval_criteria.toml \
  --direction maximize \
  --scope examples/skill-diagram/scope.yaml

anneal run --target skill-diagram --experiments 10
```

## Case Study Results

Run date: 2026-03-23. Model: `gpt-4.1` (mutation + generation + judging).

### Summary

| Metric | Value |
|--------|-------|
| Baseline score | 1.650 |
| Best score | 1.650 |
| Experiments | 6 completed (run interrupted by API timeout on exp 8) |
| Outcomes | 1 KEPT, 5 DISCARDED |
| Total cost | $3.83 |
| Wall-clock time | ~30 min |

### Trajectory

```
Experiment  Outcome     Score   Cost     Duration
──────────────────────────────────────────────────
 1          KEPT        1.650   $0.65     211s
 2          DISCARDED   2.000   $0.67     225s
 3          DISCARDED   2.300   $0.65     255s
 4          DISCARDED   1.950   $0.68     227s
 5          DISCARDED   1.650   $0.63     188s
 6          DISCARDED   2.100   $0.56     183s
```

The run was interrupted on experiment 8 by an `APITimeoutError` during stochastic
eval (network timeout on a judge call). Experiments 2–6 scored higher than
baseline on individual runs but were DISCARDED by the statistical test — the
bootstrap CI did not show statistically significant improvement at 95% confidence.

### Observations

- **Cost per experiment**: ~$0.65 (dominated by 10 samples × 4 criteria × 3 votes = 120 judge calls)
- **Stochastic eval is expensive**: 6 experiments cost $3.83 vs code-minify's 10 experiments at $2.76
- **High variance**: scores ranged from 1.65 to 2.30 across experiments, but the statistical
  test correctly rejects mutations that don't show reliable improvement across all samples
- **API reliability**: stochastic eval's concurrent judge calls stress API rate limits;
  the timeout crash was fixed by wrapping API errors into EvalError
