# Code Minify Example

Deterministic optimization target: reduce Python file character count while preserving output.

## Metric

`wc -c` on `app.py` (minimize). Returns 99999 penalty if output differs from expected.

## Usage

```bash
anneal init
anneal register \
  --name code-minify \
  --artifact examples/code-minify/app.py \
  --eval-mode deterministic \
  --run-cmd "bash examples/code-minify/eval.sh" \
  --parse-cmd "cat" \
  --direction minimize \
  --scope examples/code-minify/scope.yaml

# Greedy search
anneal run --target code-minify --experiments 10

# Simulated annealing (better for this target)
anneal run --target code-minify --experiments 10 --search annealing
```

## Baseline

~3,592 characters. The verbose naming and redundant patterns give the agent clear optimization opportunities.
