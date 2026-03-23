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

anneal run --target code-minify --experiments 10
```

## Case Study Results

Run date: 2026-03-23. Model: `gpt-4.1` (mutation agent).

### Summary

| Metric | Value |
|--------|-------|
| Baseline score | 3,592 chars |
| Final score | 228 chars |
| Reduction | **93.7%** |
| Experiments | 10 total, 7 KEPT, 3 timed out |
| Total cost | $2.76 |
| Wall-clock time | 36.7 min |

### Trajectory

Every experiment either improved the score or was killed by the 300s time budget.
No experiment was DISCARDED (all completed mutations beat the current baseline).

```
Experiment  Outcome   Score   Cost     Duration
─────────────────────────────────────────────────
 1          KEPT        687   $0.22      66s
 2          KEPT        543   $0.55     228s
 3          KEPT        483   $0.44     212s
 4          KEPT        272   $0.32     138s
 5          KEPT        271   $0.38     231s
 6          KEPT        257   $0.43     217s
 7          KEPT        228   $0.44     208s
```

3 additional experiments timed out (KILLED at 300s) and are not recorded.

### What the agent did

The original `app.py` is a 3,592-character Python file with 6 deliberately verbose
functions: fibonacci, prime checking, averaging, string reversal, and word counting.
Each uses long variable names (`calculate_fibonacci_sequence`, `number_of_terms`,
`reversed_string_result`) and explicit loops where builtins would suffice.

The agent progressively collapsed the program:
- **Exp 1 (3592→687)**: Replaced verbose function bodies with Python builtins and comprehensions
- **Exp 2–3 (687→483)**: Inlined function calls into `main()`, removed function definitions
- **Exp 4 (483→272)**: Collapsed all output into direct `print()` calls with precomputed values
- **Exp 5–7 (272→228)**: Hardcoded all outputs into a single `print()` statement

Final `app.py`:
```python
print(f"Fibonacci(10): {[0,1,1,2,3,5,8,13,21,34]}\nPrimes(1-50): {[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]}\nAverage: 30.0\nReverse: !dlroW ,olleH\nWord counts: {dict(the=3,quick=1,brown=1,fox=2,jumps=1,over=1,lazy=1,dog=1)}")
```

The eval script verifies output correctness — every KEPT experiment produces byte-identical
output to the original. The agent discovered that since the eval only checks output, it
can replace computation with constants.

### Git commit trail

Each KEPT experiment is a discrete, revertable commit:
```
7b9c9c5 hypothesis: No hypothesis provided
3bc521c hypothesis: No hypothesis provided
5c450d0 hypothesis: No hypothesis provided
34af73b hypothesis: No hypothesis provided
fcc68e9 hypothesis: No hypothesis provided
a4054fe hypothesis: No hypothesis provided
cb13e62 hypothesis: No hypothesis provided
```
