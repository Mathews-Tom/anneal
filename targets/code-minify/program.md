# code-minify — Optimization Program

## Your Role

You are a code golf optimizer. Your goal is to reduce the character count of `examples/code-minify/app.py` while preserving **identical output**.

## Metric

`wc -c` on `app.py` (lower is better). Current baseline: ~3,592 characters. If the output changes from expected, the score becomes 99,999 (penalty).

## Editable Files

- `examples/code-minify/app.py`

## What You Must Preserve

The script must produce this exact output when run with `python app.py`:

```
Fibonacci(10): [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
Primes(1-50): [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
Average: 30.0
Reverse: !dlroW ,olleH
Word counts: {'the': 3, 'quick': 1, 'brown': 1, 'fox': 2, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
```

## Optimization Strategies

1. **Shorten variable names** — `fibonacci_numbers` → `f`, `number_to_check` → `n`
2. **Use list comprehensions** instead of manual loops with `.append()`
3. **Use `Counter`** from collections instead of manual word counting
4. **Use `sum()/len()`** instead of manual average
5. **Use slice reversal** `s[::-1]` instead of character-by-character loop
6. **Remove docstrings and comments** — they add characters
7. **Remove type annotations** — they add characters
8. **Collapse functions** — inline small helpers into `main()`
9. **Remove unused imports** — `sys` and `Optional` are imported but may not be needed

## Constraints

- Only modify the file listed above
- The script must remain valid Python 3.12
- Output must be byte-identical to expected

## Response Format

Before making edits, write:

## Hypothesis

[One sentence describing your optimization approach]

After edits, write:

## Tags

[Comma-separated categories: e.g., rename-variables, list-comprehension, remove-docstrings]
