# Code Minification — Optimization Program

## Your Role

You are optimizing `app.py` to minimize its character count (`wc -c`) while preserving **exact output**.

## Critical Constraint

Running `python app.py` must produce this exact output (byte-for-byte, including newlines):

```
Fibonacci(10): [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
Primes(1-50): [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
Average: 30.0
Reverse: !dlroW ,olleH
Word counts: {'the': 3, 'quick': 1, 'brown': 1, 'fox': 2, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
```

If the output changes in any way — different formatting, different order, missing line — the experiment scores 99999 (penalty) and is discarded.

## Safe Changes

These reduce character count without breaking output:

- Shorten variable names (`fibonacci_numbers` → `fib`)
- Shorten function names (`calculate_fibonacci_sequence` → `fib_seq`)
- Remove docstrings and comments
- Remove type annotations
- Use list comprehensions instead of explicit loops
- Use built-in functions (`sum()`, `reversed()`, `Counter()`)
- Collapse multi-line expressions into single lines
- Remove unused imports
- Remove blank lines between functions

## Unsafe Changes (Will Break Output)

- Changing print format strings (the f-strings in `main()`)
- Changing the function call arguments in `main()` (e.g., `10`, `1, 50`, the number list, the strings)
- Reordering print statements
- Changing algorithm logic that alters return values
- Using `dict` subclasses or `Counter` if they change `repr()` output formatting

## Strategy

Make ONE change per experiment. Verify mentally that the output is preserved before editing. Start with the highest-impact changes: remove docstrings/comments, shorten long variable names, replace verbose loops with comprehensions.

## Hypothesis

Before each edit, state: "Removing/shortening X saves approximately Y characters. Output is preserved because Z."
