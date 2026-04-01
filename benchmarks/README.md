# Benchmarks

Validation gate benchmarks for engine components. Each benchmark validates a specific quantitative criterion.

## Running

```bash
# All benchmarks
uv run python benchmarks/bench_false_positives.py
uv run python benchmarks/bench_sa_convergence.py
uv run python benchmarks/bench_retrieval_precision.py

# Or run all sequentially
for f in benchmarks/bench_*.py; do uv run python "$f" || exit 1; done
```

## Gates

| Benchmark                      | Gate Criterion                       |
| ------------------------------ | ------------------------------------ |
| `bench_false_positives.py`     | Holm-Bonferroni reduces FP rate >50% |
| `bench_sa_convergence.py`      | Adaptive SA ≥15% better on Rastrigin |
| `bench_retrieval_precision.py` | TF-IDF precision ≥1.5x Jaccard       |

Exit code 0 = PASS, 1 = FAIL.
