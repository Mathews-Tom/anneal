# Benchmarks

Validation gate benchmarks for the enhancement plan phases. Each benchmark
validates a specific quantitative gate from `.docs/enhancement-plan.md`.

## Running

```bash
# All benchmarks
uv run python benchmarks/bench_phase2_false_positives.py
uv run python benchmarks/bench_phase3_sa_convergence.py
uv run python benchmarks/bench_phase5_retrieval_precision.py

# Or run all sequentially
for f in benchmarks/bench_*.py; do uv run python "$f" || exit 1; done
```

## Phase Gates

| Benchmark | Gate Criterion | Phase |
|-----------|---------------|-------|
| `bench_phase2_false_positives.py` | Holm-Bonferroni reduces FP rate >50% | Phase 2 |
| `bench_phase3_sa_convergence.py` | Adaptive SA ≥15% better on Rastrigin | Phase 3 |
| `bench_phase5_retrieval_precision.py` | TF-IDF precision ≥1.5x Jaccard | Phase 5 |

Exit code 0 = PASS, 1 = FAIL.
