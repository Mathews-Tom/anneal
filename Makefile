SHELL := /bin/bash
UV_CACHE_DIR ?= /private/tmp/anneal-uv-cache
export UV_CACHE_DIR

.PHONY: reproduce-paper-tables

reproduce-paper-tables:
	mkdir -p benchmarks/results benchmarks/results_seeds1-5
	set -o pipefail; \
	uv run python benchmarks/bench_false_positives.py | tee benchmarks/results/gate_false_positives.txt; fp=$${PIPESTATUS[0]}; \
	uv run python benchmarks/bench_sa_convergence.py | tee benchmarks/results/gate_sa_convergence.txt; sa=$${PIPESTATUS[0]}; \
	uv run python benchmarks/bench_retrieval_precision.py | tee benchmarks/results/gate_retrieval_precision.txt; retrieval=$${PIPESTATUS[0]}; \
	uv run python -m benchmarks.analysis.run_analysis \
		--results-dir benchmarks/raw_results_seeds1-5 \
		--output-dir benchmarks/results_seeds1-5 \
		--stats-only; analysis=$$?; \
	exit $$(( fp || sa || retrieval || analysis ))
