"""CLI entry point for benchmark statistical analysis.

Usage examples:

    # Full analysis (stats + convergence plots + cost efficiency plots + summaries)
    uv run python benchmarks/analysis/run_analysis.py --results-dir benchmarks/raw_results/

    # Statistical tests only
    uv run python benchmarks/analysis/run_analysis.py --stats-only

    # Convergence plots only
    uv run python benchmarks/analysis/run_analysis.py --convergence-only

    # Cost efficiency plots only
    uv run python benchmarks/analysis/run_analysis.py --cost-only

    # Write outputs to a specific directory
    uv run python benchmarks/analysis/run_analysis.py --output-dir benchmarks/results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.analysis.loader import group_by_target, load_results
from benchmarks.analysis.statistics import run_all_comparisons
from benchmarks.analysis.summary import (
    compute_all_summary_stats,
    generate_attribution_analysis,
    write_json_results,
    write_markdown_summary,
)


# Default configuration pairs for pairwise comparison.
_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("control", "treatment"),
    ("greedy", "treatment"),
    ("raw", "treatment"),
    ("raw", "control"),
    ("greedy", "control"),
]

# Number of benchmark targets used in Holm-Bonferroni correction.
_N_TARGETS = 5


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_analysis",
        description="Statistical analysis and visualization for anneal benchmarks.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/raw_results"),
        metavar="DIR",
        help="Directory containing <target>-<config>-seed<N>.jsonl files. "
             "Default: benchmarks/raw_results/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        metavar="DIR",
        help="Directory for generated outputs. Default: benchmarks/results/",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only run statistical tests and write summary files. Skip plots.",
    )
    parser.add_argument(
        "--convergence-only",
        action="store_true",
        help="Only generate convergence curve plots. Skip stats and cost plots.",
    )
    parser.add_argument(
        "--cost-only",
        action="store_true",
        help="Only generate cost efficiency plots. Skip stats and convergence.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Family-wise significance level for Holm-Bonferroni. Default: 0.05",
    )
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=50,
        metavar="N",
        help="Number of experiments per run for convergence plots. Default: 50",
    )
    return parser.parse_args(argv)


def _run_stats(args: argparse.Namespace) -> int:
    """Load results, run statistical tests, and write summary outputs.

    Returns exit code (0 = success, 1 = error).
    """
    print(f"Loading results from {args.results_dir} ...", flush=True)
    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: No matching JSONL files found.", file=sys.stderr)
        return 1

    by_target = group_by_target(results)
    n_targets = len(by_target)
    print(f"Loaded {len(results)} runs across {n_targets} target(s).", flush=True)

    print("Running pairwise statistical comparisons ...", flush=True)
    comparisons = run_all_comparisons(
        results,
        pairs=_DEFAULT_PAIRS,
        n_targets=n_targets,
        metric="final_score",
        alpha=args.alpha,
    )

    summary_stats = compute_all_summary_stats(results)
    attribution = generate_attribution_analysis(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "statistical_tests.json"
    write_json_results(comparisons, summary_stats, attribution, json_path)
    print(f"Written: {json_path}", flush=True)

    md_path = args.output_dir / "summary_table.md"
    write_markdown_summary(comparisons, summary_stats, attribution, md_path)
    print(f"Written: {md_path}", flush=True)

    sig_count = sum(1 for cr in comparisons if cr.significant)
    print(
        f"Significant comparisons (p_corrected < {args.alpha}): "
        f"{sig_count}/{len(comparisons)}",
        flush=True,
    )
    return 0


def _run_convergence(args: argparse.Namespace) -> int:
    """Generate convergence curve plots.

    Returns exit code (0 = success, 1 = error).
    """
    try:
        from benchmarks.analysis.convergence import plot_convergence
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Loading results from {args.results_dir} ...", flush=True)
    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: No matching JSONL files found.", file=sys.stderr)
        return 1

    by_target = group_by_target(results)
    print(
        f"Generating convergence plots for {len(by_target)} target(s) ...",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target_id in sorted(by_target):
        out = args.output_dir / f"convergence_{target_id}.png"
        try:
            plot_convergence(
                results,
                target_id=target_id,
                output_path=out,
                n_experiments=args.n_experiments,
            )
            print(f"Written: {out}", flush=True)
        except Exception as exc:  # noqa: BLE001 — propagate descriptive messages
            print(f"WARNING: Skipped {target_id}: {exc}", file=sys.stderr)

    return 0


def _run_cost(args: argparse.Namespace) -> int:
    """Generate cost efficiency plots.

    Returns exit code (0 = success, 1 = error).
    """
    try:
        from benchmarks.analysis.cost_efficiency import plot_cost_efficiency
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Loading results from {args.results_dir} ...", flush=True)
    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: No matching JSONL files found.", file=sys.stderr)
        return 1

    by_target = group_by_target(results)
    print(
        f"Generating cost efficiency plots for {len(by_target)} target(s) ...",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target_id in sorted(by_target):
        out = args.output_dir / f"cost_efficiency_{target_id}.png"
        try:
            plot_cost_efficiency(
                results,
                target_id=target_id,
                output_path=out,
            )
            print(f"Written: {out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Skipped {target_id}: {exc}", file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the benchmark analysis CLI.

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    args = _parse_args(argv)

    only_flags = [args.stats_only, args.convergence_only, args.cost_only]
    if sum(only_flags) > 1:
        print(
            "ERROR: --stats-only, --convergence-only, and --cost-only are mutually exclusive.",
            file=sys.stderr,
        )
        return 1

    run_stats = not args.convergence_only and not args.cost_only
    run_convergence = not args.stats_only and not args.cost_only
    run_cost = not args.stats_only and not args.convergence_only

    exit_code = 0

    if run_stats:
        code = _run_stats(args)
        exit_code = max(exit_code, code)

    if run_convergence:
        code = _run_convergence(args)
        exit_code = max(exit_code, code)

    if run_cost:
        code = _run_cost(args)
        exit_code = max(exit_code, code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
