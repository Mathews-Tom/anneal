"""CLI entry point for the benchmark suite runner.

Orchestrates register + run for all 5 targets across 4 configurations
and N seeds, collecting results into benchmarks/raw_results/.

Examples:
    # Preview all 200 commands without executing
    uv run python benchmarks/suite/run_suite.py --dry-run

    # Run a single target, all configurations, default 10 seeds
    uv run python benchmarks/suite/run_suite.py --target B1

    # Run all targets for the treatment configuration only
    uv run python benchmarks/suite/run_suite.py --config treatment

    # Run the full 200-run suite
    uv run python benchmarks/suite/run_suite.py --all --seeds 10

    # Run with limited concurrency to avoid API rate limits
    uv run python benchmarks/suite/run_suite.py --all --parallel 4

    # Run a single specific target + config combination
    uv run python benchmarks/suite/run_suite.py --target B1 --config treatment --seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `benchmarks.*` is importable when
# this script is invoked directly with `uv run python benchmarks/suite/run_suite.py`.
_SCRIPT_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from rich.console import Console
from rich.panel import Panel

from benchmarks.suite.config import BenchmarkConfig, BenchmarkRun
from benchmarks.suite.runner import (
    BENCHMARK_CONFIGS,
    CONFIG_BY_NAME,
    build_run_matrix,
    execute_runs,
    print_dry_run,
)
from benchmarks.suite.targets import ALL_TARGETS, get_target

console = Console()

_REPO_ROOT = Path(__file__).parent.parent.parent
_RAW_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "raw_results"

_DEFAULT_SEEDS = 10
_DEFAULT_PARALLEL = 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_suite",
        description="Benchmark suite runner for anneal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target",
        metavar="ID",
        choices=sorted(ALL_TARGETS),
        help="Run a single target (B1–B5).",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all 5 targets.",
    )

    # Config selection
    parser.add_argument(
        "--config",
        metavar="NAME",
        choices=sorted(CONFIG_BY_NAME),
        default=None,
        help="Restrict to a single configuration (raw, greedy, control, treatment).",
    )

    # Execution options
    parser.add_argument(
        "--seeds",
        type=int,
        default=_DEFAULT_SEEDS,
        metavar="N",
        help=f"Number of random seeds per (target, config) pair (default: {_DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        metavar="N",
        help="First seed value (default: 1). Seeds are [seed_start, seed_start + seeds).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=_DEFAULT_PARALLEL,
        metavar="N",
        help=f"Maximum concurrent workers (default: {_DEFAULT_PARALLEL}). "
             "Increase carefully — each worker makes live API calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_RAW_RESULTS_DIR,
        metavar="DIR",
        help="Directory for result JSONL files (default: benchmarks/raw_results/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print all commands without executing any of them.",
    )

    return parser


# ---------------------------------------------------------------------------
# Target + config resolution
# ---------------------------------------------------------------------------


def _resolve_targets(args: argparse.Namespace) -> list:
    """Return the list of BenchmarkTarget objects based on --target / --all."""
    if args.target:
        return [get_target(args.target)]
    if args.all:
        return list(ALL_TARGETS.values())
    # Neither --target nor --all: default to all targets (same as --all)
    return list(ALL_TARGETS.values())


def _resolve_configs(args: argparse.Namespace) -> list[BenchmarkConfig]:
    """Return the list of BenchmarkConfig objects based on --config."""
    if args.config:
        return [CONFIG_BY_NAME[args.config]]
    return BENCHMARK_CONFIGS


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _print_run_summary(runs: list[BenchmarkRun], dry_run: bool, parallel: int) -> None:
    target_ids = sorted({r.target.id for r in runs})
    config_names = sorted({r.config.name for r in runs})
    seeds = sorted({r.seed for r in runs})

    mode = "DRY-RUN" if dry_run else "LIVE"
    console.print(
        Panel(
            f"Targets:         {', '.join(target_ids)}\n"
            f"Configurations:  {', '.join(config_names)}\n"
            f"Seeds:           {seeds[0]}–{seeds[-1]} ({len(seeds)} total)\n"
            f"Total runs:      {len(runs)}\n"
            f"Parallel workers: {parallel}\n"
            f"Output dir:      {runs[0].output_dir if runs else 'N/A'}",
            title=f"Benchmark suite — {mode}",
            style="cyan" if dry_run else "green",
        )
    )


def _print_results_summary(results: list[dict[str, object]]) -> None:
    succeeded = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    console.print(
        Panel(
            f"Completed: {len(succeeded)}/{len(results)}\n"
            + (f"Failed:    {len(failed)}\n" if failed else ""),
            title="Suite complete",
            style="green" if not failed else "yellow",
        )
    )

    if failed:
        for r in failed:
            console.print(f"  [red]FAILED[/red] {r.get('run_id', '?')}: {r.get('error', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Require at least --target or --all when not doing dry-run from bare invocation.
    # (Bare invocation without flags defaults to all targets — that is intentional.)

    targets = _resolve_targets(args)
    configs = _resolve_configs(args)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    runs = build_run_matrix(
        targets=targets,
        configs=configs,
        seeds=seeds,
        output_dir=args.output_dir,
    )

    if not runs:
        console.print("[yellow]No runs to execute.[/yellow]")
        return 0

    _print_run_summary(runs, dry_run=args.dry_run, parallel=args.parallel)

    if args.dry_run:
        print_dry_run(runs)
        return 0

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = execute_runs(runs, dry_run=False, parallel=args.parallel)
    _print_results_summary(results)

    failed_count = sum(1 for r in results if "error" in r)
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
