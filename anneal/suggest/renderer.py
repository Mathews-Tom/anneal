"""Experiment plan renderer — human-readable summary and file writer."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from anneal.suggest.types import ExperimentSuggestion

console = Console(stderr=True)


def render_plan(suggestion: ExperimentSuggestion) -> None:
    """Print a human-readable experiment plan to the terminal."""
    lines = [
        f"[bold]{suggestion.name}[/bold]",
        "",
        f"  Problem:     {suggestion.intent.problem}",
        f"  Domain:      {suggestion.intent.domain.value}",
        f"  Eval mode:   {suggestion.eval_mode}",
        f"  Direction:   {suggestion.direction}",
        f"  Metric:      {suggestion.intent.metric_name}",
        "",
        f"  Artifacts:   {', '.join(suggestion.artifact_paths)}",
        f"  Editable:    {', '.join(suggestion.scope.editable)}",
        f"  Immutable:   {len(suggestion.scope.immutable)} files",
    ]

    if suggestion.eval_mode == "deterministic":
        lines.append(f"  Run command: {suggestion.run_command or '(not set)'}")
        lines.append(f"  Parse cmd:   {suggestion.parse_command or '(not set)'}")
    else:
        lines.append(f"  Criteria:    {len(suggestion.intent.criteria)} binary checks")
        lines.append(f"  Test prompts: {len(suggestion.test_prompts)}")

    if suggestion.warnings:
        lines.append("")
        for warning in suggestion.warnings:
            lines.append(f"  [yellow]Warning: {warning}[/yellow]")

    console.print(Panel(
        "\n".join(lines),
        title="anneal suggest — Experiment Plan",
        style="blue",
    ))


def render_criteria(suggestion: ExperimentSuggestion) -> None:
    """Print the generated evaluation criteria."""
    if not suggestion.intent.criteria:
        return

    lines: list[str] = []
    for i, c in enumerate(suggestion.intent.criteria, 1):
        lines.append(f"  {i}. [bold]{c.name}[/bold]")
        lines.append(f"     {c.question}")
        if c.pass_description:
            lines.append(f"     [green]Pass:[/green] {c.pass_description}")
        if c.fail_description:
            lines.append(f"     [red]Fail:[/red] {c.fail_description}")
        lines.append("")

    console.print(Panel(
        "\n".join(lines),
        title="Evaluation Criteria",
        style="cyan",
    ))


def write_suggestion_files(
    suggestion: ExperimentSuggestion,
    target_dir: Path,
) -> list[Path]:
    """Write all generated files to the target directory.

    Returns list of written file paths.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # scope.yaml
    scope_path = target_dir / "scope.yaml"
    scope_path.write_text(suggestion.scope.scope_yaml_content, encoding="utf-8")
    written.append(scope_path)

    # program.md
    program_path = target_dir / "program.md"
    program_path.write_text(suggestion.program_md, encoding="utf-8")
    written.append(program_path)

    # eval_criteria.toml (stochastic only)
    if suggestion.eval_criteria_toml:
        criteria_path = target_dir / "eval_criteria.toml"
        criteria_path.write_text(suggestion.eval_criteria_toml, encoding="utf-8")
        written.append(criteria_path)

    return written
