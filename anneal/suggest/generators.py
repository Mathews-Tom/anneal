"""Eval config generator and program.md generator."""

from __future__ import annotations

from anneal.suggest.types import ExperimentSuggestion, ProblemIntent, ScopeResult


def generate_eval_criteria_toml(intent: ProblemIntent) -> str:
    """Generate eval_criteria.toml content for stochastic evaluation.

    Produces binary criteria from the ProblemIntent's criteria list,
    along with test prompts and generation config.
    """
    lines = ["[meta]", f"sample_count = {max(len(intent.criteria) * 2, 10)}", "confidence_level = 0.95", ""]

    # Generation config
    lines.extend([
        "[generation]",
        'prompt_template = "Using the current artifact, generate output for: \'{test_prompt}\'"',
        'output_format = "text"',
        "",
    ])

    # Criteria
    for criterion in intent.criteria:
        lines.extend([
            "[[criteria]]",
            f'name = "{criterion.name}"',
            f'question = "{criterion.question}"',
            "",
        ])

    # If no criteria from LLM, add placeholder
    if not intent.criteria:
        lines.extend([
            "[[criteria]]",
            'name = "quality"',
            f'question = "Does the output demonstrate improvement in {intent.metric_name}? Answer YES or NO."',
            "",
        ])

    return "\n".join(lines)


def generate_test_prompts(intent: ProblemIntent) -> list[str]:
    """Generate test prompts appropriate for the domain and artifacts."""
    domain_prompts: dict[str, list[str]] = {
        "prompt": [
            "Generate a simple example output",
            "Generate output for a complex scenario",
            "Generate output with edge case inputs",
            "Generate output for the most common use case",
            "Generate output that tests all quality criteria",
        ],
        "document": [
            "A new developer reading this for the first time",
            "An experienced developer looking for API reference",
            "A manager evaluating whether to adopt this tool",
            "A contributor looking for setup instructions",
        ],
        "code": [
            "Standard execution with typical inputs",
            "Edge case: empty or minimal input",
            "Edge case: large or complex input",
            "Performance-critical execution path",
        ],
        "config": [
            "Default configuration scenario",
            "High-load production scenario",
            "Minimal resource scenario",
        ],
    }

    return domain_prompts.get(intent.domain.value, domain_prompts["code"])


def generate_program_md(
    intent: ProblemIntent,
    scope: ScopeResult,
    artifact_paths: list[str],
) -> str:
    """Generate program.md optimization instructions for the agent."""
    editable_list = "\n".join(f"- `{p}`" for p in artifact_paths)
    immutable_list = "\n".join(f"- `{p}`" for p in scope.immutable[:5])

    direction_text = "improve (higher is better)" if intent.direction == "maximize" else "reduce (lower is better)"

    sections = [
        f"# {intent.suggested_name} — Optimization Program",
        "",
        "## Your Role",
        "",
        f"You are optimizing the artifact files below to {direction_text} the metric: **{intent.metric_name}**.",
        "",
        f"Problem: {intent.problem}",
        "",
        "## Editable Files",
        "",
        editable_list,
        "",
        "## Immutable Files (DO NOT MODIFY)",
        "",
        immutable_list,
        "",
    ]

    # Constraints
    if intent.constraints:
        sections.extend([
            "## Constraints",
            "",
        ])
        for constraint in intent.constraints:
            sections.append(f"- {constraint}")
        sections.append("")

    # Domain-specific guidance
    guidance = _domain_guidance(intent)
    if guidance:
        sections.extend([
            "## Strategy",
            "",
            guidance,
            "",
        ])

    # Eval criteria awareness (stochastic)
    if intent.eval_mode == "stochastic" and intent.criteria:
        sections.extend([
            "## Evaluation Criteria",
            "",
            "Your changes will be scored against these binary criteria:",
            "",
        ])
        for c in intent.criteria:
            sections.append(f"- **{c.name}**: {c.question}")
        sections.append("")

    sections.extend([
        "## Mutation Rules",
        "",
        "- Make ONE targeted change per experiment",
        "- State your hypothesis before making edits: what you expect to change and why",
        "- Produce a `## Hypothesis` block at the start of your response",
        "- Produce a `## Tags` block with comma-separated mutation categories",
    ])

    return "\n".join(sections)


def _domain_guidance(intent: ProblemIntent) -> str:
    """Generate domain-specific optimization guidance."""
    guidance: dict[str, str] = {
        "code": (
            "Focus on one change at a time. Safe changes include:\n"
            "- Algorithmic improvements (better data structures, reduced complexity)\n"
            "- Removing dead code or unused imports\n"
            "- Simplifying complex expressions\n"
            "- Using built-in functions instead of manual implementations\n"
            "- Caching expensive computations\n\n"
            "Unsafe changes (verify output preservation):\n"
            "- Changing function signatures called by other code\n"
            "- Reordering operations with side effects\n"
            "- Changing string formatting or output format"
        ),
        "prompt": (
            "Focus on the weakest evaluation criterion first.\n"
            "- Add specific instructions that address common failures\n"
            "- Reword ambiguous instructions to be more explicit\n"
            "- Add anti-patterns (\"Do NOT do X\") for recurring mistakes\n"
            "- Move important instructions higher in the prompt (priority = position)\n"
            "- Add or improve examples that show correct behavior\n"
            "- Remove instructions that cause over-optimization for one criterion at the expense of others"
        ),
        "config": (
            "Adjust one parameter at a time to isolate its effect.\n"
            "- Start with parameters that have the largest theoretical impact\n"
            "- Use binary search on numeric parameters (halve or double)\n"
            "- Document the expected effect of each parameter change\n"
            "- Be cautious with parameters that interact (change one, hold others constant)"
        ),
        "document": (
            "Improve clarity and scannability.\n"
            "- Restructure sections for better information hierarchy\n"
            "- Replace vague language with specific instructions or examples\n"
            "- Add or improve headings for scannability\n"
            "- Remove redundant or outdated content\n"
            "- Ensure each section has a clear purpose"
        ),
    }
    return guidance.get(intent.domain.value, "")


def build_suggestion(
    intent: ProblemIntent,
    scope: ScopeResult,
    artifact_paths: list[str],
    eval_cmd: str | None = None,
    parse_cmd: str | None = None,
) -> ExperimentSuggestion:
    """Assemble all generated artifacts into an ExperimentSuggestion."""
    warnings: list[str] = []

    # Generate eval artifacts based on mode
    eval_criteria_toml: str | None = None
    test_prompts: list[str] = []

    if intent.eval_mode == "stochastic":
        eval_criteria_toml = generate_eval_criteria_toml(intent)
        test_prompts = generate_test_prompts(intent)
        if len(intent.criteria) < 3:
            warnings.append(
                "Fewer than 3 evaluation criteria generated. "
                "Consider adding more criteria for reliable scoring."
            )
    else:
        if not eval_cmd:
            warnings.append(
                "Deterministic mode selected but no --eval-cmd provided. "
                "You must provide a run command before running experiments."
            )

    # Generate program.md
    program_md = generate_program_md(intent, scope, artifact_paths)

    # Validate scope
    if not scope.editable:
        warnings.append("No editable files in scope. The agent will have nothing to modify.")

    return ExperimentSuggestion(
        name=intent.suggested_name,
        intent=intent,
        scope=scope,
        eval_criteria_toml=eval_criteria_toml,
        program_md=program_md,
        eval_mode=intent.eval_mode,
        run_command=eval_cmd,
        parse_command=parse_cmd,
        direction=intent.direction,
        artifact_paths=artifact_paths,
        test_prompts=test_prompts,
        warnings=warnings,
    )
