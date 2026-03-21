"""Template registry — curated experiment configs for common patterns."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentTemplate:
    """A curated experiment configuration template."""

    name: str
    description: str
    domain: str
    eval_mode: str
    direction: str
    metric_name: str
    artifact_patterns: list[str]
    scope_editable_patterns: list[str]
    scope_immutable_patterns: list[str]
    criteria: list[dict[str, str]] = field(default_factory=list)
    test_prompts: list[str] = field(default_factory=list)
    run_command_template: str = ""
    parse_command_template: str = ""
    program_md_guidance: str = ""


_TEMPLATES: dict[str, ExperimentTemplate] = {}


def _register(template: ExperimentTemplate) -> None:
    _TEMPLATES[template.name] = template


def get_template(name: str) -> ExperimentTemplate | None:
    return _TEMPLATES.get(name)


def list_templates() -> list[ExperimentTemplate]:
    return list(_TEMPLATES.values())


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_register(ExperimentTemplate(
    name="api-latency",
    description="Optimize API endpoint response time",
    domain="code",
    eval_mode="deterministic",
    direction="minimize",
    metric_name="p95_latency_ms",
    artifact_patterns=["src/api/**/*.py", "src/handlers/**/*.py"],
    scope_editable_patterns=["src/api/", "src/handlers/"],
    scope_immutable_patterns=["tests/", "requirements.txt", "pyproject.toml"],
    run_command_template="wrk -t4 -c100 -d10s http://localhost:8080/{endpoint} | grep Latency",
    parse_command_template="awk '{print $2}'",
    program_md_guidance=(
        "Focus on: caching, query optimization, middleware ordering, "
        "connection pooling, async I/O. Avoid: changing API contracts, "
        "removing error handling, skipping validation."
    ),
))

_register(ExperimentTemplate(
    name="bundle-size",
    description="Reduce frontend JavaScript/TypeScript bundle size",
    domain="code",
    eval_mode="deterministic",
    direction="minimize",
    metric_name="bundle_bytes",
    artifact_patterns=["src/**/*.ts", "src/**/*.tsx", "src/**/*.js"],
    scope_editable_patterns=["src/"],
    scope_immutable_patterns=["tests/", "package.json", "tsconfig.json"],
    run_command_template="npm run build && du -sb dist/ | cut -f1",
    parse_command_template="cat",
    program_md_guidance=(
        "Focus on: tree-shaking, import elimination, lazy loading, "
        "dead code removal, code splitting. Avoid: changing public API, "
        "removing polyfills without checking browser targets."
    ),
))

_register(ExperimentTemplate(
    name="test-coverage",
    description="Improve test coverage by writing better tests",
    domain="code",
    eval_mode="deterministic",
    direction="maximize",
    metric_name="coverage_percent",
    artifact_patterns=["tests/**/*.py"],
    scope_editable_patterns=["tests/"],
    scope_immutable_patterns=["src/", "pyproject.toml", "requirements.txt"],
    run_command_template="pytest --cov=src --cov-report=term | grep TOTAL",
    parse_command_template="awk '{print $4}'",
    program_md_guidance=(
        "Write new tests or improve existing ones. Focus on: uncovered "
        "branches, edge cases, error paths. Do NOT modify source code — "
        "only test files are editable."
    ),
))

_register(ExperimentTemplate(
    name="prompt-quality",
    description="Optimize a prompt or SKILL.md for output quality",
    domain="prompt",
    eval_mode="stochastic",
    direction="maximize",
    metric_name="binary_criteria_score",
    artifact_patterns=["*.md", "prompts/*.md"],
    scope_editable_patterns=["SKILL.md", "prompts/"],
    scope_immutable_patterns=["eval_criteria.toml", "scope.yaml"],
    criteria=[
        {"name": "accuracy", "question": "Is the output factually correct and complete?"},
        {"name": "formatting", "question": "Does the output follow the specified format exactly?"},
        {"name": "conciseness", "question": "Is the output free of unnecessary repetition and filler?"},
        {"name": "instruction_following", "question": "Does the output follow all instructions in the prompt?"},
    ],
    test_prompts=[
        "Generate output for a simple, common use case",
        "Generate output for a complex, multi-part request",
        "Generate output for an edge case or unusual input",
        "Generate output that tests formatting requirements",
        "Generate output that tests constraint compliance",
    ],
    program_md_guidance=(
        "Focus on the weakest criterion first. Add specific instructions, "
        "anti-patterns, or examples that address recurring failures. "
        "One change per experiment."
    ),
))

_register(ExperimentTemplate(
    name="config-tuning",
    description="Optimize configuration parameters (build, infra, database)",
    domain="config",
    eval_mode="deterministic",
    direction="minimize",
    metric_name="metric_value",
    artifact_patterns=["*.toml", "*.yaml", "*.yml", "*.json", "*.conf"],
    scope_editable_patterns=["config/"],
    scope_immutable_patterns=["src/", "tests/"],
    run_command_template="# Replace with your benchmark command",
    parse_command_template="cat",
    program_md_guidance=(
        "Adjust one parameter at a time. Use binary search on numeric "
        "parameters. Document the expected effect of each change. "
        "Be cautious with interacting parameters."
    ),
))

_register(ExperimentTemplate(
    name="doc-quality",
    description="Improve documentation clarity and scannability",
    domain="document",
    eval_mode="stochastic",
    direction="maximize",
    metric_name="doc_quality_score",
    artifact_patterns=["README.md", "docs/**/*.md"],
    scope_editable_patterns=["README.md", "docs/"],
    scope_immutable_patterns=["src/", "tests/"],
    criteria=[
        {"name": "scannable", "question": "Can a reader find the key information within 10 seconds of scanning?"},
        {"name": "complete", "question": "Does the document cover all essential topics (setup, usage, API)?"},
        {"name": "concise", "question": "Is every paragraph under 4 sentences with no redundant content?"},
        {"name": "accurate", "question": "Are all code examples, commands, and paths correct and runnable?"},
    ],
    test_prompts=[
        "A new developer reading this for the first time",
        "An experienced developer looking for API reference",
        "A contributor looking for setup and build instructions",
        "A manager evaluating whether to adopt this tool",
    ],
    program_md_guidance=(
        "Improve clarity and scannability. Restructure sections for "
        "better hierarchy. Replace vague language with specifics. "
        "Ensure each section has a clear purpose."
    ),
))

_register(ExperimentTemplate(
    name="code-minify",
    description="Reduce code size while preserving exact output",
    domain="code",
    eval_mode="deterministic",
    direction="minimize",
    metric_name="character_count",
    artifact_patterns=["*.py"],
    scope_editable_patterns=["src/"],
    scope_immutable_patterns=["tests/", "expected_output.txt"],
    run_command_template="# eval.sh that checks output correctness and returns char count",
    parse_command_template="cat",
    program_md_guidance=(
        "Shorten variable names, remove docstrings/comments, use "
        "comprehensions, use built-ins. CRITICAL: output must be "
        "preserved byte-for-byte. Verify mentally before each edit."
    ),
))
