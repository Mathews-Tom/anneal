"""Types for experiment scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Domain(Enum):
    """Detected optimization domain."""

    CODE = "code"
    PROMPT = "prompt"
    CONFIG = "config"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CriterionSuggestion:
    """A suggested binary evaluation criterion."""

    name: str
    question: str
    pass_description: str
    fail_description: str


@dataclass(frozen=True)
class ProblemIntent:
    """Structured intent parsed from natural-language problem description.

    Produced by the problem analyzer (LLM-based). Consumed by scope
    generator, eval config generator, and program.md generator.
    """

    problem: str
    domain: Domain
    metric_name: str
    direction: str  # "maximize" or "minimize"
    eval_mode: str  # "deterministic" or "stochastic"
    suggested_name: str
    constraints: list[str] = field(default_factory=list)
    criteria: list[CriterionSuggestion] = field(default_factory=list)


@dataclass(frozen=True)
class ScopeResult:
    """Generated scope configuration."""

    editable: list[str]
    immutable: list[str]
    scope_yaml_content: str


@dataclass(frozen=True)
class ExperimentSuggestion:
    """A complete experiment suggestion ready for user review.

    Contains all generated artifacts and metadata needed to register
    and run the experiment.
    """

    name: str
    intent: ProblemIntent
    scope: ScopeResult
    eval_criteria_toml: str | None  # stochastic only
    program_md: str
    eval_mode: str
    run_command: str | None  # deterministic only
    parse_command: str | None  # deterministic only
    direction: str
    artifact_paths: list[str]
    test_prompts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
