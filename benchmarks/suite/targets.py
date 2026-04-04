"""Benchmark target definitions for the five-target suite.

Defines all five targets across three domains:
  - B1, B2: Prompt optimization (stochastic)
  - B3, B4: Code optimization (deterministic)
  - B5: Documentation quality (stochastic)

Each target references artifact and scope paths relative to the repo root.
Stochastic targets carry criteria TOML paths; deterministic targets carry
run and parse commands.

Usage: uv run python benchmarks/suite/run_suite.py --dry-run
"""

from __future__ import annotations

from benchmarks.suite.config import BenchmarkTarget

# ---------------------------------------------------------------------------
# Stochastic targets — prompt and documentation domains
# ---------------------------------------------------------------------------

B1 = BenchmarkTarget(
    id="B1",
    name="classification_prompt",
    domain="prompt",
    eval_mode="stochastic",
    criteria_count=5,
    experiment_budget=50,
    artifact_path="benchmarks/suite/artifacts/B1_classification_system_prompt.md",
    scope_path="benchmarks/suite/scopes/B1_scope.yaml",
    criteria_path="benchmarks/suite/configs/B1_classification_prompt.toml",
    direction="maximize",
)

B2 = BenchmarkTarget(
    id="B2",
    name="summarization_template",
    domain="prompt",
    eval_mode="stochastic",
    criteria_count=4,
    experiment_budget=50,
    artifact_path="benchmarks/suite/artifacts/B2_summarization_few_shot.md",
    scope_path="benchmarks/suite/scopes/B2_scope.yaml",
    criteria_path="benchmarks/suite/configs/B2_summarization_prompt.toml",
    direction="maximize",
)

B5 = BenchmarkTarget(
    id="B5",
    name="documentation_quality",
    domain="documentation",
    eval_mode="stochastic",
    criteria_count=6,
    experiment_budget=50,
    artifact_path="benchmarks/suite/artifacts/B5_technical_readme.md",
    scope_path="benchmarks/suite/scopes/B5_scope.yaml",
    criteria_path="benchmarks/suite/configs/B5_documentation_quality.toml",
    direction="maximize",
)

# ---------------------------------------------------------------------------
# Deterministic targets — code optimization domain
# ---------------------------------------------------------------------------

B3 = BenchmarkTarget(
    id="B3",
    name="python_performance",
    domain="code",
    eval_mode="deterministic",
    criteria_count=0,
    experiment_budget=50,
    artifact_path="benchmarks/suite/artifacts/B3_utility_function.py",
    scope_path="benchmarks/suite/scopes/B3_scope.yaml",
    criteria_path=None,
    run_cmd="uv run python benchmarks/suite/harness/eval_b3.py",
    parse_cmd="cat",
    direction="minimize",
)

B4 = BenchmarkTarget(
    id="B4",
    name="api_code_quality",
    domain="code",
    eval_mode="deterministic",
    criteria_count=0,
    experiment_budget=50,
    artifact_path="benchmarks/suite/artifacts/B4_api_endpoint.py",
    scope_path="benchmarks/suite/scopes/B4_scope.yaml",
    criteria_path=None,
    # {repo_root} is resolved by the suite runner to an absolute path so the
    # eval command references the *main repo's* harness, not the worktree copy.
    # This prevents the optimization agent from reading the hidden test suite.
    run_cmd="uv run python {repo_root}/benchmarks/suite/harness/run_b4_composite.py",
    parse_cmd="uv run python {repo_root}/benchmarks/suite/harness/parse_composite_score.py",
    direction="maximize",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TARGETS: dict[str, BenchmarkTarget] = {
    "B1": B1,
    "B2": B2,
    "B3": B3,
    "B4": B4,
    "B5": B5,
}


def get_target(target_id: str) -> BenchmarkTarget:
    """Return the target for the given ID.

    Raises:
        KeyError: If target_id is not in ALL_TARGETS.
    """
    if target_id not in ALL_TARGETS:
        raise KeyError(
            f"Unknown target '{target_id}'. Valid targets: {sorted(ALL_TARGETS)}"
        )
    return ALL_TARGETS[target_id]
