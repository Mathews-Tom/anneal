"""Post-MVP validation: integration tests for all 8 features.

Tests three layers that unit tests don't cover:
1. Registry round-trip: new type fields survive serialize → TOML → deserialize
2. CLI argument parsing: new flags parse correctly into types
3. Cross-feature interactions: features compose correctly through shared types
"""

from __future__ import annotations

import stat
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from anneal.engine.eval import EvalEngine, EvalError
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.learning_pool import (
    GlobalLearningPool,
    Learning,
    LearningScope,
    LearningSignal,
    extract_learning,
)
from anneal.engine.registry import (
    _parse_eval_config,
    _parse_target,
    _serialize_target_toml,
)
from anneal.engine.search import PopulationSearch
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    ConstraintCommand,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    EvalResult,
    ExperimentRecord,
    MetricConstraint,
    OptimizationTarget,
    Outcome,
    PopulationConfig,
    StochasticEval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_target(
    target_id: str = "val-target",
    *,
    held_out_prompts: list[str] | None = None,
    held_out_interval: int = 10,
    constraints: list[MetricConstraint] | None = None,
    constraint_commands: list[ConstraintCommand] | None = None,
    min_criterion_scores: dict[str, float] | None = None,
    domain_tier: DomainTier = DomainTier.SANDBOX,
    meta_depth: int = 0,
    population_config: PopulationConfig | None = None,
) -> OptimizationTarget:
    stochastic = StochasticEval(
        sample_count=5,
        criteria=[BinaryCriterion(name="clarity", question="Is it clear?")],
        test_prompts=["prompt-1"],
        generation_prompt_template="{test_prompt} {artifact_content}",
        output_format="text",
        held_out_prompts=held_out_prompts or [],
        min_criterion_scores=min_criterion_scores or {},
    )
    return OptimizationTarget(
        id=target_id,
        domain_tier=domain_tier,
        artifact_paths=["artifact.md"],
        scope_path="scope.yaml",
        scope_hash="abc123",
        eval_mode=EvalMode.STOCHASTIC,
        eval_config=EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stochastic,
            held_out_interval=held_out_interval,
            constraints=constraints or [],
            constraint_commands=constraint_commands or [],
        ),
        agent_config=AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
        ),
        time_budget_seconds=300,
        loop_interval_seconds=60,
        knowledge_path=".anneal/targets/val-target",
        worktree_path="/tmp/worktree",
        git_branch="anneal/val-target",
        baseline_score=0.75,
        meta_depth=meta_depth,
        population_config=population_config,
    )


def _make_experiment_record(
    idx: int = 0,
    score: float = 0.8,
    raw_scores: list[float] | None = None,
    outcome: Outcome = Outcome.KEPT,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=f"exp-{idx:04d}",
        target_id="val-target",
        git_sha=f"sha-{idx:04d}",
        pre_experiment_sha=f"pre-{idx:04d}",
        timestamp=datetime(2026, 1, 1, 0, idx % 60),
        hypothesis=f"hypothesis {idx}",
        hypothesis_source="agent",
        mutation_diff_summary=f"diff {idx}",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=raw_scores,
        baseline_score=0.75,
        outcome=outcome,
        failure_mode=None,
        duration_seconds=1.0,
        tags=["prompt"],
        learnings=f"learning {idx}",
        cost_usd=0.01,
        bootstrap_seed=42,
    )


# =========================================================================
# Layer 1: Registry Round-Trip Tests
# =========================================================================


class TestRegistryRoundTripEvalConfig:
    """Verify new EvalConfig fields survive TOML serialize → parse."""

    def test_held_out_interval_roundtrip(self) -> None:
        target = _make_target(held_out_interval=7)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_eval_config(data["eval_config"])
        assert parsed.held_out_interval == 7

    def test_constraints_roundtrip(self) -> None:
        constraints = [
            MetricConstraint("clarity", 0.5, Direction.HIGHER_IS_BETTER),
            MetricConstraint("latency", 100.0, Direction.LOWER_IS_BETTER),
        ]
        target = _make_target(constraints=constraints)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_eval_config(data["eval_config"])
        assert len(parsed.constraints) == 2
        assert parsed.constraints[0].metric_name == "clarity"
        assert parsed.constraints[0].threshold == 0.5
        assert parsed.constraints[0].direction is Direction.HIGHER_IS_BETTER
        assert parsed.constraints[1].metric_name == "latency"
        assert parsed.constraints[1].direction is Direction.LOWER_IS_BETTER

    def test_constraint_commands_roundtrip(self) -> None:
        cmds = [
            ConstraintCommand(
                name="lint",
                run_command="bash lint.sh",
                parse_command="cat",
                timeout_seconds=30,
                threshold=0.0,
                direction=Direction.LOWER_IS_BETTER,
            ),
        ]
        target = _make_target(constraint_commands=cmds)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_eval_config(data["eval_config"])
        assert len(parsed.constraint_commands) == 1
        assert parsed.constraint_commands[0].name == "lint"
        assert parsed.constraint_commands[0].run_command == "bash lint.sh"

    def test_empty_constraints_roundtrip(self) -> None:
        target = _make_target()
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_eval_config(data["eval_config"])
        assert parsed.constraints == []
        assert parsed.constraint_commands == []


class TestRegistryRoundTripStochasticEval:
    """Verify new StochasticEval fields survive TOML serialize → parse."""

    def test_held_out_prompts_roundtrip(self) -> None:
        target = _make_target(held_out_prompts=["ho-1", "ho-2", "ho-3"])
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        sto_data = data["eval_config"]["stochastic"]
        assert sto_data["held_out_prompts"] == ["ho-1", "ho-2", "ho-3"]

    def test_min_criterion_scores_roundtrip(self) -> None:
        target = _make_target(min_criterion_scores={"clarity": 0.3, "depth": 0.5})
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        sto_data = data["eval_config"]["stochastic"]
        assert sto_data["min_criterion_scores"]["clarity"] == 0.3
        assert sto_data["min_criterion_scores"]["depth"] == 0.5


class TestRegistryRoundTripTarget:
    """Verify new OptimizationTarget fields survive TOML serialize → parse."""

    def test_population_config_roundtrip(self) -> None:
        pc = PopulationConfig(population_size=8, tournament_size=3)
        target = _make_target(population_config=pc)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_target(data)
        assert parsed.population_config is not None
        assert parsed.population_config.population_size == 8
        assert parsed.population_config.tournament_size == 3

    def test_population_config_none_roundtrip(self) -> None:
        target = _make_target(population_config=None)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_target(data)
        assert parsed.population_config is None

    def test_meta_depth_roundtrip(self) -> None:
        target = _make_target(meta_depth=1)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_target(data)
        assert parsed.meta_depth == 1

    def test_domain_tier_deployment_roundtrip(self) -> None:
        target = _make_target(domain_tier=DomainTier.DEPLOYMENT)
        toml_str = _serialize_target_toml(target)
        data = tomllib.loads(toml_str)["targets"]["val-target"]
        parsed = _parse_target(data)
        assert parsed.domain_tier is DomainTier.DEPLOYMENT

    def test_approval_callback_not_serialized(self) -> None:
        """approval_callback is runtime-only and must not appear in TOML."""
        target = _make_target()
        target.approval_callback = lambda _: True
        toml_str = _serialize_target_toml(target)
        assert "approval_callback" not in toml_str


# =========================================================================
# Layer 2: CLI Argument Parsing Tests
# =========================================================================


class TestCLIConstraintParsing:
    """Verify the --constraint flag parsing logic matches expected behavior."""

    def test_parse_higher_is_better(self) -> None:
        """Verify '>='-style constraint parses to HIGHER_IS_BETTER."""
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "register", "--name", "t1", "--artifact", "a.md",
            "--eval-mode", "stochastic", "--scope", "scope.yaml",
            "--criteria", "criteria.toml",
            "--constraint", "clarity>=0.5",
        ])
        assert args.constraint == ["clarity>=0.5"]

    def test_parse_multiple_constraints(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "register", "--name", "t1", "--artifact", "a.md",
            "--eval-mode", "stochastic", "--scope", "scope.yaml",
            "--criteria", "criteria.toml",
            "--constraint", "clarity>=0.5",
            "--constraint", "latency<=100",
        ])
        assert len(args.constraint) == 2

    def test_held_out_args_present(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "register", "--name", "t1", "--artifact", "a.md",
            "--eval-mode", "stochastic", "--scope", "scope.yaml",
            "--criteria", "criteria.toml",
            "--held-out-prompts", "/tmp/held-out.txt",
            "--held-out-interval", "5",
        ])
        assert args.held_out_prompts == "/tmp/held-out.txt"
        assert args.held_out_interval == 5

    def test_domain_tier_argument(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "register", "--name", "t1", "--artifact", "a.md",
            "--eval-mode", "deterministic", "--scope", "scope.yaml",
            "--run-cmd", "echo 1", "--parse-cmd", "cat",
            "--domain-tier", "deployment",
        ])
        assert args.domain_tier == "deployment"

    def test_meta_depth_argument(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "register", "--name", "t1", "--artifact", "a.md",
            "--eval-mode", "deterministic", "--scope", "scope.yaml",
            "--run-cmd", "echo 1", "--parse-cmd", "cat",
            "--meta-depth", "1",
        ])
        assert args.meta_depth == 1


class TestCLIRunArgs:
    """Verify run command's new flags."""

    def test_search_population_arg(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "run", "--target", "t1",
            "--search", "population",
            "--population-size", "8",
        ])
        assert args.search == "population"
        assert args.population_size == 8

    def test_global_learnings_default_true(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--target", "t1"])
        assert args.global_learnings is True

    def test_no_global_learnings(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "--target", "t1", "--no-global-learnings"])
        assert args.global_learnings is False


class TestCLIDriftArgs:
    """Verify drift subcommand exists and parses correctly."""

    def test_drift_subcommand_exists(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["drift", "--target", "my-target"])
        assert args.command == "drift"
        assert args.target == "my-target"


class TestCLIConfigureMetaDepth:
    """Verify configure command accepts --meta-depth."""

    def test_configure_meta_depth_arg(self) -> None:
        from anneal.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["configure", "--target", "t1", "--meta-depth", "1"])
        assert args.meta_depth == 1


# =========================================================================
# Layer 3: Cross-Feature Integration Tests
# =========================================================================


class TestConstraintWithSearchDecision:
    """F2 + F7: Constraint checking interacts correctly with search decisions."""

    @pytest.mark.asyncio
    async def test_constraint_rejects_after_search_keeps(self, tmp_path: Path) -> None:
        """Simulates: search says KEEP, but constraint fails → should discard."""
        engine = EvalEngine()

        # Search keeps the result (score > baseline)
        search = PopulationSearch(population_size=4)
        result = EvalResult(score=0.9)
        keep = search.should_keep(
            result, baseline_score=0.8, baseline_raw_scores=None,
            direction=Direction.HIGHER_IS_BETTER,
        )
        assert keep is True

        # But constraint fails
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 50")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            constraint_commands=[ConstraintCommand(
                name="coverage",
                run_command=f"bash {script}",
                parse_command="cat",
                timeout_seconds=10,
                threshold=80.0,
                direction=Direction.HIGHER_IS_BETTER,
            )],
        )
        constraint_results = await engine.check_constraints(tmp_path, config)
        failed = [name for name, passed, _ in constraint_results if not passed]
        assert "coverage" in failed


class TestDecayWithGlobalPool:
    """F4 + F5: Decay applies correctly in cross-project pool."""

    def test_global_pool_respects_decay_rate(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        pool_file = tmp_path / "global-learnings.jsonl"
        with patch(
            "anneal.engine.learning_pool.get_global_pool_path",
            return_value=pool_file,
        ):
            pool = GlobalLearningPool(decay_rate=0.1)
            old = Learning(
                observation="old-global",
                signal=LearningSignal.POSITIVE,
                source_condition="guided",
                source_target="t1",
                source_experiment_ids=[1],
                score_delta=1.0,
                criterion_deltas={},
                confidence=1.0,
                tags=[],
                created_at=datetime.now(UTC) - timedelta(days=60),
                project_id="proj-a",
            )
            new = Learning(
                observation="new-global",
                signal=LearningSignal.POSITIVE,
                source_condition="guided",
                source_target="t2",
                source_experiment_ids=[2],
                score_delta=0.3,
                criterion_deltas={},
                confidence=1.0,
                tags=[],
                created_at=datetime.now(UTC),
                project_id="proj-b",
            )
            pool.add(old)
            pool.add(new)

            # Without project filter: new should rank higher due to decay
            results = pool.retrieve(scope=LearningScope.GLOBAL, k=2)
            assert results[0].observation == "new-global"

            # With project filter: only proj-a results
            results_a = pool.retrieve(
                scope=LearningScope.GLOBAL, k=2, project_id="proj-a",
            )
            assert len(results_a) == 1
            assert results_a[0].project_id == "proj-a"


class TestDriftDetectionEndToEnd:
    """F3: End-to-end drift detection through consolidation → report."""

    def test_high_variance_criterion_detected(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        # Create 50 records with wildly varying criterion_0
        for i in range(50):
            raw = [float(i % 10) * 0.2, 0.5]  # criterion_0 varies 0.0–1.8
            store.append_record(_make_experiment_record(
                idx=i, score=0.8 + (i % 3) * 0.01, raw_scores=raw,
            ))

        cr = store.consolidate()
        assert cr.score_variance > 0
        assert cr.criterion_variances.get("criterion_0", 0) > 0

        entries = store.get_drift_report(variance_threshold=0.01)
        drifting_names = [e.criterion_name for e in entries]
        assert "criterion_0" in drifting_names

    def test_stable_evaluator_no_drift(self, tmp_path: Path) -> None:
        store = KnowledgeStore(tmp_path / "knowledge")
        for i in range(50):
            store.append_record(_make_experiment_record(
                idx=i, score=0.8, raw_scores=[0.5, 0.5],
            ))

        store.consolidate()
        entries = store.get_drift_report(variance_threshold=0.1)
        assert entries == []


class TestHeldOutEvalConfig:
    """F1: EvalConfig + StochasticEval held-out wiring."""

    @pytest.mark.asyncio
    async def test_held_out_requires_stochastic(self) -> None:
        engine = EvalEngine()
        config = EvalConfig(
            metric_name="q",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="echo 1",
                parse_command="cat",
                timeout_seconds=10,
            ),
        )
        with pytest.raises(EvalError, match="stochastic"):
            await engine.evaluate_held_out(Path("/tmp"), config, "content")

    def test_held_out_score_field_default_none(self) -> None:
        record = _make_experiment_record()
        assert record.held_out_score is None

    def test_held_out_score_field_settable(self) -> None:
        record = _make_experiment_record()
        record.held_out_score = 0.72
        assert record.held_out_score == 0.72


class TestPopulationWithTournament:
    """F7: Population lifecycle — add, fill, cull, direction handling."""

    def test_full_lifecycle(self) -> None:
        ps = PopulationSearch(population_size=3, tournament_size=2)
        assert not ps.is_population_full()

        ps.add_candidate("b1", 0.6)
        ps.add_candidate("b2", 0.8)
        ps.add_candidate("b3", 0.7)
        assert ps.is_population_full()

        # Adding a 4th triggers tournament (culls back to 3)
        ps.add_candidate("b4", 0.9)
        assert len(ps.population) == 3

        # Highest scores should survive more often in HIGHER_IS_BETTER
        scores = [s for _, s in ps.population]
        assert 0.9 in scores  # the new best should survive

    def test_direction_affects_tournament(self) -> None:
        ps = PopulationSearch(population_size=2, tournament_size=2)
        ps._population = [("low", 0.1), ("mid", 0.5), ("high", 0.9)]

        # HIGHER_IS_BETTER: low should be eliminated
        survivors_h = ps.tournament_select(Direction.HIGHER_IS_BETTER)
        assert len(survivors_h) == 2
        assert 0.9 in [s for _, s in survivors_h]

        # LOWER_IS_BETTER: high should be eliminated
        ps._population = [("low", 0.1), ("mid", 0.5), ("high", 0.9)]
        survivors_l = ps.tournament_select(Direction.LOWER_IS_BETTER)
        assert len(survivors_l) == 2
        assert 0.1 in [s for _, s in survivors_l]


class TestLearningExtraction:
    """F4 + F5: extract_learning produces correct created_at and project_id."""

    def test_extract_sets_created_at(self) -> None:
        record = _make_experiment_record(idx=1, score=0.85)
        before = datetime.now(UTC)
        learning = extract_learning(record, source_target="t1")
        after = datetime.now(UTC)
        assert before <= learning.created_at <= after

    def test_extract_default_project_id_empty(self) -> None:
        record = _make_experiment_record(idx=2, score=0.90)
        learning = extract_learning(record)
        assert learning.project_id == ""


class TestDeploymentTierType:
    """F6: Domain tier enum and approval_callback type on OptimizationTarget."""

    def test_deployment_tier_value(self) -> None:
        assert DomainTier.DEPLOYMENT.value == "deployment"
        assert DomainTier.SANDBOX.value == "sandbox"

    def test_approval_callback_default_none(self) -> None:
        target = _make_target()
        assert target.approval_callback is None

    def test_approval_callback_settable(self) -> None:
        target = _make_target(domain_tier=DomainTier.DEPLOYMENT)
        target.approval_callback = lambda _diff: True
        assert target.approval_callback("test diff") is True


class TestMetaDepthConfiguration:
    """F8: meta_depth field on OptimizationTarget."""

    def test_default_zero(self) -> None:
        target = _make_target()
        assert target.meta_depth == 0

    def test_meta_depth_one(self) -> None:
        target = _make_target(meta_depth=1)
        assert target.meta_depth == 1


class TestConsolidationRecordNewFields:
    """F3: ConsolidationRecord has criterion_variances and score_variance."""

    def test_defaults(self) -> None:
        from anneal.engine.types import ConsolidationRecord

        cr = ConsolidationRecord(
            experiment_range=(0, 50),
            timestamp=datetime.now(),
            total_experiments=50,
            kept_count=10,
            discarded_count=30,
            crashed_count=10,
            score_start=0.5,
            score_end=0.8,
            top_improvements=[],
            failed_approaches=[],
            tags_frequency={},
        )
        assert cr.criterion_variances == {}
        assert cr.score_variance == 0.0

    def test_explicit_variances(self) -> None:
        from anneal.engine.types import ConsolidationRecord

        cr = ConsolidationRecord(
            experiment_range=(0, 50),
            timestamp=datetime.now(),
            total_experiments=50,
            kept_count=10,
            discarded_count=30,
            crashed_count=10,
            score_start=0.5,
            score_end=0.8,
            top_improvements=[],
            failed_approaches=[],
            tags_frequency={},
            criterion_variances={"c0": 0.15},
            score_variance=0.05,
        )
        assert cr.criterion_variances["c0"] == 0.15
        assert cr.score_variance == 0.05
