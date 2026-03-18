"""Tests for CompositeMetric constraint mode (F2): EvalEngine.check_constraints."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from anneal.engine.eval import EvalEngine
from anneal.engine.types import (
    BinaryCriterion,
    ConstraintCommand,
    Direction,
    EvalConfig,
    StochasticEval,
)


def _make_eval_config(
    *,
    min_criterion_scores: dict[str, float] | None = None,
    constraint_commands: list[ConstraintCommand] | None = None,
    stochastic: bool = False,
) -> EvalConfig:
    stochastic_cfg = None
    if stochastic or min_criterion_scores:
        stochastic_cfg = StochasticEval(
            sample_count=5,
            criteria=[BinaryCriterion(name="relevant", question="Is it relevant?")],
            test_prompts=["prompt"],
            generation_prompt_template="Generate: {test_prompt}",
            output_format="text",
            min_criterion_scores=min_criterion_scores or {},
        )
    return EvalConfig(
        metric_name="accuracy",
        direction=Direction.HIGHER_IS_BETTER,
        stochastic=stochastic_cfg,
        constraint_commands=constraint_commands or [],
    )


class TestCheckConstraintsStochastic:
    """Tests for stochastic min_criterion_scores constraints."""

    @pytest.mark.asyncio
    async def test_all_criteria_pass(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(
            min_criterion_scores={"relevant": 0.7, "coherent": 0.5},
        )
        per_criterion = {"relevant": 0.8, "coherent": 0.6}
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=per_criterion,
        )
        assert len(results) == 2
        assert all(passed for _, passed, _ in results)

    @pytest.mark.asyncio
    async def test_one_criterion_fails(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(
            min_criterion_scores={"relevant": 0.7, "coherent": 0.9},
        )
        per_criterion = {"relevant": 0.8, "coherent": 0.5}
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=per_criterion,
        )
        names_and_pass = {name: passed for name, passed, _ in results}
        assert names_and_pass["relevant"] is True
        assert names_and_pass["coherent"] is False

    @pytest.mark.asyncio
    async def test_missing_criterion_defaults_to_zero(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(
            min_criterion_scores={"missing_criterion": 0.5},
        )
        per_criterion = {"relevant": 0.8}
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=per_criterion,
        )
        assert len(results) == 1
        name, passed, actual = results[0]
        assert name == "missing_criterion"
        assert passed is False
        assert actual == 0.0

    @pytest.mark.asyncio
    async def test_exact_threshold_passes(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(min_criterion_scores={"relevant": 0.7})
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores={"relevant": 0.7},
        )
        _, passed, _ = results[0]
        assert passed is True

    @pytest.mark.asyncio
    async def test_no_constraints_returns_empty(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config()
        results = await engine.check_constraints(tmp_path, config)
        assert results == []

    @pytest.mark.asyncio
    async def test_no_per_criterion_scores_skips_stochastic(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(min_criterion_scores={"relevant": 0.7})
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=None,
        )
        assert results == []


class TestCheckConstraintsDeterministic:
    """Tests for deterministic constraint_commands."""

    @pytest.mark.asyncio
    async def test_constraint_command_passes(self, tmp_path: Path) -> None:
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 95")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )
        engine = EvalEngine()
        config = _make_eval_config(constraint_commands=[cmd])
        results = await engine.check_constraints(tmp_path, config)
        assert len(results) == 1
        name, passed, actual = results[0]
        assert name == "coverage"
        assert passed is True
        assert actual == pytest.approx(95.0)

    @pytest.mark.asyncio
    async def test_constraint_command_fails(self, tmp_path: Path) -> None:
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 85")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )
        engine = EvalEngine()
        config = _make_eval_config(constraint_commands=[cmd])
        results = await engine.check_constraints(tmp_path, config)
        name, passed, actual = results[0]
        assert name == "coverage"
        assert passed is False
        assert actual == pytest.approx(85.0)

    @pytest.mark.asyncio
    async def test_lower_is_better_direction(self, tmp_path: Path) -> None:
        script = tmp_path / "latency.sh"
        script.write_text("#!/bin/bash\necho 50")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="latency",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=100.0,
            direction=Direction.LOWER_IS_BETTER,
        )
        engine = EvalEngine()
        config = _make_eval_config(constraint_commands=[cmd])
        results = await engine.check_constraints(tmp_path, config)
        _, passed, _ = results[0]
        assert passed is True

    @pytest.mark.asyncio
    async def test_mixed_stochastic_and_deterministic(self, tmp_path: Path) -> None:
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 95")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )
        config = EvalConfig(
            metric_name="accuracy",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=StochasticEval(
                sample_count=5,
                criteria=[BinaryCriterion(name="relevant", question="?")],
                test_prompts=["p"],
                generation_prompt_template="G: {test_prompt}",
                output_format="text",
                min_criterion_scores={"relevant": 0.7},
            ),
            constraint_commands=[cmd],
        )
        engine = EvalEngine()
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores={"relevant": 0.8},
        )
        assert len(results) == 2
        names = [name for name, _, _ in results]
        assert "relevant" in names
        assert "coverage" in names
