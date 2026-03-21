"""Evaluation engine: deterministic and stochastic modes with bootstrap CI."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
from pathlib import Path

import numpy as np
import openai

from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    ConstraintCommand,
    DeterministicEval,
    Direction,
    EvalConfig,
    EvalResult,
    MetricConstraint,
    StochasticEval,
)

logger = logging.getLogger(__name__)

_API_SEMAPHORE = asyncio.Semaphore(10)


class EvalError(Exception):
    """Raised on evaluation failures."""


def _make_client(model: str) -> openai.AsyncOpenAI:
    """Build an OpenAI-compatible async client routed by model prefix."""
    if model.startswith("gemini-"):
        return openai.AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _bootstrap_ci(
    scores: list[float],
    n_resamples: int = 10_000,
    seed: int | None = None,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval on the mean of *scores*.

    Returns (lower, upper) percentile bounds.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


class DeterministicEvaluator:
    """Runs a shell command, parses a number from output."""

    async def evaluate(
        self,
        worktree_path: Path,
        config: DeterministicEval,
    ) -> EvalResult:
        # Run the command with timeout
        run_proc: asyncio.subprocess.Process | None = None
        try:
            run_proc = await asyncio.create_subprocess_shell(
                config.run_command,
                cwd=worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            run_stdout, run_stderr = await asyncio.wait_for(
                run_proc.communicate(),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            if run_proc is not None:
                run_proc.kill()
                await run_proc.wait()
            raise EvalError(
                f"run_command timed out after {config.timeout_seconds}s: {config.run_command}"
            )

        assert run_proc is not None  # Assigned above; TimeoutError re-raises
        if run_proc.returncode != 0:
            raise EvalError(
                f"run_command exited with code {run_proc.returncode}: "
                f"{run_stderr.decode(errors='replace').strip()}"
            )

        # Pipe stdout through parse_command
        parse_proc: asyncio.subprocess.Process | None = None
        try:
            parse_proc = await asyncio.create_subprocess_shell(
                config.parse_command,
                cwd=worktree_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            parse_stdout, parse_stderr = await asyncio.wait_for(
                parse_proc.communicate(input=run_stdout),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            if parse_proc is not None:
                parse_proc.kill()
                await parse_proc.wait()
            raise EvalError(
                f"parse_command timed out after {config.timeout_seconds}s: {config.parse_command}"
            )

        assert parse_proc is not None  # Assigned above; TimeoutError re-raises
        if parse_proc.returncode != 0:
            raise EvalError(
                f"parse_command exited with code {parse_proc.returncode}: "
                f"{parse_stderr.decode(errors='replace').strip()}"
            )

        output = parse_stdout.decode().strip()
        try:
            score = float(output)
        except ValueError:
            raise EvalError(f"Cannot parse score as float from parse_command output: {output!r}")

        return EvalResult(score=score)


class StochasticEvaluator:
    """Generates N samples from fixed test prompts, scores each against K criteria."""

    async def evaluate(
        self,
        worktree_path: Path,
        config: StochasticEval,
        artifact_content: str,
    ) -> EvalResult:
        return await self._evaluate_with_prompts(
            worktree_path, config, artifact_content, config.test_prompts,
        )

    async def evaluate_held_out(
        self,
        worktree_path: Path,
        config: StochasticEval,
        artifact_content: str,
    ) -> EvalResult:
        """Evaluate using held_out_prompts instead of test_prompts.

        Same logic as evaluate() but uses config.held_out_prompts.
        Raises EvalError if held_out_prompts is empty.
        """
        if not config.held_out_prompts:
            raise EvalError("No held_out_prompts configured for held-out evaluation")
        return await self._evaluate_with_prompts(
            worktree_path, config, artifact_content, config.held_out_prompts,
        )

    async def _evaluate_with_prompts(
        self,
        worktree_path: Path,
        config: StochasticEval,
        artifact_content: str,
        prompts: list[str],
    ) -> EvalResult:
        """Core evaluation logic shared between regular and held-out eval."""
        gen_agent_config = config.generation_agent_config
        if gen_agent_config is None:
            raise EvalError("StochasticEval requires generation_agent_config")

        gen_client = _make_client(gen_agent_config.model)
        eval_client = _make_client(gen_agent_config.evaluator_model)

        total_cost = 0.0

        # 1. Generate N samples from fixed prompts
        samples: list[str] = []
        gen_tasks = [
            self._generate_sample(
                gen_client,
                gen_agent_config.model,
                config.generation_prompt_template.format(
                    test_prompt=prompt,
                    artifact_content=artifact_content,
                ),
                config.output_format,
            )
            for prompt in prompts
        ]
        gen_results = await asyncio.gather(*gen_tasks)
        for text, cost in gen_results:
            samples.append(text)
            total_cost += cost

        # 2. Score each sample against K criteria independently
        votes = config.judgment_votes
        per_sample_scores: list[float] = []
        for i, sample in enumerate(samples):
            # Randomize criterion order per sample to prevent anchoring
            shuffled_criteria = list(config.criteria)
            random.shuffle(shuffled_criteria)

            score_tasks = [
                self._score_criterion(
                    eval_client,
                    gen_agent_config.evaluator_model,
                    sample,
                    criterion,
                    votes=votes,
                )
                for criterion in shuffled_criteria
            ]
            criterion_results = await asyncio.gather(*score_tasks)
            sample_score = 0.0
            for binary_val, cost in criterion_results:
                sample_score += binary_val
                total_cost += cost
            per_sample_scores.append(sample_score)

        # 3. Aggregate
        aggregate_score = float(np.mean(per_sample_scores))

        # 4. Bootstrap CI with reproducible seed from hash of scores
        score_bytes = ",".join(str(s) for s in per_sample_scores).encode()
        seed = int(hashlib.sha256(score_bytes).hexdigest(), 16) % 2**32
        ci_lower, ci_upper = _bootstrap_ci(
            per_sample_scores,
            seed=seed,
            confidence=config.confidence_level,
        )

        return EvalResult(
            score=aggregate_score,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            raw_scores=per_sample_scores,
            cost_usd=total_cost,
        )

    async def _generate_sample(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        prompt: str,
        output_format: str,
    ) -> tuple[str, float]:
        """Generate a single sample. Returns (text, cost_usd)."""
        async with _API_SEMAPHORE:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate output in {output_format} format.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        text = response.choices[0].message.content or ""
        cost = _extract_cost(response)
        return text, cost

    async def _score_criterion_once(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        sample: str,
        criterion: BinaryCriterion,
    ) -> tuple[float, float]:
        """Single judgment call for a (sample, criterion) pair. Returns (0|1, cost_usd)."""
        async with _API_SEMAPHORE:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluator. Answer the following question about the "
                            "provided output with exactly YES or NO. No explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"## Criterion: {criterion.name}\n\n"
                            f"{criterion.question}\n\n"
                            f"## Output to evaluate\n\n{sample}"
                        ),
                    },
                ],
            )
        answer = (response.choices[0].message.content or "").strip().upper()
        binary = 1.0 if answer.startswith("YES") else 0.0
        cost = _extract_cost(response)
        return binary, cost

    async def _score_criterion(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        sample: str,
        criterion: BinaryCriterion,
        votes: int = 1,
    ) -> tuple[float, float]:
        """Score a (sample, criterion) pair with majority voting.

        When votes=1, behaves identically to a single judgment call.
        When votes>1, makes N independent calls and returns the majority
        answer. This eliminates judgment variance from non-deterministic
        API responses while preserving generation diversity.
        """
        if votes <= 1:
            return await self._score_criterion_once(client, model, sample, criterion)

        vote_tasks = [
            self._score_criterion_once(client, model, sample, criterion)
            for _ in range(votes)
        ]
        vote_results = await asyncio.gather(*vote_tasks)

        total_cost = sum(cost for _, cost in vote_results)
        yes_count = sum(1 for binary, _ in vote_results if binary > 0.5)
        majority = 1.0 if yes_count > votes / 2 else 0.0
        return majority, total_cost


def _extract_cost(response: object) -> float:  # openai.types.chat.ChatCompletion
    """Extract cost from response usage. Returns 0.0 if unavailable."""
    usage = response.usage
    if usage is None:
        return 0.0
    # OpenAI responses don't expose cost directly; approximate from tokens.
    # The runner accumulates actual costs from billing. Return 0.0 here
    # when the API doesn't provide cost metadata.
    return 0.0


class EvalEngine:
    """Dispatcher that routes to the appropriate evaluator."""

    def __init__(self) -> None:
        self._deterministic = DeterministicEvaluator()
        self._stochastic = StochasticEvaluator()

    async def evaluate(
        self,
        worktree_path: Path,
        eval_config: EvalConfig,
        artifact_content: str | None = None,
    ) -> EvalResult:
        if eval_config.deterministic is not None:
            return await self._deterministic.evaluate(worktree_path, eval_config.deterministic)

        if eval_config.stochastic is not None:
            if artifact_content is None:
                raise EvalError("Stochastic evaluation requires artifact_content")
            return await self._stochastic.evaluate(
                worktree_path,
                eval_config.stochastic,
                artifact_content,
            )

        raise EvalError("EvalConfig has neither deterministic nor stochastic configuration")

    async def evaluate_held_out(
        self,
        worktree_path: Path,
        eval_config: EvalConfig,
        artifact_content: str,
    ) -> EvalResult:
        """Run held-out evaluation. Only for stochastic targets."""
        if eval_config.stochastic is None:
            raise EvalError("Held-out evaluation requires stochastic config")
        if not eval_config.stochastic.held_out_prompts:
            raise EvalError("No held_out_prompts configured")
        return await self._stochastic.evaluate_held_out(
            worktree_path, eval_config.stochastic, artifact_content,
        )

    async def check_constraints(
        self,
        worktree_path: Path,
        eval_config: EvalConfig,
        artifact_content: str | None = None,
        per_criterion_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, bool, float]]:
        """Check all constraints. Returns list of (name, passed, actual_value).

        Checks two types of constraints:
        1. Stochastic min_criterion_scores: per-criterion floor values
        2. Deterministic constraint_commands: secondary eval commands with thresholds
        """
        results: list[tuple[str, bool, float]] = []

        # 1. Stochastic min_criterion_scores
        if (
            eval_config.stochastic is not None
            and eval_config.stochastic.min_criterion_scores
            and per_criterion_scores is not None
        ):
            for criterion_name, threshold in eval_config.stochastic.min_criterion_scores.items():
                actual = per_criterion_scores.get(criterion_name, 0.0)
                passed = actual >= threshold
                results.append((criterion_name, passed, actual))

        # 2. Deterministic constraint_commands
        for cmd in eval_config.constraint_commands:
            det_eval = DeterministicEval(
                run_command=cmd.run_command,
                parse_command=cmd.parse_command,
                timeout_seconds=cmd.timeout_seconds,
            )
            eval_result = await self._deterministic.evaluate(worktree_path, det_eval)
            if cmd.direction is Direction.HIGHER_IS_BETTER:
                passed = eval_result.score >= cmd.threshold
            else:
                passed = eval_result.score <= cmd.threshold
            results.append((cmd.name, passed, eval_result.score))

        return results
