"""Evaluation engine: deterministic and stochastic modes with bootstrap CI."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import openai

from anneal.engine.client import compute_cost, make_client, strip_provider_prefix
from anneal.engine.types import (
    BinaryCriterion,
    DeterministicEval,
    Direction,
    EvalConfig,
    EvalResult,
    StochasticEval,
)

logger = logging.getLogger(__name__)

_API_SEMAPHORE = asyncio.Semaphore(10)


class EvalError(Exception):
    """Raised on evaluation failures."""



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


class BradleyTerryScorer:
    """Bradley-Terry pairwise comparison model for criterion scoring.

    Instead of majority voting (binary YES/NO), estimates the
    probability that a sample satisfies a criterion using pairwise
    comparison with a reference. Provides calibrated uncertainty
    estimates that enable early stopping.
    """

    @staticmethod
    def estimate_strength(
        yes_count: int,
        total_votes: int,
    ) -> tuple[float, float]:
        """Estimate Bradley-Terry strength parameter with uncertainty.

        Uses Bayesian estimation with Beta(1,1) prior.
        Returns (mean_strength, uncertainty_width).

        mean = (yes + 1) / (total + 2)  # Laplace smoothing
        uncertainty = 1.96 * sqrt(mean * (1-mean) / (total + 2))  # Normal approx
        """
        alpha = yes_count + 1  # Beta prior
        beta = (total_votes - yes_count) + 1
        mean = alpha / (alpha + beta)
        n = alpha + beta
        uncertainty = 1.96 * (mean * (1 - mean) / n) ** 0.5
        return (mean, uncertainty)

    @staticmethod
    def should_stop_early(
        mean: float,
        uncertainty: float,
        threshold: float = 0.5,
    ) -> bool:
        """Check if we can stop voting early.

        Stop if the 95% CI is entirely above or below the threshold.
        """
        lower = mean - uncertainty
        upper = mean + uncertainty
        return lower > threshold or upper < threshold


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

        gen_client = make_client(gen_agent_config.model)
        eval_client = make_client(gen_agent_config.evaluator_model)
        gen_model = strip_provider_prefix(gen_agent_config.model)
        eval_model = strip_provider_prefix(gen_agent_config.evaluator_model)

        total_cost = 0.0

        # 1. Generate N samples from fixed prompts
        samples: list[str] = []
        gen_tasks = [
            self._generate_sample(
                gen_client,
                gen_model,
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
        comparison_mode = getattr(config, "comparison_mode", "majority_vote")
        per_sample_scores: list[float] = []
        per_criterion_totals: dict[str, list[float]] = {c.name: [] for c in config.criteria}
        for sample in samples:
            if votes >= 2:
                # Split votes between forward and reverse criterion order to cancel position bias
                forward_criteria = list(config.criteria)
                reverse_criteria = list(reversed(config.criteria))
                forward_votes = votes // 2
                reverse_votes = votes - forward_votes

                forward_tasks = [
                    self._score_criterion(
                        eval_client, eval_model, sample, criterion,
                        votes=forward_votes, comparison_mode=comparison_mode,
                    )
                    for criterion in forward_criteria
                ]
                forward_results = await asyncio.gather(*forward_tasks)

                reverse_tasks = [
                    self._score_criterion(
                        eval_client, eval_model, sample, criterion,
                        votes=reverse_votes, comparison_mode=comparison_mode,
                    )
                    for criterion in reverse_criteria
                ]
                reverse_results = await asyncio.gather(*reverse_tasks)

                # Merge: average forward and reverse scores per criterion
                criterion_scores: dict[str, tuple[float, float]] = {}
                for criterion, (binary_val, cost) in zip(forward_criteria, forward_results):
                    criterion_scores[criterion.name] = (binary_val, cost)
                for criterion, (binary_val, cost) in zip(reverse_criteria, reverse_results):
                    fwd_val, fwd_cost = criterion_scores[criterion.name]
                    criterion_scores[criterion.name] = (
                        (fwd_val + binary_val) / 2,
                        fwd_cost + cost,
                    )

                sample_score = 0.0
                for crit_name, (avg_val, cost) in criterion_scores.items():
                    sample_score += avg_val
                    total_cost += cost
                    per_criterion_totals[crit_name].append(avg_val)
            else:
                # Single vote: use random shuffle (existing behavior)
                shuffled_criteria = list(config.criteria)
                random.shuffle(shuffled_criteria)
                score_tasks = [
                    self._score_criterion(
                        eval_client, eval_model, sample, criterion,
                        votes=votes, comparison_mode=comparison_mode,
                    )
                    for criterion in shuffled_criteria
                ]
                criterion_results = await asyncio.gather(*score_tasks)
                sample_score = 0.0
                for criterion, (binary_val, cost) in zip(shuffled_criteria, criterion_results):
                    sample_score += binary_val
                    total_cost += cost
                    per_criterion_totals[criterion.name].append(binary_val)
            per_sample_scores.append(sample_score)

        # 3. Aggregate
        aggregate_score = float(np.mean(per_sample_scores))

        # Criterion names in stable original order
        criterion_names = [c.name for c in config.criteria]

        # Per-criterion mean scores
        per_criterion_scores = {
            name: float(np.mean(per_criterion_totals[name]))
            for name in criterion_names
        }

        # 4. Bootstrap CI with reproducible seed from hash of scores
        score_bytes = ",".join(f"{s:.6f}" for s in per_sample_scores).encode()
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
            criterion_names=criterion_names,
            per_criterion_scores=per_criterion_scores,
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
        cost = _extract_cost(response, model)
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
        cost = _extract_cost(response, model)
        return binary, cost

    async def _score_criterion(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        sample: str,
        criterion: BinaryCriterion,
        votes: int = 1,
        comparison_mode: str = "majority_vote",
    ) -> tuple[float, float]:
        """Score a (sample, criterion) pair with majority voting or Bradley-Terry.

        comparison_mode="majority_vote": existing binary voting behavior.
        comparison_mode="bradley_terry": Bradley-Terry strength estimation
            with early stopping when confidence is sufficient.

        When votes=1 and comparison_mode="majority_vote", behaves identically
        to a single judgment call.
        """
        if votes <= 1 and comparison_mode == "majority_vote":
            return await self._score_criterion_once(client, model, sample, criterion)

        yes_count = 0
        total_cost = 0.0

        for i in range(votes):
            binary, cost = await self._score_criterion_once(client, model, sample, criterion)
            yes_count += int(binary)
            total_cost += cost

            if comparison_mode == "bradley_terry" and i >= 1:
                mean, uncertainty = BradleyTerryScorer.estimate_strength(yes_count, i + 1)
                if BradleyTerryScorer.should_stop_early(mean, uncertainty):
                    return (mean, total_cost)

        if comparison_mode == "bradley_terry":
            mean, _ = BradleyTerryScorer.estimate_strength(yes_count, votes)
            return (mean, total_cost)

        # Majority vote (existing behavior)
        majority = 1.0 if yes_count > votes / 2 else 0.0
        return majority, total_cost


def _extract_cost(response: object, model: str = "") -> float:
    """Compute cost from response token usage and model pricing.

    Uses the unified compute_cost() from client.py which has per-model
    token pricing. Returns 0.0 for local models or if usage is unavailable.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return compute_cost(model, input_tokens, output_tokens)


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
