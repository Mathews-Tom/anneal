"""Evaluation engine: deterministic and stochastic modes with bootstrap CI."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import openai

from anneal.engine.agent import AgentInvoker
from anneal.engine.client import compute_cost, make_client, strip_provider_prefix
from anneal.engine.eval_cache import EvalCache
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    DeterministicEval,
    Direction,
    EvalConfig,
    EvalResult,
    StochasticEval,
    VerifierCommand,
)

logger = logging.getLogger(__name__)

_API_SEMAPHORE = asyncio.Semaphore(10)
_CLAUDE_CODE_SEMAPHORE = asyncio.Semaphore(3)  # Limit concurrent Claude Code subprocesses


class EvalError(Exception):
    """Raised on evaluation failures."""


async def run_verifiers(
    worktree_path: Path,
    verifiers: list[VerifierCommand],
) -> list[tuple[str, bool, str]]:
    """Run binary pass/fail verifiers sequentially. Fail-fast on first failure.

    Returns list of (name, passed, stderr_output) tuples.
    Only returns results up to and including the first failure.
    """
    results: list[tuple[str, bool, str]] = []
    for verifier in verifiers:
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                verifier.run_command,
                cwd=worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=verifier.timeout_seconds,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            results.append((verifier.name, False, f"timed out after {verifier.timeout_seconds}s"))
            return results

        assert proc is not None
        if proc.returncode != 0:
            stderr_content = stderr_bytes.decode(errors="replace").strip()
            results.append((verifier.name, False, stderr_content))
            return results

        results.append((verifier.name, True, ""))

    return results


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
    """Runs a shell command, parses a number from output.

    Supports retry on failure and flake detection (median of 3 runs)
    for network-dependent eval commands.
    """

    async def evaluate(
        self,
        worktree_path: Path,
        config: DeterministicEval,
    ) -> EvalResult:
        if config.flake_detection:
            scores = []
            for i in range(3):
                result = await self._evaluate_with_retry(worktree_path, config)
                scores.append(result.score)
                logger.debug("Flake detection run %d/3: score=%.4f", i + 1, result.score)
            median = float(sorted(scores)[1])
            return EvalResult(score=median)
        return await self._evaluate_with_retry(worktree_path, config)

    async def _evaluate_with_retry(
        self,
        worktree_path: Path,
        config: DeterministicEval,
    ) -> EvalResult:
        last_error: EvalError | None = None
        for attempt in range(config.max_retries):
            try:
                return await self._evaluate_once(worktree_path, config)
            except EvalError as exc:
                last_error = exc
                if attempt < config.max_retries - 1:
                    logger.warning(
                        "Eval attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, config.max_retries, exc, config.retry_delay_seconds,
                    )
                    await asyncio.sleep(config.retry_delay_seconds)
        raise last_error  # type: ignore[misc]

    async def _evaluate_once(
        self,
        worktree_path: Path,
        config: DeterministicEval,
    ) -> EvalResult:
        """Single evaluation attempt: run command, parse output, return score."""
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

        assert run_proc is not None
        if run_proc.returncode != 0:
            raise EvalError(
                f"run_command exited with code {run_proc.returncode}: "
                f"{run_stderr.decode(errors='replace').strip()}"
            )

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

        assert parse_proc is not None
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

    def __init__(self) -> None:
        self._invoker = AgentInvoker()

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
        gen_cfg = config.generation_agent_config
        if gen_cfg is None:
            raise EvalError("StochasticEval requires generation_agent_config")

        # Build judgment config: explicit or synthesized from gen_cfg.evaluator_model
        judge_cfg = config.judgment_agent_config
        if judge_cfg is None:
            judge_cfg = AgentConfig(
                mode="api",
                model=gen_cfg.evaluator_model,
                evaluator_model=gen_cfg.evaluator_model,
                max_budget_usd=gen_cfg.max_budget_usd,
                temperature=1.0,
            )

        total_cost = 0.0

        # 1. Generate N samples from fixed prompts
        samples: list[str] = []
        gen_tasks = [
            self._generate_sample(
                gen_cfg,
                config.generation_prompt_template.format(
                    test_prompt=prompt,
                    artifact_content=artifact_content,
                ),
                config.output_format,
                worktree_path,
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
                        judge_cfg, sample, criterion, worktree_path,
                        votes=forward_votes, comparison_mode=comparison_mode,
                    )
                    for criterion in forward_criteria
                ]
                forward_results = await asyncio.gather(*forward_tasks)

                reverse_tasks = [
                    self._score_criterion(
                        judge_cfg, sample, criterion, worktree_path,
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
                        judge_cfg, sample, criterion, worktree_path,
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
        config: AgentConfig,
        prompt: str,
        output_format: str,
        worktree_path: Path,
    ) -> tuple[str, float]:
        """Generate a single sample. Returns (text, cost_usd).

        Dispatches to Claude Code subprocess or API based on config.mode.
        """
        if config.mode == "claude_code":
            async with _CLAUDE_CODE_SEMAPHORE:
                full_prompt = f"Generate output in {output_format} format.\n\n{prompt}"
                result = await self._invoker.invoke(
                    config, full_prompt, worktree_path,
                    time_budget_seconds=120, deployment_mode=True,
                )
                return result.raw_output, result.cost_usd

        # API path (default)
        client = make_client(config.model)
        model = strip_provider_prefix(config.model)
        try:
            async with _API_SEMAPHORE:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=config.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Generate output in {output_format} format.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise EvalError(f"Generation API call failed: {exc}") from exc
        text = response.choices[0].message.content or ""
        cost = _extract_cost(response, model)
        return text, cost

    async def _score_criterion_once(
        self,
        config: AgentConfig,
        sample: str,
        criterion: BinaryCriterion,
        worktree_path: Path,
    ) -> tuple[float, float]:
        """Single judgment call for a (sample, criterion) pair. Returns (0|1, cost_usd).

        Dispatches to Claude Code subprocess or API based on config.mode.
        """
        system_msg = (
            "You are an evaluator. Answer the following question about the "
            "provided output with exactly YES or NO. No explanation."
        )
        user_msg = (
            f"## Criterion: {criterion.name}\n\n"
            f"{criterion.question}\n\n"
            f"## Output to evaluate\n\n{sample}"
        )

        if config.mode == "claude_code":
            async with _CLAUDE_CODE_SEMAPHORE:
                full_prompt = f"{system_msg}\n\n{user_msg}"
                result = await self._invoker.invoke(
                    config, full_prompt, worktree_path,
                    time_budget_seconds=60, deployment_mode=True,
                )
                return _parse_yes_no(result.raw_output), result.cost_usd

        # API path (default)
        client = make_client(config.model)
        model = strip_provider_prefix(config.model)
        try:
            async with _API_SEMAPHORE:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=1.0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise EvalError(f"Scoring API call failed for {criterion.name}: {exc}") from exc
        raw_answer = response.choices[0].message.content or ""
        cost = _extract_cost(response, model)
        return _parse_yes_no(raw_answer), cost

    async def _score_criterion(
        self,
        config: AgentConfig,
        sample: str,
        criterion: BinaryCriterion,
        worktree_path: Path,
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
            return await self._score_criterion_once(config, sample, criterion, worktree_path)

        yes_count = 0
        total_cost = 0.0

        for i in range(votes):
            binary, cost = await self._score_criterion_once(config, sample, criterion, worktree_path)
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


def _parse_yes_no(raw: str) -> float:
    """Parse a YES/NO judgment response into a binary score (1.0 or 0.0)."""
    return 1.0 if raw.strip().upper().startswith("YES") else 0.0


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

    def __init__(self, cache: EvalCache | None = None) -> None:
        self._deterministic = DeterministicEvaluator()
        self._stochastic = StochasticEvaluator()
        self._cache = cache

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
            criteria_names = [c.name for c in eval_config.stochastic.criteria]

            if self._cache is not None:
                cached = self._cache.get(artifact_content, criteria_names)
                if cached is not None:
                    logger.info(
                        "EvalCache hit (rate=%.1f%%)", self._cache.hit_rate * 100,
                    )
                    return EvalResult(
                        score=cached.score,
                        raw_scores=list(cached.raw_scores),
                        criterion_names=list(cached.criterion_names),
                    )

            result = await self._stochastic.evaluate(
                worktree_path,
                eval_config.stochastic,
                artifact_content,
            )

            if self._cache is not None:
                self._cache.put(
                    artifact_content,
                    criteria_names,
                    result.score,
                    result.raw_scores or [],
                )

            return result

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
