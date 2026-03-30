from __future__ import annotations

from anneal.engine.search import (
    GreedySearch,
    ParetoSearch,
    PopulationSearch,
    SimulatedAnnealingSearch,
)
from anneal.engine.types import (
    AgentConfig,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    EvalResult,
    OptimizationTarget,
    PopulationConfig,
)


def make_search_strategy(target: OptimizationTarget):
    """Instantiate search strategy from target configuration."""
    strategy_name = "greedy"
    if target.population_config:
        strategy_name = target.population_config.search_strategy

    match strategy_name:
        case "greedy":
            return GreedySearch()
        case "simulated_annealing":
            return SimulatedAnnealingSearch()
        case "population":
            pop_cfg = target.population_config
            return PopulationSearch(
                population_size=pop_cfg.population_size if pop_cfg else 4,
                tournament_size=pop_cfg.tournament_size if pop_cfg else 2,
            )
        case "pareto":
            return ParetoSearch()
        case _:
            return GreedySearch()


def _make_target(population_config: PopulationConfig | None = None) -> OptimizationTarget:
    return OptimizationTarget(
        id="test-target",
        domain_tier=DomainTier.SANDBOX,
        artifact_paths=["test.py"],
        scope_path="scope.yaml",
        scope_hash="abc123",
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="score",
            direction=Direction.HIGHER_IS_BETTER,
        ),
        agent_config=AgentConfig(
            mode="api",
            model="test-model",
            evaluator_model="test-model",
        ),
        time_budget_seconds=60,
        loop_interval_seconds=0,
        knowledge_path=".anneal/targets/test",
        worktree_path="/tmp/test-worktree",
        git_branch="test",
        baseline_score=0.5,
        population_config=population_config,
    )


def test_make_search_strategy_greedy_default():
    target = _make_target()
    strategy = make_search_strategy(target)
    assert isinstance(strategy, GreedySearch)


def test_make_search_strategy_pareto():
    target = _make_target(PopulationConfig(search_strategy="pareto"))
    strategy = make_search_strategy(target)
    assert isinstance(strategy, ParetoSearch)


def test_make_search_strategy_population():
    target = _make_target(PopulationConfig(search_strategy="population", population_size=8, tournament_size=3))
    strategy = make_search_strategy(target)
    assert isinstance(strategy, PopulationSearch)
    assert strategy._population_size == 8
    assert strategy._tournament_size == 3


def test_make_search_strategy_simulated_annealing():
    target = _make_target(PopulationConfig(search_strategy="simulated_annealing"))
    strategy = make_search_strategy(target)
    assert isinstance(strategy, SimulatedAnnealingSearch)


def test_make_search_strategy_unknown_falls_back_to_greedy():
    target = _make_target(PopulationConfig(search_strategy="nonexistent"))
    strategy = make_search_strategy(target)
    assert isinstance(strategy, GreedySearch)


def test_population_config_default_strategy_is_greedy():
    config = PopulationConfig()
    assert config.search_strategy == "greedy"


def test_pareto_search_deterministic_fallback():
    search = ParetoSearch()
    result = EvalResult(score=0.8, per_criterion_scores=None)
    kept = search.should_keep(
        challenger_result=result,
        baseline_score=0.5,
        baseline_raw_scores=None,
        direction=Direction.HIGHER_IS_BETTER,
    )
    assert kept is True
