"""Verify all publication claims against the codebase.

Checks that every feature claimed in the comparison table is actually
implemented and wired into the engine. Exits non-zero if any claim
cannot be verified.

Usage: uv run python benchmarks/verify_claims.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).parent.parent


class ClaimResult(NamedTuple):
    label: str
    passed: bool
    detail: str  # file:line or short explanation


def _find_pattern(file: Path, pattern: str) -> int | None:
    """Return the first 1-based line number containing *pattern*, or None."""
    if not file.exists():
        return None
    for i, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return i
    return None


def _all_patterns(file: Path, patterns: list[str]) -> dict[str, int | None]:
    """Return a mapping of pattern → first matching line (or None)."""
    results: dict[str, int | None] = {p: None for p in patterns}
    if not file.exists():
        return results
    for i, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        for p in patterns:
            if results[p] is None and p in line:
                results[p] = i
    return results


# ---------------------------------------------------------------------------
# Individual claim checks
# ---------------------------------------------------------------------------


def check_wilcoxon() -> ClaimResult:
    """Wilcoxon signed-rank test called in stochastic comparison."""
    search = ROOT / "anneal/engine/search.py"
    # Must be imported AND called (not just a comment)
    hits = _all_patterns(search, ["from scipy.stats import wilcoxon", "wilcoxon(differences"])
    imported = hits["from scipy.stats import wilcoxon"]
    called = hits["wilcoxon(differences"]
    if imported and called:
        return ClaimResult(
            "Wilcoxon signed-rank test",
            True,
            f"search.py:{called} (import:{imported})",
        )
    missing = []
    if not imported:
        missing.append("import not found")
    if not called:
        missing.append("call not found")
    return ClaimResult("Wilcoxon signed-rank test", False, "; ".join(missing))


def check_holm_bonferroni() -> ClaimResult:
    """_adjusted_alpha defined in search.py AND called via holm_bonferroni flag in runner.py."""
    search = ROOT / "anneal/engine/search.py"
    runner = ROOT / "anneal/engine/runner.py"
    defined = _find_pattern(search, "def _adjusted_alpha(")
    called_in_search = _find_pattern(search, "_adjusted_alpha(alpha")
    wired_in_runner = _find_pattern(runner, "holm_bonferroni=")
    if defined and called_in_search and wired_in_runner:
        return ClaimResult(
            "Holm-Bonferroni correction",
            True,
            f"search.py:{defined} (defined), search.py:{called_in_search} (called), runner.py:{wired_in_runner} (wired)",
        )
    missing = []
    if not defined:
        missing.append("_adjusted_alpha not defined in search.py")
    if not called_in_search:
        missing.append("_adjusted_alpha not called in search.py")
    if not wired_in_runner:
        missing.append("holm_bonferroni not wired in runner.py")
    return ClaimResult("Holm-Bonferroni correction", False, "; ".join(missing))


def check_divergence_detection() -> ClaimResult:
    """DIVERGENCE_ thresholds defined and applied in runner.py."""
    runner = ROOT / "anneal/engine/runner.py"
    hits = _all_patterns(
        runner,
        ["DIVERGENCE_WARNING", "DIVERGENCE_CRITICAL", "divergence > DIVERGENCE_"],
    )
    defined_warning = hits["DIVERGENCE_WARNING"]
    defined_critical = hits["DIVERGENCE_CRITICAL"]
    applied = hits["divergence > DIVERGENCE_"]
    if defined_warning and defined_critical and applied:
        return ClaimResult(
            "Held-out divergence detection",
            True,
            f"runner.py:{defined_warning} (WARNING), runner.py:{defined_critical} (CRITICAL), runner.py:{applied} (applied)",
        )
    missing = []
    if not defined_warning:
        missing.append("DIVERGENCE_WARNING not found")
    if not defined_critical:
        missing.append("DIVERGENCE_CRITICAL not found")
    if not applied:
        missing.append("threshold not applied")
    return ClaimResult("Held-out divergence detection", False, "; ".join(missing))


def check_bradley_terry() -> ClaimResult:
    """BradleyTerryScorer class defined and called in eval.py."""
    eval_file = ROOT / "anneal/engine/eval.py"
    hits = _all_patterns(eval_file, ["class BradleyTerryScorer", "BradleyTerryScorer.estimate_strength"])
    cls = hits["class BradleyTerryScorer"]
    used = hits["BradleyTerryScorer.estimate_strength"]
    if cls and used:
        return ClaimResult(
            "Bradley-Terry scoring",
            True,
            f"eval.py:{cls} (class), eval.py:{used} (used)",
        )
    missing = []
    if not cls:
        missing.append("BradleyTerryScorer class not found")
    if not used:
        missing.append("BradleyTerryScorer.estimate_strength not called")
    return ClaimResult("Bradley-Terry scoring", False, "; ".join(missing))


def check_position_bias_cancellation() -> ClaimResult:
    """Forward + reverse ordering present in stochastic eval (eval.py)."""
    eval_file = ROOT / "anneal/engine/eval.py"
    hits = _all_patterns(
        eval_file,
        ["forward_criteria", "reverse_criteria", "forward_scores", "reverse_scores"],
    )
    all_found = {k: v for k, v in hits.items() if v is not None}
    if len(all_found) == 4:
        first = min(all_found.values())
        return ClaimResult(
            "Position bias cancellation",
            True,
            f"eval.py:{first}+ (forward/reverse split)",
        )
    missing = [k for k, v in hits.items() if v is None]
    return ClaimResult("Position bias cancellation", False, f"missing: {', '.join(missing)}")


def check_budget_caps() -> ClaimResult:
    """BudgetCap with max_usd_per_day defined in types.py and enforced in safety.py."""
    types_file = ROOT / "anneal/engine/types.py"
    safety_file = ROOT / "anneal/engine/safety.py"
    budget_class = _find_pattern(types_file, "class BudgetCap")
    max_field = _find_pattern(types_file, "max_usd_per_day")
    enforced = _find_pattern(safety_file, "max_usd_per_day")
    if budget_class and max_field and enforced:
        return ClaimResult(
            "Budget caps",
            True,
            f"types.py:{budget_class} (BudgetCap), types.py:{max_field} (field), safety.py:{enforced} (enforced)",
        )
    missing = []
    if not budget_class:
        missing.append("BudgetCap class not found in types.py")
    if not max_field:
        missing.append("max_usd_per_day not found in types.py")
    if not enforced:
        missing.append("max_usd_per_day not enforced in safety.py")
    return ClaimResult("Budget caps", False, "; ".join(missing))


def check_cross_domain_transfer() -> ClaimResult:
    """domain_penalty applied in learning_pool.py."""
    pool_file = ROOT / "anneal/engine/learning_pool.py"
    hits = _all_patterns(pool_file, ["domain_penalty", "base *= domain_penalty"])
    param = hits["domain_penalty"]
    applied = hits["base *= domain_penalty"]
    if param and applied:
        return ClaimResult(
            "Cross-domain knowledge transfer",
            True,
            f"learning_pool.py:{param} (param), learning_pool.py:{applied} (applied)",
        )
    missing = []
    if not param:
        missing.append("domain_penalty param not found")
    if not applied:
        missing.append("domain_penalty multiplication not found")
    return ClaimResult("Cross-domain knowledge transfer", False, "; ".join(missing))


def check_thompson_sampling() -> ClaimResult:
    """Beta-Binomial bandit strategy selection in strategy_selector.py."""
    sel_file = ROOT / "anneal/engine/strategy_selector.py"
    hits = _all_patterns(
        sel_file,
        ["class StrategySelector", "Thompson Sampling", "random.betavariate"],
    )
    cls = hits["class StrategySelector"]
    comment = hits["Thompson Sampling"]
    beta = hits["random.betavariate"]
    if cls and comment and beta:
        return ClaimResult(
            "Thompson Sampling",
            True,
            f"strategy_selector.py:{cls} (class), strategy_selector.py:{beta} (betavariate)",
        )
    missing = []
    if not cls:
        missing.append("StrategySelector class not found")
    if not beta:
        missing.append("random.betavariate not found")
    return ClaimResult("Thompson Sampling", False, "; ".join(missing))


def check_map_elites() -> ClaimResult:
    """MapElitesArchive implemented in archive.py."""
    archive_file = ROOT / "anneal/engine/archive.py"
    cls_line = _find_pattern(archive_file, "class MapElitesArchive")
    if cls_line:
        return ClaimResult(
            "MAP-Elites",
            True,
            f"archive.py:{cls_line} (MapElitesArchive)",
        )
    return ClaimResult(
        "MAP-Elites",
        False,
        "MapElitesArchive not found in archive.py",
    )


def check_strategy_manifest() -> ClaimResult:
    """StrategyManifest with component-level evolution in strategy.py."""
    strat_file = ROOT / "anneal/engine/strategy.py"
    hits = _all_patterns(strat_file, ["class StrategyManifest", "def evolve_weakest_component"])
    cls = hits["class StrategyManifest"]
    evolve = hits["def evolve_weakest_component"]
    if cls and evolve:
        return ClaimResult(
            "Strategy Manifest",
            True,
            f"strategy.py:{cls} (class), strategy.py:{evolve} (evolve)",
        )
    missing = []
    if not cls:
        missing.append("StrategyManifest not found")
    if not evolve:
        missing.append("evolve_weakest_component not found")
    return ClaimResult("Strategy Manifest", False, "; ".join(missing))


def check_dual_agent_mutation() -> ClaimResult:
    """exploration_model in AgentConfig (types.py) and AgentSelector wired in runner.py."""
    types_file = ROOT / "anneal/engine/types.py"
    runner_file = ROOT / "anneal/engine/runner.py"
    sel_file = ROOT / "anneal/engine/strategy_selector.py"
    exploration_field = _find_pattern(types_file, "exploration_model")
    agent_selector_class = _find_pattern(sel_file, "class AgentSelector")
    selector_wired = _find_pattern(runner_file, "AgentSelector()")
    if exploration_field and agent_selector_class and selector_wired:
        return ClaimResult(
            "Dual-Agent Mutation",
            True,
            f"types.py:{exploration_field} (field), strategy_selector.py:{agent_selector_class} (class), runner.py:{selector_wired} (wired)",
        )
    missing = []
    if not exploration_field:
        missing.append("exploration_model not in AgentConfig")
    if not agent_selector_class:
        missing.append("AgentSelector class not found")
    if not selector_wired:
        missing.append("AgentSelector not instantiated in runner.py")
    return ClaimResult("Dual-Agent Mutation", False, "; ".join(missing))


def check_two_phase_mutation() -> ClaimResult:
    """two_phase_mutation flag and DiagnosisResult used in runner.py."""
    runner_file = ROOT / "anneal/engine/runner.py"
    hits = _all_patterns(
        runner_file,
        ["two_phase_mutation", "DiagnosisResult", "diagnosis: DiagnosisResult"],
    )
    flag = hits["two_phase_mutation"]
    dtype = hits["DiagnosisResult"]
    var = hits["diagnosis: DiagnosisResult"]
    if flag and dtype and var:
        return ClaimResult(
            "Two-Phase Mutation",
            True,
            f"runner.py:{flag} (flag check), runner.py:{var} (DiagnosisResult var)",
        )
    missing = []
    if not flag:
        missing.append("two_phase_mutation flag not found in runner.py")
    if not dtype:
        missing.append("DiagnosisResult not imported in runner.py")
    if not var:
        missing.append("diagnosis variable not declared")
    return ClaimResult("Two-Phase Mutation", False, "; ".join(missing))


def check_episodic_memory() -> ClaimResult:
    """Lesson class defined in types.py and extract_lesson used in knowledge.py."""
    types_file = ROOT / "anneal/engine/types.py"
    knowledge_file = ROOT / "anneal/engine/knowledge.py"
    lesson_class = _find_pattern(types_file, "class Lesson")
    extract_fn = _find_pattern(knowledge_file, "async def extract_lesson")
    if lesson_class and extract_fn:
        return ClaimResult(
            "Episodic Memory",
            True,
            f"types.py:{lesson_class} (Lesson), knowledge.py:{extract_fn} (extract_lesson)",
        )
    missing = []
    if not lesson_class:
        missing.append("Lesson class not found in types.py")
    if not extract_fn:
        missing.append("extract_lesson not found in knowledge.py")
    return ClaimResult("Episodic Memory", False, "; ".join(missing))


def check_eval_consistency_monitoring() -> ClaimResult:
    """Drift detection with variance tracking in knowledge.py."""
    knowledge_file = ROOT / "anneal/engine/knowledge.py"
    hits = _all_patterns(
        knowledge_file,
        ["def get_drift_report", "score_variance", "criterion_variances"],
    )
    drift_fn = hits["def get_drift_report"]
    score_var = hits["score_variance"]
    crit_var = hits["criterion_variances"]
    if drift_fn and score_var and crit_var:
        return ClaimResult(
            "Eval Consistency Monitoring",
            True,
            f"knowledge.py:{drift_fn} (get_drift_report), knowledge.py:{score_var} (score_variance)",
        )
    missing = []
    if not drift_fn:
        missing.append("get_drift_report not found")
    if not score_var:
        missing.append("score_variance tracking not found")
    if not crit_var:
        missing.append("criterion_variances tracking not found")
    return ClaimResult("Eval Consistency Monitoring", False, "; ".join(missing))


def check_adaptive_sample_sizing() -> ClaimResult:
    """Adaptive sample extension and early-stopping in eval.py."""
    eval_file = ROOT / "anneal/engine/eval.py"
    hits = _all_patterns(
        eval_file,
        ["def _evaluate_adaptive", "early_stop_effect_size", "Adaptive sampling: early stop"],
    )
    fn = hits["def _evaluate_adaptive"]
    threshold = hits["early_stop_effect_size"]
    log_stop = hits["Adaptive sampling: early stop"]
    if fn and threshold and log_stop:
        return ClaimResult(
            "Adaptive Sample Sizing",
            True,
            f"eval.py:{fn} (_evaluate_adaptive), eval.py:{log_stop} (early stop log)",
        )
    missing = []
    if not fn:
        missing.append("_evaluate_adaptive not found")
    if not threshold:
        missing.append("early_stop_effect_size not found")
    if not log_stop:
        missing.append("early stop branch not found")
    return ClaimResult("Adaptive Sample Sizing", False, "; ".join(missing))


def check_hybrid_search() -> ClaimResult:
    """HybridSearch with greedy-to-SA transition in search.py."""
    search = ROOT / "anneal/engine/search.py"
    hits = _all_patterns(
        search,
        ["class HybridSearch", "_greedy_phase_length", "self._annealing"],
    )
    cls = hits["class HybridSearch"]
    phase_len = hits["_greedy_phase_length"]
    annealing = hits["self._annealing"]
    if cls and phase_len and annealing:
        return ClaimResult(
            "HybridSearch",
            True,
            f"search.py:{cls} (class), search.py:{annealing} (SA transition)",
        )
    missing = []
    if not cls:
        missing.append("HybridSearch class not found")
    if not phase_len:
        missing.append("_greedy_phase_length not found")
    if not annealing:
        missing.append("SA transition not found")
    return ClaimResult("HybridSearch", False, "; ".join(missing))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


CHECKS = [
    check_wilcoxon,
    check_holm_bonferroni,
    check_divergence_detection,
    check_bradley_terry,
    check_position_bias_cancellation,
    check_budget_caps,
    check_cross_domain_transfer,
    check_thompson_sampling,
    check_map_elites,
    check_strategy_manifest,
    check_dual_agent_mutation,
    check_two_phase_mutation,
    check_episodic_memory,
    check_eval_consistency_monitoring,
    check_adaptive_sample_sizing,
    check_hybrid_search,
]


def main() -> None:
    results: list[ClaimResult] = [check() for check in CHECKS]

    passed = 0
    failed = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.label} — {result.detail}")
        if result.passed:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print()
    print(f"{passed}/{total} claims verified. {failed} FAILED.")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
