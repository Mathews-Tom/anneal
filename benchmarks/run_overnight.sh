#!/usr/bin/env bash
#
# Benchmark suite overnight orchestrator
#
# Sequentially iterates the (target, config, seed) matrix. Each combination
# is a single `run_suite.py --target X --config Y --seeds 1 --seed-start Z`
# invocation. Runs that already produced a non-empty JSONL in OUT_DIR are skipped,
# so this script is safe to restart
# after any crash, Ctrl-C, or power loss.
#
# To resume from a specific point: comment or reorder the arrays below.
# To re-run a specific combination from scratch: delete its JSONL file under
# OUT_DIR before running this script.
#
# Usage:
#   export GEMINI_API_KEY=...
#   bash benchmarks/run_overnight.sh
#
# Focused B3/B5 25-seed replication from .docs/anneal-enhancement-development-plan.md:
#   EXPERIMENT=replication bash benchmarks/run_overnight.sh
#
# Lower-cost B3/B5 10-seed smoke replication:
#   EXPERIMENT=smoke bash benchmarks/run_overnight.sh
#
# GPT-5.4 mutation route, with GPT-5.4-mini diagnosis/judge:
#   export OPENAI_API_KEY=sk-...
#   EXPERIMENT=gpt54 bash benchmarks/run_overnight.sh
#
# Use BACKEND=codex_exec to make anneal invoke Codex CLI for mutation and
# stochastic generation/judgment calls instead of direct LLM API calls.
#
# Recommended for overnight runs:
#   tmux new -s anneal
#   bash benchmarks/run_overnight.sh
#   # Ctrl-B D to detach; `tmux attach -t anneal` to reattach
#
# Symmetric expansion (all 5 targets at 10 seeds — fills B1/B2/B4 seeds 6-10):
#   SYMMETRIC=1 bash benchmarks/run_overnight.sh
#
# Custom extended-target set:
#   EXTENDED_TARGETS="B1 B4" bash benchmarks/run_overnight.sh
#
# General overrides:
#   OUT_DIR=benchmarks/raw_results_custom
#   ANALYSIS_DIR=benchmarks/results_custom
#   SEED_START=6 SEED_COUNT=25
#   TARGETS_OVERRIDE="B3 B5" CONFIGS_OVERRIDE="greedy control treatment"
#   MUTATION_MODEL=gpt-5.5 DIAGNOSIS_MODEL=gpt-5.4-mini JUDGE_MODEL=gpt-5.4-mini
#   BACKEND=codex_exec
#   DRY_RUN=1             # print runner commands without executing API work
#   TEE_SESSION_LOG=0     # disable process-substitution tee in restricted shells
#   STREAM_RUN_LOG=0      # disable live per-run log streaming
#   BENCHMARK_NO_PROGRESS_TIMEOUT_SECONDS=3600  # child run watchdog
#   B5_EXPERIMENT_BUDGET=5 B5_SAMPLE_COUNT=1 B5_JUDGMENT_VOTES=1
#                           # lower-fidelity Codex-only B5 exploratory runs

set -u  # undefined variables are errors; do NOT use -e (continue on per-run failure)

# --------------------------------------------------------------------------
# Editable scope — adjust to restart, narrow, or extend the run
# --------------------------------------------------------------------------

# Iteration order: seed → target → config.
#
# Seeds below EXTENDED_SEED_MIN run for every target.
# Seeds at or above EXTENDED_SEED_MIN run only for targets in EXTENDED_TARGETS.
#
# Default asymmetric design: B3/B5 expanded to 10 seeds for statistical power
# (Wilcoxon at N=5 cannot survive Holm-Bonferroni correction; N=10 can).
# B1/B2/B4 stay at 5 seeds because B1/B2 are eval-noise-limited and B4
# saturates quickly — additional seeds yield diminishing returns there.
#
# Env overrides:
#   SYMMETRIC=1            → run all 5 targets at all seeds (200-run grid)
#   EXTENDED_TARGETS="..." → space-separated target IDs; overrides default
EXPERIMENT="${EXPERIMENT:-default}"
TARGETS=(B3 B4 B1 B2 B5)
CONFIGS=(raw greedy control treatment)
SEEDS=(2 3 4 5 6 7 8 9 10)
EXTENDED_SEED_MIN=6
RUN_ANALYSIS="${RUN_ANALYSIS:-}"
STREAM_RUN_LOG="${STREAM_RUN_LOG:-1}"
EXTENDED_TARGETS_ENV="${EXTENDED_TARGETS:-}"
EXTENDED_TARGETS=()
B5_EXPERIMENT_BUDGET="${B5_EXPERIMENT_BUDGET:-}"
B5_SAMPLE_COUNT="${B5_SAMPLE_COUNT:-}"
B5_JUDGMENT_VOTES="${B5_JUDGMENT_VOTES:-}"

SESSION_STAMP="$(date +%Y%m%d-%H%M%S)"

build_seeds() {
  local start="$1"
  local count="$2"
  local end=$(( start + count ))
  SEEDS=()
  local seed
  for (( seed=start; seed<end; seed++ )); do
    SEEDS+=("$seed")
  done
}

split_override() {
  local raw="${1//,/ }"
  # shellcheck disable=SC2206  # word-split normalized env string into array intentionally
  echo ${raw}
}

case "$EXPERIMENT" in
  default)
    OUT_DIR="${OUT_DIR:-benchmarks/raw_results}"
    RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
    ;;
  replication)
    TARGETS=(B3 B5)
    CONFIGS=(greedy control treatment)
    build_seeds "${SEED_START:-6}" "${SEED_COUNT:-25}"
    EXTENDED_TARGETS=(B3 B5)
    OUT_DIR="${OUT_DIR:-benchmarks/raw_results_replication_${SESSION_STAMP}}"
    ANALYSIS_DIR="${ANALYSIS_DIR:-benchmarks/results_replication_${SESSION_STAMP}}"
    RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
    ;;
  smoke)
    TARGETS=(B3 B5)
    CONFIGS=(greedy treatment)
    build_seeds "${SEED_START:-6}" "${SEED_COUNT:-10}"
    EXTENDED_TARGETS=(B3 B5)
    OUT_DIR="${OUT_DIR:-benchmarks/raw_results_smoke_${SESSION_STAMP}}"
    ANALYSIS_DIR="${ANALYSIS_DIR:-benchmarks/results_smoke_${SESSION_STAMP}}"
    RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
    ;;
  gpt54 | gpt55)
    TARGETS=(B3 B5)
    CONFIGS=(treatment)
    build_seeds "${SEED_START:-6}" "${SEED_COUNT:-10}"
    EXTENDED_TARGETS=(B3 B5)
    OUT_DIR="${OUT_DIR:-benchmarks/raw_results_gpt54_${SESSION_STAMP}}"
    ANALYSIS_DIR="${ANALYSIS_DIR:-benchmarks/analysis_gpt54_${SESSION_STAMP}}"
    MUTATION_MODEL="${MUTATION_MODEL:-gpt-5.4}"
    DIAGNOSIS_MODEL="${DIAGNOSIS_MODEL:-gpt-5.4-mini}"
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
    BACKEND="${BACKEND:-codex_exec}"
    RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
    ;;
  *)
    echo "FATAL: unknown EXPERIMENT=$EXPERIMENT. Use default, replication, smoke, or gpt55." >&2
    exit 1
    ;;
esac

if [[ -n "${TARGETS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2207  # split_override intentionally emits words
  TARGETS=( $(split_override "$TARGETS_OVERRIDE") )
fi

if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2207
  CONFIGS=( $(split_override "$CONFIGS_OVERRIDE") )
fi

if [[ -n "${SEED_LIST:-}" ]]; then
  # shellcheck disable=SC2207
  SEEDS=( $(split_override "$SEED_LIST") )
elif [[ -n "${SEED_START:-}" && -n "${SEED_COUNT:-}" && "$EXPERIMENT" == "default" ]]; then
  build_seeds "$SEED_START" "$SEED_COUNT"
fi

if [[ "${SYMMETRIC:-0}" == "1" ]]; then
  EXTENDED_TARGETS=("${TARGETS[@]}")
elif [[ -n "$EXTENDED_TARGETS_ENV" ]]; then
  # shellcheck disable=SC2207
  EXTENDED_TARGETS=( $(split_override "$EXTENDED_TARGETS_ENV") )
elif (( ${#EXTENDED_TARGETS[@]} == 0 )); then
  EXTENDED_TARGETS=(B3 B5)
fi

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROUTE_TEXT="${MUTATION_MODEL:-} ${DIAGNOSIS_MODEL:-} ${JUDGE_MODEL:-}"
if [[ "${BACKEND:-api}" != "codex_exec" && "$ROUTE_TEXT" == *gpt-* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY is not set for GPT model route." >&2
  exit 1
fi

if [[ "${BACKEND:-api}" == "api" && -z "${MUTATION_MODEL:-}" && -z "${DIAGNOSIS_MODEL:-}" && -z "${JUDGE_MODEL:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "FATAL: GEMINI_API_KEY is not set for the default Gemini model route." >&2
  exit 1
fi

if [[ ! -f benchmarks/suite/run_suite.py ]]; then
  echo "FATAL: not in repo root or run_suite.py missing." >&2
  exit 1
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/anneal-uv-cache}"
export UV_CACHE_DIR

LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

SESSION_LOG="$LOG_DIR/session-${SESSION_STAMP}.log"

# Tee all output to the session log while still printing to the terminal
if [[ "${TEE_SESSION_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "$SESSION_LOG") 2>&1
fi

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

is_done() {
  local result_path="$1"
  local done_marker="${result_path}.done"
  [[ -f "$done_marker" ]]
}

count_lines() {
  [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0
}

expected_records() {
  local target="$1"
  local config="$2"
  if [[ "$config" == "raw" ]]; then
    echo 1
  elif [[ "$target" == "B5" && -n "$B5_EXPERIMENT_BUDGET" ]]; then
    echo "$B5_EXPERIMENT_BUDGET"
  else
    echo 50
  fi
}

has_complete_legacy_result() {
  local result_path="$1"
  local expected="$2"
  local lines
  lines=$(count_lines "$result_path")
  (( lines >= expected ))
}

format_duration() {
  local seconds="$1"
  local hours=$(( seconds / 3600 ))
  local minutes=$(( (seconds % 3600) / 60 ))
  local secs=$(( seconds % 60 ))
  if (( hours > 0 )); then
    printf "%dh %02dm %02ds" "$hours" "$minutes" "$secs"
  elif (( minutes > 0 )); then
    printf "%dm %02ds" "$minutes" "$secs"
  else
    printf "%ds" "$secs"
  fi
}

start_log_stream() {
  local log_file="$1"
  RUN_LOG_TAIL_PID=""
  if [[ "$STREAM_RUN_LOG" == "1" ]]; then
    tail -n +1 -f "$log_file" &
    RUN_LOG_TAIL_PID="$!"
  fi
}

stop_log_stream() {
  local stream_pid="$1"
  if [[ -n "$stream_pid" ]]; then
    sleep 0.2
    kill "$stream_pid" 2>/dev/null || true
    wait "$stream_pid" 2>/dev/null || true
  fi
}

ts() { date +%Y-%m-%dT%H:%M:%S%z; }

# --------------------------------------------------------------------------
# Signal handling
# --------------------------------------------------------------------------
#
# Ctrl-C in the terminal delivers SIGINT to the entire foreground process
# group, which kills the in-flight `uv run` child with rc=130 and returns
# control to bash. Without a trap, bash treats that as "this command
# failed" and proceeds to the next loop iteration — so a single Ctrl-C
# only kills the current run, not the batch.
#
# This trap sets a STOP flag; the loop body checks it after each run and
# breaks out of all three nested loops when set. A second Ctrl-C bypasses
# the graceful path and exits immediately, so the operator is never stuck
# waiting for a stubborn run.

STOP=0
INTERRUPTED_AT=""

_on_interrupt() {
  STOP=$(( STOP + 1 ))
  if (( STOP >= 2 )); then
    echo
    echo "[$(ts)] second interrupt received — exiting immediately"
    exit 130
  fi
  echo
  echo "[$(ts)] interrupt received — stopping after current run completes"
  echo "[$(ts)] press Ctrl-C again to exit immediately"
  INTERRUPTED_AT=$(ts)
}
trap _on_interrupt INT TERM

# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

completed=0
skipped=0
failed=0
failed_list=()

# Bash 3.2 (macOS default) has no array-membership operator — encode the
# extended-target set as a padded string so we can match with the [[ glob.
_extended_set=" ${EXTENDED_TARGETS[*]} "

# Plan the matrix once so the startup banner is honest about what runs.
total=0
for seed in "${SEEDS[@]}"; do
  for target in "${TARGETS[@]}"; do
    if (( seed >= EXTENDED_SEED_MIN )) && [[ "$_extended_set" != *" $target "* ]]; then
      continue
    fi
    total=$(( total + ${#CONFIGS[@]} ))
  done
done
current=0

echo "[$(ts)] session start — $total combinations planned"
echo "[$(ts)] EXPERIMENT=$EXPERIMENT OUT_DIR=$OUT_DIR"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[$(ts)] DRY_RUN=1"
fi
if [[ "${RUN_ANALYSIS:-0}" == "1" ]]; then
  echo "[$(ts)] ANALYSIS_DIR=${ANALYSIS_DIR:-}"
fi
if [[ -n "${MUTATION_MODEL:-}" || -n "${DIAGNOSIS_MODEL:-}" || -n "${JUDGE_MODEL:-}" ]]; then
  echo "[$(ts)] MODEL_ROUTE mutation=${MUTATION_MODEL:-default} diagnosis=${DIAGNOSIS_MODEL:-default} judge=${JUDGE_MODEL:-default}"
fi
echo "[$(ts)] BACKEND=${BACKEND:-api}"
echo "[$(ts)] STREAM_RUN_LOG=$STREAM_RUN_LOG"
echo "[$(ts)] NO_PROGRESS_TIMEOUT=${BENCHMARK_NO_PROGRESS_TIMEOUT_SECONDS:-3600}s"
if [[ -n "$B5_EXPERIMENT_BUDGET" || -n "$B5_SAMPLE_COUNT" || -n "$B5_JUDGMENT_VOTES" ]]; then
  echo "[$(ts)] B5_OVERRIDES experiment_budget=${B5_EXPERIMENT_BUDGET:-default} sample_count=${B5_SAMPLE_COUNT:-default} judgment_votes=${B5_JUDGMENT_VOTES:-default}"
fi
echo "[$(ts)] TARGETS=(${TARGETS[*]}) CONFIGS=(${CONFIGS[*]}) SEEDS=(${SEEDS[*]})"
echo "[$(ts)] seeds >= $EXTENDED_SEED_MIN restricted to: ${EXTENDED_TARGETS[*]}"
echo

for seed in "${SEEDS[@]}"; do
  for target in "${TARGETS[@]}"; do
    if (( seed >= EXTENDED_SEED_MIN )) && [[ "$_extended_set" != *" $target "* ]]; then
      continue
    fi
    for config in "${CONFIGS[@]}"; do
      current=$(( current + 1 ))
      run_id="${target}-${config}-seed${seed}"
      result_path="$OUT_DIR/${run_id}.jsonl"
      done_marker="${result_path}.done"
      run_log="$LOG_DIR/${run_id}-${SESSION_STAMP}.log"
      expected="$(expected_records "$target" "$config")"

      if is_done "$result_path" || has_complete_legacy_result "$result_path" "$expected"; then
        lines=$(count_lines "$result_path")
        echo "[$(ts)] [$current/$total] SKIP $run_id (already has $lines/$expected records)"
        skipped=$(( skipped + 1 ))
        continue
      fi

      echo "[$(ts)] [$current/$total] RUN  $run_id → $run_log"
      : > "$run_log"
      RUN_LOG_TAIL_PID=""
      start_log_stream "$run_log"
      start=$(date +%s)
      cmd=(
        uv run python benchmarks/suite/run_suite.py
        --target "$target"
        --config "$config"
        --seeds 1
        --seed-start "$seed"
        --output-dir "$OUT_DIR"
      )
      if [[ -n "${MUTATION_MODEL:-}" ]]; then
        cmd+=(--mutation-model "$MUTATION_MODEL")
      fi
      if [[ -n "${DIAGNOSIS_MODEL:-}" ]]; then
        cmd+=(--diagnosis-model "$DIAGNOSIS_MODEL")
      fi
      if [[ -n "${JUDGE_MODEL:-}" ]]; then
        cmd+=(--judge-model "$JUDGE_MODEL")
      fi
      if [[ -n "${BACKEND:-}" ]]; then
        cmd+=(--agent-mode "$BACKEND")
      fi
      if [[ "$target" == "B5" && -n "$B5_EXPERIMENT_BUDGET" ]]; then
        cmd+=(--experiment-budget "$B5_EXPERIMENT_BUDGET")
      fi
      if [[ "$target" == "B5" && -n "$B5_SAMPLE_COUNT" ]]; then
        cmd+=(--sample-count "$B5_SAMPLE_COUNT")
      fi
      if [[ "$target" == "B5" && -n "$B5_JUDGMENT_VOTES" ]]; then
        cmd+=(--judgment-votes "$B5_JUDGMENT_VOTES")
      fi
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        cmd+=(--dry-run)
      fi

      if "${cmd[@]}" > "$run_log" 2>&1; then
        rc=0
      else
        rc=$?
      fi
      stop_log_stream "$RUN_LOG_TAIL_PID"
      elapsed=$(( $(date +%s) - start ))
      elapsed_text="$(format_duration "$elapsed")"

      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[$(ts)] [$current/$total] DRY  $run_id (${elapsed_text}, rc=$rc)"
        completed=$(( completed + 1 ))
      elif (( rc == 0 )) && [[ -s "$result_path" ]]; then
        lines=$(count_lines "$result_path")
        : > "$done_marker"
        echo "[$(ts)] [$current/$total] DONE $run_id ($lines/$expected records, ${elapsed_text}, rc=$rc)"
        completed=$(( completed + 1 ))
      else
        lines=$(count_lines "$result_path")
        # rc=130 means the child received SIGINT (likely propagated from
        # our Ctrl-C). Tag it as interrupted rather than failed so the
        # post-run summary is honest about what happened.
        if (( rc == 130 )) || (( STOP )); then
          echo "[$(ts)] [$current/$total] INTR $run_id (interrupted after ${elapsed_text}, $lines records saved — partial state in .anneal/targets/${run_id%-seed*})"
        else
          if (( lines > 0 )); then
            echo "[$(ts)] [$current/$total] FAIL $run_id ($lines records saved, ${elapsed_text}, rc=$rc — see $run_log)"
          else
            echo "[$(ts)] [$current/$total] FAIL $run_id (no records, ${elapsed_text}, rc=$rc — see $run_log)"
          fi
          failed=$(( failed + 1 ))
          failed_list+=("$run_id")
        fi
      fi

      if (( STOP )); then
        echo "[$(ts)] stopping — interrupt requested at $INTERRUPTED_AT"
        break 3
      fi
    done
  done
done

analysis_rc=0
if [[ "${RUN_ANALYSIS:-0}" == "1" && "${DRY_RUN:-0}" != "1" && "$failed" -eq 0 && "$STOP" -eq 0 ]]; then
  if [[ -z "${ANALYSIS_DIR:-}" ]]; then
    echo "[$(ts)] FAIL analysis requested but ANALYSIS_DIR is empty"
    analysis_rc=1
  else
    echo "[$(ts)] ANALYZE $OUT_DIR → $ANALYSIS_DIR"
    analysis_log="$LOG_DIR/analysis-${SESSION_STAMP}.log"
    if uv run python -m benchmarks.analysis.run_analysis \
          --results-dir "$OUT_DIR" \
          --output-dir "$ANALYSIS_DIR" \
          --stats-only > "$analysis_log" 2>&1; then
      echo "[$(ts)] DONE analysis (log: $analysis_log)"
    else
      analysis_rc=$?
      echo "[$(ts)] FAIL analysis (rc=$analysis_rc, log: $analysis_log)"
    fi
  fi
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

echo
echo "================================================================"
if (( STOP )); then
  echo "[$(ts)] session interrupted (stopped at combination $current/$total)"
else
  echo "[$(ts)] session complete"
fi
echo "  Completed:              $completed"
echo "  Skipped (pre-existing): $skipped"
echo "  Failed:                 $failed"
if [[ "${RUN_ANALYSIS:-0}" == "1" ]]; then
  echo "  Analysis rc:            $analysis_rc"
fi
if (( STOP )); then
  echo "  Remaining:              $(( total - current ))"
fi
if (( ${#failed_list[@]} > 0 )); then
  echo
  echo "Failed runs (re-run this script to retry):"
  for r in "${failed_list[@]}"; do
    echo "  $r  (log: $LOG_DIR/${r}-${SESSION_STAMP}.log)"
  done
fi
echo "================================================================"
echo "Session log: $SESSION_LOG"

# Exit code reflects whether anything failed
(( failed == 0 && analysis_rc == 0 ))
