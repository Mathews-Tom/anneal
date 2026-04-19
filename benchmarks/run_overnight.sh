#!/usr/bin/env bash
#
# Benchmark suite overnight orchestrator
#
# Sequentially iterates the (target, config, seed) matrix. Each combination
# is a single `run_suite.py --target X --config Y --seeds 1 --seed-start Z`
# invocation. Runs that already produced a non-empty JSONL in
# benchmarks/raw_results/ are skipped, so this script is safe to restart
# after any crash, Ctrl-C, or power loss.
#
# To resume from a specific point: comment or reorder the arrays below.
# To re-run a specific combination from scratch: delete its JSONL file
# under benchmarks/raw_results/ before running this script.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash benchmarks/run_overnight.sh
#
# Recommended for overnight runs:
#   tmux new -s anneal
#   bash benchmarks/run_overnight.sh
#   # Ctrl-B D to detach; `tmux attach -t anneal` to reattach

set -u  # undefined variables are errors; do NOT use -e (continue on per-run failure)

# --------------------------------------------------------------------------
# Editable scope — adjust to restart, narrow, or extend the run
# --------------------------------------------------------------------------

# Iteration order: seed → target → config.
# Leading with deterministic targets (B3, B4) validates the pipeline cheaply
# before committing stochastic budget on B1/B2/B5.
TARGETS=(B3 B4 B1 B2 B5)
CONFIGS=(raw greedy control treatment)
SEEDS=(2 3 4 5)

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY is not set. Export it before running." >&2
  exit 1
fi

if [[ ! -f benchmarks/suite/run_suite.py ]]; then
  echo "FATAL: not in repo root or run_suite.py missing." >&2
  exit 1
fi

LOG_DIR="benchmarks/raw_results/_logs"
mkdir -p "$LOG_DIR"

SESSION_STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_LOG="$LOG_DIR/session-${SESSION_STAMP}.log"

# Tee all output to the session log while still printing to the terminal
exec > >(tee -a "$SESSION_LOG") 2>&1

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

is_done() {
  # Returns 0 iff the JSONL result exists and is non-empty.
  [[ -s "$1" ]]
}

count_lines() {
  [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0
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

total=$(( ${#SEEDS[@]} * ${#TARGETS[@]} * ${#CONFIGS[@]} ))
current=0

echo "[$(ts)] session start — $total combinations planned"
echo "[$(ts)] TARGETS=(${TARGETS[*]}) CONFIGS=(${CONFIGS[*]}) SEEDS=(${SEEDS[*]})"
echo

for seed in "${SEEDS[@]}"; do
  for target in "${TARGETS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      current=$(( current + 1 ))
      run_id="${target}-${config}-seed${seed}"
      result_path="benchmarks/raw_results/${run_id}.jsonl"
      run_log="$LOG_DIR/${run_id}-${SESSION_STAMP}.log"

      if is_done "$result_path"; then
        lines=$(count_lines "$result_path")
        echo "[$(ts)] [$current/$total] SKIP $run_id (already has $lines records)"
        skipped=$(( skipped + 1 ))
        continue
      fi

      echo "[$(ts)] [$current/$total] RUN  $run_id → $run_log"
      start=$(date +%s)
      if uv run python benchmarks/suite/run_suite.py \
            --target "$target" \
            --config "$config" \
            --seeds 1 \
            --seed-start "$seed" > "$run_log" 2>&1; then
        rc=0
      else
        rc=$?
      fi
      elapsed=$(( $(date +%s) - start ))

      if is_done "$result_path"; then
        lines=$(count_lines "$result_path")
        echo "[$(ts)] [$current/$total] DONE $run_id ($lines records, ${elapsed}s, rc=$rc)"
        completed=$(( completed + 1 ))
      else
        # rc=130 means the child received SIGINT (likely propagated from
        # our Ctrl-C). Tag it as interrupted rather than failed so the
        # post-run summary is honest about what happened.
        if (( rc == 130 )) || (( STOP )); then
          echo "[$(ts)] [$current/$total] INTR $run_id (interrupted, ${elapsed}s — partial state in .anneal/targets/${run_id%-seed*})"
        else
          echo "[$(ts)] [$current/$total] FAIL $run_id (no records, ${elapsed}s, rc=$rc — see $run_log)"
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
(( failed == 0 ))
