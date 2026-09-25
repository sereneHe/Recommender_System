#!/usr/bin/env bash
# Orchestrate the remaining Stage 2 phases.
#
# Stage 2 is sequential: Round 2B needs the selected Round 2A optimizer,
# Round 2C needs the two selected network configurations, and confirmation
# needs the final winner.  This wrapper therefore requires those selections
# explicitly; it never chooses them from a holdout or from incomplete runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/02_industry_hc_tuning_new.sh"

RUN_MODE="${RUN_MODE:-all}"
RUN_OPTIMIZER="${RUN_OPTIMIZER:-0}"
W_MODE="${W_MODE:-fixed}"
SEEDS="${SEEDS:-42 43 44}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-42 43 44 45 46}"
ENFORCE_PROTOCOL="${ENFORCE_PROTOCOL:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "Set ${name} before RUN_MODE=all."
}

run_phase() {
  local phase="$1"
  echo
  echo "=== Stage 2 ${phase} ==="
  STAGE="${phase}" \
  W_MODE="${W_MODE}" \
  SEEDS="${SEEDS}" \
  CONFIRM_SEEDS="${CONFIRM_SEEDS}" \
  ENFORCE_PROTOCOL="${ENFORCE_PROTOCOL}" \
  bash "${BASE_SCRIPT}"
}

case "${RUN_MODE}" in
  optimizer)
    run_phase optimizer
    ;;

  all)
    require_env SELECTED_LR
    require_env SELECTED_WEIGHT_DECAY
    require_env SELECTED_CONFIGS
    require_env WINNER_LR
    require_env WINNER_WEIGHT_DECAY
    require_env WINNER_HIDDEN_DIM
    require_env WINNER_DEPTH
    require_env WINNER_N_OUTER
    require_env WINNER_N_INNER

    if [[ "${RUN_OPTIMIZER}" == "1" ]]; then
      run_phase optimizer
    else
      echo "Skipping Round 2A; using the supplied selected configuration values."
    fi

    run_phase capacity
    run_phase budget
    run_phase confirm
    ;;

  *)
    die "Unknown RUN_MODE=${RUN_MODE}. Use optimizer or all."
    ;;
esac

echo
echo "Stage 2 orchestration completed. Review paired validation summaries before using any result."
