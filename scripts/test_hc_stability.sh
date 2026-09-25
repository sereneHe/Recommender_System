#!/bin/sh

# Staged, single-factor stability study for the plain HC predictor (ALM DAG
# constraints) on monthly industry data.
#
# Usage:
#   sh scripts/test_hc_stability.sh
#   HC_STAGE=weight_decay SELECTED_LR=0.02 sh scripts/test_hc_stability.sh
#   HC_STAGE=capacity SELECTED_LR=0.02 SELECTED_WEIGHT_DECAY=0.05 \
#     sh scripts/test_hc_stability.sh
#   HC_STAGE=validate WINNER_LR=0.02 WINNER_WEIGHT_DECAY=0.05 \
#     WINNER_HIDDEN_DIM=16 WINNER_DEPTH=1 sh scripts/test_hc_stability.sh
#
# This script deliberately does not set shell-level random seeds.  It also no
# longer launches the invalid 3x3 lr/weight_decay grid from job 23754902: every
# candidate there failed before producing cv_errors.yaml because Gurobi sessions
# were exhausted.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"

SOLVER="${SOLVER:-hc_predictor}"
EXPERIMENT="${EXPERIMENT:-ALM_INDUSTRY_RECOMMENDER}"
TIME_LIMIT="${TIME_LIMIT:-120}"
HC_STAGE="${HC_STAGE:-learning_rate}"

# Plain HC is an ALM-vs-SPBM optimizer study; CE controls do not apply here.
export HC_CONSTRAINT_BACKEND=alm

# W must remain recalculated for industry_eu: its initial W_est is zero.  The
# CV implementation supplies only the current outer-training rows to each DAG
# solve and reuses that W for the fold.  This removes the former outer-test
# leakage and avoids initial/final full-data DAG solves.
COMMON_OVERRIDES="
  solver.time_limit=${TIME_LIMIT}
  solver.n_outer=10
  solver.n_inner=100
  solver.recalculate_dag=true
  solver.constrained=true
"

run_target_validation() {
  target_problem="$1"
  shift
  ${CMD} experiment="${EXPERIMENT}" solver="${SOLVER}" problem="${target_problem}" \
    ${COMMON_OVERRIDES} \
    solver.learning_rate="${WINNER_LR}" \
    solver.weight_decay="${WINNER_WEIGHT_DECAY}" \
    solver.hidden_dim="${WINNER_HIDDEN_DIM}" \
    solver.depth="${WINNER_DEPTH}" \
    "$@"
}

case "${HC_STAGE}" in
  learning_rate)
    # Stage 1: retain the old capacity and weight_decay=.25, varying only lr.
    ${CMD} experiment="${EXPERIMENT}" solver="${SOLVER}" \
      problem="${PROBLEM:-FRED_16country_monthly/industry_eu_ltu}" \
      ${COMMON_OVERRIDES} \
      solver.hidden_dim=32 \
      solver.depth=2 \
      solver.learning_rate=0.01,0.02,0.05 \
      solver.weight_decay=0.25 \
      "$@"
    ;;

  weight_decay)
    : "${SELECTED_LR:?Run HC_STAGE=learning_rate first, then set SELECTED_LR to its winning value.}"
    # Stage 2: now hold the selected lr fixed and vary only regularization.
    ${CMD} experiment="${EXPERIMENT}" solver="${SOLVER}" \
      problem="${PROBLEM:-FRED_16country_monthly/industry_eu_ltu}" \
      ${COMMON_OVERRIDES} \
      solver.hidden_dim=32 \
      solver.depth=2 \
      solver.learning_rate="${SELECTED_LR}" \
      solver.weight_decay=0.01,0.05,0.25 \
      "$@"
    ;;

  capacity)
    : "${SELECTED_LR:?Set SELECTED_LR from the learning-rate stage.}"
    : "${SELECTED_WEIGHT_DECAY:?Set SELECTED_WEIGHT_DECAY from the weight-decay stage.}"
    # Stage 3: test reduced capacity only after optimization parameters win.
    ${CMD} experiment="${EXPERIMENT}" solver="${SOLVER}" \
      problem="${PROBLEM:-FRED_16country_monthly/industry_eu_ltu}" \
      ${COMMON_OVERRIDES} \
      solver.learning_rate="${SELECTED_LR}" \
      solver.weight_decay="${SELECTED_WEIGHT_DECAY}" \
      solver.hidden_dim=8,16 \
      solver.depth=1 \
      "$@"
    ;;

  validate)
    : "${WINNER_LR:?Set the final selected learning rate.}"
    : "${WINNER_WEIGHT_DECAY:?Set the final selected weight decay.}"
    : "${WINNER_HIDDEN_DIM:?Set the final selected hidden dimension.}"
    : "${WINNER_DEPTH:?Set the final selected depth.}"
    # Do not choose a global configuration from LTU alone.  Validate the one
    # winner on LTU, the previously stronger LUX, and the weaker NLD.
    run_target_validation FRED_16country_monthly/industry_eu_ltu "$@"
    run_target_validation FRED_16country_monthly/industry_eu_lux "$@"
    run_target_validation FRED_16country_monthly/industry_eu_nld "$@"
    ;;

  *)
    echo "Unknown HC_STAGE=${HC_STAGE}; use learning_rate, weight_decay, capacity, or validate." >&2
    exit 2
    ;;
esac
