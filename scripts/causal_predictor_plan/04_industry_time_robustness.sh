#!/usr/bin/env bash
# Phase 4: causal-safe lag, time-trend, regime-indicator, and robust-loss
# experiments for Industry/FRED. Every run uses expanding-window CV.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "04" "industry time and robust-loss experiments"

GROUP="${GROUP:-FRED_16country_monthly}"
PROBLEMS="${PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN04_INDUSTRY_TIME}"
STAGE="${STAGE:-lag}"
SOLVER="${SOLVER:-hc_predictor}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
SELECTED_LAG="${SELECTED_LAG:-1}"

export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE=0

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.cv_strategy=time_series"
  "solver.cv_time_test_size=${TIME_TEST_SIZE}"
  "solver.cv_time_gap=0"
  "solver.recalculate_dag=true"
  "solver.feature_selector=none"
  "solver.learning_rate=${LR}"
  "solver.weight_decay=${WEIGHT_DECAY}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.constrained=true"
)

run_time_case() {
  local label="$1"
  shift
  run_seeded "${label}" "${EXPERIMENT_PREFIX}" "${SOLVER}" "${PROBLEMS}" false \
    "${COMMON[@]}" "$@"
}

case "${STAGE}" in
  lag)
    for lag in 0 1 3 6 12; do
      run_time_case "lag${lag}" "+problem.feature_lag=${lag}" solver.prediction_loss=mse
    done
    ;;

  trend)
    run_time_case "lag${SELECTED_LAG}_no_trend" "+problem.feature_lag=${SELECTED_LAG}" "+problem.add_time_trend=false" solver.prediction_loss=mse
    run_time_case "lag${SELECTED_LAG}_with_trend" "+problem.feature_lag=${SELECTED_LAG}" "+problem.add_time_trend=true" solver.prediction_loss=mse
    ;;

  loss)
    run_time_case "mse" "+problem.feature_lag=${SELECTED_LAG}" solver.prediction_loss=mse
    for delta in 0.5 1.0 2.0; do
      run_time_case "huber_delta${delta}" "+problem.feature_lag=${SELECTED_LAG}" solver.prediction_loss=huber "solver.huber_delta=${delta}"
    done
    ;;

  regime)
    require_env REGIME_BREAK_DATE
    run_time_case "regime_${REGIME_BREAK_DATE}" \
      "+problem.feature_lag=${SELECTED_LAG}" \
      "+problem.add_time_trend=true" \
      "+problem.regime_break_date=${REGIME_BREAK_DATE}" \
      solver.prediction_loss=mse
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use lag, trend, loss, or regime."
    ;;
esac
