#!/usr/bin/env bash
# Phase 2: staged HC parameter selection on time-ordered Industry/FRED data.
#
# The reported CV scores are development evidence, not the final held-out
# result. Run one STAGE at a time and carry only the selected settings forward.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "02" "industry HC staged tuning"

GROUP="${GROUP:-FRED_16country_monthly}"
SCREEN_PROBLEMS="${SCREEN_PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld}"
ALL_PROBLEMS="${ALL_PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_bel,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_esp,${GROUP}/industry_eu_est,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_fra,${GROUP}/industry_eu_grc,${GROUP}/industry_eu_irl,${GROUP}/industry_eu_ita,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld,${GROUP}/industry_eu_prt,${GROUP}/industry_eu_svk,${GROUP}/industry_eu_svn}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN02_INDUSTRY_HC}"
STAGE="${STAGE:-optimizer}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"

export HC_CONSTRAINT_BACKEND=alm
export HC_WEIBULL_GAUSSIANIZE=0

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.cv_strategy=time_series"
  "solver.cv_time_test_size=${TIME_TEST_SIZE}"
  "solver.cv_time_gap=0"
  "solver.recalculate_dag=true"
  "solver.constrained=true"
  "solver.feature_selector=none"
  "solver.prediction_loss=mse"
)

run_candidate() {
  local label="$1"
  shift
  run_seeded "${label}" "${EXPERIMENT_PREFIX}" "hc_predictor" "${SCREEN_PROBLEMS}" false \
    "${COMMON[@]}" "$@"
}

case "${STAGE}" in
  optimizer)
    # Six deliberately selected pairs, not a Cartesian grid.
    run_candidate "opt_lr025_wd025" solver.learning_rate=0.25 solver.weight_decay=0.25 solver.hidden_dim=32 solver.depth=2 solver.n_outer=10 solver.n_inner=100
    run_candidate "opt_lr010_wd010" solver.learning_rate=0.10 solver.weight_decay=0.10 solver.hidden_dim=32 solver.depth=2 solver.n_outer=10 solver.n_inner=100
    run_candidate "opt_lr010_wd001" solver.learning_rate=0.10 solver.weight_decay=0.01 solver.hidden_dim=32 solver.depth=2 solver.n_outer=10 solver.n_inner=100
    run_candidate "opt_lr003_wd001" solver.learning_rate=0.03 solver.weight_decay=0.01 solver.hidden_dim=32 solver.depth=2 solver.n_outer=10 solver.n_inner=100
    run_candidate "opt_lr003_wd000" solver.learning_rate=0.03 solver.weight_decay=0.00 solver.hidden_dim=32 solver.depth=2 solver.n_outer=10 solver.n_inner=100
    run_candidate "opt_lr001_wd001" solver.learning_rate=0.01 solver.weight_decay=0.01 solver.hidden_dim=32 solver.depth=2 solver.n_outer=10 solver.n_inner=100
    ;;

  capacity)
    require_env SELECTED_LR
    require_env SELECTED_WEIGHT_DECAY
    for spec in "16 1" "32 1" "32 2" "64 2" "64 3"; do
      read -r hidden depth <<< "${spec}"
      run_candidate "capacity_h${hidden}_d${depth}" \
        "solver.learning_rate=${SELECTED_LR}" \
        "solver.weight_decay=${SELECTED_WEIGHT_DECAY}" \
        "solver.hidden_dim=${hidden}" "solver.depth=${depth}" \
        solver.n_outer=10 solver.n_inner=100
    done
    ;;

  budget)
    require_env SELECTED_LR
    require_env SELECTED_WEIGHT_DECAY
    require_env SELECTED_HIDDEN_DIM
    require_env SELECTED_DEPTH
    for spec in "5 100" "10 100" "10 200" "20 100"; do
      read -r outer inner <<< "${spec}"
      run_candidate "budget_o${outer}_i${inner}" \
        "solver.learning_rate=${SELECTED_LR}" \
        "solver.weight_decay=${SELECTED_WEIGHT_DECAY}" \
        "solver.hidden_dim=${SELECTED_HIDDEN_DIM}" "solver.depth=${SELECTED_DEPTH}" \
        "solver.n_outer=${outer}" "solver.n_inner=${inner}"
    done
    ;;

  confirm)
    require_env WINNER_LR
    require_env WINNER_WEIGHT_DECAY
    require_env WINNER_HIDDEN_DIM
    require_env WINNER_DEPTH
    require_env WINNER_N_OUTER
    require_env WINNER_N_INNER
    run_seeded "confirm" "${EXPERIMENT_PREFIX}" "hc_predictor" "${ALL_PROBLEMS}" false \
      "${COMMON[@]}" \
      "solver.learning_rate=${WINNER_LR}" \
      "solver.weight_decay=${WINNER_WEIGHT_DECAY}" \
      "solver.hidden_dim=${WINNER_HIDDEN_DIM}" "solver.depth=${WINNER_DEPTH}" \
      "solver.n_outer=${WINNER_N_OUTER}" "solver.n_inner=${WINNER_N_INNER}"
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use optimizer, capacity, budget, or confirm."
    ;;
esac
