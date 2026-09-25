#!/usr/bin/env bash
# Phase 2: clean, paired Industry HC tuning.
#
# The fixed-W mode is deliberately fail-closed.  In the current repository
# recalculate_dag=false does not by itself load per-fold W matrices, so this
# script refuses to submit a supposedly fixed-W experiment until the W_fold
# cache loader is present.  Set W_MODE=dynamic only for an explicitly
# exploratory run with fold-local DAG estimation.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "02" "industry HC clean staged tuning"

GROUP="${GROUP:-FRED_16country_monthly}"
SCREEN_PROBLEMS="${SCREEN_PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld}"
ALL_PROBLEMS="${ALL_PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_bel,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_esp,${GROUP}/industry_eu_est,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_fra,${GROUP}/industry_eu_grc,${GROUP}/industry_eu_irl,${GROUP}/industry_eu_ita,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld,${GROUP}/industry_eu_prt,${GROUP}/industry_eu_svk,${GROUP}/industry_eu_svn}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN02_INDUSTRY_HC_S2A_CLEAN}"
CONFIRM_EXPERIMENT_PREFIX="${CONFIRM_EXPERIMENT_PREFIX:-PLAN02_INDUSTRY_HC_S2_CONFIRM}"
STAGE="${STAGE:-optimizer}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"
W_MODE="${W_MODE:-fixed}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-42 43 44 45 46}"
ENFORCE_PROTOCOL="${ENFORCE_PROTOCOL:-1}"
W_FOLD_CACHE_MANIFEST="${W_FOLD_CACHE_MANIFEST:-${W_FOLD_CACHE_DIR:-}/manifest.yaml}"

export HC_CONSTRAINT_BACKEND=alm
export HC_WEIBULL_GAUSSIANIZE=0

case "${W_MODE}" in
  fixed)
    if [[ -z "${W_FOLD_CACHE_DIR:-}" ]]; then
      die "W_MODE=fixed requires W_FOLD_CACHE_DIR and a real per-fold W cache."
    fi
    [[ -d "${W_FOLD_CACHE_DIR}" ]] || die "W_FOLD_CACHE_DIR does not exist: ${W_FOLD_CACHE_DIR}"
    if ! find "${W_FOLD_CACHE_DIR}" -type f -print -quit | grep -q .; then
      die "W_FOLD_CACHE_DIR is empty: ${W_FOLD_CACHE_DIR}"
    fi
    [[ -f "${W_FOLD_CACHE_MANIFEST}" ]] || die "W_FOLD_CACHE_MANIFEST is missing: ${W_FOLD_CACHE_MANIFEST}"
    if ! grep -Eq "W_FOLD_CACHE_DIR|w_fold_cache_dir|W_fold" \
      "${REPO_ROOT}/recommender_estimator.py" \
      "${REPO_ROOT}/recommender_utils.py" \
      "${REPO_ROOT}/run_experiments.py"; then
      die "The current Python code has no W_fold cache loader. Implement it before fixed-W Stage 2. Use W_MODE=dynamic only for exploratory runs."
    fi
    export HC_W_FOLD_CACHE_DIR="${W_FOLD_CACHE_DIR}"
    export HC_W_FOLD_CACHE_REQUIRED=1
    W_RECALCULATE=false
    ;;
  dynamic)
    echo "WARNING: W_MODE=dynamic re-estimates W per outer fold; results are exploratory, not fixed-W Stage 2."
    echo "WARNING: TIME_LIMIT=${TIME_LIMIT}; six configs × six targets × three seeds × four folds can require 432 DAG solves."
    export HC_W_FOLD_CACHE_REQUIRED=0
    W_RECALCULATE=true
    if [[ "${EXPERIMENT_PREFIX}" != *_DYNAMIC_W ]]; then
      EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}_DYNAMIC_W"
    fi
    ;;
  *)
    die "Unknown W_MODE=${W_MODE}. Use fixed or dynamic."
    ;;
esac

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.cv_strategy=time_series"
  "solver.cv_time_test_size=${TIME_TEST_SIZE}"
  "solver.cv_time_gap=0"
  "solver.recalculate_dag=${W_RECALCULATE}"
  "solver.constrained=true"
  "solver.use_w_constraints=true"
  "solver.use_ci_penalty=false"
  "solver.feature_selector=none"
  "solver.prediction_loss=mse"
  "solver.hidden_dim=32"
  "solver.depth=2"
  "solver.n_outer=10"
  "solver.n_inner=100"
)

if [[ "${ENFORCE_PROTOCOL}" == "1" ]]; then
  IFS=',' read -r -a SCREEN_PROBLEM_LIST <<< "${SCREEN_PROBLEMS}"
  IFS=',' read -r -a ALL_PROBLEM_LIST <<< "${ALL_PROBLEMS}"
  [[ "${#SCREEN_PROBLEM_LIST[@]}" -eq 6 ]] || die "Round 2 screening requires exactly 6 targets; got ${#SCREEN_PROBLEM_LIST[@]}."
  [[ "${#ALL_PROBLEM_LIST[@]}" -eq 16 ]] || die "Confirmation requires exactly 16 targets; got ${#ALL_PROBLEM_LIST[@]}."
  [[ "${N_RUNS}" == "4" ]] || die "Round 2 protocol requires N_RUNS=4; got ${N_RUNS}."
  [[ "${#PLAN_SEEDS[@]}" -eq 3 ]] || die "Round 2A-C require exactly 3 seeds; set SEEDS to three values."
fi

run_seeded_with_seeds() {
  local seed_list="$1"
  shift
  local -a saved_seeds=("${PLAN_SEEDS[@]}")
  read -r -a PLAN_SEEDS <<< "${seed_list//,/ }"
  run_seeded "$@"
  PLAN_SEEDS=("${saved_seeds[@]}")
}

run_candidate() {
  local label="$1"
  shift
  run_seeded "${label}" "${EXPERIMENT_PREFIX}" "hc_predictor" "${SCREEN_PROBLEMS}" false \
    "${COMMON[@]}" "$@"
}

case "${STAGE}" in
  optimizer)
    echo "Round 2A: 6 paired LR/WD configurations × 6 targets × ${#PLAN_SEEDS[@]} seeds × ${N_RUNS} origins."
    run_candidate "opt_lr025_wd025" solver.learning_rate=0.25 solver.weight_decay=0.25
    run_candidate "opt_lr010_wd010" solver.learning_rate=0.10 solver.weight_decay=0.10
    run_candidate "opt_lr010_wd001" solver.learning_rate=0.10 solver.weight_decay=0.01
    run_candidate "opt_lr003_wd001" solver.learning_rate=0.03 solver.weight_decay=0.01
    run_candidate "opt_lr003_wd000" solver.learning_rate=0.03 solver.weight_decay=0.00
    run_candidate "opt_lr001_wd001" solver.learning_rate=0.01 solver.weight_decay=0.01
    ;;

  capacity)
    require_env SELECTED_LR
    require_env SELECTED_WEIGHT_DECAY
    echo "Round 2B: fixed optimizer settings, five network capacities."
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
    require_env SELECTED_CONFIGS
    IFS=';' read -r -a budget_configs <<< "${SELECTED_CONFIGS}"
    [[ "${#budget_configs[@]}" -eq 2 ]] || die "Round 2C requires exactly two selected network configs. Format: lr,wd,hidden,depth;lr,wd,hidden,depth"
    echo "Round 2C: 2 selected network configurations × 4 training budgets."
    config_index=0
    for config in "${budget_configs[@]}"; do
      IFS=',' read -r selected_lr selected_wd selected_hidden selected_depth extra <<< "${config}"
      [[ -n "${selected_lr:-}" && -n "${selected_wd:-}" && -n "${selected_hidden:-}" && -n "${selected_depth:-}" && -z "${extra:-}" ]] || die "Invalid SELECTED_CONFIGS item: ${config}"
      config_index=$((config_index + 1))
      for spec in "5 100" "10 100" "10 200" "20 100"; do
        read -r outer inner <<< "${spec}"
        run_candidate "budget_c${config_index}_o${outer}_i${inner}" \
          "solver.learning_rate=${selected_lr}" \
          "solver.weight_decay=${selected_wd}" \
          "solver.hidden_dim=${selected_hidden}" \
          "solver.depth=${selected_depth}" \
          "solver.n_outer=${outer}" "solver.n_inner=${inner}"
      done
    done
    ;;

  confirm)
    require_env WINNER_LR
    require_env WINNER_WEIGHT_DECAY
    require_env WINNER_HIDDEN_DIM
    require_env WINNER_DEPTH
    require_env WINNER_N_OUTER
    require_env WINNER_N_INNER
    if [[ "${ENFORCE_PROTOCOL}" == "1" ]]; then
      read -r -a confirm_seed_list <<< "${CONFIRM_SEEDS//,/ }"
      [[ "${#confirm_seed_list[@]}" -eq 5 ]] || die "Confirmation requires exactly 5 seeds; got ${#confirm_seed_list[@]}."
    fi
    echo "Confirmation: selected configuration on all 16 Industry targets using seeds ${CONFIRM_SEEDS}."
    run_seeded_with_seeds "${CONFIRM_SEEDS}" \
      "confirm" "${CONFIRM_EXPERIMENT_PREFIX}" "hc_predictor" "${ALL_PROBLEMS}" false \
      "${COMMON[@]}" \
      "solver.learning_rate=${WINNER_LR}" \
      "solver.weight_decay=${WINNER_WEIGHT_DECAY}" \
      "solver.hidden_dim=${WINNER_HIDDEN_DIM}" \
      "solver.depth=${WINNER_DEPTH}" \
      "solver.n_outer=${WINNER_N_OUTER}" \
      "solver.n_inner=${WINNER_N_INNER}"
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use optimizer, capacity, budget, or confirm."
    ;;
esac
