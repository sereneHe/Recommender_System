#!/usr/bin/env bash
# Phase 5: run Mark, Mark-CC, and locked HC-family baselines under the same
# time-ordering, targets, and seeds. This is an end-to-end comparison; it is
# not a claim that all methods have identical graph-learning internals.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "05" "Mark / Mark-CC / HC comparison"

GROUP="${GROUP:-FRED_16country_monthly}"
PROBLEMS="${PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_bel,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_esp,${GROUP}/industry_eu_est,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_fra,${GROUP}/industry_eu_grc,${GROUP}/industry_eu_irl,${GROUP}/industry_eu_ita,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld,${GROUP}/industry_eu_prt,${GROUP}/industry_eu_svk,${GROUP}/industry_eu_svn}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN05_METHOD_COMPARISON}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"

export HC_CONSTRAINT_BACKEND=alm
export HC_WEIBULL_GAUSSIANIZE=0

MARK_COMMON=(
  "solver.n_runs=${N_RUNS}"
  "+solver.cv_strategy=time_series"
  "+solver.cv_time_test_size=${TIME_TEST_SIZE}"
  "+solver.cv_time_gap=0"
  "solver.recalculate_dag=true"
  "solver.feature_selector=none"
)

# Mark and Mark-CC are run with their native configurations.
run_seeded "mark" "${EXPERIMENT_PREFIX}" "mark" "${PROBLEMS}" false "${MARK_COMMON[@]}"
run_seeded "mark_cc" "${EXPERIMENT_PREFIX}" "mark_with_cc" "${PROBLEMS}" false "${MARK_COMMON[@]}"

NN_COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.learning_rate=${LR}"
  "solver.weight_decay=${WEIGHT_DECAY}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.prediction_loss=mse"
)

# HC uses its own declared cv fields, so no Hydra '+' prefix is required.
run_seeded "hc_w_alm" "${EXPERIMENT_PREFIX}" "hc_predictor" "${PROBLEMS}" false \
  "${NN_COMMON[@]}" \
  "solver.cv_strategy=time_series" "solver.cv_time_test_size=${TIME_TEST_SIZE}" "solver.cv_time_gap=0" \
  "solver.constrained=true"

run_seeded "hce_independent" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${NN_COMMON[@]}" \
  "solver.cv_strategy=time_series" "solver.cv_time_test_size=${TIME_TEST_SIZE}" "solver.cv_time_gap=0" \
  "solver.constrained=true" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=alm_pbm" \
  "solver.ci_add_dsep_independence=true" \
  "solver.ci_add_collider_marginal_independence=false" \
  "solver.ci_add_shielded_collider_dependence=false" \
  "solver.ce_use_balanced_batches=false" \
  "solver.validation_split_strategy=time" "solver.dag_fit_scope=inner_train" \
  "solver.early_stopping_patience=2"
