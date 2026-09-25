#!/usr/bin/env bash
# Phase 3: separate the effects of W, independent CE constraints, and balanced
# batches on continuous Industry/FRED targets.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "03" "industry W/CE/batch ablation"

GROUP="${GROUP:-FRED_16country_monthly}"
PROBLEMS="${PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN03_INDUSTRY_CONSTRAINTS}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"
CE_BACKEND="${CE_BACKEND:-alm_pbm}"

# Carry the winner from phase 2 through the environment. Defaults make this
# script runnable for a pilot but should be replaced for confirmation.
LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"

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
  "solver.prediction_loss=mse"
)

CE_COMMON=(
  "solver.constrained=true"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_constraint_backend=${CE_BACKEND}"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.validation_split_strategy=time"
  "solver.dag_fit_scope=inner_train"
  "solver.early_stopping_patience=2"
)

# I0: same HC-CE network path, without graph constraints.
run_seeded "i0_dnn_no_constraint" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}" \
  "solver.constrained=false" "solver.use_w_constraints=false" "solver.use_ci_penalty=false"

# I1: historical W-only HC. Its optimizer selector is an environment variable.
export HC_CONSTRAINT_BACKEND=alm
run_seeded "i1_w_alm" "${EXPERIMENT_PREFIX}" "hc_predictor" "${PROBLEMS}" false \
  "${COMMON[@]}" "solver.constrained=true"

# I2/I3: CE-only, differing solely in batch reweighting.
run_seeded "i2_ce_ordinary_batch" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}" "${CE_COMMON[@]}" \
  "solver.use_w_constraints=false" "solver.ce_use_balanced_batches=false"
run_seeded "i3_ce_balanced_batch" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}" "${CE_COMMON[@]}" \
  "solver.use_w_constraints=false" "solver.ce_use_balanced_batches=true"

# I4/I5: W plus the same CE constraints, again isolating the batch policy.
run_seeded "i4_w_ce_ordinary_batch" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}" "${CE_COMMON[@]}" \
  "solver.use_w_constraints=true" "solver.ce_use_balanced_batches=false"
run_seeded "i5_w_ce_balanced_batch" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}" "${CE_COMMON[@]}" \
  "solver.use_w_constraints=true" "solver.ce_use_balanced_batches=true"

echo "NOTE: each arm presently re-estimates a fold-local W. Treat this as a development ablation."
echo "      A cached fold-W mechanism is required before calling the arms a final paired-W comparison."
