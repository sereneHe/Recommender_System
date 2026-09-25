#!/usr/bin/env bash
# Phase 8: approximate posterior-stable constraints from multiple birth-death
# chains. Run only after phases 1 and 3 show that independent constraints are
# feasible and useful. This is restricted to continuous Industry/FRED data.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "08" "multi-chain birth-death posterior constraints"

GROUP="${GROUP:-FRED_16country_monthly}"
PROBLEMS="${PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN08_BD_POSTERIOR}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC=1
export HC_CE_BD_CHAINS="${HC_CE_BD_CHAINS:-4}"
export HC_CE_BD_STEPS="${HC_CE_BD_STEPS:-2000}"
export HC_CE_BD_BURN_IN="${HC_CE_BD_BURN_IN:-500}"
export HC_CE_BD_INITIAL_THRESHOLD="${HC_CE_BD_INITIAL_THRESHOLD:-0.1}"
export HC_CE_BD_EDGE_PENALTY="${HC_CE_BD_EDGE_PENALTY:-1.0}"
export HC_CE_BD_TEMPERATURE="${HC_CE_BD_TEMPERATURE:-1.0}"
export HC_CE_BD_ALLOW_REVERSE="${HC_CE_BD_ALLOW_REVERSE:-1}"
export HC_CE_BD_SEED_BASE="${HC_CE_BD_SEED_BASE:-${HC_CE_BD_SEED:-20260916}}"
export HC_CE_BD_CHAIN_SEED_STRIDE="${HC_CE_BD_CHAIN_SEED_STRIDE:-1009}"
export HC_CE_BD_INDEPENDENCE_SUPPORT="${HC_CE_BD_INDEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_DEPENDENCE_SUPPORT="${HC_CE_BD_DEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_MAX_CONFLICT_SUPPORT="${HC_CE_BD_MAX_CONFLICT_SUPPORT:-0.2}"

echo "BD settings: chains=${HC_CE_BD_CHAINS} steps=${HC_CE_BD_STEPS} burn_in=${HC_CE_BD_BURN_IN}"

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
  "solver.constrained=true"
  "solver.use_w_constraints=false"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_constraint_backend=alm_pbm"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ce_use_balanced_batches=false"
  "solver.validation_split_strategy=time"
  "solver.dag_fit_scope=inner_train"
  "solver.early_stopping_patience=2"
)

# B1 uses the historical single thresholded DAG. It is required to attribute a
# posterior-stability effect instead of comparing only against no CE at all.
export HC_CE_BD_MCMC=0
run_seeded "b1_single_dag_independent" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}"

# B2: posterior-stable independent constraints from equal-mass chains.
export HC_CE_BD_MCMC=1
run_seeded "b2_posterior_independent" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
  "${COMMON[@]}"

echo "The artifact records per-chain aggregate diagnostics, but not ESS/R-hat: edge traces are not retained yet."
echo "Do not describe this as a converged Bayesian posterior until trace diagnostics are implemented."
