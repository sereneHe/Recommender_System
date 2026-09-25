#!/usr/bin/env bash
# Phase 6: CoDiet predictive and discrete-CMI screen.
#
# It intentionally does not claim to perform CLR/ILR, Gaussian-copula, or a
# mixed-variable statistic. Those require a reviewed feature-type map and are
# excluded rather than silently coercing categorical values to pseudo-continuous
# values.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "06" "CoDiet raw/type-screen baseline"

PROBLEMS="${PROBLEMS:-codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN06_CODIET}"
STAGE="${STAGE:-baseline}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
N_RUNS="${N_RUNS:-5}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"

export HC_CONSTRAINT_BACKEND=alm
export HC_WEIBULL_GAUSSIANIZE=0

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.recalculate_dag=true"
  "solver.feature_selector=none"
)
NN_COMMON=(
  "solver.learning_rate=${LR}"
  "solver.weight_decay=${WEIGHT_DECAY}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.prediction_loss=mse"
)

case "${STAGE}" in
  baseline)
    run_seeded "mark" "${EXPERIMENT_PREFIX}" "mark" "${PROBLEMS}" false \
      "solver.n_runs=${N_RUNS}" "solver.recalculate_dag=true" "solver.feature_selector=none"
    run_seeded "hce_no_constraint" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" "${NN_COMMON[@]}" \
      "solver.constrained=false" "solver.use_w_constraints=false" "solver.use_ci_penalty=false"
    ;;

  discrete_cmi)
    run_seeded "hce_discrete_cmi_independent" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" "${NN_COMMON[@]}" \
      "solver.constrained=true" "solver.use_w_constraints=false" \
      "solver.use_ci_penalty=true" "solver.ci_penalty_kind=discrete_conditional_independence" \
      "solver.ci_discrete_n_bins=4" "solver.ci_discrete_soft_temperature=0.2" \
      "solver.ce_constraint_backend=alm_pbm" \
      "solver.ci_add_dsep_independence=true" \
      "solver.ci_add_collider_marginal_independence=false" \
      "solver.ci_add_shielded_collider_dependence=false" \
      "solver.ce_use_balanced_batches=false" \
      "solver.validation_split_strategy=random" "solver.dag_fit_scope=inner_train" \
      "solver.early_stopping_patience=2"
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use baseline or discrete_cmi."
    ;;
esac

echo "Before any copula/CLR/mixed-variable run, complete the feature-type map specified in the Markdown plan."
