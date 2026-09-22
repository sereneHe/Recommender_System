#!/usr/bin/env bash
# CE New-Recommender screens on CoDiet data.
#
# CoDiet: small graphs with known causal chains, mixed continuous/discrete
# features, moderate sample size, frequent non-linear associations, and a
# site/group stratification.  For this data type we therefore use:
#   - discrete conditional independence (CMI) for mixed-type constraints
#   - quantile (median) residualization to handle nonlinear mean structures
#   - balanced batches when the target is discrete/categorical
#   - cross-window SE where windows correspond to site/group strata
#   - W constraints enabled as a stable baseline
#
# The STAGE=baseline arm mirrors the historical Plan-06 CoDiet baseline
# (mark and unconstrained HC-CE) under identical seeds.  STAGE=discrete_cmi
# runs the upgraded CMI-CE arm.  Both must be run on the same seeds before any
# paired comparison is valid.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

announce_stage "CE-NEW-CODIET" "CoDiet mixed/type-aware CMI-CE screen"

PROBLEMS="${PROBLEMS:-codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_CODIET}"
STAGE="${STAGE:-baseline}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
N_RUNS="${N_RUNS:-5}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"

export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE=0
# CoDiet has no exact ground-truth graph for all problems; the knowledge graph
# is approximate, so posterior stability filtering is useful but less
# aggressive than the time-series macro panel.
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-1}"
export HC_CE_BD_CHAINS="${HC_CE_BD_CHAINS:-3}"
export HC_CE_BD_STEPS="${HC_CE_BD_STEPS:-1200}"
export HC_CE_BD_BURN_IN="${HC_CE_BD_BURN_IN:-250}"
export HC_CE_BD_INDEPENDENCE_SUPPORT="${HC_CE_BD_INDEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_MAX_CONFLICT_SUPPORT="${HC_CE_BD_MAX_CONFLICT_SUPPORT:-0.2}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
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

# Upgraded type-aware CMI-CE for CoDiet.  Note ci_discrete_max_z_states=128:
# CoDiet graphs are small, so we allow more joint conditioning states before
# pruning than the macro panel would tolerate.
CMI_CE_COMMON=(
  "solver.constrained=true"
  "solver.use_w_constraints=true"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=discrete_conditional_independence"
  "solver.ci_discrete_n_bins=4"
  "solver.ci_discrete_soft_temperature=0.2"
  "solver.ci_discrete_max_z_states=128"
  "solver.ce_statistic_shrinkage=0.2"
  "solver.ce_residualize_method=quantile"
  "solver.ce_se_method=window"
  "solver.ce_cross_window_n_windows=4"
  "solver.ce_tolerance_sd_multiplier=2.0"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ce_use_balanced_batches=true"
  "solver.ce_sensitive_group_source=auto"
  "solver.ce_batch_size=128"
  "solver.ci_pbm_penalty_update=adapt"
  "solver.ci_pbm_epoch_len=100"
  "solver.validation_split_strategy=random"
  "solver.dag_fit_scope=inner_train"
  "solver.early_stopping_patience=2"
  "solver.ce_pbm_backend=stochastic_pbm"
)

case "${STAGE}" in
  baseline)
    # Historical baseline: mark and unconstrained HC-CE, unchanged.
    run_seeded "mark" "${EXPERIMENT_PREFIX}" "mark" "${PROBLEMS}" false \
      "solver.n_runs=${N_RUNS}" "solver.recalculate_dag=true" "solver.feature_selector=none"

    run_seeded "hce_no_constraint" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "solver.constrained=false" "solver.use_w_constraints=false" "solver.use_ci_penalty=false"
    ;;

  discrete_cmi)
    # Upgraded type-aware CMI-CE arm.  Run on the same seeds and CV as the
    # baseline before drawing paired conclusions.
    run_seeded "hce_cmi_upgraded" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "${CMI_CE_COMMON[@]}"
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use 'baseline' or 'discrete_cmi'."
    ;;
esac

echo "=== CE-NEW-CODIET complete ==="