#!/usr/bin/env bash
# CE New-Recommender screens on CoDiet data.
#
# CoDiet: small graphs with known causal chains, mixed continuous/discrete
# features, moderate sample size, frequent non-linear associations, and a
# site/group stratification.  For this data type we therefore use:
#   - soft discrete conditional mutual information for mixed-type variables
#   - W + CMI compared against a matched W-only predictor
#   - ordinary batches in both arms, avoiding a sampling-policy confound
#
# STAGE=baseline runs hc_predictor, mark, mark_with_cc, unconstrained HC-CE,
# and a W-only HC-CE control. STAGE=discrete_cmi runs the W+CMI arm. Use the
# same seeds; compare CMI+W against W-only to isolate the CMI contribution.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

announce_stage "CE-NEW-CODIET" "CoDiet mixed/type-aware CMI-CE screen"

PROBLEMS="${PROBLEMS:-codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_CODIET_V2}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
STAGE="${STAGE:-baseline}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
N_RUNS="${N_RUNS:-5}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
CMI_INDEPENDENCE_TOLERANCE="${CMI_INDEPENDENCE_TOLERANCE:-0.0}"
INCLUDE_CLASSICAL_BASELINES="${INCLUDE_CLASSICAL_BASELINES:-1}"
HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND:-alm}"

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
# Separate the birth-death chain RNG seed from the data/graph seed (see the
# industry wrapper); without a base every data seed reuses one chain seed.
export HC_CE_BD_SEED_BASE="${HC_CE_BD_SEED_BASE:-20260916}"

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
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}"
  "solver.use_w_constraints=true"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=discrete_conditional_independence"
  "solver.ci_discrete_n_bins=4"
  "solver.ci_discrete_soft_temperature=0.2"
  "solver.ci_discrete_max_z_states=128"
  "solver.ce_tolerance_mode=fixed"
  "solver.ce_independence_tolerance=${CMI_INDEPENDENCE_TOLERANCE}"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.validation_split_strategy=random"
  "solver.dag_fit_scope=inner_train"
  "solver.early_stopping_patience=2"
  "solver.constraint_audit_enabled=true"
)

run_classical_hc_baseline() {
  local label="$1"
  local backend="$2"
  shift 2

  local seed
  for seed in "${PLAN_SEEDS[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed ${seed}."
    export HC_SPBM_RANDOM_SEED="${seed}"
    export HC_CONSTRAINT_BACKEND="${backend}"
    if [[ -n "${HC_CE_BD_SEED_BASE:-}" ]]; then
      export HC_CE_BD_SEED="$((HC_CE_BD_SEED_BASE + seed))"
    fi
    local -a seed_overrides=(
      "solver.random_state=${seed}"
      "solver.cv_random_state=$((seed + 10000))"
      "solver.validation_random_state=$((seed + 20000))"
    )
    if (( ${#PLAN_EXTRA_OVERRIDES[@]} > 0 )); then
      run_hydra "${EXPERIMENT_PREFIX}_${label}_seed${seed}" "hc_predictor" "${PROBLEMS}" \
        "${seed_overrides[@]}" "$@" "${PLAN_EXTRA_OVERRIDES[@]}"
    else
      run_hydra "${EXPERIMENT_PREFIX}_${label}_seed${seed}" "hc_predictor" "${PROBLEMS}" \
        "${seed_overrides[@]}" "$@"
    fi
  done
}

run_classical_tree_baseline() {
  local label="$1"
  local solver="$2"
  shift 2

  local seed
  for seed in "${PLAN_SEEDS[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed ${seed}."
    local -a seed_overrides=(
      "solver.random_state=${seed}"
      "solver.n_runs=${N_RUNS}"
      "solver.recalculate_dag=true"
      "solver.feature_selector=none"
      "+solver.cv_strategy=site_gender"
      "+solver.cv_time_test_size=null"
      "+solver.cv_time_gap=0"
    )
    if (( ${#PLAN_EXTRA_OVERRIDES[@]} > 0 )); then
      run_hydra "${EXPERIMENT_PREFIX}_${label}_seed${seed}" "${solver}" "${PROBLEMS}" \
        "${seed_overrides[@]}" "$@" "${PLAN_EXTRA_OVERRIDES[@]}"
    else
      run_hydra "${EXPERIMENT_PREFIX}_${label}_seed${seed}" "${solver}" "${PROBLEMS}" \
        "${seed_overrides[@]}" "$@"
    fi
  done
}

case "${STAGE}" in
  baseline)
    # Predictor baselines share the same problems, graph estimation policy,
    # model/CV seeds, and training budget as the HC-CE controls below.
    if [[ "${INCLUDE_CLASSICAL_BASELINES}" == "1" ]]; then
      run_classical_hc_baseline "b0_hc_predictor_${HC_BASELINE_BACKEND}" "${HC_BASELINE_BACKEND}" \
        "${COMMON[@]}" \
        "solver.constrained=true" \
        "solver.use_w_constraints=true" \
        "solver.use_ci_penalty=false" \
        "solver.w_matrix_space=model_standardized" \
        "solver.validation_split_strategy=random" \
        "solver.dag_fit_scope=inner_train" \
        "solver.ce_use_balanced_batches=false" \
        "solver.ce_batch_size=128" \
        "solver.constraint_audit_enabled=true"
      run_classical_tree_baseline "b1_mark" "mark"
      run_classical_tree_baseline "b2_mark_with_cc" "mark_with_cc" \
        "solver.n_outer=${N_OUTER}" "solver.time_limit=${TIME_LIMIT}"
    fi

    run_seeded "hce_no_constraint" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "solver.constrained=false" "solver.use_w_constraints=false" "solver.use_ci_penalty=false"

    # Matched W-only control for the CMI arm: same HC-CE model and training
    # settings, with the only difference being whether CMI constraints are on.
    run_seeded "hce_w_only" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "${CMI_CE_COMMON[@]}" \
      "solver.use_ci_penalty=false" \
      "solver.constraint_audit_enabled=true"
    ;;

  discrete_cmi)
    # Upgraded type-aware CMI-CE arm.  Run on the same seeds and CV as the
    # baseline before drawing paired conclusions.
    run_seeded "hce_cmi_upgraded" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "${CMI_CE_COMMON[@]}"

    if [[ "${DRY_RUN}" != "1" ]]; then
      AUDIT_ROOT="${AUDIT_ROOT:-multirun}"
      AUDIT_SUMMARY_DIR="${AUDIT_SUMMARY_DIR:-results/ce_codiet}"
      "${PYTHON_BIN}" scripts/causal_predictor_plan/summarize_ce_preaudit.py \
        --root "${AUDIT_ROOT}" --output-dir "${AUDIT_SUMMARY_DIR}" \
        --experiment-prefix "${EXPERIMENT_PREFIX}" --stage full
    fi
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use 'baseline' or 'discrete_cmi'."
    ;;
esac

echo "=== CE-NEW-CODIET complete ==="
