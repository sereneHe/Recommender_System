#!/usr/bin/env bash
# STAGE-driven driver for the HC-NN-CE program.
#
# STAGE=p0                  : no-constraint vs the legacy global W moment;
# STAGE=p1                  : CI-statistic oracle audit + CE-only training arm;
# STAGE=p1_target_residual  : target-related W conditional moments (NOT CE);
# STAGE=p2                  : optimization variants; requires P1_VERIFIED=1.
#
# Runs on the synthetic ER generator with the true graph (recalculate_dag=false),
# so the comparison isolates the predictor and the constraint, not the DAG
# learner.  CoDiet/Industry keep their own type-aware / time-series paths.
#
# Local use:
#   STAGE=p0 bash scripts/experiments_ce_new_recommender_er_p1p2.sh
#   DRY_RUN=1 STAGE=p1 bash scripts/experiments_ce_new_recommender_er_p1p2.sh
#
# PBS use:
#   qsub -v EXPERIMENT_SCRIPT=scripts/experiments_ce_new_recommender_er_p1p2.sh,STAGE=p0 \
#     cluster_computing/run_metacentrum.pbs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-42 43 44}}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"

PROBLEMS="${PROBLEMS:-synthetic_er}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_ER_P1P2}"
STAGE="${STAGE:-p0}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-10}"
CI_AUDIT_SEM_TYPES="${CI_AUDIT_SEM_TYPES:-gauss nonlinear}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-0}"

read -r -a CE_NEW_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"
if [[ ${#CE_NEW_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi

announce_stage "ER-P1P2" "STAGE=${STAGE}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.constraint_audit_enabled=true"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_stochastic_constrained_optimizer=false"
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
)

CE_ONLY=(
  "solver.constrained=true"
  "solver.use_w_constraints=false"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_statistic_kind=partial_correlation"
  "solver.ce_statistic_shrinkage=0.05"
  "solver.ce_residualize_method=linear"
  "solver.ce_se_method=window"
  "solver.ce_cross_window_n_windows=5"
  "solver.ce_tolerance_sd_multiplier=1.96"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
)

run_arm() {
  local label="$1"
  shift 1
  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${CE_NEW_NOISE_SEEDS[@]}"; do
      [[ "${noise_seed}" =~ ^[0-9]+$ ]] || die "Invalid NOISE_SEEDS value ${noise_seed}."
      model_seed="$((graph_seed * 100000 + noise_seed))"
      export HC_SPBM_RANDOM_SEED="${model_seed}"
      local -a seed_overrides=(
        "solver.random_state=${model_seed}"
        "solver.cv_random_state=$((model_seed + 10000))"
        "solver.validation_random_state=$((model_seed + 20000))"
        "++problem.seed=${graph_seed}"
        "++problem.graph_seed=${graph_seed}"
        "++problem.noise_seed=${noise_seed}"
      )
      if (( ${#PLAN_EXTRA_OVERRIDES[@]} > 0 )); then
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "hc_predictor_ce" "${PROBLEMS}" "${seed_overrides[@]}" "$@" "${PLAN_EXTRA_OVERRIDES[@]}"
      else
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "hc_predictor_ce" "${PROBLEMS}" "${seed_overrides[@]}" "$@"
      fi
    done
  done
}

run_ci_audit() {
  local sem_type
  for sem_type in ${CI_AUDIT_SEM_TYPES}; do
    local out="results/ci_audit/ci_audit_ER_${sem_type}.csv"
    echo "--- CI statistic oracle audit: ER / ${sem_type} -> ${out} ---"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN: would audit ER/${sem_type}"
      continue
    fi
    "${PYTHON_BIN}" scripts/causal_predictor_plan/audit_nonlinear_ci.py \
      --graph-type ER \
      --sem-type "${sem_type}" \
      --graph-seeds ${GRAPH_SEEDS} \
      --noise-seeds ${NOISE_SEEDS} \
      --n-samples "${N_SAMPLES}" \
      --n-nodes "${N_NODES}" \
      --expected-edges "${EXPECTED_EDGES}" \
      --output "${out}"
  done
}

case "${STAGE}" in
  p0)
    # No-constraint reference vs the legacy global W mean moment.
    run_arm "p0_no_constraint" \
      "${COMMON[@]}" \
      "solver.constrained=false" \
      "solver.use_w_constraints=false" \
      "solver.use_ci_penalty=false" \
      "solver.constraint_audit_oracle=none"
    run_arm "p0_legacy_global_w" \
      "${COMMON[@]}" \
      "solver.constrained=true" \
      "solver.use_w_constraints=true" \
      "solver.use_ci_penalty=false" \
      "solver.w_constraint_mode=legacy_global" \
      "solver.constraint_audit_oracle=synthetic_linear_sem"
    ;;

  p1)
    # Oracle audit of the CE statistics first, then the CE-only training arm.
    run_ci_audit
    run_arm "p1_ce_only" \
      "${COMMON[@]}" \
      "${CE_ONLY[@]}"
    ;;

  p1_target_residual)
    # Target-related W conditional moments.  This is a W constraint form, not CE.
    run_arm "p1tr_target_residual" \
      "${COMMON[@]}" \
      "solver.constrained=true" \
      "solver.use_w_constraints=true" \
      "solver.use_ci_penalty=false" \
      "solver.w_constraint_mode=target_residual" \
      "solver.constraint_audit_oracle=synthetic_linear_sem"
    ;;

  p2)
    if [[ "${P1_VERIFIED:-0}" != "1" ]]; then
      die "STAGE=p2 requires P1_VERIFIED=1 after reviewing the STAGE=p1 audit and CE-only results."
    fi
    run_arm "p2_warmup_ramp" \
      "${COMMON[@]}" \
      "solver.constrained=true" \
      "solver.use_w_constraints=true" \
      "solver.use_ci_penalty=false" \
      "solver.w_constraint_mode=target_residual" \
      "solver.gradient_log_interval=${GRAD_LOG_INTERVAL}" \
      "solver.w_warmup_fraction=${WARMUP_FRACTION:-0.3}" \
      "solver.w_warmup_ramp_fraction=${WARMUP_RAMP_FRACTION:-0.2}" \
      "solver.constraint_audit_oracle=synthetic_linear_sem"
    run_arm "p2_cap_0p1" \
      "${COMMON[@]}" \
      "solver.constrained=true" \
      "solver.use_w_constraints=true" \
      "solver.use_ci_penalty=false" \
      "solver.w_constraint_mode=target_residual" \
      "solver.gradient_log_interval=${GRAD_LOG_INTERVAL}" \
      "solver.w_grad_ratio_cap=0.1" \
      "solver.constraint_audit_oracle=synthetic_linear_sem"
    run_arm "p2_cap_0p3" \
      "${COMMON[@]}" \
      "solver.constrained=true" \
      "solver.use_w_constraints=true" \
      "solver.use_ci_penalty=false" \
      "solver.w_constraint_mode=target_residual" \
      "solver.gradient_log_interval=${GRAD_LOG_INTERVAL}" \
      "solver.w_grad_ratio_cap=0.3" \
      "solver.constraint_audit_oracle=synthetic_linear_sem"
    run_arm "p2_cap_1p0" \
      "${COMMON[@]}" \
      "solver.constrained=true" \
      "solver.use_w_constraints=true" \
      "solver.use_ci_penalty=false" \
      "solver.w_constraint_mode=target_residual" \
      "solver.gradient_log_interval=${GRAD_LOG_INTERVAL}" \
      "solver.w_grad_ratio_cap=1.0" \
      "solver.constraint_audit_oracle=synthetic_linear_sem"
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use p0, p1, p1_target_residual, or p2."
    ;;
esac

echo
echo "=== ER P1P2 STAGE=${STAGE} complete ==="
