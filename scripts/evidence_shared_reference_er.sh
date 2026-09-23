#!/usr/bin/env bash
# Shared-reference synthetic ER cohort for the missing A2/A3/A5 evidence nodes.
#
# One cohort, frozen nuisances, and a small set of shared references so each
# treatment arm only changes its declared factor:
#
#   ref_nn            no constraint (shared reference for W / capacity arms)
#   ref_ce            CE only, partial correlation, window SE (shared CE ref)
#   ref_dep           CE + collider dependence, filter off (sign-flip ref)
#
#   A2: a2_w_global, a2_w_target_residual, a2_w_mask, a2_w_bias_calibration,
#       a2_balanced_batch, a2_pruning
#   A3: a3_covariance, a3_hac_se, a3_sign_flip_filter
#   A5: a5_hidden_depth, a5_lr_wd, a5_n_outer_inner
#
# Submit:
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_shared_reference_er.sh \
#     cluster_computing/run_metacentrum.pbs
#   GRAPH_SEEDS="42 43 44 45 46 47 48 49 50 51" qsub -v \
#     EXPERIMENT_SCRIPT=scripts/evidence_shared_reference_er.sh,GRAPH_SEEDS=... \
#     cluster_computing/run_metacentrum.pbs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"
read -r -a SR_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"

PROBLEMS="${PROBLEMS:-synthetic_er}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_EVIDENCE_SHARED_ER}"
EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-shared_er_$(date -u +%Y%m%dT%H%M%SZ)}"

TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"

# Frozen nuisances: identical for every arm, only the treatment differs.
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
DROPOUT="${DROPOUT:-0.15}"
LR="${LR:-0.25}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.25}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-0}"

if [[ ${#SR_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi

announce_stage "EVIDENCE-SHARED" "shared-reference ER cohort; batch=${EVIDENCE_BATCH_ID}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.dropout=${DROPOUT}"
  "solver.learning_rate=${LR}"
  "solver.weight_decay=${WEIGHT_DECAY}"
  "solver.grad_clip_norm=${GRAD_CLIP_NORM}"
  "solver.use_validation=true"
  "solver.restore_best_validation_model=true"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
  "solver.constraint_audit_enabled=true"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_stochastic_constrained_optimizer=false"
  "solver.ce_constraint_backend=alm_pbm"
  "solver.ce_pbm_backend=stochastic_pbm"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.evidence_batch_id=${EVIDENCE_BATCH_ID}"
  "++problem.sem_type=gauss"
)

CE_BASE=(
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
  shift
  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${SR_NOISE_SEEDS[@]}"; do
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

# ---- shared references ----
run_arm "ref_nn" "${COMMON[@]}" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

run_arm "ref_ce" "${COMMON[@]}" "${CE_BASE[@]}"

run_arm "ref_dep" "${COMMON[@]}" "${CE_BASE[@]}" \
  "solver.ci_add_collider_conditional_dependence=true"

# ---- A2: constraint embedding ----
run_arm "a2_w_global" "${COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=legacy_global" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "a2_w_target_residual" "${COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=target_residual" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "a2_w_mask" "${COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=legacy_global" \
  "solver.w_prediction_dependent_mask=true" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "a2_w_bias_calibration" "${COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.w_bias_calibration=true" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "a2_balanced_batch" "${COMMON[@]}" "${CE_BASE[@]}" \
  "solver.ce_use_balanced_batches=true"

run_arm "a2_pruning" "${COMMON[@]}" "${CE_BASE[@]}" \
  "solver.ci_prune_redundant=true" "solver.ci_max_dsep_separator_size=3"

# ---- A3: CI/CE statistic & uncertainty ----
run_arm "a3_covariance" "${COMMON[@]}" "${CE_BASE[@]}" \
  "solver.ce_statistic_kind=covariance"

run_arm "a3_hac_se" "${COMMON[@]}" "${CE_BASE[@]}" \
  "solver.ce_se_method=hac" "solver.ce_hac_max_lag=6"

run_arm "a3_sign_flip_filter" "${COMMON[@]}" "${CE_BASE[@]}" \
  "solver.ci_add_collider_conditional_dependence=true" \
  "solver.ci_dependent_statistic=signed" \
  "solver.ce_window_filter_enabled=true" "solver.ce_window_max_sign_flip_rate=0.30"

# ---- A5: capacity & budget (unconstrained capacity contrast) ----
run_arm "a5_hidden_depth" "${COMMON[@]}" \
  "solver.hidden_dim=64" "solver.depth=3" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

run_arm "a5_lr_wd" "${COMMON[@]}" \
  "solver.learning_rate=0.05" "solver.weight_decay=0.10" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

run_arm "a5_n_outer_inner" "${COMMON[@]}" \
  "solver.n_outer=5" "solver.n_inner=50" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

echo
echo "=== EVIDENCE-SHARED complete: batch=${EVIDENCE_BATCH_ID} ==="
echo "Refresh the tree after sync: SKIP_SYNC=0 bash scripts/refresh_progress_tree.sh"
