#!/usr/bin/env bash
# Clean, paired repair batch for the ER HC-CE evidence tree.
#
# Why this exists
# ---------------
# Older ER artifacts cannot be used as strict evidence because:
#   1) some E6 nonlinear W+CE runs resolved use_w_constraints=false;
#   2) nominally paired arms used different dropout / gradient-clipping values;
#   3) the old report aggregation averaged those non-identical runs.
#
# The historical E6 bug was already corrected in later server jobs.  Therefore
# STAGE=audit (the default) rebuilds the cohort-aware evidence tree and submits
# no redundant model run.  STAGE=full_clean_replicate is optional: it creates a
# new self-contained regression cohort if an independent post-fix replication
# is desired.  Every arm in that cohort has the same model, data, split and CE
# settings unless that field is the declared treatment.
#
# Submit, for example:
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree_config_repair.sh \
#     cluster_computing/run_metacentrum.pbs
#   STAGE=full_clean_replicate qsub -v \
#     EXPERIMENT_SCRIPT=scripts/evidence_tree_config_repair.sh,STAGE=full_clean_replicate \
#     cluster_computing/run_metacentrum.pbs
#
# Optional environment overrides:
#   GRAPH_SEEDS="42 43 44" NOISE_SEEDS="101" EVIDENCE_BATCH_ID=repair_er_v2
#   DRY_RUN=1 bash scripts/evidence_tree_config_repair.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"
read -r -a REPAIR_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"

PROBLEMS="${PROBLEMS:-synthetic_er}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_EVIDENCE_REPAIR_ER}"
# A generated default prevents accidental merging of two independently
# submitted repair cohorts.  Set it explicitly when a submission must be
# resumed under the same cohort identifier.
EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-repair_er_$(date -u +%Y%m%dT%H%M%SZ)}"

TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
STAGE="${STAGE:-audit}"
# PBS executes in a scratch copy where metacentrum_runs/ is deliberately not
# mounted.  Dashboard refresh therefore belongs on the persistent checkout
# after the PBS wrapper has synced artifacts back.
REFRESH_EVIDENCE="${REFRESH_EVIDENCE:-0}"

# Explicitly freeze every training nuisance that was previously implicit in
# one or more arms.  Do not change one of these values within this batch.
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
DROPOUT="${DROPOUT:-0.15}"
LEARNING_RATE="${LEARNING_RATE:-0.25}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.25}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC=0

announce_stage "EVIDENCE-REPAIR" "${STAGE}; batch=${EVIDENCE_BATCH_ID}"
echo "paired units: graph seeds=${PLAN_SEEDS[*]}; noise seeds=${REPAIR_NOISE_SEEDS[*]}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.dropout=${DROPOUT}"
  "solver.learning_rate=${LEARNING_RATE}"
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
)

LINEAR_CE=(
  "solver.constrained=true"
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
)

run_arm() {
  local label="$1"
  shift
  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${REPAIR_NOISE_SEEDS[@]}"; do
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

run_full_clean_replicate() {
  # Linear, Gaussian mechanism controls.  The direct W+CE versus CE comparison
  # is now available, rather than inferring a W contribution from W+CE vs W.
  run_arm "lin_nn" \
  "${COMMON[@]}" "++problem.sem_type=gauss" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"
  run_arm "lin_ce" \
  "${COMMON[@]}" "${LINEAR_CE[@]}" "++problem.sem_type=gauss" \
  "solver.use_w_constraints=false" "solver.constraint_audit_oracle=synthetic_linear_sem"
  run_arm "lin_w" \
  "${COMMON[@]}" "++problem.sem_type=gauss" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=synthetic_linear_sem"
  run_arm "lin_w_ce" \
  "${COMMON[@]}" "${LINEAR_CE[@]}" "++problem.sem_type=gauss" \
  "solver.use_w_constraints=true" "solver.constraint_audit_oracle=synthetic_linear_sem"

  # Nonlinear stress controls.  There is no linear-SEM oracle for these arms.
  # In particular, nonlin_w_ce must resolve use_w_constraints=true; this is the
  # regression check for the historical E6 configuration bug.
  run_arm "nonlin_nn" \
  "${COMMON[@]}" "++problem.sem_type=nonlinear" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"
  run_arm "nonlin_ce" \
  "${COMMON[@]}" "${LINEAR_CE[@]}" "++problem.sem_type=nonlinear" \
  "solver.use_w_constraints=false" "solver.constraint_audit_oracle=none"
  run_arm "nonlin_quantile_ce" \
  "${COMMON[@]}" "${LINEAR_CE[@]}" "++problem.sem_type=nonlinear" \
  "solver.ce_residualize_method=quantile" "solver.use_w_constraints=false" \
  "solver.constraint_audit_oracle=none"
  run_arm "nonlin_w_ce" \
  "${COMMON[@]}" "${LINEAR_CE[@]}" "++problem.sem_type=nonlinear" \
  "solver.use_w_constraints=true" "solver.constraint_audit_oracle=none"
}

case "${STAGE}" in
  audit)
    echo "No mandatory model rerun: the old E6 error is superseded by later corrected cohorts."
    ;;
  full_clean_replicate)
    run_full_clean_replicate
    ;;
  *)
    die "Unknown STAGE=${STAGE}; use audit or full_clean_replicate."
    ;;
esac

if [[ "${REFRESH_EVIDENCE}" == "1" && "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/scan_progress_tree.py
  "${PYTHON_BIN}" scripts/build_evidence_index.py
  "${PYTHON_BIN}" scripts/build_evidence_nodes.py
  "${PYTHON_BIN}" scripts/render_evidence_tree.py
fi

echo "=== EVIDENCE-REPAIR complete: batch=${EVIDENCE_BATCH_ID} ==="
echo "Refresh the persistent dashboard after sync: SKIP_SYNC=0 bash scripts/refresh_progress_tree.sh"
