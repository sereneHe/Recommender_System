#!/usr/bin/env bash
# Advance the most promising, but still unconfirmed, HC-CE hypothesis.
#
# H1: On correctly specified Gaussian ER SEMs, target-related CE-only provides
#     a practically useful improvement over the identical unconstrained NN.
#
# The default stage is a powered confirmation, not an optimizer sweep.  It uses
# 20 independent graph x noise units and fixed nuisance settings.  This is the
# smallest next experiment that can decide whether CE deserves more tuning.
#
# Stages:
#   ce_confirm (default): NN-only vs CE-only; 10 graph x 2 noise = 20 pairs.
#   optimizer: ALM vs PBM vs stochastic-PBM, enabled only with
#              CE_GATE_CONFIRMED=1 after reviewing ce_confirm.
#   sco: stochastic constrained optimizer layer vs SPBM, also gated.
#
# Examples:
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree_ce_priority.sh \
#     cluster_computing/run_metacentrum.pbs
#   STAGE=optimizer CE_GATE_CONFIRMED=1 qsub -v \
#     EXPERIMENT_SCRIPT=scripts/evidence_tree_ce_priority.sh,STAGE=optimizer,CE_GATE_CONFIRMED=1 \
#     cluster_computing/run_metacentrum.pbs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

STAGE="${STAGE:-ce_confirm}"
GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
NOISE_SEEDS="${NOISE_SEEDS:-101 102}"
SEEDS="${GRAPH_SEEDS}"
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"
read -r -a PRIORITY_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"

PROBLEMS="${PROBLEMS:-synthetic_er}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_EVIDENCE_PRIORITY_ER}"
EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-priority_ce_$(date -u +%Y%m%dT%H%M%SZ)}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
# The PBS worker has only scratch multirun/mlruns outputs.  Refresh evidence
# after its wrapper copies those artifacts into the persistent metacentrum_runs
# mirror, rather than generating an empty dashboard inside scratch.
REFRESH_EVIDENCE="${REFRESH_EVIDENCE:-0}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC=0

announce_stage "EVIDENCE-CE-PRIORITY" "${STAGE}; batch=${EVIDENCE_BATCH_ID}"
echo "paired units: graph seeds=${PLAN_SEEDS[*]}; noise seeds=${PRIORITY_NOISE_SEEDS[*]}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.hidden_dim=32"
  "solver.depth=2"
  "solver.dropout=0.15"
  "solver.learning_rate=0.25"
  "solver.weight_decay=0.25"
  "solver.grad_clip_norm=5.0"
  "solver.use_validation=true"
  "solver.restore_best_validation_model=true"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
  "solver.constraint_audit_enabled=true"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_w_constraints=false"
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
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
  "++problem.evidence_batch_id=${EVIDENCE_BATCH_ID}"
)

run_arm() {
  local label="$1"
  shift
  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${PRIORITY_NOISE_SEEDS[@]}"; do
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

run_ce_confirmation() {
  run_arm "nn" \
    "${COMMON[@]}" \
    "solver.constrained=false" "solver.use_ci_penalty=false" \
    "solver.constraint_audit_oracle=none"
  run_arm "ce" \
    "${COMMON[@]}" \
    "solver.constrained=true" "solver.use_ci_penalty=true" \
    "solver.ce_constraint_backend=alm_pbm" \
    "solver.ce_pbm_backend=stochastic_pbm" \
    "solver.use_stochastic_constrained_optimizer=false"
}

require_confirmed_ce() {
  [[ "${CE_GATE_CONFIRMED:-0}" == "1" ]] || \
    die "STAGE=${STAGE} is gated. Run and review STAGE=ce_confirm first; then set CE_GATE_CONFIRMED=1."
}

run_optimizer_comparison() {
  # Independent-only CE makes the algorithmic contrast interpretable:
  # ALM-all routes each CE constraint to ALM; PBM-all routes the identical
  # constraints to deterministic PBM or local stochastic PBM.  Hybrid is not
  # included because without dependent constraints it is algebraically the
  # same routing as ALM-all, not a fourth optimizer.
  run_arm "opt_alm_all" \
    "${COMMON[@]}" "solver.constrained=true" "solver.use_ci_penalty=true" \
    "solver.ce_constraint_backend=alm_all" \
    "solver.ce_pbm_backend=stochastic_pbm" \
    "solver.use_stochastic_constrained_optimizer=false"
  run_arm "opt_pbm_all" \
    "${COMMON[@]}" "solver.constrained=true" "solver.use_ci_penalty=true" \
    "solver.ce_constraint_backend=pbm_all" \
    "solver.ce_pbm_backend=humancompatible_pbm" \
    "solver.ci_pbm_penalty_update=dimin" \
    "solver.use_stochastic_constrained_optimizer=false"
  run_arm "opt_spbm_all" \
    "${COMMON[@]}" "solver.constrained=true" "solver.use_ci_penalty=true" \
    "solver.ce_constraint_backend=pbm_all" \
    "solver.ce_pbm_backend=stochastic_pbm" \
    "solver.ci_pbm_penalty_update=dimin" \
    "solver.use_stochastic_constrained_optimizer=false"
}

run_sco_comparison() {
  # SCO is a training layer, not another constraint encoder.  Re-run its
  # SPBM reference in this cohort so STAGE=sco is self-contained even when it
  # is submitted after a separate optimizer job with a different batch id.
  run_arm "opt_spbm_all" \
    "${COMMON[@]}" "solver.constrained=true" "solver.use_ci_penalty=true" \
    "solver.ce_constraint_backend=pbm_all" \
    "solver.ce_pbm_backend=stochastic_pbm" \
    "solver.ci_pbm_penalty_update=dimin" \
    "solver.use_stochastic_constrained_optimizer=false"
  run_arm "sco_spbm_all" \
    "${COMMON[@]}" "solver.constrained=true" "solver.use_ci_penalty=true" \
    "solver.ce_constraint_backend=pbm_all" \
    "solver.ce_pbm_backend=stochastic_pbm" \
    "solver.ci_pbm_penalty_update=dimin" \
    "solver.use_stochastic_constrained_optimizer=true"
}

case "${STAGE}" in
  ce_confirm) run_ce_confirmation ;;
  optimizer) require_confirmed_ce; run_optimizer_comparison ;;
  sco) require_confirmed_ce; run_sco_comparison ;;
  *) die "Unknown STAGE=${STAGE}; use ce_confirm, optimizer, or sco." ;;
esac

if [[ "${REFRESH_EVIDENCE}" == "1" && "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/scan_progress_tree.py
  "${PYTHON_BIN}" scripts/build_evidence_index.py
  "${PYTHON_BIN}" scripts/build_evidence_nodes.py
  "${PYTHON_BIN}" scripts/render_evidence_tree.py
fi

echo "=== EVIDENCE-CE-PRIORITY complete: stage=${STAGE}; batch=${EVIDENCE_BATCH_ID} ==="
echo "Refresh the persistent dashboard after sync: SKIP_SYNC=0 bash scripts/refresh_progress_tree.sh"
