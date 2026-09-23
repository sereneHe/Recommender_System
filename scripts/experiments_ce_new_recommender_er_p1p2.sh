#!/usr/bin/env bash
# P1/P2 driver: target-residual conditional moments and gradient-conflict logging.
#
# Arms (all on the synthetic ER generator with the true graph, W only, no CE):
#   p0_legacy_global          : current global (W-I)[Xbar, Ybar] mean moment;
#   p1_target_residual        : target-parent conditional moments
#                               E[phi_k(Pa_Y) * (Yhat - f_W(Pa_Y))] = 0;
#   p2_target_residual_gradlog: p1 plus per-step gradient-conflict diagnostics
#                               (cos between MSE and constraint grads, norm ratio, ||g||).
#
# Compare p1 - p0 (does the conditional residual help?) and read p2's logged
# cos/ratio to see whether the constraint fights the MSE objective.
#
# Usage:
#   bash scripts/experiments_ce_new_recommender_er_p1p2.sh
#   DRY_RUN=1 bash scripts/experiments_ce_new_recommender_er_p1p2.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

announce_stage "ER-P1P2" "target-residual conditional moments and gradient diagnostics"

GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-42 43 44}}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"

PROBLEMS="${PROBLEMS:-synthetic_er}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_ER_P1P2}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-10}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-0}"

read -r -a CE_NEW_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"
if [[ ${#CE_NEW_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi

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
  "solver.constrained=true"
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
  "solver.use_w_constraints=true"
  "solver.use_ci_penalty=false"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
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

run_arm "p0_legacy_global" \
  "${COMMON[@]}" \
  "solver.w_constraint_mode=legacy_global"

run_arm "p1_target_residual" \
  "${COMMON[@]}" \
  "solver.w_constraint_mode=target_residual"

run_arm "p2_target_residual_gradlog" \
  "${COMMON[@]}" \
  "solver.w_constraint_mode=target_residual" \
  "solver.gradient_log_interval=${GRAD_LOG_INTERVAL}"

echo
echo "=== ER P1/P2 complete ==="
