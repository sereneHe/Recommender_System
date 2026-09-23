#!/usr/bin/env bash
# CE New-Recommender screens on Synthetic Scale-Free (SF) graphs.
#
# SF: power-law DAG with hubs, iid, n=1000, true graph known.
# Compared with ER there are fewer d-separations around hubs, so per-constraint
# variance is higher and redundant deep conditioning sets should be pruned.
# We therefore use slightly higher shrinkage, keep the W constraint enabled
# (hub structures make W moment constraints reliable), and prune redundant
# separators.  recalculate_dag=false selects the true SEM / true DAG.  B0--B2
# provide the matched hc_predictor, mark, and mark_with_cc references.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-42 43 44}}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
# See the ER wrapper: PLAN_SEEDS was parsed before GRAPH_SEEDS existed.
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"

announce_stage "CE-NEW-SF" "Synthetic Scale-Free: W-Plus-CE with hub pruning"

PROBLEMS="${PROBLEMS:-synthetic_sf}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_SF}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"

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
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
)

# SF-specific CE upgrades: W kept on, shrinkage 0.1, redundant-separator pruning
# to stop hub paths from generating overly deep conditioning sets.
CE_COMMON=(
  "solver.constrained=true"
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}"
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
  "solver.use_w_constraints=true"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_statistic_kind=partial_correlation"
  "solver.ce_statistic_shrinkage=0.1"
  "solver.ce_residualize_method=linear"
  "solver.ce_se_method=window"
  "solver.ce_cross_window_n_windows=5"
  "solver.ce_tolerance_sd_multiplier=1.96"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ci_max_dsep_separator_size=3"
  "solver.ci_prune_redundant=true"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
)

# hc_predictor does not define use_stochastic_constrained_optimizer, so keep
# its matched baseline overrides separate from the HC-CE array above.
BASELINE_NN_COMMON=(
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
  "solver.recalculate_dag=false"
  "solver.constrained=true"
  "solver.use_w_constraints=true"
  "solver.w_matrix_space=raw_sem"
  "solver.use_ci_penalty=false"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
)

run_phase1_synthetic_ce_new() {
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

run_phase1_synthetic_nn_baseline() {
  local label="$1"
  local solver="$2"
  local backend="$3"
  shift 3

  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${CE_NEW_NOISE_SEEDS[@]}"; do
      [[ "${noise_seed}" =~ ^[0-9]+$ ]] || die "Invalid NOISE_SEEDS value ${noise_seed}."
      model_seed="$((graph_seed * 100000 + noise_seed))"
      export HC_SPBM_RANDOM_SEED="${model_seed}"
      export HC_CONSTRAINT_BACKEND="${backend}"
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
          "${solver}" "${PROBLEMS}" "${seed_overrides[@]}" "$@" "${PLAN_EXTRA_OVERRIDES[@]}"
      else
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "${solver}" "${PROBLEMS}" "${seed_overrides[@]}" "$@"
      fi
    done
  done
}

run_phase1_synthetic_tree_baseline() {
  local label="$1"
  local solver="$2"
  shift 2

  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${CE_NEW_NOISE_SEEDS[@]}"; do
      [[ "${noise_seed}" =~ ^[0-9]+$ ]] || die "Invalid NOISE_SEEDS value ${noise_seed}."
      model_seed="$((graph_seed * 100000 + noise_seed))"
      local -a seed_overrides=(
        "solver.random_state=${model_seed}"
        "solver.n_runs=${N_RUNS}"
        "solver.recalculate_dag=false"
        "+solver.cv_strategy=site_gender"
        "+solver.cv_random_state=$((model_seed + 10000))"
        "+solver.cv_time_test_size=null"
        "+solver.cv_time_gap=0"
        "++problem.seed=${graph_seed}"
        "++problem.graph_seed=${graph_seed}"
        "++problem.noise_seed=${noise_seed}"
      )
      if (( ${#PLAN_EXTRA_OVERRIDES[@]} > 0 )); then
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "${solver}" "${PROBLEMS}" "${seed_overrides[@]}" "$@" "${PLAN_EXTRA_OVERRIDES[@]}"
      else
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "${solver}" "${PROBLEMS}" "${seed_overrides[@]}" "$@"
      fi
    done
  done
}

# E0: no-constraint reference with the same HC-CE network and CV protocol.
run_phase1_synthetic_ce_new "e0_no_constraint" \
  "${COMMON[@]}" \
  "solver.constrained=false" \
  "solver.recalculate_dag=false" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" \
  "solver.constraint_audit_oracle=none"

# E1: true-W reference.
run_phase1_synthetic_ce_new "e1_true_w" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}" \
  "solver.recalculate_dag=false" \
  "solver.w_matrix_space=raw_sem" \
  "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# E2: upgraded pure-CE (W disabled, so hub d-separations alone carry the task).
run_phase1_synthetic_ce_new "e2_pure_ce_upgraded" \
  "${COMMON[@]}" \
  "${CE_COMMON[@]}" \
  "solver.use_w_constraints=false"

# E3: W + upgraded CE (recommended production arm for hub-heavy graphs).
run_phase1_synthetic_ce_new "e3_w_plus_ce_upgraded" \
  "${COMMON[@]}" \
  "${CE_COMMON[@]}"

# B0--B2 are additional predictor baselines, paired with E0--E3 on the same
# graph, innovation, model, and CV seeds.  Set INCLUDE_BASELINES=0 to omit
# them during a quick CE-only screen.
if [[ "${INCLUDE_BASELINES:-1}" == "1" ]]; then
  HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND:-alm}"
  run_phase1_synthetic_nn_baseline "b0_hc_predictor_${HC_BASELINE_BACKEND}" \
    "hc_predictor" "${HC_BASELINE_BACKEND}" "${BASELINE_NN_COMMON[@]}"
  run_phase1_synthetic_tree_baseline "b1_mark" "mark"
  run_phase1_synthetic_tree_baseline "b2_mark_with_cc" "mark_with_cc" \
    "solver.n_outer=${N_OUTER}" "solver.time_limit=${TIME_LIMIT}" \
    "+solver.w_matrix_space=raw_sem"
fi

echo "=== CE-NEW-SF complete ==="
