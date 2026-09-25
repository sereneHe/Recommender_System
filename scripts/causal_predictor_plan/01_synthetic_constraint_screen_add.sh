#!/usr/bin/env bash
# Phase 1A: estimated-DAG stabilization screen on the Gaussian synthetic
# oracle.  This is deliberately separate from the original Phase-1 arms so
# that the reference and each stabilization proposal share the same graph,
# noise, model seed, CV budget, and audit settings.
#
# Implemented arms:
#   S0  MSE-only baseline
#   S1  true-W moment-constraint control
#   S2  true-DAG independent constraints (feasibility control)
#   S4R single estimated-DAG independent constraints (reference)
#   S5  multi-chain birth-death posterior-stable independent constraints
#   S6  PBM independent constraints with zero tolerance (control)
#   S7  PBM independent constraints with an explicit tolerance
#   S8  all candidate d-separators without pruning
#   S9  all candidate d-separators with subset pruning
#
# Important scope boundary:
#   S5 is the repository's implemented posterior-stability approximation;
#   it is not B=50--100 bootstrap exDBN stability selection.  S6 uses the
#   existing PBM tolerance (|c|-tol <= 0), not a Gurobi slack-variable model.
#   S9 is a heuristic ablation, not a logically equivalent CI reduction.

# A single graph seed is the safe PBS default.  Set PHASE1A_FULL=1 to use the
# formal 20-graph design, or pass one graph seed per PBS job through SEEDS.
if [[ "${PHASE1A_FULL:-0}" == "1" ]]; then
  DEFAULT_GRAPH_SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"
else
  DEFAULT_GRAPH_SEEDS="42"
fi
GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-${DEFAULT_GRAPH_SEEDS}}}"
SEEDS="${GRAPH_SEEDS}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "01A" "synthetic estimated-DAG stabilization screen"

PROBLEMS="${PROBLEMS:-synthetic_er,synthetic_sf}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN01A_SYNTHETIC}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
NOISE_SEEDS="${NOISE_SEEDS:-101,102,103,104,105}"
SOFT_TOLERANCE="${SOFT_TOLERANCE:-0.05}"

read -r -a PHASE1A_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"
if [[ ${#PHASE1A_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi

export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE=0
PHASE1A_W_CACHE_DIR="${PHASE1A_W_CACHE_DIR:-${REPO_ROOT}/results/phase1_add/w_cache}"
export HC_CE_W_CACHE_DIR="${PHASE1A_W_CACHE_DIR}"

# Keep all arms on the same training and audit budget.  Only the graph source,
# posterior aggregation, PBM tolerance, or pruning switch changes below.
COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.constraint_audit_enabled=true"
  "solver.ci_mode=manual"
  "solver.ci_manual_from_training_dag=true"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ci_max_dsep_separator_size=3"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_stochastic_constrained_optimizer=false"
  "++solver.gurobi_seed=0"
  "++solver.gurobi_threads=1"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
)

run_phase1a_synthetic() {
  local label="$1"
  local solver="$2"
  shift 2

  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${PHASE1A_NOISE_SEEDS[@]}"; do
      [[ "${noise_seed}" =~ ^[0-9]+$ ]] || die "Invalid NOISE_SEEDS value ${noise_seed}."
      model_seed="$((graph_seed * 100000 + noise_seed))"
      export HC_SPBM_RANDOM_SEED="${model_seed}"
      export HC_CE_BD_SEED="${model_seed}"
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

# S0: paired predictive baseline.
export HC_CE_BD_MCMC=0
run_phase1a_synthetic "s0_no_constraint" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=false" \
  "solver.recalculate_dag=false" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=none"

# S1: retain the original true-W control so the stabilization arms can be
# compared with the first-moment constraint as well as with MSE-only DNN.
run_phase1a_synthetic "s1_true_w" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=false" \
  "solver.use_w_constraints=true" \
  "solver.w_matrix_space=raw_sem" \
  "solver.use_ci_penalty=false" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# S2: true-DAG control.  Stable counts here show whether the CI construction
# itself is feasible before estimated-DAG variation is introduced.
run_phase1a_synthetic "s2_true_independent" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=false" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=alm_pbm" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# S4R: estimated-DAG reference.  Keep the label distinct from the original S4
# so the added summary cannot overwrite the previous phase-1 artifacts.
run_phase1a_synthetic "s4_estimated_independent_reference" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=true" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=alm_pbm" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# S5: posterior-stable independent constraints.  This uses the implemented
# multi-chain birth-death sampler.  It is deliberately not called bootstrap
# stability selection because exDBN bootstrap aggregation is not implemented.
export HC_CE_BD_MCMC=1
export HC_CE_BD_CHAINS="${HC_CE_BD_CHAINS:-4}"
export HC_CE_BD_STEPS="${HC_CE_BD_STEPS:-500}"
export HC_CE_BD_BURN_IN="${HC_CE_BD_BURN_IN:-100}"
export HC_CE_BD_INITIAL_THRESHOLD="${HC_CE_BD_INITIAL_THRESHOLD:-0.1}"
export HC_CE_BD_EDGE_PENALTY="${HC_CE_BD_EDGE_PENALTY:-1.0}"
export HC_CE_BD_TEMPERATURE="${HC_CE_BD_TEMPERATURE:-1.0}"
export HC_CE_BD_ALLOW_REVERSE="${HC_CE_BD_ALLOW_REVERSE:-1}"
export HC_CE_BD_CHAIN_SEED_STRIDE="${HC_CE_BD_CHAIN_SEED_STRIDE:-1009}"
export HC_CE_BD_INDEPENDENCE_SUPPORT="${HC_CE_BD_INDEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_DEPENDENCE_SUPPORT="${HC_CE_BD_DEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_MAX_CONFLICT_SUPPORT="${HC_CE_BD_MAX_CONFLICT_SUPPORT:-0.2}"
run_phase1a_synthetic "s5_posterior_stable_independent" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=true" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=alm_pbm" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# S6: PBM zero-tolerance control.  This is required to separate the PBM
# backend effect from the tolerance effect in S7.
export HC_CE_BD_MCMC=0
run_phase1a_synthetic "s6_pbm_zero_tolerance_independent" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=true" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=pbm_all" \
  "solver.ce_independence_tolerance=0.0" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# S7: soft-tolerance proxy.  PBM receives |c|-tolerance <= 0; this allows a
# predeclared residual band but is not the proposed Gurobi slack-variable model.
run_phase1a_synthetic "s7_pbm_soft_tolerance_independent" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=true" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=pbm_all" \
  "solver.ce_independence_tolerance=${SOFT_TOLERANCE}" \
  "solver.ci_prune_redundant=false" \
  "solver.ci_dsep_all_separators=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# S8/S9 make the pruning comparison nontrivial by enumerating all d-separators
# up to the fixed size bound.  S8 is the unpruned candidate-pool control;
# S9 keeps only minimal conditioning sets for each endpoint pair.
run_phase1a_synthetic "s8_all_dsep_unpruned" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=true" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=alm_pbm" \
  "solver.ci_dsep_all_separators=true" \
  "solver.ci_prune_redundant=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_phase1a_synthetic "s9_all_dsep_pruned" "hc_predictor_ce" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.recalculate_dag=true" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=true" \
  "solver.ci_penalty_kind=conditional_expectation" \
  "solver.ce_constraint_backend=alm_pbm" \
  "solver.ci_dsep_all_separators=true" \
  "solver.ci_prune_redundant=true" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

if [[ "${SKIP_PHASE1A_SUMMARY:-0}" != "1" && "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/causal_predictor_plan/summarize_phase1.py \
    --scan-root "${PHASE1A_SCAN_ROOT:-multirun}" \
    --output-dir "${PHASE1A_SUMMARY_DIR:-results/phase1_add}" \
    --experiment-prefix "${EXPERIMENT_PREFIX}"
fi

echo "Phase 1A boundaries: S5=posterior-stability approximation; S7=PBM tolerance proxy; S9=non-equivalent subset-pruning ablation."
