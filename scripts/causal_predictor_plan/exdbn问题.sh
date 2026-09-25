#!/usr/bin/env bash
# Preliminary ExDBN failure-isolation screen.
#
# The reference and every diagnostic arm use the same synthetic ER/SF data,
# seeds, neural budget and CE constraint builder.  All arms use MIPGap=0.01;
# only the requested solver/structural control is changed.  The long-time arm
# is the exception in TimeLimit by design: it tests whether 120 seconds is too
# short.  The remaining diagnostic arms use TimeLimit=120 seconds.
#
# Arms:
#   s0_reference       current structural settings, time=120, gap=0.01
#   s1_time1800        time=1800, gap=0.01
#   s2_clique5         milp_clique backend, max clique size=5, time=120
#   s3_edge_penalty    explicit edge-cardinality penalty, time=120
#   s4_parent_limit    maximum parents per child, time=120
#
# This script intentionally keeps weights_bound (the Big-M-like coefficient
# bound) fixed.  Big-M sensitivity is a separate experiment.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PROBLEMS="${PROBLEMS:-synthetic_er,synthetic_sf}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_EXDBN_ISSUE}"
GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-42}}"
SEEDS="${GRAPH_SEEDS}"
# Re-read PLAN_SEEDS after replacing the shared default with graph seeds.
read -r -a PLAN_SEEDS <<< "${GRAPH_SEEDS//,/ }"

announce_stage "EXDBN" "ExDBN problem isolation: time, clique, edge penalty, parents"

NOISE_SEEDS="${NOISE_SEEDS:-101,102,103,104,105}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"

# Keep the current Big-M-like coefficient bound fixed in all five arms.
WEIGHTS_BOUND="${WEIGHTS_BOUND:-1.0}"

BASE_TIME_LIMIT="${BASE_TIME_LIMIT:-120}"
LONG_TIME_LIMIT="${LONG_TIME_LIMIT:-1800}"
MIP_GAP="${MIP_GAP:-0.01}"

EDGE_PENALTY="${EDGE_PENALTY:-0.1}"
MAX_PARENTS="${MAX_PARENTS:-4}"
MAX_CLIQUE_SIZE="${MAX_CLIQUE_SIZE:-5}"

read -r -a EXDBN_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"
if [[ ${#EXDBN_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi
if [[ ${#PLAN_SEEDS[@]} -eq 0 ]]; then
  die "GRAPH_SEEDS must contain at least one integer."
fi

export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC=0

COMMON=(
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.constraint_audit_enabled=true"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
  "solver.ci_mode=manual"
  "solver.ci_manual_from_training_dag=true"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ci_skip_if_direct_edge=true"
  "solver.ci_max_dsep_separator_size=3"
  "solver.ci_prune_redundant=false"
  "solver.ci_dsep_all_separators=false"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_stochastic_constrained_optimizer=false"
  "solver.use_w_constraints=false"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_constraint_backend=alm_pbm"
  "solver.constrained=true"
  "solver.recalculate_dag=true"
  "solver.edge_penalty=0.0"
  "solver.max_parents=null"
  "solver.enable_clique_constraints=false"
  "solver.max_clique_size=null"
  "solver.weights_bound=${WEIGHTS_BOUND}"
  "solver.nonzero_threshold=0.01"
  "solver.reg_type=l2"
  "solver.lambda1=0.1"
  "++solver.gurobi_seed=0"
  "++solver.gurobi_threads=1"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
)

run_exdbn_arm() {
  local label="$1"
  shift
  local time_limit="$1"
  shift
  local graph_seed noise_seed model_seed

  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${EXDBN_NOISE_SEEDS[@]}"; do
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

      run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
        "hc_predictor_ce" "${PROBLEMS}" \
        "${seed_overrides[@]}" \
        "${COMMON[@]}" \
        "solver.time_limit=${time_limit}" \
        "solver.target_mip_gap=${MIP_GAP}" \
        "$@"
    done
  done
}

# Reference: same graph-estimation path as the problematic runs, with the
# requested common 1% MIP gap and the short 120-second budget.
run_exdbn_arm "s0_reference" "${BASE_TIME_LIMIT}"

# Diagnostic 1: only the solver budget is extended to 1800 seconds.
run_exdbn_arm "s1_time1800" "${LONG_TIME_LIMIT}"

# Diagnostic 2: only the DAG solver backend/maximum clique control changes.
run_exdbn_arm "s2_clique5" "${BASE_TIME_LIMIT}" \
  "solver.dag_solver_backend=milp_clique" \
  "solver.enable_clique_constraints=true" \
  "solver.max_clique_size=${MAX_CLIQUE_SIZE}" \
  "solver.clique_callback_mode=all_cliques" \
  "solver.clique_cut_formulation=k_plus_1" \
  "solver.clique_cut_selection=all" \
  "solver.max_clique_cuts_per_callback=0" \
  "solver.deduplicate_clique_cuts=true"

# Diagnostic 3: only explicit edge-cardinality regularisation changes.
run_exdbn_arm "s3_edge_penalty" "${BASE_TIME_LIMIT}" \
  "solver.edge_penalty=${EDGE_PENALTY}"

# Diagnostic 4: only the maximum number of parents per child changes.
run_exdbn_arm "s4_parent_limit" "${BASE_TIME_LIMIT}" \
  "solver.max_parents=${MAX_PARENTS}"

cat <<EOF

ExDBN issue screen submitted.
  reference:     time=${BASE_TIME_LIMIT}, mip_gap=${MIP_GAP}, weights_bound=${WEIGHTS_BOUND}
  time arm:      time=${LONG_TIME_LIMIT}, mip_gap=${MIP_GAP}
  clique arm:    time=${BASE_TIME_LIMIT}, max_clique_size=${MAX_CLIQUE_SIZE}
  edge arm:      time=${BASE_TIME_LIMIT}, edge_penalty=${EDGE_PENALTY}
  parent arm:    time=${BASE_TIME_LIMIT}, max_parents=${MAX_PARENTS}

Compare active_w_edges, edge precision/recall/F1, SHD, independent/dependent
constraint counts, solver status, actual MIP gap, timeout rate and test NMSE.
EOF
