#!/usr/bin/env bash
# Phase 1 dense-graph screen: identify which existing ExDBN controls can
# prevent a near-complete estimated DAG and the resulting CI/CE instability.
#
# Every arm uses the same ER/SF graph seed, innovation-noise seed, model seed,
# NN budget and target-related independent CE builder.  Only one structural
# or solver control changes per arm, so edge density, solver tail behaviour,
# constraint counts and predictive NMSE can be compared pairwise.
#
# Arms:
#   s0  current estimated-DAG reference
#   s1  longer Gurobi time limit (diagnostic; not a sparsity fix)
#   s2  big-M/weight bound sensitivity (diagnostic; not expected to fix density)
#   s3  historical l1 regularisation (diagnostic; mixes coefficient/edge terms)
#   s4  larger post-solve nonzero threshold (changes released W/CI graph only)
#   s5  all d-separators with minimal-set pruning
#   s6  implemented multi-chain BD graph for CI generation
#   s7  explicit edge-cardinality penalty in the ExDBN objective
#   s8  maximum-parent constraint per child
#   s9  edge-cardinality penalty plus maximum-parent constraint
#   s10 skeleton-clique cap (lazy cuts ported from the clique solver)
#
# Optional supplement (PHASE1D_SUPPLEMENT=1):
#   s11--s14 edge-penalty grid
#   s15 max-parent sensitivity
#   s16--s17 edge-penalty plus parent-cap candidates
# The supplement uses one common solver budget for all candidates.  It is
# disabled by default because it adds a second grid on top of the diagnostic
# screen above.
#
# Important interpretation boundaries:
#   * s1 and s2 test solver sensitivity, not a causal improvement claim.
#   * s4/s5 only affect the graph/constraint object passed downstream; they do
#     not reduce the MILP search space.
#   * s6 stabilises the CI graph only; the original W estimate is unchanged.
#   * s7--s10 are the actual sparsity interventions.  They are opt-in solver
#     controls with defaults disabled in hc_predictor.yaml.
#   * A clique cap bounds dense local regions of the undirected skeleton.  It
#     is not an edge-count constraint; for d=10, a K4-free graph can still
#     have 33 edges.  Report it alongside the explicit edge penalty.

if [[ "${PHASE1D_FULL:-0}" == "1" ]]; then
  DEFAULT_GRAPH_SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"
else
  DEFAULT_GRAPH_SEEDS="42"
fi
GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-${DEFAULT_GRAPH_SEEDS}}}"
SEEDS="${GRAPH_SEEDS}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "01D" "synthetic dense-graph / ExDBN stability screen"

PROBLEMS="${PROBLEMS:-synthetic_er,synthetic_sf}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN01D_SYNTHETIC_DENSE}"
TIME_LIMIT="${TIME_LIMIT:-120}"
LONG_TIME_LIMIT="${LONG_TIME_LIMIT:-1800}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
NOISE_SEEDS="${NOISE_SEEDS:-101,102,103,104,105}"
BIG_M="${BIG_M:-20}"
L1_LAMBDA="${L1_LAMBDA:-0.1}"
EDGE_PENALTY="${EDGE_PENALTY:-0.1}"
# ER42's true graph has maximum in-degree 4.  A cap of 3 would make that true
# graph infeasible, so it would be an aggressive misspecified ablation rather
# than a fair stabilization test.  Override deliberately if that is intended.
MAX_PARENTS="${MAX_PARENTS:-4}"
CLIQUE_MAX_SIZE="${CLIQUE_MAX_SIZE:-3}"
RELEASE_THRESHOLD="${RELEASE_THRESHOLD:-0.1}"
PHASE1D_SUPPLEMENT="${PHASE1D_SUPPLEMENT:-0}"
SUPPLEMENT_TIME_LIMIT="${SUPPLEMENT_TIME_LIMIT:-1800}"
SUPPLEMENT_TARGET_MIP_GAP="${SUPPLEMENT_TARGET_MIP_GAP:-0.01}"
SUPPLEMENT_GUROBI_SEEDS="${SUPPLEMENT_GUROBI_SEEDS:-0}"

read -r -a PHASE1D_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"
if [[ ${#PHASE1D_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi

export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC=0
# The cache key includes the fold-local input and every W-solver control.  It
# therefore makes s0/s4/s5/s6 reuse *exactly* the same estimated W, while a
# structural arm (time limit, big-M, penalty, parent cap, or clique cap) still
# computes its own W.  Atomic writes make this safe when PBS jobs share it.
PHASE1D_W_CACHE_DIR="${PHASE1D_W_CACHE_DIR:-${REPO_ROOT}/results/phase1_dense/w_cache}"
export HC_CE_W_CACHE_DIR="${PHASE1D_W_CACHE_DIR}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.constraint_audit_enabled=true"
  # Synthetic truth is used only for post-fit oracle/graph diagnostics; it is
  # never supplied to ExDBN or the neural optimizer.
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
  "solver.weights_bound=1.0"
  "solver.nonzero_threshold=0.01"
  "solver.reg_type=l2"
  "solver.lambda1=0.1"
  "solver.ci_dsep_all_separators=false"
  "solver.ci_prune_redundant=false"
  "++solver.gurobi_seed=0"
  "++solver.gurobi_threads=1"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.sem_type=gauss"
)

run_phase1d_synthetic() {
  local label="$1"
  shift
  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${PHASE1D_NOISE_SEEDS[@]}"; do
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
          "hc_predictor_ce" "${PROBLEMS}" "${seed_overrides[@]}" "$@" \
          "${PLAN_EXTRA_OVERRIDES[@]}"
      else
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "hc_predictor_ce" "${PROBLEMS}" "${seed_overrides[@]}" "$@"
      fi
    done
  done
}

# The common reference is the configuration that produced the dense 41--45
# edge estimates in the previous Phase-1 screen.
run_phase1d_synthetic "s0_dense_reference" "${COMMON[@]}"

# Time limit and big-M are diagnostic arms.  A longer limit can improve the
# incumbent/gap but cannot itself make the objective prefer sparse graphs.
run_phase1d_synthetic "s1_dense_time_limit" "${COMMON[@]}" \
  "solver.time_limit=${LONG_TIME_LIMIT}"
run_phase1d_synthetic "s2_dense_big_m" "${COMMON[@]}" \
  "solver.weights_bound=${BIG_M}"

# Historical l1 mode replaces the L2 coefficient regularizer with an edge
# count term.  It is retained as a legacy ablation, but is not a pure isolated
# sparsity comparison with s7 because it simultaneously removes L2 shrinkage.
run_phase1d_synthetic "s3_dense_l1" "${COMMON[@]}" \
  "solver.reg_type=l1" "solver.lambda1=${L1_LAMBDA}"

# This threshold is applied after the MILP solve.  It tests whether downstream
# CI instability is caused by weak released edges, not whether the MILP became
# sparse.
run_phase1d_synthetic "s4_dense_release_threshold" "${COMMON[@]}" \
  "solver.nonzero_threshold=${RELEASE_THRESHOLD}"

# Constraint-pool control: enumerate and retain only minimal conditioning sets.
run_phase1d_synthetic "s5_dense_pruned_ci" "${COMMON[@]}" \
  "solver.ci_dsep_all_separators=true" "solver.ci_prune_redundant=true"

# The repository's implemented posterior-stability approximation changes the
# graph used for CI generation.  It does not change the W MILP objective.
export HC_CE_BD_MCMC=1
export HC_CE_BD_CHAINS="${HC_CE_BD_CHAINS:-4}"
export HC_CE_BD_STEPS="${HC_CE_BD_STEPS:-500}"
export HC_CE_BD_BURN_IN="${HC_CE_BD_BURN_IN:-100}"
export HC_CE_BD_INDEPENDENCE_SUPPORT="${HC_CE_BD_INDEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_DEPENDENCE_SUPPORT="${HC_CE_BD_DEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_MAX_CONFLICT_SUPPORT="${HC_CE_BD_MAX_CONFLICT_SUPPORT:-0.2}"
run_phase1d_synthetic "s6_dense_posterior_stable" "${COMMON[@]}"
export HC_CE_BD_MCMC=0

# Actual structural-sparsity interventions.
run_phase1d_synthetic "s7_dense_edge_penalty" "${COMMON[@]}" \
  "solver.edge_penalty=${EDGE_PENALTY}"
run_phase1d_synthetic "s8_dense_max_parents" "${COMMON[@]}" \
  "solver.max_parents=${MAX_PARENTS}"
run_phase1d_synthetic "s9_dense_edge_penalty_max_parents" "${COMMON[@]}" \
  "solver.edge_penalty=${EDGE_PENALTY}" "solver.max_parents=${MAX_PARENTS}"

# Lazy clique cuts are ported from
# project-bestdagsolverintheworld/dagsolvers/solve_milp_clique.py.  ``k_plus_1``
# is the locally strongest cap formulation: every K_(k+1) subset must omit at
# least one skeleton edge.  ``all_cliques`` makes the Phase-1 test deterministic
# and complete at each integer incumbent; use a capped selection only for a
# later runtime-engineering study.
run_phase1d_synthetic "s10_dense_clique_cap" "${COMMON[@]}" \
  "solver.dag_solver_backend=milp_clique" \
  "solver.enable_clique_constraints=true" \
  "solver.max_clique_size=${CLIQUE_MAX_SIZE}" \
  "solver.clique_callback_mode=all_cliques" \
  "solver.clique_cut_formulation=k_plus_1" \
  "solver.clique_cut_selection=all" \
  "solver.max_clique_cuts_per_callback=0" \
  "solver.deduplicate_clique_cuts=true"

# Supplementary configuration-selection grid.  These arms are intentionally
# separate from s7--s10: the latter are diagnostic controls, whereas these
# candidates are compared under one common 1800-second solver budget.  The
# parent-cap values are checked after the run by summarize_phase1.py against
# W_true.csv; a candidate that excludes the known synthetic graph is rejected.
if [[ "${PHASE1D_SUPPLEMENT}" == "1" ]]; then
  read -r -a PHASE1D_SUPPLEMENT_SOLVER_SEEDS <<< "${SUPPLEMENT_GUROBI_SEEDS//,/ }"
  if [[ ${#PHASE1D_SUPPLEMENT_SOLVER_SEEDS[@]} -eq 0 ]]; then
    die "SUPPLEMENT_GUROBI_SEEDS must contain at least one integer."
  fi

  run_phase1d_supplement_arm() {
    local label="$1"
    local penalty="$2"
    local parents="$3"
    shift 3
    local solver_seed
    local -a supplement_common=()
    local override
    for override in "${COMMON[@]}"; do
      [[ "${override}" == "++solver.gurobi_seed=0" ]] || supplement_common+=("${override}")
    done
    for solver_seed in "${PHASE1D_SUPPLEMENT_SOLVER_SEEDS[@]}"; do
      [[ "${solver_seed}" =~ ^[0-9]+$ ]] || die "Invalid SUPPLEMENT_GUROBI_SEEDS value ${solver_seed}."
      run_phase1d_synthetic "${label}_solver${solver_seed}" "${supplement_common[@]}" \
        "solver.time_limit=${SUPPLEMENT_TIME_LIMIT}" \
        "solver.target_mip_gap=${SUPPLEMENT_TARGET_MIP_GAP}" \
        "solver.edge_penalty=${penalty}" \
        "solver.max_parents=${parents}" \
        "++solver.gurobi_seed=${solver_seed}" \
        "$@"
    done
  }

  # The explicit grid is deliberately small enough for a pilot.  Use
  # PHASE1D_FULL=1 for the 20-graph confirmation after selecting finalists.
  run_phase1d_supplement_arm "s11_edge_penalty_001" "0.01" "null"
  run_phase1d_supplement_arm "s12_edge_penalty_003" "0.03" "null"
  run_phase1d_supplement_arm "s13_edge_penalty_010" "0.10" "null"
  run_phase1d_supplement_arm "s14_edge_penalty_030" "0.30" "null"
  run_phase1d_supplement_arm "s15_max_parents_006" "0.0" "6"
  run_phase1d_supplement_arm "s16_edge_penalty_003_max_parents_004" "0.03" "4"
  run_phase1d_supplement_arm "s17_edge_penalty_003_max_parents_006" "0.03" "6"
fi

if [[ "${SKIP_PHASE1D_SUMMARY:-0}" != "1" && "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/causal_predictor_plan/summarize_phase1.py \
    --scan-root "${PHASE1D_SCAN_ROOT:-multirun}" \
    --output-dir "${PHASE1D_SUMMARY_DIR:-results/phase1_dense}" \
    --experiment-prefix "${EXPERIMENT_PREFIX}"
fi

cat <<'EOF'
Phase 1D interpretation:
  * Compare s0/s1/s2 for solver sensitivity, not a causal improvement claim.
  * Compare s0/s4/s5/s6 for released-graph/CI-pool stabilization.
  * Compare s0/s7/s8/s9/s10 for actual graph sparsity interventions.
  * Before interpreting s8/s9/s10, verify in phase1_run_summary.csv that
    ``structural_prior_excludes_true_graph`` is false for every realization.
  * Report active W edges, solver status/runtime/gap, CI counts and zero-rate
    together with paired test NMSE.  A lower edge count alone is not success.
  * ``constraint_counts.csv`` now records the applied edge penalty/parent cap,
    estimated/true edge counts, directed precision/recall/F1 and SHD; use
    these fields to verify that s7--s9 were read by the solver.
  * With PHASE1D_SUPPLEMENT=1, use the s11--s17 rows to select a candidate only
    after checking structural_prior_excludes_true_graph, edge-count CV, edge
    Jaccard, timeout/MIP-gap rates and zero-constraint rate.
EOF
