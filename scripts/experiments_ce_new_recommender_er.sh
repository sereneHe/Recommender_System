#!/usr/bin/env bash
# CE New-Recommender screens on Synthetic ER graphs.
#
# ER: sparse random DAG, iid, n=1000, true graph known.
# Best-case environment for CE: low shrinkage, tight tolerance, pure-CE arm
# (W disabled) plus a W+CE combined arm.  recalculate_dag=false so constraints
# are derived from the true SEM / true DAG, not from a fitted MILP graph.
#
# Requires the CE statistical upgrades already merged into hc_predictor_ce.py:
#   - ce_statistic_kind=partial_correlation
#   - ce_statistic_shrinkage
#   - ce_residualize_method (linear|quantile)
#   - ce_se_method (window|hac) + ce_tolerance_sd_multiplier
#   - cross-window sign-flip constraint pruning
#
# Arms E0..E3 run the Gaussian linear SEM.  Arms E4..E6 repeat the CE screen on
# the nonlinear SEM (++problem.sem_type=nonlinear) as a misspecification stress
# test; set INCLUDE_NONLINEAR=0 to skip them.  The nonlinear conditional mean is
# not described by the linear synthetic oracle or the raw-SEM W moment, so the
# oracle is disabled for those arms and E6's W term is a deliberately
# misspecified control.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

GRAPH_SEEDS="${GRAPH_SEEDS:-${SEEDS:-42 43 44}}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
# _common.sh parsed PLAN_SEEDS from SEEDS before this block ran, so re-derive
# it here; otherwise a GRAPH_SEEDS-only override is silently ignored.
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"

announce_stage "CE-NEW-ER" "Synthetic ER: pure-CE vs true-W vs W+CE"

PROBLEMS="${PROBLEMS:-synthetic_er}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_ER}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"

export HC_WEIBULL_GAUSSIANIZE=0
# For synthetic data the graph is known, so BD post-hoc filtering is not needed
# when constraints come from the true DAG.
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

# Upgraded CE defaults.  ER is iid with n=1000, so we use low shrinkage,
# linear residualization, cross-window SE, and a tight 95% tolerance.
CE_COMMON=(
  "solver.constrained=true"
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}"
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
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

# E0: no-constraint baseline with the same network/validation path.
run_phase1_synthetic_ce_new "e0_no_constraint" \
  "${COMMON[@]}" \
  "solver.constrained=false" \
  "solver.recalculate_dag=false" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" \
  "solver.constraint_audit_oracle=none"

# E1: true-W reference (the upper bound of one-shot moment constraints).
run_phase1_synthetic_ce_new "e1_true_w" \
  "${COMMON[@]}" \
  "solver.constrained=true" \
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}" \
  "solver.recalculate_dag=false" \
  "solver.w_matrix_space=raw_sem" \
  "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# E2: upgraded pure-CE (partial correlation, shrinkage, window SE tolerance).
run_phase1_synthetic_ce_new "e2_pure_ce_upgraded" \
  "${COMMON[@]}" \
  "${CE_COMMON[@]}" \
  "solver.use_w_constraints=false"

# E3: W + upgraded CE (the recommended production arm).
run_phase1_synthetic_ce_new "e3_w_plus_ce_upgraded" \
  "${COMMON[@]}" \
  "${CE_COMMON[@]}" \
  "solver.use_w_constraints=true"

# Nonlinear ER stress test (INCLUDE_NONLINEAR=0 to skip).  The partial-
# correlation CE statistic assumes a linear conditional mean, so these arms
# measure graceful degradation, not correctness.  The synthetic linear oracle
# and the raw-SEM W moment do not describe the nonlinear mean, so the oracle is
# disabled and the E6 W term is read as a misspecified control.
if [[ "${INCLUDE_NONLINEAR:-1}" == "1" ]]; then
  run_phase1_synthetic_ce_new "e4_nonlinear_pure_ce" \
    "${COMMON[@]}" \
    "${CE_COMMON[@]}" \
    "++problem.sem_type=nonlinear" \
    "solver.use_w_constraints=false" \
    "solver.constraint_audit_oracle=none"

  run_phase1_synthetic_ce_new "e5_nonlinear_quantile_ce" \
    "${COMMON[@]}" \
    "${CE_COMMON[@]}" \
    "++problem.sem_type=nonlinear" \
    "solver.ce_residualize_method=quantile" \
    "solver.use_w_constraints=false" \
    "solver.constraint_audit_oracle=none"

  run_phase1_synthetic_ce_new "e6_nonlinear_w_plus_ce" \
    "${COMMON[@]}" \
    "${CE_COMMON[@]}" \
    "++problem.sem_type=nonlinear" \
    "solver.constraint_audit_oracle=none"
fi

echo "=== CE-NEW-ER complete ==="
