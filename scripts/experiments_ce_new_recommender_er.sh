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
# Arms E0..E3 run the Gaussian linear SEM.  Arms E4..E7 repeat the CE screen on
# the nonlinear SEM (++problem.sem_type=nonlinear) as a misspecification stress
# test; set INCLUDE_NONLINEAR=0 to skip them.  The nonlinear conditional mean is
# not described by the linear synthetic oracle or the raw-SEM W moment, so the
# oracle is disabled for those arms and E6's W term is a deliberately
# misspecified control.  E7 is the nonlinear no-constraint reference.
#
# Arms B0..B2 are method baselines (INCLUDE_BASELINES=0 to skip): B0 is the HC
# NN predictor with the true W moment (hc_predictor + ALM), B1/B2 are mark and
# mark_with_cc.  They follow scripts/replay_er_sf_baselines.sh and use the true
# DAG so they are comparable with E0..E7.

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
)

# Keep the SEM type explicit for every arm.  In particular, do not add a
# nonlinear override after a Gaussian override: the resolved Hydra config must
# contain one unambiguous generator type for a seed-matched comparison.
LINEAR_SEM=(
  "++problem.sem_type=gauss"
)
NONLINEAR_SEM=(
  "++problem.sem_type=nonlinear"
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

# hc_predictor baseline overrides.  Only keys that exist in hc_predictor.yaml
# are passed (hc_predictor has no use_stochastic_constrained_optimizer key).
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

# Baseline runner for the HC NN predictor (hc_predictor).  Keeps the same
# graph/noise seed pairing as the CE arms; HC_CONSTRAINT_BACKEND selects the
# ALM/SPBM dual backend used by replay_er_sf_baselines.sh.
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

# Baseline runner for the tree/XGB methods (mark, mark_with_cc).  Their configs
# do not define the NN keys, so only keys they own are passed and keys absent
# from their schema use Hydra's ``+`` append form.  Constraint audit is not
# enabled: those estimators have no constraint_audit_rows.
run_phase1_synthetic_tree() {
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
      run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
        "${solver}" "${PROBLEMS}" "${seed_overrides[@]}" "$@"
    done
  done
}

if [[ "${INCLUDE_MAIN:-1}" == "1" ]]; then
# E0: no-constraint baseline with the same network/validation path.
run_phase1_synthetic_ce_new "e0_no_constraint" \
  "${COMMON[@]}" \
  "${LINEAR_SEM[@]}" \
  "solver.constrained=false" \
  "solver.recalculate_dag=false" \
  "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" \
  "solver.constraint_audit_oracle=none"

# E1: true-W reference (the upper bound of one-shot moment constraints).
run_phase1_synthetic_ce_new "e1_true_w" \
  "${COMMON[@]}" \
  "${LINEAR_SEM[@]}" \
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
  "${LINEAR_SEM[@]}" \
  "${CE_COMMON[@]}" \
  "solver.use_w_constraints=false"

# E3: W + upgraded CE (the recommended production arm).
run_phase1_synthetic_ce_new "e3_w_plus_ce_upgraded" \
  "${COMMON[@]}" \
  "${LINEAR_SEM[@]}" \
  "${CE_COMMON[@]}" \
  "solver.use_w_constraints=true"

# W-variant ablation (INCLUDE_W_VARIANTS=0 to skip).  Tests whether the legacy
# global W mean-moment adds anything beyond a prediction-dependent mean shift.
if [[ "${INCLUDE_W_VARIANTS:-1}" == "1" ]]; then
  # W1: ALM only on prediction-dependent W rows (target-local moments).
  run_phase1_synthetic_ce_new "w1_true_w_masked" \
    "${COMMON[@]}" \
    "${LINEAR_SEM[@]}" \
    "solver.constrained=true" \
    "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}" \
    "solver.recalculate_dag=false" \
    "solver.w_matrix_space=raw_sem" \
    "solver.use_w_constraints=true" \
    "solver.use_ci_penalty=false" \
    "solver.w_prediction_dependent_mask=true" \
    "solver.constraint_audit_oracle=synthetic_linear_sem"

  # W2: no-retrain mean calibration only (no W ALM).
  run_phase1_synthetic_ce_new "w2_true_w_calib_only" \
    "${COMMON[@]}" \
    "${LINEAR_SEM[@]}" \
    "solver.constrained=true" \
    "solver.recalculate_dag=false" \
    "solver.w_matrix_space=raw_sem" \
    "solver.use_w_constraints=false" \
    "solver.use_ci_penalty=false" \
    "solver.w_bias_calibration=true" \
    "solver.constraint_audit_oracle=synthetic_linear_sem"

  # W3: masked W ALM plus the calibration intercept.
  run_phase1_synthetic_ce_new "w3_true_w_masked_calib" \
    "${COMMON[@]}" \
    "${LINEAR_SEM[@]}" \
    "solver.constrained=true" \
    "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}" \
    "solver.recalculate_dag=false" \
    "solver.w_matrix_space=raw_sem" \
    "solver.use_w_constraints=true" \
    "solver.use_ci_penalty=false" \
    "solver.w_prediction_dependent_mask=true" \
    "solver.w_bias_calibration=true" \
    "solver.constraint_audit_oracle=synthetic_linear_sem"
fi

# Capacity x W interaction (INCLUDE_CAPACITY_VARIANTS=0 to skip).  N0/N1 are the
# default-capacity e0/e1 already.  N2/N3 make the MLP high-variance (wide/deep,
# dropout 0, weight decay 0, validation off) to test whether W helps a
# high-variance model.  Verify N2 has lower train and higher test NMSE before
# calling it high-variance; lr is lowered via HC_HICAP_LR for stability.
if [[ "${INCLUDE_CAPACITY_VARIANTS:-1}" == "1" ]]; then
  HICAP=(
    "solver.hidden_dim=${HC_HICAP_HIDDEN_DIM:-128}"
    "solver.depth=${HC_HICAP_DEPTH:-3}"
    "solver.dropout=${HC_HICAP_DROPOUT:-0.0}"
    "solver.weight_decay=${HC_HICAP_WEIGHT_DECAY:-0.0}"
    "solver.learning_rate=${HC_HICAP_LR:-0.1}"
    "solver.use_validation=false"
    "solver.restore_best_validation_model=false"
  )
  run_phase1_synthetic_ce_new "n2_hicap_no_w" \
    "${COMMON[@]}" \
    "${LINEAR_SEM[@]}" \
    "${HICAP[@]}" \
    "solver.constrained=false" \
    "solver.recalculate_dag=false" \
    "solver.use_w_constraints=false" \
    "solver.use_ci_penalty=false" \
    "solver.constraint_audit_oracle=synthetic_linear_sem"
  run_phase1_synthetic_ce_new "n3_hicap_true_w" \
    "${COMMON[@]}" \
    "${LINEAR_SEM[@]}" \
    "${HICAP[@]}" \
    "solver.constrained=true" \
    "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}" \
    "solver.recalculate_dag=false" \
    "solver.w_matrix_space=raw_sem" \
    "solver.use_w_constraints=true" \
    "solver.use_ci_penalty=false" \
    "solver.constraint_audit_oracle=synthetic_linear_sem"
fi

# Nonlinear ER stress test (INCLUDE_NONLINEAR=0 to skip).  The partial-
# correlation CE statistic assumes a linear conditional mean, so these arms
# measure graceful degradation, not correctness.  The synthetic linear oracle
# and the raw-SEM W moment do not describe the nonlinear mean, so the oracle is
# disabled and the E6 W term is read as a misspecified control.
if [[ "${INCLUDE_NONLINEAR:-1}" == "1" ]]; then
  run_phase1_synthetic_ce_new "e4_nonlinear_pure_ce" \
    "${COMMON[@]}" \
    "${CE_COMMON[@]}" \
    "${NONLINEAR_SEM[@]}" \
    "solver.use_w_constraints=false" \
    "solver.constraint_audit_oracle=none"

  run_phase1_synthetic_ce_new "e5_nonlinear_quantile_ce" \
    "${COMMON[@]}" \
    "${CE_COMMON[@]}" \
    "${NONLINEAR_SEM[@]}" \
    "solver.ce_residualize_method=quantile" \
    "solver.use_w_constraints=false" \
    "solver.constraint_audit_oracle=none"

  run_phase1_synthetic_ce_new "e6_nonlinear_w_plus_ce" \
    "${COMMON[@]}" \
    "${CE_COMMON[@]}" \
    "${NONLINEAR_SEM[@]}" \
    "solver.use_w_constraints=true" \
    "solver.constraint_audit_oracle=none"

  # E7: nonlinear no-constraint reference.  Without it the nonlinear CE arms
  # cannot be separated from the harder nonlinear target itself.
  run_phase1_synthetic_ce_new "e7_nonlinear_no_constraint" \
    "${COMMON[@]}" \
    "${NONLINEAR_SEM[@]}" \
    "solver.constrained=false" \
    "solver.recalculate_dag=false" \
    "solver.use_w_constraints=false" \
    "solver.use_ci_penalty=false" \
    "solver.constraint_audit_oracle=none"
fi
fi

# Nonlinear ER method baselines (INCLUDE_NONLINEAR_BASELINES=1).  These answer
# whether the classical methods keep their ranking on the nonlinear generator;
# mark_with_cc's W moment is misspecified there (nonlinear mean).
if [[ "${INCLUDE_NONLINEAR_BASELINES:-0}" == "1" ]]; then
  HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND:-alm}"
  run_phase1_synthetic_nn_baseline "b0nl_hc_predictor_${HC_BASELINE_BACKEND}" "hc_predictor" "${HC_BASELINE_BACKEND}" \
    "${BASELINE_NN_COMMON[@]}" \
    "${NONLINEAR_SEM[@]}" \
    "solver.constraint_audit_oracle=none"
  run_phase1_synthetic_tree "b1nl_mark" "mark" \
    "${NONLINEAR_SEM[@]}"
  run_phase1_synthetic_tree "b2nl_mark_with_cc" "mark_with_cc" \
    "solver.n_outer=${N_OUTER}" \
    "solver.time_limit=${TIME_LIMIT}" \
    "${NONLINEAR_SEM[@]}" \
    "+solver.w_matrix_space=raw_sem"
fi

# Method baselines (INCLUDE_BASELINES=0 to skip).  Conventions follow
# scripts/replay_er_sf_baselines.sh.  All use the true DAG
# (recalculate_dag=false) so the comparison isolates the predictor.
if [[ "${INCLUDE_BASELINES:-1}" == "1" ]]; then
  # B0: HC NN predictor with the true W moment (ALM by default, no CE).
  HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND:-alm}"
  run_phase1_synthetic_nn_baseline "b0_hc_predictor_${HC_BASELINE_BACKEND}" "hc_predictor" "${HC_BASELINE_BACKEND}" \
    "${BASELINE_NN_COMMON[@]}" \
    "${LINEAR_SEM[@]}"

  # B1/B2: MARK and MARK-CC tree baselines.
  run_phase1_synthetic_tree "b1_mark" "mark" \
    "${LINEAR_SEM[@]}"
  run_phase1_synthetic_tree "b2_mark_with_cc" "mark_with_cc" \
    "solver.n_outer=${N_OUTER}" \
    "solver.time_limit=${TIME_LIMIT}" \
    "${LINEAR_SEM[@]}" \
    "+solver.w_matrix_space=raw_sem"

  # M0/M1: fair ablation of the W penalty on the XGB + Lagrangian path.  Both
  # arms use mark_with_cc's fitting procedure (n_outer x n_estimators = 100
  # trees, standardized y, same objective/API); M0 sets rho0=0 so the W penalty
  # and its dual are identically zero.  Only M1 - M0 identifies the effect of
  # the W graph-moment constraint.
  run_phase1_synthetic_tree "m0_xgb100_no_w" "mark_with_cc" \
    "solver.n_outer=${N_OUTER}" \
    "solver.time_limit=${TIME_LIMIT}" \
    "solver.rho0=0.0" \
    "${LINEAR_SEM[@]}" \
    "+solver.w_matrix_space=raw_sem"
  run_phase1_synthetic_tree "m1_xgb100_w" "mark_with_cc" \
    "solver.n_outer=${N_OUTER}" \
    "solver.time_limit=${TIME_LIMIT}" \
    "${LINEAR_SEM[@]}" \
    "+solver.w_matrix_space=raw_sem"
fi

echo "=== CE-NEW-ER complete ==="
