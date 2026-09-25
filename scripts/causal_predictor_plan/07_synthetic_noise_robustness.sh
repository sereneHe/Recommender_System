#!/usr/bin/env bash
# Phase 7: distribution-shift stress test for the supported continuous linear
# SEMs. Laplace and Student-t have variance matched to Gaussian noise_scale.
# Count and zero-inflated SEMs are deliberately not launched: the repository
# does not yet contain a valid non-linear count SEM generator.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "07" "synthetic continuous-noise robustness"

PROBLEMS="${PROBLEMS:-synthetic_er,synthetic_sf}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN07_SYNTHETIC_NOISE}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
STUDENT_T_DF="${STUDENT_T_DF:-5}"

export HC_CONSTRAINT_BACKEND=alm
export HC_WEIBULL_GAUSSIANIZE=0

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.feature_selector=none"
  "solver.learning_rate=0.03"
  "solver.weight_decay=0.01"
  "solver.hidden_dim=32"
  "solver.depth=2"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
)

for noise in gauss laplace student_t; do
  extra=("++problem.sem_type=${noise}")
  if [[ "${noise}" == "student_t" ]]; then
    extra+=("++problem.noise_df=${STUDENT_T_DF}")
  fi

  # Oracle graph: recalculate_dag=false preserves the synthetic ground truth.
  run_seeded "${noise}_true_w" "${EXPERIMENT_PREFIX}" "hc_predictor" "${PROBLEMS}" true \
    "${COMMON[@]}" "${extra[@]}" \
    "solver.constrained=true" "solver.recalculate_dag=false"

  run_seeded "${noise}_true_independent" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" true \
    "${COMMON[@]}" "${extra[@]}" \
    "solver.constrained=true" "solver.recalculate_dag=false" \
    "solver.use_w_constraints=false" "solver.use_ci_penalty=true" \
    "solver.ci_penalty_kind=conditional_expectation" \
    "solver.ce_constraint_backend=alm_pbm" \
    "solver.ci_add_dsep_independence=true" \
    "solver.ci_add_collider_marginal_independence=false" \
    "solver.ci_add_shielded_collider_dependence=false" \
    "solver.ce_use_balanced_batches=false" \
    "solver.use_stochastic_constrained_optimizer=false"

  # Estimated graph: isolates graph error from the true-W result above.
  run_seeded "${noise}_estimated_independent" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" true \
    "${COMMON[@]}" "${extra[@]}" \
    "solver.constrained=true" "solver.recalculate_dag=true" \
    "solver.use_w_constraints=false" "solver.use_ci_penalty=true" \
    "solver.ci_penalty_kind=conditional_expectation" \
    "solver.ce_constraint_backend=alm_pbm" \
    "solver.ci_add_dsep_independence=true" \
    "solver.ci_add_collider_marginal_independence=false" \
    "solver.ci_add_shielded_collider_dependence=false" \
    "solver.ce_use_balanced_batches=false" \
    "solver.use_stochastic_constrained_optimizer=false"
done

echo "Count/zero-inflated stress tests are intentionally blocked until a valid count SEM generator is added."
