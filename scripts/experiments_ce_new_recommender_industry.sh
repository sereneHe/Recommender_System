#!/usr/bin/env bash
# CE-Lite recommender: FRED monthly industry pre-audit and full CE-Lite runs.
#
# Two stages:
#
#   STAGE=preaudit    (default) run the paired DNN-only vs W+CE-Lite screen on
#                     the FRED targets, write the audit artifacts, and emit the
#                     three-stage Go/No-Go gate.  Use the gate to decide which
#                     countries are worth a full run.
#
#   STAGE=ce_lite_go  run the full CE-Lite arm (W + high-confidence CE with
#                     partial-correlation/shrinkage/time-series SE) on the
#                     selected targets, over the seed list.
#
# Usage:
#   STAGE=preaudit bash scripts/experiments_ce_new_recommender_industry.sh
#   STAGE=ce_lite_go \
#     PROBLEMS=FRED_16country_monthly/industry_eu_ita,FRED_16country_monthly/industry_eu_svn \
#     bash scripts/experiments_ce_new_recommender_industry.sh
#
# Recognised environment: STAGE, PROBLEMS, DATASET_SCOPE (all|pilot), SEEDS,
# TIME_LIMIT, N_RUNS, N_OUTER, N_INNER, CV_TIME_TEST_SIZE, CV_TIME_GAP,
# EXPERIMENT_PREFIX, SUMMARY_ROOT, DRY_RUN, plus trailing Hydra overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="${SCRIPT_DIR}/causal_predictor_plan"

STAGE="${STAGE:-preaudit}"
export HC_WEIBULL_GAUSSIANIZE=0

case "${STAGE}" in
  preaudit)
    export DATASET_GROUPS="industry"
    export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-CE_NEW_INDUSTRY_PREAUDIT}"
    echo "=== CE-Lite industry pre-audit (Go/No-Go screen) ==="
    bash "${PLAN_DIR}/09_ce_preaudit.sh" "$@"
    if [[ "${DRY_RUN:-0}" != "1" && -n "${SUMMARY_ROOT:-}" ]]; then
      "${PYTHON_BIN:-python3}" "${PLAN_DIR}/summarize_ce_preaudit.py" \
        --root "${SUMMARY_ROOT}" --output-dir "${SUMMARY_DIR:-${SUMMARY_ROOT}}"
    else
      echo
      echo "Next: summarize the runs to get the Go/No-Go gate, e.g."
      echo "  python3 ${PLAN_DIR}/summarize_ce_preaudit.py --root <hydra-output-root>"
      echo "or set SUMMARY_ROOT=<hydra-output-root> to run it automatically."
    fi
    ;;

  ce_lite_go)
    # Source the shared launcher helpers with the trailing overrides so that
    # run_seeded forwards them to every Hydra call.
    source "${PLAN_DIR}/_common.sh" "$@"
    require_env PROBLEMS
    export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-CE_NEW_INDUSTRY_CE_LITE}"
    TIME_LIMIT="${TIME_LIMIT:-120}"
    N_RUNS="${N_RUNS:-5}"
    N_OUTER="${N_OUTER:-10}"
    N_INNER="${N_INNER:-100}"
    CV_TIME_TEST_SIZE="${CV_TIME_TEST_SIZE:-12}"
    CV_TIME_GAP="${CV_TIME_GAP:-0}"

    announce_stage "CE-Lite" "full FRED CE-Lite on selected targets"

    COMMON=(
      "solver.time_limit=${TIME_LIMIT}"
      "solver.n_runs=${N_RUNS}"
      "solver.n_outer=${N_OUTER}"
      "solver.n_inner=${N_INNER}"
      "solver.cv_strategy=time_series"
      "solver.cv_time_test_size=${CV_TIME_TEST_SIZE}"
      "solver.cv_time_gap=${CV_TIME_GAP}"
      "solver.validation_split_strategy=time"
      "solver.recalculate_dag=true"
      "solver.dag_fit_scope=inner_train"
      "solver.feature_selector=none"
      "solver.constraint_audit_enabled=true"
      "solver.ci_mode=manual"
      "solver.ci_manual_from_training_dag=true"
      "solver.ci_target_related_only=true"
      "solver.ci_add_dsep_independence=true"
      "solver.ci_add_collider_marginal_independence=false"
      "solver.ci_add_collider_conditional_dependence=false"
      "solver.ci_add_shielded_collider_dependence=false"
      "solver.ci_penalty_kind=conditional_expectation"
      "solver.ce_constraint_backend=alm_pbm"
      "solver.ce_independence_tolerance=0.0"
      "solver.ce_tolerance_mode=standard_error"
      "solver.ce_tolerance_sd_multiplier=1.96"
      "solver.ce_se_method=auto"
      "solver.ce_window_n_windows=5"
      "solver.ce_window_min_size=100"
      "solver.ce_hac_max_lag=6"
      "solver.ce_window_filter_enabled=true"
      "solver.ce_window_max_sign_flip_rate=0.30"
      "solver.ce_statistic_kind=partial_correlation"
      "solver.ce_statistic_shrinkage=${CE_SHRINKAGE:-0.1}"
      "solver.ce_residualize_method=${CE_RESIDUALIZE_METHOD:-linear}"
      "solver.constrained=true"
      "solver.use_ci_penalty=true"
      "solver.use_w_constraints=true"
    )
    run_seeded "ce_lite" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}"
    ;;

  *)
    echo "ERROR: STAGE must be 'preaudit' or 'ce_lite_go', got '${STAGE}'." >&2
    exit 2
    ;;
esac
