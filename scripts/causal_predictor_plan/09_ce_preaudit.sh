#!/usr/bin/env bash
# Phase 09: CE feasibility/stability pre-audit across synthetic, CoDiet and
# Industry. This reports diagnostics; it does not automatically label a data
# set Go/No-Go or select a winner from the same folds used for screening.

SEEDS="${SEEDS:-42 43 44}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

announce_stage "09" "CE constraint pre-audit"

DATASET_SCOPE="${DATASET_SCOPE:-all}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN09_CE_PREAUDIT}"

SYNTH_PROBLEMS="${SYNTH_PROBLEMS:-synthetic_er,synthetic_sf}"
CODIET_PROBLEMS="${CODIET_PROBLEMS:-codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr}"
INDUSTRY_PROBLEMS="${INDUSTRY_PROBLEMS:-FRED_16country_monthly/industry_eu_aut,FRED_16country_monthly/industry_eu_bel,FRED_16country_monthly/industry_eu_deu,FRED_16country_monthly/industry_eu_esp,FRED_16country_monthly/industry_eu_est,FRED_16country_monthly/industry_eu_fin,FRED_16country_monthly/industry_eu_fra,FRED_16country_monthly/industry_eu_grc,FRED_16country_monthly/industry_eu_irl,FRED_16country_monthly/industry_eu_ita,FRED_16country_monthly/industry_eu_ltu,FRED_16country_monthly/industry_eu_lux,FRED_16country_monthly/industry_eu_nld,FRED_16country_monthly/industry_eu_prt,FRED_16country_monthly/industry_eu_svk,FRED_16country_monthly/industry_eu_svn}"

if [[ "${DATASET_SCOPE}" == "pilot" ]]; then
  CODIET_PROBLEMS="${CODIET_PILOT_PROBLEMS:-codiet,codiet_glu,codiet_hdl}"
  INDUSTRY_PROBLEMS="${INDUSTRY_PILOT_PROBLEMS:-FRED_16country_monthly/industry_eu_aut,FRED_16country_monthly/industry_eu_est,FRED_16country_monthly/industry_eu_ita}"
elif [[ "${DATASET_SCOPE}" != "all" ]]; then
  die "DATASET_SCOPE must be 'pilot' or 'all'."
fi

run_preaudit_group() {
  local label="$1"
  local problems="$2"
  local cv_strategy="$3"
  local ci_kind="$4"
  local tolerance_mode="$5"
  shift 5
  local validation_strategy="${VALIDATION_SPLIT_STRATEGY:-}"
  if [[ -z "${validation_strategy}" ]]; then
    validation_strategy="random"
    [[ "${cv_strategy}" == "time_series" ]] && validation_strategy="time"
  fi

  local -a common=(
    "solver.time_limit=${TIME_LIMIT}"
    "solver.n_runs=${N_RUNS}"
    "solver.n_outer=${N_OUTER}"
    "solver.n_inner=${N_INNER}"
    "solver.cv_strategy=${cv_strategy}"
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
    "solver.ci_penalty_kind=${ci_kind}"
    "solver.ce_constraint_backend=alm_pbm"
    "solver.ce_independence_tolerance=0.0"
    "solver.ce_tolerance_mode=${tolerance_mode}"
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
    "solver.validation_split_strategy=${validation_strategy}"
    "$@"
  )

# Pair a no-constraint neural baseline with W+CE-Lite using the same data,
# fold-generation seeds, graph settings, and training budget.
  run_seeded "${label}_dnn_only" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${problems}" false \
    "${common[@]}" "solver.constrained=false" "solver.use_ci_penalty=false" "solver.use_w_constraints=false"
  run_seeded "${label}_ce_active" "${EXPERIMENT_PREFIX}" "hc_predictor_ce" "${problems}" false \
    "${common[@]}" "solver.constrained=true" "solver.use_ci_penalty=true" "solver.use_w_constraints=true"
}

export HC_WEIBULL_GAUSSIANIZE=0

if [[ "${DATASET_SCOPE}" == "all" ]]; then
  echo "Full scope: 27 problems x ${#PLAN_SEEDS[@]} seeds x 2 arms = $((27 * ${#PLAN_SEEDS[@]} * 2)) Hydra runs; this is a cluster batch, not a short local check."
else
  echo "Pilot scope: ${#PLAN_SEEDS[@]} seeds x 2 arms across ER/SF, 3 CoDiet targets, and 3 FRED targets."
fi

# Synthetic and CoDiet use shuffled/stratified folds. CoDiet is deliberately
# audited with its discrete-CMI statistic; continuous partial correlation is
# not treated as valid for its mixed categorical/continuous variables.
run_preaudit_group "synthetic" "${SYNTH_PROBLEMS}" "site_gender" \
  "conditional_expectation" "standard_error"
run_preaudit_group "codiet" "${CODIET_PROBLEMS}" "site_gender" \
  "discrete_conditional_independence" "fixed"

# Industry uses expanding-window validation; ce_se_method=auto therefore
# resolves to Newey-West HAC, not the iid partial-correlation approximation.
run_preaudit_group "industry" "${INDUSTRY_PROBLEMS}" "time_series" \
  "conditional_expectation" "standard_error" \
  "solver.cv_time_test_size=${CV_TIME_TEST_SIZE:-12}" \
  "solver.cv_time_gap=${CV_TIME_GAP:-0}"

echo
echo "The run writes constraint_metadata.csv, constraint_stat_audit.csv, cv_fold_metrics.csv, and (when used) ci_window_filter_audit.csv per Hydra run."
echo "After completion, summarize them with:"
echo "  ${PYTHON_BIN} scripts/causal_predictor_plan/summarize_ce_preaudit.py --root <hydra-output-root>"
echo "Review stability, HAC tolerances, constraint violations, and fold errors together; no single diagnostic is an automatic Go/No-Go gate."
