#!/usr/bin/env bash
# CE New-Recommender screens on FRED 16-country monthly industry data.
#
# Industry: near-dense, strongly co-moving monthly macro panel; small n
# (~150-250 observations), no known ground-truth DAG, strong autocorrelation.
# This is the hardest case for CE.  The configuration therefore uses:
#   - W constraints as the primary stable one-shot moments
#   - birth-death posterior CI filtering (HC_CE_BD_MCMC=1, high support)
#   - partial correlation + strong shrinkage (0.3)
#   - Newey-West HAC standard errors (monthly autocorrelation), tolerance k=2.5
#   - dependent/shielded collider constraints always disabled
#
# Run the pre-audit first (constraint count/support plus descriptive null-fit
# diagnostics). SNR/sign-flip thresholds apply only to signed dependence
# constraints; this script deliberately adds independent constraints only.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/causal_predictor_plan/_common.sh"

announce_stage "CE-NEW-INDUSTRY" "FRED 16-country monthly: CE-Lite with W + BD-filtered CE"

GROUP="${GROUP:-FRED_16country_monthly}"
PREAUDIT_PROBLEMS="${PREAUDIT_PROBLEMS:-${GROUP}/industry_eu_est,${GROUP}/industry_eu_ita,${GROUP}/industry_eu_svn,${GROUP}/industry_eu_aut,${GROUP}/industry_eu_deu}"
ALL_PROBLEMS="${ALL_PROBLEMS:-${GROUP}/industry_eu_aut,${GROUP}/industry_eu_bel,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_esp,${GROUP}/industry_eu_est,${GROUP}/industry_eu_fin,${GROUP}/industry_eu_fra,${GROUP}/industry_eu_grc,${GROUP}/industry_eu_irl,${GROUP}/industry_eu_ita,${GROUP}/industry_eu_ltu,${GROUP}/industry_eu_lux,${GROUP}/industry_eu_nld,${GROUP}/industry_eu_prt,${GROUP}/industry_eu_svk,${GROUP}/industry_eu_svn}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-PLAN_CE_NEW_INDUSTRY}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
STAGE="${STAGE:-preaudit}"
PREAUDIT_EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}_PREAUDIT"
CELITE_EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}_CELITE"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-4}"
TIME_TEST_SIZE="${TIME_TEST_SIZE:-12}"

LR="${LR:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"

export HC_WEIBULL_GAUSSIANIZE=0

# Posterior-stable constraints are mandatory for industry: single-MILP d-sep
# constraints are too unstable across folds / graph seeds.
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-1}"
export HC_CE_BD_CHAINS="${HC_CE_BD_CHAINS:-4}"
export HC_CE_BD_STEPS="${HC_CE_BD_STEPS:-1500}"
export HC_CE_BD_BURN_IN="${HC_CE_BD_BURN_IN:-300}"
export HC_CE_BD_INDEPENDENCE_SUPPORT="${HC_CE_BD_INDEPENDENCE_SUPPORT:-0.85}"
export HC_CE_BD_MAX_CONFLICT_SUPPORT="${HC_CE_BD_MAX_CONFLICT_SUPPORT:-0.15}"
# Separate the birth-death chain RNG seed from the data/graph seed.  _common.sh
# derives HC_CE_BD_SEED = HC_CE_BD_SEED_BASE + seed; without a base the chains
# reuse one RNG seed across data seeds, which confounds cross-seed support.
export HC_CE_BD_SEED_BASE="${HC_CE_BD_SEED_BASE:-20260916}"

COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.cv_strategy=time_series"
  "solver.cv_time_test_size=${TIME_TEST_SIZE}"
  "solver.cv_time_gap=0"
  "solver.recalculate_dag=true"
  "solver.feature_selector=none"
  "solver.prediction_loss=mse"
  "solver.learning_rate=${LR}"
  "solver.weight_decay=${WEIGHT_DECAY}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
)

# Upgraded CE-Lite for small/time-series macro panels.
CE_LITE_COMMON=(
  "solver.constrained=true"
  "solver.ce_constraint_backend=${CE_CONSTRAINT_BACKEND}"
  "solver.use_w_constraints=true"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_statistic_kind=partial_correlation"
  "solver.ce_statistic_shrinkage=0.3"
  "solver.ce_residualize_method=linear"
  "solver.ce_se_method=hac"
  "solver.ce_hac_max_lag=6"
  "solver.ce_tolerance_sd_multiplier=2.5"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.validation_split_strategy=time"
  "solver.dag_fit_scope=inner_train"
  "solver.early_stopping_patience=2"
  "solver.constraint_audit_enabled=true"
)

case "${STAGE}" in
  preaudit)
    # Stage P: constraint audit. Runs the CE arm with the audit enabled and a
    # deliberately small training budget (constraints come from the DAG, not
    # from a converged predictor). Independent constraints target zero, so
    # their SNR/sign flips are descriptive rather than a validity gate. The
    # pre-audit advances datasets with >=3 constraints and >=60% seed support
    # to the paired CE-Lite screen; it is not a final efficacy verdict.
    AUDIT_N_OUTER="${AUDIT_N_OUTER:-1}"
    AUDIT_N_INNER="${AUDIT_N_INNER:-1}"
    echo "Running CE pre-audit on: ${PREAUDIT_PROBLEMS}"
    run_seeded "ce_audit" "${PREAUDIT_EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PREAUDIT_PROBLEMS}" false \
      "${COMMON[@]}" \
      "${CE_LITE_COMMON[@]}" \
      "solver.n_outer=${AUDIT_N_OUTER}" \
      "solver.n_inner=${AUDIT_N_INNER}"

    if [[ "${DRY_RUN}" != "1" ]]; then
      AUDIT_ROOT="${AUDIT_ROOT:-multirun}"
      AUDIT_SUMMARY_DIR="${AUDIT_SUMMARY_DIR:-results/ce_preaudit}"
      "${PYTHON_BIN}" scripts/causal_predictor_plan/summarize_ce_preaudit.py \
        --root "${AUDIT_ROOT}" --output-dir "${AUDIT_SUMMARY_DIR}" \
        --experiment-prefix "${PREAUDIT_EXPERIMENT_PREFIX}" --stage preaudit
      echo "Pre-audit gate: ${AUDIT_SUMMARY_DIR}/ce_preaudit_gate.csv"
      echo "The Go list means 'advance to paired CE-Lite screening', not proven efficacy."
      echo "Set PROBLEMS to selected Go countries and run:"
      echo "  STAGE=ce_lite_go PROBLEMS=<comma list> bash $0"
    else
      echo "DRY_RUN: skipping summarize_ce_preaudit.py"
    fi
    ;;

  ce_lite_go)
    # Stage L: full CE-Lite only on problems that passed pre-audit.
    # Set PROBLEMS to the audited Go countries first.
    run_seeded "hce_lite" "${CELITE_EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "${CE_LITE_COMMON[@]}"

    # Baseline: same network / CV / seed but no constraints.
    run_seeded "s0_no_constraint" "${CELITE_EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "solver.constrained=false" \
      "solver.use_w_constraints=false" \
      "solver.use_ci_penalty=false"

    # Matched HC-CE reference: W constraints only (no CE), with identical
    # architecture, validation, graph scope, and minibatch policy.
    run_seeded "hce_w_only" "${CELITE_EXPERIMENT_PREFIX}" "hc_predictor_ce" "${PROBLEMS}" false \
      "${COMMON[@]}" \
      "${CE_LITE_COMMON[@]}" \
      "solver.use_ci_penalty=false"

    if [[ "${DRY_RUN}" != "1" ]]; then
      AUDIT_ROOT="${AUDIT_ROOT:-multirun}"
      AUDIT_SUMMARY_DIR="${AUDIT_SUMMARY_DIR:-results/ce_industry}"
      "${PYTHON_BIN}" scripts/causal_predictor_plan/summarize_ce_preaudit.py \
        --root "${AUDIT_ROOT}" --output-dir "${AUDIT_SUMMARY_DIR}" \
        --experiment-prefix "${CELITE_EXPERIMENT_PREFIX}" --stage full
    fi
    ;;

  *)
    die "Unknown STAGE=${STAGE}. Use 'preaudit' or 'ce_lite_go'."
    ;;
esac

echo "=== CE-NEW-INDUSTRY complete ==="
