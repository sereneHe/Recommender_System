#!/usr/bin/env bash
# Combined launcher for the still-unrun CE-new work:
#   1. SF: only the previously uncompleted E3 plus matched HC/Mark/Mark-CC.
#   2. FRED industry CE pre-audit.
#   3. ER: nonlinear method baselines (hc_predictor / mark / mark_with_cc).
#
# The ER main screen (E0--E7, linear baselines, M0/M1) is launched separately
# with experiments_ce_new_recommender_er.sh, so it is not repeated here.
# CoDiet is excluded; run experiments_ce_new_recommender_codiet_all.sh.
#
# Run with the PBS wrapper:
#   qsub -v EXPERIMENT_SCRIPT=scripts/experiments_ce_new_recommender_er_sf_industry.sh \
#     cluster_computing/run_metacentrum.pbs

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ER_NONLINEAR_BASELINES="${RUN_ER_NONLINEAR_BASELINES:-1}"
RUN_SF="${RUN_SF:-1}"
RUN_INDUSTRY_PREAUDIT="${RUN_INDUSTRY_PREAUDIT:-1}"
RUN_INDUSTRY_CE_LITE="${RUN_INDUSTRY_CE_LITE:-0}"

# Synthetic scripts use graph/noise seeds; FRED uses model/CV seeds.
GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${SEEDS:-42 43 44}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND:-alm}"
DRY_RUN="${DRY_RUN:-0}"
INDUSTRY_GO_PROBLEMS="${INDUSTRY_GO_PROBLEMS:-}"

# Positional Hydra directory overrides supplied by run_metacentrum.pbs must
# reach every component script unchanged.
declare -a EXTRA_OVERRIDES=()
EXTRA_OVERRIDE_COUNT="$#"
if (( $# > 0 )); then
  EXTRA_OVERRIDES=("$@")
fi

require_bool() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || {
    echo "${name} must be 0 or 1, got ${value}." >&2
    exit 2
  }
}

for toggle in RUN_ER_NONLINEAR_BASELINES RUN_SF RUN_INDUSTRY_PREAUDIT RUN_INDUSTRY_CE_LITE; do
  require_bool "${toggle}" "${!toggle}"
done

run_stage() {
  local label="$1"
  shift
  echo
  echo "================================================================================"
  echo "=== ${label} ==="
  echo "================================================================================"
  "$@"
}

run_child() {
  local script="$1"
  shift
  if (( EXTRA_OVERRIDE_COUNT > 0 )); then
    "$@" "${script}" "${EXTRA_OVERRIDES[@]}"
  else
    "$@" "${script}"
  fi
}

echo "=== CE-new SF + industry + nonlinear-ER-baselines launcher ==="
echo "repo=${REPO_ROOT} graph_seeds=${GRAPH_SEEDS} noise_seeds=${NOISE_SEEDS} seeds=${SEEDS} dry_run=${DRY_RUN}"

if [[ "${RUN_ER_NONLINEAR_BASELINES}" == "1" ]]; then
  run_stage "ER: nonlinear method baselines (HC/Mark/Mark-CC)" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_er.sh" env \
      GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" \
      CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" \
      INCLUDE_MAIN=0 INCLUDE_BASELINES=0 INCLUDE_NONLINEAR_BASELINES=1 DRY_RUN="${DRY_RUN}" \
      bash
fi

if [[ "${RUN_SF}" == "1" ]]; then
  run_stage "SF: resume E3 + matched HC/Mark/Mark-CC baselines" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_sf.sh" env \
      GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" \
      CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" \
      RUN_SF_E0=0 RUN_SF_E1=0 RUN_SF_E2=0 RUN_SF_E3=1 INCLUDE_BASELINES=1 DRY_RUN="${DRY_RUN}" \
      bash
fi

if [[ "${RUN_INDUSTRY_PREAUDIT}" == "1" ]]; then
  run_stage "Industry: CE pre-audit" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_industry.sh" env \
      SEEDS="${SEEDS}" CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      STAGE=preaudit DRY_RUN="${DRY_RUN}" \
      bash
fi

if [[ "${RUN_INDUSTRY_CE_LITE}" == "1" ]]; then
  if [[ -z "${INDUSTRY_GO_PROBLEMS}" ]]; then
    echo "RUN_INDUSTRY_CE_LITE=1 requires INDUSTRY_GO_PROBLEMS from the pre-audit gate." >&2
    exit 2
  fi
  run_stage "Industry: CE-Lite, matched controls, HC/Mark/Mark-CC" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_industry.sh" env \
      SEEDS="${SEEDS}" CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" INCLUDE_CLASSICAL_BASELINES=1 \
      STAGE=ce_lite_go PROBLEMS="${INDUSTRY_GO_PROBLEMS}" DRY_RUN="${DRY_RUN}" \
      bash
fi

echo "=== CE-new launcher complete ==="
