#!/usr/bin/env bash
# Unified launcher for the repaired CE-new experiment suite.
#
# Default sequence:
#   1. ER: E0--E7, including nonlinear E6 (W+CE), nonlinear E7 (no-constraint),
#      and the matched B0/B1/B2 hc_predictor / mark / mark_with_cc baselines.
#   2. SF: E0--E3 plus the matched B0/B1/B2 classical baselines.
#   3. Industry: CE pre-audit only.  CE-Lite remains gated on its Go list.
#   4. CoDiet: HC/Mark/Mark-CC, matched HC-CE controls, and type-aware CMI.
#
# Run with the PBS wrapper:
#   qsub -v EXPERIMENT_SCRIPT=scripts/experiments_ce_new_recommender_suite.sh \
#     cluster_computing/run_metacentrum.pbs
#
# To run Industry CE-Lite after pre-audit:
#   qsub -v EXPERIMENT_SCRIPT=scripts/experiments_ce_new_recommender_suite.sh,\
#RUN_INDUSTRY_CE_LITE=1,INDUSTRY_GO_PROBLEMS=FRED_16country_monthly/industry_eu_ita \
#     cluster_computing/run_metacentrum.pbs

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Each component may be disabled without editing the file.  Industry CE-Lite
# is deliberately off until the pre-audit identifies a Go set.
RUN_ER="${RUN_ER:-1}"
RUN_SF="${RUN_SF:-1}"
RUN_INDUSTRY_PREAUDIT="${RUN_INDUSTRY_PREAUDIT:-1}"
RUN_INDUSTRY_CE_LITE="${RUN_INDUSTRY_CE_LITE:-0}"
RUN_CODIET_BASELINE="${RUN_CODIET_BASELINE:-1}"
RUN_CODIET_CMI="${RUN_CODIET_CMI:-1}"

# Shared reproducibility controls.  Synthetic scripts use graph/noise seeds;
# FRED and CoDiet use model/CV seeds.
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

for toggle in RUN_ER RUN_SF RUN_INDUSTRY_PREAUDIT RUN_INDUSTRY_CE_LITE RUN_CODIET_BASELINE RUN_CODIET_CMI; do
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

echo "=== CE-new unified suite ==="
echo "repo=${REPO_ROOT} graph_seeds=${GRAPH_SEEDS} noise_seeds=${NOISE_SEEDS} seeds=${SEEDS} dry_run=${DRY_RUN}"

if [[ "${RUN_ER}" == "1" ]]; then
  # ER owns the fully matched classical baselines.  INCLUDE_BASELINES=1 gives:
  # B0 hc_predictor (ALM, true W), B1 mark, and B2 mark_with_cc.
  run_stage "ER: E0--E7 + matched HC/Mark/Mark-CC baselines" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_er.sh" env \
      GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" \
      CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" \
      INCLUDE_NONLINEAR=1 \
      INCLUDE_BASELINES=1 DRY_RUN="${DRY_RUN}" \
      bash
fi

if [[ "${RUN_SF}" == "1" ]]; then
  run_stage "SF: E0--E3 + matched HC/Mark/Mark-CC baselines" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_sf.sh" env \
      GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" \
      CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" INCLUDE_BASELINES=1 DRY_RUN="${DRY_RUN}" \
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

if [[ "${RUN_CODIET_BASELINE}" == "1" ]]; then
  run_stage "CoDiet: HC/Mark/Mark-CC, no-constraint, and W-only controls" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_codiet.sh" env \
      SEEDS="${SEEDS}" CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" INCLUDE_CLASSICAL_BASELINES=1 \
      STAGE=baseline DRY_RUN="${DRY_RUN}" \
      bash
fi

if [[ "${RUN_CODIET_CMI}" == "1" ]]; then
  run_stage "CoDiet: matched W+CMI arm" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_codiet.sh" env \
      SEEDS="${SEEDS}" CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      STAGE=discrete_cmi DRY_RUN="${DRY_RUN}" \
      bash
fi

echo "=== CE-new unified suite complete ==="
