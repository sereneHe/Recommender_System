#!/usr/bin/env bash
# Batch for the evidence-tree items that were outside the previous test waves.
# It deliberately separates runnable audits from two still-unimplemented
# methods (bootstrap stability selection and a non-Gaussian-moment prior).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
RUN_CI_AUDIT="${RUN_CI_AUDIT:-1}"
RUN_SF="${RUN_SF:-1}"
RUN_NONLINEAR_ER="${RUN_NONLINEAR_ER:-1}"
RUN_NON_GAUSSIAN="${RUN_NON_GAUSSIAN:-1}"
RUN_CODIET="${RUN_CODIET:-1}"
SEEDS="${SEEDS:-42 43 44}"
GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"

run_or_print() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if [[ "${RUN_CI_AUDIT}" == "1" ]]; then
  run_or_print env GRAPH_TYPES="ER SF" GRAPH_SEEDS="${GRAPH_SEEDS}" \
    NOISE_SEEDS="${NOISE_SEEDS}" OUTPUT_DIR="${OUTPUT_DIR:-reports/evidence_ci/a3_${EVIDENCE_BATCH_ID:-local}}" \
    bash scripts/evidence_tree/A3_nonlinear_ci_audit.sh
fi

if [[ "${RUN_SF}" == "1" ]]; then
  run_or_print env SEEDS="${GRAPH_SEEDS}" GRAPH_SEEDS="${GRAPH_SEEDS}" \
    NOISE_SEEDS="${NOISE_SEEDS}" RUN_SF_E0=0 RUN_SF_E1=0 RUN_SF_E2=0 RUN_SF_E3=1 \
    INCLUDE_BASELINES=1 bash scripts/experiments_ce_new_recommender_sf.sh
fi

if [[ "${RUN_NONLINEAR_ER}" == "1" ]]; then
  run_or_print env GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" \
    INCLUDE_MAIN=0 INCLUDE_BASELINES=0 INCLUDE_NONLINEAR_BASELINES=1 \
    bash scripts/experiments_ce_new_recommender_er.sh
fi

if [[ "${RUN_NON_GAUSSIAN}" == "1" ]]; then
  # Supported Laplace/Student-t stress tests are runnable.  Count/ZI SEM and a
  # LiNGAM-style higher-moment prior remain blocked by the data generator.
  run_or_print env SEEDS="${GRAPH_SEEDS}" PROBLEMS=synthetic_er,synthetic_sf \
    bash scripts/causal_predictor_plan/07_synthetic_noise_robustness.sh
  echo "[BLOCKED] A4.non_gaussian_moments: no implemented higher-moment ExDBN prior."
fi

if [[ "${RUN_CODIET}" == "1" ]]; then
  run_or_print env SEEDS="${SEEDS}" STAGE=baseline \
    bash scripts/experiments_ce_new_recommender_codiet.sh
  run_or_print env SEEDS="${SEEDS}" STAGE=discrete_cmi \
    bash scripts/experiments_ce_new_recommender_codiet.sh
fi

echo "[BLOCKED] A4.stability_selection: classical B-bootstrap edge selection is not the existing BD approximation."
echo "[NEXT] After these runs, refresh with scripts/refresh_progress_tree.sh; do not call the blocked nodes verified."
