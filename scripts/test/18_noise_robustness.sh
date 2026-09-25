#!/usr/bin/env bash
# Task 18 — noise robustness (Laplace / Student-t), synthetic ER + SF.
#
# Only the supported noise families are runnable here.  Count / zero-inflated
# SEM and a LiNGAM-style higher-moment prior remain blocked by the generator and
# are reported as NOT implemented rather than filled in from other logs.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/18_noise_robustness.sh \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
: "${GRAPH_SEEDS:=42 43 44}"
BATCH="${EVIDENCE_BATCH_ID:-et_noise_robustness_v1}"

echo "### noise robustness (Laplace/Student-t)  batch=${BATCH}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: would run causal_predictor_plan/07_synthetic_noise_robustness.sh"
  exit 0
fi
EVIDENCE_BATCH_ID="${BATCH}" SEEDS="${GRAPH_SEEDS}" PROBLEMS=synthetic_er,synthetic_sf \
  bash scripts/causal_predictor_plan/07_synthetic_noise_robustness.sh

echo "[NOT IMPLEMENTED] A4.non_gaussian_moments: no higher-moment ExDBN prior."
