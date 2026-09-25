#!/usr/bin/env bash
# CoDiet type-aware CE pre-audit.  Keeps CoDiet separate from continuous ER,
# SF and FRED: only the discrete/CMI branch is eligible for a CoDiet claim.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/python_runtime.sh
project_python_require "${ROOT}"
DATASET_SCOPE="${DATASET_SCOPE:-pilot}"
DATASET_GROUPS=codiet
SEEDS="${SEEDS:-42 43 44}"
CODIET_PILOT_PROBLEMS="${CODIET_PILOT_PROBLEMS:-codiet,codiet_glu,codiet_hdl}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-ET_C0_CODIET}"
EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-et_c0_codiet_v1}"
export DATASET_SCOPE DATASET_GROUPS SEEDS CODIET_PILOT_PROBLEMS EXPERIMENT_PREFIX EVIDENCE_BATCH_ID

echo "[C0] CoDiet baseline + type-aware discrete-CMI pre-audit"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: would run scripts/causal_predictor_plan/09_ce_preaudit.sh for ${CODIET_PILOT_PROBLEMS}"
  exit 0
fi
bash scripts/causal_predictor_plan/09_ce_preaudit.sh
echo "[C0] summarize the generated ce_preaudit_gate.csv separately; no Gaussian CE claim is made."
