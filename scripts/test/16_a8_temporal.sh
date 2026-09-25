#!/usr/bin/env bash
# Task 16 — A8.5 smooth temporal SEM (lag-aware dynamic mechanism).
#
# CE is NOT tested here: without a lag-aware oracle, nn_ce_true / nn_w_true /
# nn_w_ce_true cannot be interpreted.  A8_ARMS is forced to "nn0,xgb100" (data +
# prediction smoke only).  The tree keeps this node as
# implementation=implemented / oracle_audit=not_audited / claim=unverified.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
A8_MECHANISM="temporal_smooth"; export A8_MECHANISM
source "${ROOT}/scripts/test/_a8_stage.sh"   # forces A8_ARMS=nn0,xgb100 for temporal
BATCH="${EVIDENCE_BATCH_ID:-et_a8_temporal_${A8_STAGE}_v1}"

echo "### A8.5 temporal_smooth  stage=${A8_STAGE}  graph=${GRAPH_SEEDS}  noise=${NOISE_SEEDS}  arms=${A8_ARMS}  batch=${BATCH}"
echo "### NOTE: CE arms disabled (no lag-aware oracle); prediction-only evidence."
MECHANISM="${A8_MECHANISM}" A8_STAGE="${A8_STAGE}" A8_ARMS="${A8_ARMS}" \
  GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" EVIDENCE_BATCH_ID="${BATCH}" DRY_RUN="${DRY_RUN}" \
  bash scripts/evidence_tree/A8_nn_favorable_synthetic.sh
