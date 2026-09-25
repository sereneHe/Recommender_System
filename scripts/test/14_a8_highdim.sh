#!/usr/bin/env bash
# Task 14 — A8.3 high-dimensional smooth SEM (p=50, tanh + interactions).
# Report runtime/capacity for this mechanism separately from the p=20 ones.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
A8_MECHANISM="highdim_smooth"; export A8_MECHANISM
source "${ROOT}/scripts/test/_a8_stage.sh"
BATCH="${EVIDENCE_BATCH_ID:-et_a8_highdim_${A8_STAGE}_v1}"

echo "### A8.3 highdim_smooth  stage=${A8_STAGE}  graph=${GRAPH_SEEDS}  noise=${NOISE_SEEDS}  arms=${A8_ARMS}  batch=${BATCH}"
MECHANISM="${A8_MECHANISM}" A8_STAGE="${A8_STAGE}" A8_ARMS="${A8_ARMS}" \
  GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" EVIDENCE_BATCH_ID="${BATCH}" DRY_RUN="${DRY_RUN}" \
  bash scripts/evidence_tree/A8_nn_favorable_synthetic.sh
