#!/usr/bin/env bash
# Task 13 — A8.2 deep compositional ER (layered tanh).  See 12 for stage policy.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
A8_MECHANISM="compositional"; export A8_MECHANISM
source "${ROOT}/scripts/test/_a8_stage.sh"
BATCH="${EVIDENCE_BATCH_ID:-et_a8_compositional_${A8_STAGE}_v1}"

echo "### A8.2 compositional  stage=${A8_STAGE}  graph=${GRAPH_SEEDS}  noise=${NOISE_SEEDS}  arms=${A8_ARMS}  batch=${BATCH}"
MECHANISM="${A8_MECHANISM}" A8_STAGE="${A8_STAGE}" A8_ARMS="${A8_ARMS}" \
  GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" EVIDENCE_BATCH_ID="${BATCH}" DRY_RUN="${DRY_RUN}" \
  bash scripts/evidence_tree/A8_nn_favorable_synthetic.sh
