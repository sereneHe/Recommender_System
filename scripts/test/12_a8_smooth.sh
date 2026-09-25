#!/usr/bin/env bash
# Task 12 — A8.1 smooth nonlinear ER (tanh/sin/softplus).
#
# Stage-driven (A8_STAGE=smoke|pilot|confirm|locked); see scripts/test/_a8_stage.sh.
#   smoke   : graph 42            x noise 101        (5 arms)
#   pilot   : graph 42-44         x noise 101-102    (nn0, nn_ce_true, xgb100)
#   confirm : graph 52-61         x noise 101-102    (new; nn0, nn_ce_true, xgb100)
#   locked  : graph 1001-1010     x noise 501-502    (F0 reserved)
#
# Submit (example: L1 smoke):
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/12_a8_smooth.sh,A8_STAGE=smoke \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
A8_MECHANISM="smooth_additive"; export A8_MECHANISM
source "${ROOT}/scripts/test/_a8_stage.sh"
BATCH="${EVIDENCE_BATCH_ID:-et_a8_smooth_${A8_STAGE}_v1}"

echo "### A8.1 smooth_additive  stage=${A8_STAGE}  graph=${GRAPH_SEEDS}  noise=${NOISE_SEEDS}  arms=${A8_ARMS}  batch=${BATCH}"
MECHANISM="${A8_MECHANISM}" A8_STAGE="${A8_STAGE}" A8_ARMS="${A8_ARMS}" \
  GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" EVIDENCE_BATCH_ID="${BATCH}" DRY_RUN="${DRY_RUN}" \
  bash scripts/evidence_tree/A8_nn_favorable_synthetic.sh
