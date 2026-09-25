#!/usr/bin/env bash
# Evidence-tree test batch 2: ExDBN/MILP graph-prior experiments on ER.
#
# A4 contains 11 arms x 3 graph seeds = 33 MILP runs.  The default 300-second
# MIP limit keeps this shard near 7--10 hours; MIP_TIME_LIMIT=1800 is useful
# only as a separate long diagnostic and is not a 12-hour batch.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/02_graph_prior_er.sh,MIP_TIME_LIMIT=300 \
#     cluster_computing/run_metacentrum.pbs

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

: "${GRAPH_SEEDS:=42 43 44}"
: "${NOISE_SEEDS:=101}"
: "${MIP_TIME_LIMIT:=300}"
: "${EVIDENCE_BATCH_ID:=et_a4_er_v2}"

GRAPH_SEEDS="${GRAPH_SEEDS}" \
NOISE_SEEDS="${NOISE_SEEDS}" \
MIP_TIME_LIMIT="${MIP_TIME_LIMIT}" \
EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID}" \
  bash "${ROOT}/scripts/evidence_tree/A4_graph_prior.sh"

echo "=== ER graph-prior evidence batch complete (MIP_TIME_LIMIT=${MIP_TIME_LIMIT}) ==="
