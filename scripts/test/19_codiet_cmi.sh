#!/usr/bin/env bash
# Task 19 — CoDiet baseline + type-aware discrete-CMI (kept separate).
#
# CoDiet must be reported on its own: its results are never pooled into the
# continuous ER / SF / FRED averages, and it does not supply evidence for the
# synthetic-ER A3.discrete_CMI node (see README gap note).
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/19_codiet_cmi.sh \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
BATCH="${EVIDENCE_BATCH_ID:-et_c0_codiet_v1}"
: "${SEEDS:=42 43 44}"

echo "### CoDiet baseline + discrete-CMI  batch=${BATCH}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: would run C0_codiet_preaudit.sh (baseline + discrete_cmi)"
  exit 0
fi
EVIDENCE_BATCH_ID="${BATCH}" SEEDS="${SEEDS}" \
  bash scripts/evidence_tree/C0_codiet_preaudit.sh

echo "[SCOPE] report CoDiet separately; do not pool with ER/FRED."
