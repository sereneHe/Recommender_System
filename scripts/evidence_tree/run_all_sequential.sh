#!/usr/bin/env bash
# Sequential evidence run in ONE PBS job (no parallel cohorts).
#
#   1) CE priority continuation (same cohort as the earlier priority job)
#   2) batch1: A1 + A2 + A3 + A5 + B0 + G0   (synthetic ER)
#   3) batch2: A4 + A6                       (MILP + FRED)
#
# Submit (single job):
#   qsub -l walltime=24:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/evidence_tree/run_all_sequential.sh,MIP_TIME_LIMIT=300 \
#     cluster_computing/run_metacentrum.pbs
#
# Estimated total ~10-15 h.  Use walltime 24:00:00.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# batch2 (A4 MILP + A6 FRED) uses Gurobi; hold one slot around it so concurrent
# Gurobi jobs queue instead of being killed.
source "${SCRIPT_DIR}/gurobi_slot.sh"

# A cohort must have exactly ONE writer.  The CE-priority stage re-runs the
# NN reference arm, so reusing an earlier batch id would put a second reference
# run into that cohort; the evidence builder then (correctly) excludes the whole
# pairing as ``invalid_pairing:duplicate_within_cohort``.  Generate a fresh
# cohort id per submission instead (override with CE_BATCH_ID only for a
# deliberate, single-writer continuation).
CE_BATCH_ID="${CE_BATCH_ID:-priority_ce_$(date -u +%Y%m%dT%H%M%SZ)}"
MIP_TIME_LIMIT="${MIP_TIME_LIMIT:-300}"

echo "################################################################################"
echo "### 1/3  CE priority continuation   batch=${CE_BATCH_ID}"
echo "################################################################################"
EVIDENCE_BATCH_ID="${CE_BATCH_ID}" \
  bash "${PROJ_DIR}/scripts/evidence_tree_ce_priority.sh"

echo "################################################################################"
echo "### 2/3  batch1 synthetic (A1 A2 A3 A5 B0 G0)"
echo "################################################################################"
bash "${SCRIPT_DIR}/batch1_synthetic_light.sh"

echo "################################################################################"
echo "### 3/3  batch2 MILP/FRED (A4 A6)   MIP_TIME_LIMIT=${MIP_TIME_LIMIT}"
echo "################################################################################"
gurobi_slot_acquire
MIP_TIME_LIMIT="${MIP_TIME_LIMIT}" bash "${SCRIPT_DIR}/batch2_milp_fred.sh"
gurobi_slot_release

echo
echo "=== sequential evidence run complete ==="
