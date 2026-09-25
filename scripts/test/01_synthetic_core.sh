#!/usr/bin/env bash
# Evidence-tree test batch 1: all non-MILP synthetic ER experiments.
#
# Covered axes: A1 (optimizer), A2 (constraint embedding), A3 (CI/CE
# statistic), A5 (capacity/budget), B0 (canonical baselines), and the cheap
# G0 contract audit.  All axes use the same synthetic ER data contract and the
# same graph/noise seed lists.  The batch IDs are deliberately versioned so a
# rerun cannot be merged with older evidence.
#
# Expected full workload: 90 training commands (15+21+15+15+24), usually
# about 2--6 hours on one PBS allocation.  It is a safe 12-hour shard.
#
# The task1 recovery mode invokes only the axes that had not run when A3
# failed on 2026-09-24.  It deliberately does not rerun the already-complete
# A1/A2 cohorts, so completed evidence is not duplicated under one cohort id.
#
# Submit on MetaCentrum:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/01_synthetic_core.sh \
#     cluster_computing/run_metacentrum.pbs

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
TREE="${ROOT}/scripts/evidence_tree"

: "${GRAPH_SEEDS:=42 43 44}"
: "${NOISE_SEEDS:=101}"
: "${TIME_LIMIT:=120}"
: "${EVIDENCE_VERSION:=v2}"

run_axis() {
  local script="$1" batch="$2"
  echo
  echo "################################################################################"
  echo "### ${script}  (batch=${batch})"
  echo "################################################################################"
  GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" TIME_LIMIT="${TIME_LIMIT}" \
    EVIDENCE_BATCH_ID="${batch}" bash "${TREE}/${script}"
}

run_axis A1_optimizer.sh            "et_a1_er_${EVIDENCE_VERSION}"
run_axis A2_constraint_embedding.sh "et_a2_er_${EVIDENCE_VERSION}"
run_axis A3_ci_statistic.sh         "et_a3_er_${EVIDENCE_VERSION}"
run_axis A5_capacity_budget.sh      "et_a5_er_${EVIDENCE_VERSION}"
run_axis B0_baselines.sh            "et_b0_er_${EVIDENCE_VERSION}"

# G0 is an audit of the SYNCED evidence index, not of scratch output.  In-job it
# has no mirror to read and always false-fails, so it is DEFERRED by default and
# evaluated by the post-sync refresh.  Set RUN_G0=1 to force an in-job run.
if [[ "${RUN_G0:-0}" == "1" ]]; then
  G0_STATUS="ok"
  EVIDENCE_BATCH_ID="et_g0_er_${EVIDENCE_VERSION}" bash "${TREE}/G0_contract_audit.sh" || G0_STATUS="FAILED"
  if [[ "${G0_STATUS}" != "ok" ]]; then
    echo
    echo "!!! in-job G0 did not pass (expected without synced results; refresh G0 is authoritative) !!!" >&2
  fi
else
  G0_STATUS="deferred_to_refresh"
  echo "G0 audit deferred to the post-sync refresh (bash scripts/refresh_progress_tree.sh)."
fi

echo "=== synthetic core evidence batch complete (G0=${G0_STATUS}) ==="
