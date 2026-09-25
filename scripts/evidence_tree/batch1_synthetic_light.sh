#!/usr/bin/env bash
# Evidence-tree batch 1 (light, synthetic): A1 + A2 + A3 + A5 + B0 + G0.
#
# Expected ~5-6 h at the default 3 graph seeds; use walltime 12:00:00.  Each
# axis keeps its own cohort id (et_<axis>_er_v1) so the Evidence Builder groups
# arms correctly.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/evidence_tree/batch1_synthetic_light.sh \
#     cluster_computing/run_metacentrum.pbs
# Optional: pass SEEDS via -v to enlarge the cohort (time scales linearly).

set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

EXTRA=("$@")
if (( $# == 0 )); then
  EXTRA=()
fi

run_axis() {
  local script="$1"
  local batch="$2"
  shift 2
  echo
  echo "################################################################################"
  echo "### ${script}  (batch=${batch})"
  echo "################################################################################"
  EVIDENCE_BATCH_ID="${batch}" bash "${SCRIPT_DIR}/${script}" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
}

run_axis A1_optimizer.sh                 et_a1_er_v1
run_axis A2_constraint_embedding.sh      et_a2_er_v1
run_axis A3_ci_statistic.sh              et_a3_er_v1
run_axis A5_capacity_budget.sh           et_a5_er_v1
run_axis B0_baselines.sh                 et_b0_er_v1

# G0 audits the SYNCED evidence index; inside a training job there is no mirror,
# so in-job G0 false-fails.  Defer to the post-sync refresh by default; set
# RUN_G0=1 for an informational in-job run.
echo
if [[ "${RUN_G0:-0}" == "1" ]]; then
  echo "### in-job G0 contract audit (INFORMATIONAL; no synced index here) ###"
  G0_STATUS="ok"
  EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-et_g0_v1}" bash "${SCRIPT_DIR}/G0_contract_audit.sh" || G0_STATUS="FAILED"
  [[ "${G0_STATUS}" == "ok" ]] || echo "!!! in-job G0 did not pass (expected without synced results) !!!" >&2
else
  G0_STATUS="deferred_to_refresh"
  echo "### G0 deferred to post-sync refresh: bash scripts/refresh_progress_tree.sh ###"
fi

echo
if [[ "${G0_STATUS}" == "ok" ]]; then
  echo "=== evidence batch 1 complete (in-job G0 ok) ==="
elif [[ "${G0_STATUS}" == "deferred_to_refresh" ]]; then
  echo "=== evidence batch 1 complete (G0 deferred to refresh) ==="
else
  echo "=== evidence batch 1 finished but in-job G0 FAILED ==="
fi
