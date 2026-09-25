#!/usr/bin/env bash
# Task 1 / 3  —  synthetic ER core + graph prior
#
#   1) 01_synthetic_core.sh   A1 + A2 + A3 + A5 + B0 + G0
#   2) 02_graph_prior_er.sh   A4 (ExDBN / MILP)
#
# ``TASK1_MODE=full`` runs the complete pre-registered core plus A4.
# ``TASK1_MODE=resume`` runs only the experiments missing after the 2026-09-24
# A3 covariance compatibility failure: A3, A5, B0, G0, and A4.  It assigns
# fresh v3 cohort IDs and never duplicates the completed A1/A2 v2 cohorts.
# Resume is the default while the current evidence batch is being repaired.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/task1_synthetic.sh,MIP_TIME_LIMIT=300,TASK1_MODE=resume \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# Gurobi is token-limited: acquire one of GUROBI_MAX (default 2) slots before the
# MILP phase so extra tasks QUEUE instead of being killed.
source "${SCRIPT_DIR}/../evidence_tree/gurobi_slot.sh"

: "${GRAPH_SEEDS:=42 43 44}"
: "${NOISE_SEEDS:=101}"
: "${TIME_LIMIT:=120}"
: "${MIP_TIME_LIMIT:=300}"
: "${TASK1_MODE:=resume}"
: "${TASK1_RECOVERY_VERSION:=v3}"
export GRAPH_SEEDS NOISE_SEEDS TIME_LIMIT MIP_TIME_LIMIT

run_axis() {
  local script="$1" batch="$2"
  echo
  echo "--- ${script} (batch=${batch}) ---"
  EVIDENCE_BATCH_ID="${batch}" bash "${SCRIPT_DIR}/../evidence_tree/${script}"
}

run_g0() {
  local batch="$1"
  # The fair-comparison G0 audit needs the SYNCED evidence index.  Inside a
  # training job that index does not exist (results are still on scratch), so an
  # in-job audit can only fail with a false alarm.  G0 runs in the post-sync
  # refresh instead.  Set RUN_INJOB_G0=1 to force an informational in-job run.
  if [[ "${RUN_INJOB_G0:-0}" == "1" ]]; then
    echo "--- in-job G0 (INFORMATIONAL ONLY; no synced index in this job) ---"
    EVIDENCE_BATCH_ID="${batch}" bash "${SCRIPT_DIR}/../evidence_tree/G0_contract_audit.sh" \
      || echo "!!! in-job G0 failed; expected without synced results (see refresh G0)." >&2
    return 0
  fi
  echo "--- G0 fair-comparison audit deferred to the post-sync refresh ---"
  echo "    after results are mirrored back: bash scripts/refresh_progress_tree.sh"
}

echo "################################################################################"
echo "### TASK 1/3  synthetic core + graph prior"
echo "###   mode=${TASK1_MODE} graph_seeds=${GRAPH_SEEDS} noise_seeds=${NOISE_SEEDS} MIP_TIME_LIMIT=${MIP_TIME_LIMIT}"
echo "################################################################################"

case "${TASK1_MODE}" in
  full)
    echo
    echo "--- 1/2  01_synthetic_core.sh ---"
    EVIDENCE_VERSION="${TASK1_RECOVERY_VERSION}" RUN_G0=0 \
      bash "${SCRIPT_DIR}/01_synthetic_core.sh"

    echo
    echo "--- 2/2  02_graph_prior_er.sh ---"
    gurobi_slot_acquire
    EVIDENCE_BATCH_ID="et_a4_er_${TASK1_RECOVERY_VERSION}" \
      bash "${SCRIPT_DIR}/02_graph_prior_er.sh"
    gurobi_slot_release

    echo
    echo "--- final  G0 contract audit ---"
    run_g0 "et_g0_er_${TASK1_RECOVERY_VERSION}"
    ;;
  resume)
    echo
    echo "--- recovery 1/5  A3 CI/CE statistics ---"
    run_axis A3_ci_statistic.sh "et_a3_er_${TASK1_RECOVERY_VERSION}"

    echo
    echo "--- recovery 2/5  A5 capacity and training budget ---"
    run_axis A5_capacity_budget.sh "et_a5_er_${TASK1_RECOVERY_VERSION}"

    echo
    echo "--- recovery 3/5  B0 comparable baselines ---"
    run_axis B0_baselines.sh "et_b0_er_${TASK1_RECOVERY_VERSION}"

    echo
    echo "--- recovery 4/5  A4 graph priors / MILP ---"
    gurobi_slot_acquire
    EVIDENCE_BATCH_ID="et_a4_er_${TASK1_RECOVERY_VERSION}" \
      bash "${SCRIPT_DIR}/02_graph_prior_er.sh"
    gurobi_slot_release

    echo
    echo "--- recovery 5/5  G0 contract audit ---"
    run_g0 "et_g0_er_${TASK1_RECOVERY_VERSION}"
    ;;
  *)
    echo "TASK1_MODE must be 'full' or 'resume', got '${TASK1_MODE}'." >&2
    exit 2
    ;;
esac

echo
echo "=== TASK 1/3 complete ==="
