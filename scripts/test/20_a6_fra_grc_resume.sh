#!/usr/bin/env bash
# Task 20 — resume A6 FRED for FRA/GRC only (task2 / job 23972039), filling
# EXACTLY the missing (country, arm, seed) units without recomputing the rest.
#
# Missing set (verified from 23972039): FRA fred_ref seed44 (partial) plus all
# four variation arms (lag/trend/regime/huber) x seeds 42/43/44 for FRA and GRC;
# GRC fred_ref seed44.  Completed FRA/GRC fred_ref 42/43 and the first six
# countries are NOT rerun.
#
# Reuses EVIDENCE_BATCH_ID=et_a6_fred_v2 so the filled units join the SAME
# cohort (no duplicate keys are produced, so pairing stays clean).
#
# Gurobi: acquires one of GUROBI_MAX=2 slots and waits if the license is busy,
# so extra tasks QUEUE instead of being killed.
#
# Submit:
#   qsub -l walltime=08:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/20_a6_fra_grc_resume.sh \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"

source "${ROOT}/scripts/evidence_tree/gurobi_slot.sh"

A6="${ROOT}/scripts/evidence_tree/A6_data_time_robustness.sh"
export EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-et_a6_fred_v2}"
export TIME_LIMIT="${TIME_LIMIT:-120}"
export N_OUTER="${N_OUTER:-10}"
export N_INNER="${N_INNER:-100}"

FRA="FRED_16country_monthly/industry_eu_fra"
GRC="FRED_16country_monthly/industry_eu_grc"
VARIATION="lag,trend,regime,huber"

run_a6() {  # <country-problem> <arms> <graph-seeds>
  local problem="$1" arms="$2" seeds="$3"
  echo "--- A6 resume: problem=${problem##*/} arms=${arms} seeds=${seeds}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    DRY_RUN: would run A6 ARMS=${arms} GRAPH_SEEDS='${seeds}' PROBLEMS=${problem}"
    return 0
  fi
  PROBLEMS="${problem}" A6_ARMS="${arms}" GRAPH_SEEDS="${seeds}" NOISE_SEEDS="${NOISE_SEEDS:-101}" \
    bash "${A6}"
}

echo "### A6 FRA/GRC resume  batch=${EVIDENCE_BATCH_ID}"
if [[ "${DRY_RUN}" != "1" ]]; then
  gurobi_slot_acquire
fi

# FRA: ref seed44 (partial), then the four variation arms for all seeds
run_a6 "${FRA}" "ref" "44"
run_a6 "${FRA}" "${VARIATION}" "42 43 44"
# GRC: ref seed44, then the four variation arms for all seeds
run_a6 "${GRC}" "ref" "44"
run_a6 "${GRC}" "${VARIATION}" "42 43 44"

[[ "${DRY_RUN}" == "1" ]] || gurobi_slot_release
echo "=== A6 FRA/GRC resume complete: batch=${EVIDENCE_BATCH_ID} ==="
