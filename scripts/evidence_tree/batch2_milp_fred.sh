#!/usr/bin/env bash
# Evidence-tree batch 2 (MILP / FRED): A4 graph prior + A6 data/time robustness.
#
# Run count at the default 3 graph seeds:
#   A4 = 11 configs x 3 = 33 runs
#   A6 =  5 configs x 3 = 15 runs
#   total = 48 runs   (NOT 51)
#
# A4 is MILP-bound.  Bound the long arm with MIP_TIME_LIMIT (e.g. 300) so the
# walltime estimate is real; otherwise it defaults to 1800 s per fold.
# Expected ~7-10 h at 3 seeds; use walltime 15:00:00.
#
# Submit:
#   qsub -l walltime=15:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/evidence_tree/batch2_milp_fred.sh,MIP_TIME_LIMIT=300 \
#     cluster_computing/run_metacentrum.pbs

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

run_axis A4_graph_prior.sh              et_a4_er_v1
run_axis A6_data_time_robustness.sh     et_a6_fred_v1

echo
echo "=== evidence batch 2 complete ==="
