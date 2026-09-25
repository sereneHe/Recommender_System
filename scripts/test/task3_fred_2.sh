#!/usr/bin/env bash
# Task 3 / 3  —  FRED part 2 (A6 data/time/robustness, last 8 countries)
#
#   07_fred_irl_ita.sh   IRL, ITA
#   08_fred_ltu_lux.sh   LTU, LUX
#   09_fred_nld_prt.sh   NLD, PRT
#   10_fred_svk_svn.sh   SVK, SVN
#
# Runs the four shards SEQUENTIALLY in one PBS job.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/task3_fred_2.sh \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# A6/FRED uses MILP (Gurobi) on every shard: hold ONE Gurobi slot for the whole
# job so concurrent Gurobi tasks queue instead of being killed by the license.
source "${SCRIPT_DIR}/../evidence_tree/gurobi_slot.sh"

: "${GRAPH_SEEDS:=42 43 44}"
: "${NOISE_SEEDS:=101}"
: "${FRED_TIME_LIMIT:=120}"
export GRAPH_SEEDS NOISE_SEEDS FRED_TIME_LIMIT

SHARDS=(
  07_fred_irl_ita.sh
  08_fred_ltu_lux.sh
  09_fred_nld_prt.sh
  10_fred_svk_svn.sh
)

echo "################################################################################"
echo "### TASK 3/3  FRED part 2 (${#SHARDS[@]} shards)"
echo "###   graph_seeds=${GRAPH_SEEDS}  noise_seeds=${NOISE_SEEDS}  FRED_TIME_LIMIT=${FRED_TIME_LIMIT}"
echo "################################################################################"

i=0
gurobi_slot_acquire
for s in "${SHARDS[@]}"; do
  i=$((i + 1))
  echo
  echo "--- ${i}/${#SHARDS[@]}  ${s} ---"
  bash "${SCRIPT_DIR}/${s}"
done
gurobi_slot_release

echo
echo "=== TASK 3/3 complete ==="
