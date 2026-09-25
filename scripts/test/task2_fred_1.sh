#!/usr/bin/env bash
# Task 2 / 3  —  FRED part 1 (A6 data/time/robustness, first 8 countries)
#
#   03_fred_aut_bel.sh   AUT, BEL
#   04_fred_deu_esp.sh   DEU, ESP
#   05_fred_est_fin.sh   EST, FIN
#   06_fred_fra_grc.sh   FRA, GRC
#
# Runs the four shards SEQUENTIALLY in one PBS job.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/task2_fred_1.sh \
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
  03_fred_aut_bel.sh
  04_fred_deu_esp.sh
  05_fred_est_fin.sh
  06_fred_fra_grc.sh
)

echo "################################################################################"
echo "### TASK 2/3  FRED part 1 (${#SHARDS[@]} shards)"
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
echo "=== TASK 2/3 complete ==="
