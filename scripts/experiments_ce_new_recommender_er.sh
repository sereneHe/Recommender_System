#!/usr/bin/env bash
# CE-Lite recommender screen: synthetic Erdos-Renyi (ER) mechanism check.
#
# This is a thin wrapper over the phase-1 synthetic constraint screen.  It
# keeps a single source of truth for the arms (S0..S4) while pinning the graph
# family to ER so it can be launched on its own.
#
# Usage:
#   bash scripts/experiments_ce_new_recommender_er.sh
#   DRY_RUN=1 bash scripts/experiments_ce_new_recommender_er.sh
#   SEEDS="42 43 44" NOISE_SEEDS="101 102" \
#     bash scripts/experiments_ce_new_recommender_er.sh
#
# Recognised environment: SEEDS/GRAPH_SEEDS, NOISE_SEEDS, TIME_LIMIT, N_RUNS,
# N_OUTER, N_INNER, N_SAMPLES, N_NODES, EXPECTED_EDGES, INCLUDE_DEPENDENT,
# EXPERIMENT_PREFIX, DRY_RUN, plus trailing Hydra overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="${SCRIPT_DIR}/causal_predictor_plan"

export PROBLEMS="${PROBLEMS:-synthetic_er}"
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-CE_NEW_ER}"
export HC_WEIBULL_GAUSSIANIZE=0

echo "=== CE-Lite ER screen ==="
echo "problems=${PROBLEMS} experiment_prefix=${EXPERIMENT_PREFIX}"
exec bash "${PLAN_DIR}/01_synthetic_constraint_screen.sh" "$@"
