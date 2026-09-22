#!/usr/bin/env bash
# CE-Lite recommender screen: synthetic scale-free (SF) mechanism check.
#
# Thin wrapper over the phase-1 synthetic constraint screen, pinned to the SF
# graph family.  See the ER wrapper for the shared environment knobs.
#
# Usage:
#   bash scripts/experiments_ce_new_recommender_sf.sh
#   DRY_RUN=1 bash scripts/experiments_ce_new_recommender_sf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="${SCRIPT_DIR}/causal_predictor_plan"

export PROBLEMS="${PROBLEMS:-synthetic_sf}"
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-CE_NEW_SF}"
export HC_WEIBULL_GAUSSIANIZE=0

echo "=== CE-Lite SF screen ==="
echo "problems=${PROBLEMS} experiment_prefix=${EXPERIMENT_PREFIX}"
exec bash "${PLAN_DIR}/01_synthetic_constraint_screen.sh" "$@"
