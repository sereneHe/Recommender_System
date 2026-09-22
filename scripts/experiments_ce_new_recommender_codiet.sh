#!/usr/bin/env bash
# CE-Lite recommender: CoDiet raw baseline vs discrete-CMI constraint arm.
#
# Thin wrapper over the phase-6 CoDiet type screen so the same seeds and
# training budget are used for both arms, making the comparison paired.
#
# Usage:
#   STAGE=baseline      bash scripts/experiments_ce_new_recommender_codiet.sh
#   STAGE=discrete_cmi  bash scripts/experiments_ce_new_recommender_codiet.sh
#   DRY_RUN=1 STAGE=discrete_cmi \
#     bash scripts/experiments_ce_new_recommender_codiet.sh
#
# STAGE=baseline runs the raw predictive baselines (Mark and the unconstrained
# HC-CE network).  STAGE=discrete_cmi runs the quantile-binned discrete-CMI
# constraint arm.  Recognised environment: PROBLEMS, SEEDS, TIME_LIMIT, N_RUNS,
# LR, WEIGHT_DECAY, HIDDEN_DIM, DEPTH, N_OUTER, N_INNER, EXPERIMENT_PREFIX,
# DRY_RUN, plus trailing Hydra overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIR="${SCRIPT_DIR}/causal_predictor_plan"

STAGE="${STAGE:-baseline}"
case "${STAGE}" in
  baseline|discrete_cmi) ;;
  *) echo "ERROR: STAGE must be 'baseline' or 'discrete_cmi', got '${STAGE}'." >&2; exit 2 ;;
esac

export STAGE
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-CE_NEW_CODIET}"
export HC_WEIBULL_GAUSSIANIZE=0

echo "=== CE-Lite CoDiet screen ==="
echo "stage=${STAGE} experiment_prefix=${EXPERIMENT_PREFIX}"
exec bash "${PLAN_DIR}/06_codiet_type_screen.sh" "$@"
