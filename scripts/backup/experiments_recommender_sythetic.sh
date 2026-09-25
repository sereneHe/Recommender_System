#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE="${HC_WEIBULL_GAUSSIANIZE:-0}"
export HC_WEIBULL_MODE="${HC_WEIBULL_MODE:-rand}"

source "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CONFIG_NAME="${CONFIG_NAME:-config}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-SYNTHETIC_RECOMMENDER_ER_SF}"
SOLVER="${SOLVER:-hc_predictor}"
PROBLEMS="${PROBLEMS:-synthetic_er,synthetic_sf}"
SYNTHETIC_RECALCULATE_DAG="${SYNTHETIC_RECALCULATE_DAG:-false}"

case "${HC_WEIBULL_GAUSSIANIZE}" in
  1|true|TRUE|yes|YES|on|ON)
    # Gaussianization changes the linear coefficients, so learn W in the
    # transformed space instead of reusing the raw synthetic ground truth.
    SYNTHETIC_RECALCULATE_DAG=true
    ;;
esac

echo "Synthetic ER/SF: solver=${SOLVER}, backend=${HC_CONSTRAINT_BACKEND}, weibull=${HC_WEIBULL_GAUSSIANIZE}/${HC_WEIBULL_MODE}, recalculate_dag=${SYNTHETIC_RECALCULATE_DAG}"

EXTRA_OVERRIDES=("solver.recalculate_dag=${SYNTHETIC_RECALCULATE_DAG}")
if [[ "${SYNTHETIC_RECALCULATE_DAG}" == "true" ]]; then
  EXTRA_OVERRIDES+=("solver.knowledge_graph_filename=null")
fi

"${PYTHON_BIN}" run_experiments.py \
  --multirun \
  --config-name="${CONFIG_NAME}" \
  experiment="${EXPERIMENT_NAME}" \
  solver="${SOLVER}" \
  problem="${PROBLEMS}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
