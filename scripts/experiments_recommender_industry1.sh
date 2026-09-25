#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"
export HC_WEIBULL_GAUSSIANIZE="${HC_WEIBULL_GAUSSIANIZE:-0}"
export HC_WEIBULL_MODE="${HC_WEIBULL_MODE:-rand}"

source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CONFIG_NAME="${CONFIG_NAME:-config}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-INDUSTRY_RECOMMENDER_FRED_16COUNTRY_MONTHLY}"
SOLVER="${SOLVER:-hc_predictor}"
PROBLEM_GROUP="${PROBLEM_GROUP:-FRED_16country_monthly}"
PROBLEMS="${PROBLEMS:-${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,${PROBLEM_GROUP}/industry_eu_deu,${PROBLEM_GROUP}/industry_eu_esp,${PROBLEM_GROUP}/industry_eu_est,${PROBLEM_GROUP}/industry_eu_fin,${PROBLEM_GROUP}/industry_eu_fra,${PROBLEM_GROUP}/industry_eu_grc,${PROBLEM_GROUP}/industry_eu_irl,${PROBLEM_GROUP}/industry_eu_ita,${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn}"

echo "FRED 16-country monthly: solver=${SOLVER}, backend=${HC_CONSTRAINT_BACKEND}, weibull=${HC_WEIBULL_GAUSSIANIZE}/${HC_WEIBULL_MODE}"

"${PYTHON_BIN}" run_experiments.py \
  --multirun \
  --config-name="${CONFIG_NAME}" \
  experiment="${EXPERIMENT_NAME}" \
  solver="${SOLVER}" \
  problem="${PROBLEMS}" \
  "$@"
