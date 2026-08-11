#!/bin/sh

set -e

ROOT="/Users/xiaoyuhe/Recommender_Pavel"
PYTHON_EXEC="${ROOT}/.venv/bin/python"

# PROBLEM_GROUPS="FRED_9country_monthly,FRED_9country_quarterly,OECD_9country_monthly,OECD_9country_quarterly,FRED_16country_monthly,FRED_16country_quarterly,OECD_16country_monthly,OECD_16country_quarterly"
# SOLVERS="hc_predictor,hc_predictor_ci,hc_predictor_ce"
PROBLEM_GROUPS="${PROBLEM_GROUPS:-FRED_16country_monthly}"
SOLVERS="${SOLVERS:-hc_predictor,hc_predictor_ci,hc_predictor_ce}"

REPORTS_ROOT="${REPORTS_ROOT:-${ROOT}/reports}"
MULTIRUN_ROOT="${MULTIRUN_ROOT:-${ROOT}/multirun}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.1}"
TOP_K_EDGES="${TOP_K_EDGES:-10}"
FILTER_EDGES="${FILTER_EDGES:-1}"

FILTER_EDGES_ARG="--filter-edges"
case "${FILTER_EDGES}" in
  0|false|FALSE|no|NO) FILTER_EDGES_ARG="--no-filter-edges" ;;
esac

cd "${ROOT}"

"${PYTHON_EXEC}" plot_map.py \
  --grid-report-adjacency \
  --problem-group "${PROBLEM_GROUPS}" \
  --solver "${SOLVERS}" \
  --reports-root "${REPORTS_ROOT}" \
  --multirun-root "${MULTIRUN_ROOT}" \
  ${FILTER_EDGES_ARG} \
  --edge-threshold "${EDGE_THRESHOLD}" \
  --top-k-edges "${TOP_K_EDGES}"
