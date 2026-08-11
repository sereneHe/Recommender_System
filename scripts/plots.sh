#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
PYTHON_EXEC="${PYTHON_EXEC:-${PROJECT_ROOT}/.venv/bin/python}"

PROBLEM_GROUPS="${PROBLEM_GROUPS:-FRED_16country_monthly}"
SOLVERS="${SOLVERS:-${solver:-hc_predictor_ce,hc_predictor,mark_with_cc,mark}}"
MAP_SOLVER="${MAP_SOLVER:-hc_predictor_ce_shielded_collider_limit}"
PROBLEM_TEMPLATE="${PROBLEM_TEMPLATE:-}"
if [ -z "${PROBLEM_TEMPLATE}" ]; then
  PROBLEM_TEMPLATE='${PROBLEM_GROUP}/industry_eu_fin'
fi

PROBLEM_GROUP="${PROBLEM_GROUPS%%,*}"
PROBLEM_GROUP=$(printf '%s\n' "${PROBLEM_GROUP}" | sed 's/^ *//;s/ *$//')
PROBLEM=$(printf '%s\n' "${PROBLEM_TEMPLATE}" | sed "s|\${PROBLEM_GROUP}|${PROBLEM_GROUP}|g")
case "${PROBLEM}" in
  */*) ;;
  *) PROBLEM="${PROBLEM_GROUP}/${PROBLEM}" ;;
esac
TARGET="${TARGET:-$(printf '%s\n' "${PROBLEM}" | sed 's/.*industry_eu_//' | tr '[:lower:]' '[:upper:]')}"
OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${PROJECT_ROOT}/reports/plots/${PROBLEM_GROUP}}"
REPORT_FAMILY="${PROBLEM_GROUP%%_*}"
REPORTS_ROOT="${REPORTS_ROOT:-${PROJECT_ROOT}/reports/${REPORT_FAMILY}}"
MULTIRUN_ROOT="${MULTIRUN_ROOT:-${PROJECT_ROOT}/multirun}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.05}"
TOP_K_EDGES="${TOP_K_EDGES:-20}"

cd "${PROJECT_ROOT}"

PROBLEM_GROUPS="${PROBLEM_GROUP}" SOLVERS="${SOLVERS}" PROBLEM="${PROBLEM}" TARGET="${TARGET}" PLOT_OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON_EXEC}" "plot_fin_test_mean.py"
PROBLEM_GROUPS="${PROBLEM_GROUP}" SOLVERS="${SOLVERS}" PROBLEM="${PROBLEM}" TARGET="${TARGET}" PLOT_OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON_EXEC}" "plot_test_mean_summary.py"
PROBLEM_GROUPS="${PROBLEM_GROUP}" SOLVERS="${SOLVERS}" PROBLEM="${PROBLEM}" TARGET="${TARGET}" PLOT_OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON_EXEC}" "plot_methods_heatmap.py"
"${PYTHON_EXEC}" "plot_map.py" --grid-report-adjacency --problem-group "${PROBLEM_GROUP}" --solver "${MAP_SOLVER}" --reports-root "${REPORTS_ROOT}" --multirun-root "${MULTIRUN_ROOT}" --output-root "${OUTPUT_DIR}" --group-output-name --edge-threshold "${EDGE_THRESHOLD}" --top-k-edges "${TOP_K_EDGES}"
