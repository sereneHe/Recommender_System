#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

PROBLEM_GROUPS="${PROBLEM_GROUPS:-FRED_16country_monthly}"
SOLVERS="${SOLVERS:-${solver:-hc_predictor_ce,hc_predictor,mark_with_cc}}"
MAP_SOLVER="${SOLVERS%%,*}"
MAP_SOLVER=$(printf '%s\n' "${MAP_SOLVER}" | sed 's/^ *//;s/ *$//')
if [ "${PROBLEM+x}" ]; then
  PROBLEM_TEMPLATE="${PROBLEM}"
elif [ "${problem+x}" ]; then
  PROBLEM_TEMPLATE="${problem}"
else
  PROBLEM_TEMPLATE='${PROBLEM_GROUP}/industry_eu_fin'
fi

if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"

PROBLEM_GROUP="${PROBLEM_GROUPS%%,*}"
PROBLEM_GROUP=$(printf '%s\n' "${PROBLEM_GROUP}" | sed 's/^ *//;s/ *$//')
PROBLEM=$(printf '%s\n' "${PROBLEM_TEMPLATE}" | sed "s|\${PROBLEM_GROUP}|${PROBLEM_GROUP}|g")
case "${PROBLEM}" in
  */*) ;;
  *) PROBLEM="${PROBLEM_GROUP}/${PROBLEM}" ;;
esac
TARGET="${TARGET:-$(printf '%s\n' "${PROBLEM}" | sed 's/.*industry_eu_//' | tr '[:lower:]' '[:upper:]')}"
OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${PROJECT_ROOT}/reports/plots/${PROBLEM_GROUP}}"
REPORTS_ROOT="${REPORTS_ROOT:-${PROJECT_ROOT}/reports}"
MULTIRUN_ROOT="${MULTIRUN_ROOT:-${PROJECT_ROOT}/multirun}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.05}"
TOP_K_EDGES="${TOP_K_EDGES:-20}"

echo "[PLOT] problem group: ${PROBLEM_GROUP}"
echo "[PLOT] solvers: ${SOLVERS}"
echo "[PLOT] map solver: ${MAP_SOLVER}"
echo "[PLOT] problem: ${PROBLEM}"
echo "[PLOT] output dir: ${OUTPUT_DIR}"

PROBLEM_GROUPS="${PROBLEM_GROUP}" SOLVERS="${SOLVERS}" PROBLEM="${PROBLEM}" TARGET="${TARGET}" PLOT_OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON}" "plot_fin_test_mean.py"
PROBLEM_GROUPS="${PROBLEM_GROUP}" SOLVERS="${SOLVERS}" PROBLEM="${PROBLEM}" TARGET="${TARGET}" PLOT_OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON}" "plot_test_mean_summary.py"
PROBLEM_GROUPS="${PROBLEM_GROUP}" SOLVERS="${SOLVERS}" PROBLEM="${PROBLEM}" TARGET="${TARGET}" PLOT_OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON}" "plot_methods_heatmap.py"
"${PYTHON}" "plot_map.py" --grid-report-adjacency --problem-group "${PROBLEM_GROUP}" --solver "${MAP_SOLVER}" --reports-root "${REPORTS_ROOT}" --multirun-root "${MULTIRUN_ROOT}" --output-root "${OUTPUT_DIR}" --group-output-name --edge-threshold "${EDGE_THRESHOLD}" --top-k-edges "${TOP_K_EDGES}"
