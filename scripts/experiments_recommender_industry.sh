#!/bin/sh

set -e

export PYTHONPATH=$PYTHONPATH:../
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH}"

PROBLEM_GROUP="${PROBLEM_GROUP:-FRED_16country_quarterly}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-recommender_industry}"
SOLVER_NAME="${SOLVER_NAME:-hc_predictor}"
SOLVER_DATA_PATH="${SOLVER_DATA_PATH:-}"
SOLVER_KNOWLEDGE_GRAPH_FILENAME="${SOLVER_KNOWLEDGE_GRAPH_FILENAME:-}"
PYTHON_EXEC="${PYTHON_EXEC:-./.venv/bin/python3}"
if [ ! -x "${PYTHON_EXEC}" ]; then
  PYTHON_EXEC="python3"
fi

PROBLEM_LIST=$(
${PYTHON_EXEC} - <<PY
from pathlib import Path

group = "${PROBLEM_GROUP}"
root = Path("experiments_conf/problem") / group
files = sorted(root.glob("industry_eu_*.yaml"))
if not files:
    raise SystemExit(f"No industry problem YAML files found in {root}")
print(",".join(f"{group}/{path.stem}" for path in files))
PY
)

CMD="${PYTHON_EXEC} run_experiments.py --multirun --config-name=config_industry"
SOLVER_OVERRIDES=""
if [ -n "${SOLVER_DATA_PATH}" ]; then
SOLVER_OVERRIDES="${SOLVER_OVERRIDES} solver.data_path='${SOLVER_DATA_PATH}'"
fi
if [ -n "${SOLVER_KNOWLEDGE_GRAPH_FILENAME}" ]; then
SOLVER_OVERRIDES="${SOLVER_OVERRIDES} solver.knowledge_graph_filename='${SOLVER_KNOWLEDGE_GRAPH_FILENAME}'"
fi

${CMD} experiment="${EXPERIMENT_NAME}" solver="${SOLVER_NAME}"${SOLVER_OVERRIDES} problem="${PROBLEM_LIST}"
