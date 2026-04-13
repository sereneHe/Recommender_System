#!/bin/sh

set -e

export PYTHONPATH=$PYTHONPATH:../
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH}"

PROBLEM_GROUP="${PROBLEM_GROUP:-FRED_16country_quarterly}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-recommender_industry}"
SOLVER_NAME="${SOLVER_NAME:-mark,hc_predictor}"
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

${CMD} experiment="${EXPERIMENT_NAME}" solver="${SOLVER_NAME}" problem="${PROBLEM_LIST}"
