#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1
. "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"

"${PYTHON_BIN}" multirun_summary_mlflow.py \
  --group FRED_16country_monthly \
  --reports-dir reports/FRED/FRED_16country_monthly \
  --source multirun \
  --solver hc_predictor_ce,mark_with_cc,mark,hc_predictor_ci,hc_predictor\
  --split-settings

"${PYTHON_BIN}" constraints_count.py \
