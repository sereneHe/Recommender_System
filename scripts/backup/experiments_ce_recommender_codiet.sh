#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
. "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"

# HDL (mg/dL)、LDL (mg/dL)、TRIG (mg/dL)
# ${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_microbiome_metabolome" solver.recalculate_dag=true
# feature_select=none
${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_compact" solver.recalculate_dag=true
