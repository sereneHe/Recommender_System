#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${PYTHON_EXEC:-}" ]; then PYTHON_BIN="${PYTHON_EXEC}"; fi
. "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"

# ${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_compact" solver.recalculate_dag=true solver.ce_constraint_backend=pbm_all

# ${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_compact" solver.recalculate_dag=true solver.ce_constraint_backend=alm_all

#${CMD} experiment="CODIET_CE_RECOMMENDER" solver="mark,mark_with_cc" problem="codiet_hdl_compact" solver.recalculate_dag=false
${CMD} experiment="CODIET_RECOMMENDER3" solver="mark_with_cc" solver.N_SELECT_FEATURES="range(2,101)"  problem.target="'GLU (mg/dL)', 'HDL (mg/dL)', 'LDL (mg/dL)', 'TRIG (mg/dL)', 'HbA1c (%)', 'Systolic Blood Pressure (mm Hg)', 'Diastolic Blood Pressure (mm Hg)', 'CRP (mg/dL)', 'whtr(waist-height_ratio)'"  solver.recalculate_dag=false
