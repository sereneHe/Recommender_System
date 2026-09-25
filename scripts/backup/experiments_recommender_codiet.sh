#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
. "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config-cluster"


#${CMD} experiment="CODIET_RECOMMENDER3" solver="mark" solver.N_SELECT_FEATURES="range(2,101)"  problem.target="'GLU (mg/dL)', 'HDL (mg/dL)', 'LDL (mg/dL)', 'TRIG (mg/dL)', 'HbA1c (%)', 'Systolic Blood Pressure (mm Hg)', 'Diastolic Blood Pressure (mm Hg)', 'CRP (mg/dL)', 'whtr(waist-height_ratio)'"

${CMD} experiment="CODIET_RECOMMENDER4" solver="mark_with_cc" solver.rho0="0.1,1,0.01" solver.rho_mult="1.5,2" solver.n_outer="1,3,6,8,10,12,15,20"  problem="codiet, codiet_diast, codiet_glu, codiet_hba, codiet_hdl, codiet_ldl, codiet_syst, codiet_trig, codiet_whtr"

