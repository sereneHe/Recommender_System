#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
. "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"

# Base hc_predictor runtime options. Override these in the environment, e.g.
# qsub -v HC_CONSTRAINT_BACKEND=spbm,HC_WEIBULL_GAUSSIANIZE=1 cluster_computing/run_metacentrum.pbs
export HC_CONSTRAINT_BACKEND="${HC_CONSTRAINT_BACKEND:-alm}"        # alm | spbm
export HC_WEIBULL_GAUSSIANIZE="${HC_WEIBULL_GAUSSIANIZE:-0}"       # 0 | 1
export HC_WEIBULL_MODE="${HC_WEIBULL_MODE:-rand}"                  # rand | mid

echo "hc_predictor: backend=${HC_CONSTRAINT_BACKEND}, weibull=${HC_WEIBULL_GAUSSIANIZE}, weibull_mode=${HC_WEIBULL_MODE}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"

# Base HC example; HC_CONSTRAINT_BACKEND selects ALM or SPBM above.
# ${CMD} experiment="CODIET_RECOMMENDER" solver="hc_predictor" problem="codiet_hdl_compact" solver.recalculate_dag=true

# ${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_compact" solver.recalculate_dag=true solver.ce_constraint_backend=pbm_all

# ${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_compact" solver.recalculate_dag=true solver.ce_constraint_backend=alm_all

${CMD} experiment="CODIET_CE_RECOMMENDER" solver="mark,mark_with_cc" problem="codiet_hdl_compact" solver.recalculate_dag=true 

${CMD} experiment="CODIET_CE_RECOMMENDER" solver="mark,mark_with_cc" problem="codiet_hdl_compact" solver.recalculate_dag=true 
