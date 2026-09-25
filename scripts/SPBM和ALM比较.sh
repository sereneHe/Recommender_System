#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"
feature_select=none
time_limit=120
recalculate_dag=true
solver="hc_predictor"
samples="200"
edge_ratios="5"
lag_num="0"
vars="10"
time_limits="120,1800"

export HC_CONSTRAINT_BACKEND="alm"
PROBLEM_GROUP="FRED_16country_monthly"
problem="${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"
${CMD} experiment="ALM_INDUSTRY_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" "$@"
problem="codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr"
${CMD} experiment="ALM_CODIET_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" "$@"
problem="synthetic_er,synthetic_sf"
${CMD} experiment="ALM_SYN_RECOMMENDER" solver="${solver}" problem="${problem}" solver.recalculate_dag="${recalculate_dag}" problem.n_nodes="${vars}" problem.n_samples="${samples}" problem.expected_edges="${edge_ratios}" solver.time_limit="${time_limits}" "$@"

export HC_CONSTRAINT_BACKEND="spbm"
problem="${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"
${CMD} experiment="SPBM_INDUSTRY_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" "$@"
problem="codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr"
${CMD} experiment="SPBM_CODIET_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" "$@"
problem="synthetic_er,synthetic_sf"
${CMD} experiment="SPBM_SYN_RECOMMENDER" solver="${solver}" problem="${problem}" solver.recalculate_dag="${recalculate_dag}" problem.n_nodes="${vars}" problem.n_samples="${samples}" problem.expected_edges="${edge_ratios}" solver.time_limit="${time_limits}" "$@"
