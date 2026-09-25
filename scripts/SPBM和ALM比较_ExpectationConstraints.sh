#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"

# Optional posterior-stable CI/CE constraints from DAG birth-death MCMC.
# This is OFF by default, preserving the historical single-thresholded-W path.
# Every value can be overridden locally or through qsub -v, for example:
#   qsub -v HC_CE_BD_MCMC=1,HC_CE_BD_STEPS=2000,HC_CE_BD_BURN_IN=400 \
#     cluster_computing/run_metacentrum.pbs
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-0}"
export HC_CE_BD_STEPS="${HC_CE_BD_STEPS:-1000}"
export HC_CE_BD_BURN_IN="${HC_CE_BD_BURN_IN:-200}"
export HC_CE_BD_INITIAL_THRESHOLD="${HC_CE_BD_INITIAL_THRESHOLD:-0.1}"
export HC_CE_BD_EDGE_PENALTY="${HC_CE_BD_EDGE_PENALTY:-1.0}"
export HC_CE_BD_TEMPERATURE="${HC_CE_BD_TEMPERATURE:-1.0}"
export HC_CE_BD_ALLOW_REVERSE="${HC_CE_BD_ALLOW_REVERSE:-1}"
export HC_CE_BD_SEED="${HC_CE_BD_SEED:-20260916}"
export HC_CE_BD_INDEPENDENCE_SUPPORT="${HC_CE_BD_INDEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_DEPENDENCE_SUPPORT="${HC_CE_BD_DEPENDENCE_SUPPORT:-0.8}"
export HC_CE_BD_MAX_CONFLICT_SUPPORT="${HC_CE_BD_MAX_CONFLICT_SUPPORT:-0.2}"

CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"
PROBLEM_GROUP="FRED_16country_monthly"
problem="${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"
feature_select=none
time_limit=120
recalculate_dag=true
solver="hc_predictor_ce"

echo "HC-CE birth-death MCMC: enabled=${HC_CE_BD_MCMC}, steps=${HC_CE_BD_STEPS}, burn_in=${HC_CE_BD_BURN_IN}, seed=${HC_CE_BD_SEED}"

# CoDiet CRP diagnosis (job 23754856): the learned DAG generated 17 target
# d-separation independence constraints and zero dependence constraints.  With
# W constraints disabled, a near-constant prediction satisfies all 17 CE
# constraints and was selected by validation.  The CoDiet commands below run:
#
# 1. a no-CE-constraint predictive baseline; and
# 2. each CE backend with automatic d-separation independences disabled.
#
# Compare outer-fold cv_errors.yaml, not only validation_history.yaml:
# - If the no-CE baseline is better, the CE constraint set is harmful.
# - If no-dsep CE matches the baseline but the historical full-dsep run is
#   worse, the automatically generated independence constraints caused the
#   collapse.
# - If no-dsep CE is better than the baseline, retain only its remaining
#   collider/manual constraints and inspect them before enabling d-separation.
#
# This script intentionally does not set any seed.  The inner validation split
# is still not target-stratified; that requires an estimator code change.

# problem="${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"
# ${CMD} experiment="SPBM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=pbm_all "$@"
# ${CMD} experiment="ALM_PBM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=alm_pbm "$@"
# ${CMD} experiment="ALM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=alm_all "$@"

problem="codiet,codiet_diast,codiet_glu,codiet_hba,codiet_hdl,codiet_ldl,codiet_syst,codiet_trig,codiet_whtr"
# Unconstrained predictor baseline.  use_ci_penalty=false disables both the
# diagnostic CI penalty and the PBM/ALM expectation constraints.
${CMD} experiment="CE_NO_CONSTRAINT_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.use_ci_penalty=false "$@"

# Keep CE enabled but suppress the automatic DAG d-separation independences.
# Collider/manual constraints, if any, remain available for each backend.
${CMD} experiment="SPBM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=pbm_all solver.use_ci_penalty=true solver.ci_add_dsep_independence=false "$@"
${CMD} experiment="ALM_PBM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=alm_pbm solver.use_ci_penalty=true solver.ci_add_dsep_independence=false "$@"
${CMD} experiment="ALM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=alm_all solver.use_ci_penalty=true solver.ci_add_dsep_independence=false "$@"

problem="synthetic_er,synthetic_sf"
samples="200"
edge_ratios="5"
lag_num="0"
vars="10"
time_limits="120,1800"
${CMD} experiment="SPBM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=pbm_all problem.n_nodes="${vars}" problem.n_samples="${samples}" problem.expected_edges="${edge_ratios}" solver.time_limit="${time_limits}" "$@"
${CMD} experiment="ALM_PBM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=alm_pbm problem.n_nodes="${vars}" problem.n_samples="${samples}" problem.expected_edges="${edge_ratios}" solver.time_limit="${time_limits}" "$@"
${CMD} experiment="ALM_CE_RECOMMENDER" solver="${solver}" problem="${problem}" solver.recalculate_dag="${recalculate_dag}" solver.ce_constraint_backend=alm_all problem.n_nodes="${vars}" problem.n_samples="${samples}" problem.expected_edges="${edge_ratios}" solver.time_limit="${time_limits}" "$@"
