#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/.."

CMD="python3 run_experiments.py --multirun --config-name=config"

PROBLEM_GROUP="FRED_16country_monthly"

# #active edge 阈值
# ${CMD} \
#   experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce" problem="${PROBLEM_GROUP}/industry_eu_bel,${PROBLEM_GROUP}/industry_eu_deu,${PROBLEM_GROUP}/industry_eu_esp,${PROBLEM_GROUP}/industry_eu_est,${PROBLEM_GROUP}/industry_eu_fin,${PROBLEM_GROUP}/industry_eu_fra,${PROBLEM_GROUP}/industry_eu_grc,${PROBLEM_GROUP}/industry_eu_irl,${PROBLEM_GROUP}/industry_eu_ita,${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn" \
#   solver.lambda1=0.01 \
#   solver.lambda2=0.01 \
#   solver.ci_threshold=0.05 \
#   solver.nonzero_threshold=0.05 \
#   solver.ci_add_shielded_collider_dependence=false

# shielded_collider
${CMD} \
  experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce" problem="${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,${PROBLEM_GROUP}/industry_eu_deu,${PROBLEM_GROUP}/industry_eu_esp,${PROBLEM_GROUP}/industry_eu_est,${PROBLEM_GROUP}/industry_eu_fin,${PROBLEM_GROUP}/industry_eu_fra,${PROBLEM_GROUP}/industry_eu_grc,${PROBLEM_GROUP}/industry_eu_irl,${PROBLEM_GROUP}/industry_eu_ita,${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn" \
  solver.lambda1=0.01 \
  solver.lambda2=0.01 \
  solver.ci_threshold=0.0001 \
  solver.nonzero_threshold=0.01 \
  solver.ci_add_shielded_collider_dependence=true \
  solver.ci_use_shielded_collider_limits=false
# # milp
# ${CMD} \
#   experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce" problem="${PROBLEM_GROUP}/industry_eu_bel,${PROBLEM_GROUP}/industry_eu_deu,${PROBLEM_GROUP}/industry_eu_esp,${PROBLEM_GROUP}/industry_eu_est,${PROBLEM_GROUP}/industry_eu_fin,${PROBLEM_GROUP}/industry_eu_fra,${PROBLEM_GROUP}/industry_eu_grc,${PROBLEM_GROUP}/industry_eu_irl,${PROBLEM_GROUP}/industry_eu_ita,${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn" \
#   solver.lambda1=0.05 \
#   solver.lambda2=0.05 \
#   solver.ci_threshold=0.0001 \
#   solver.nonzero_threshold=0.01 \
#   solver.ci_add_shielded_collider_dependence=false
