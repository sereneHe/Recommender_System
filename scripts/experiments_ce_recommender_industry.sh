#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config"
PROBLEM_GROUP="FRED_16country_monthly"
problem="${PROBLEM_GROUP}/industry_eu_ltu"
# problem="${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"
feature_select=none
time_limit=120
recalculate_dag=true

# CE defaults
${CMD} experiment="INDUSTRY_CE_RECOMMENDER" solver="hc_predictor_ce" problem="${problem}" solver.time_limit="${time_limit}"

#active edge 阈值
${CMD} experiment="INDUSTRY_CE_RECOMMENDER" solver="hc_predictor_ce" problem="${problem}" solver.lambda1=0.01 solver.lambda2=0.01 solver.ci_threshold=0.05 solver.nonzero_threshold=0.05 solver.ci_add_shielded_collider_dependence=false solver.feature_selector="${feature_select}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}"

# shielded_collider
${CMD} experiment="INDUSTRY_CE_RECOMMENDER" solver="hc_predictor_ce" problem="${problem}" solver.lambda1=0.01 solver.lambda2=0.01 solver.ci_threshold=0.0001 solver.nonzero_threshold=0.01 solver.ci_add_shielded_collider_dependence=true solver.ci_use_shielded_collider_limits=false solver.feature_selector="${feature_select}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}"

# shielded_collider + limits
${CMD} experiment="INDUSTRY_CE_RECOMMENDER" solver="hc_predictor_ce" problem="${problem}" solver.lambda1=0.01 solver.lambda2=0.01 solver.ci_threshold=0.0001 solver.nonzero_threshold=0.01 solver.ci_add_shielded_collider_dependence=true solver.ci_use_shielded_collider_limits=true solver.feature_selector="${feature_select}" solver.time_limit="${time_limit}" solver.recalculate_dag="${recalculate_dag}"

# ${CMD} \
#   experiment="INDUSTRY_CE_RECOMMENDER" solver="mark_with_cc,mark,hc_predictor_ci,hc_predictor" problem="${problem}" \
#   solver.feature_selector="${feature_select}" \
#   ++solver.time_limit="${time_limit}" \
#   solver.recalculate_dag="${recalculate_dag}"
