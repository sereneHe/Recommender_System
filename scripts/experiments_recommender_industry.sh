#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config"

PROBLEM_GROUP="FRED_16country_monthly"

${CMD} experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce,hc_predictor,mark_with_cc,mark" problem="${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,${PROBLEM_GROUP}/industry_eu_deu,${PROBLEM_GROUP}/industry_eu_esp,${PROBLEM_GROUP}/industry_eu_est,${PROBLEM_GROUP}/industry_eu_fin,${PROBLEM_GROUP}/industry_eu_fra,${PROBLEM_GROUP}/industry_eu_grc,${PROBLEM_GROUP}/industry_eu_irl,${PROBLEM_GROUP}/industry_eu_ita,${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"
