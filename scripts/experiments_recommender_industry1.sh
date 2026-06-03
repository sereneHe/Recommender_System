#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config"

#RUN 9country Single#########################################################################################################################################################################################################################################################################################################################################################################################################################
# PROBLEM_GROUP="FRED_9country_monthly"
# # "FRED_9country_quarterly"
# # "OECD_9country_monthly"
# # "OECD_9country_quarterly"

# ${CMD} experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce,mark_with_cc" problem="${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,/${PROBLEM_GROUP}/industry_eu_deu,/${PROBLEM_GROUP}/industry_eu_fin,/${PROBLEM_GROUP}/industry_eu_fra,/${PROBLEM_GROUP}/industry_eu_ita,/${PROBLEM_GROUP}/industry_eu_lux,/${PROBLEM_GROUP}/industry_eu_nld,/${PROBLEM_GROUP}/industry_eu_prt"

#RUN 9country Group########################################################################################################################################################################################################################################################################################################################################################################################################################
PROBLEM_GROUPS=(
  "FRED_9country_monthly"
  "FRED_9country_quarterly"
  "OECD_9country_monthly"
  "OECD_9country_quarterly"
)

for PROBLEM_GROUP in "${PROBLEM_GROUPS[@]}"; do
  echo "Running PROBLEM_GROUP=${PROBLEM_GROUP}"

  ${CMD} experiment="INDUSTRY_RECOMMENDER" solver="mark_with_cc,hc_predictor_ce" problem="${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,/${PROBLEM_GROUP}/industry_eu_deu,/${PROBLEM_GROUP}/industry_eu_fin,/${PROBLEM_GROUP}/industry_eu_fra,/${PROBLEM_GROUP}/industry_eu_ita,/${PROBLEM_GROUP}/industry_eu_lux,/${PROBLEM_GROUP}/industry_eu_nld,/${PROBLEM_GROUP}/industry_eu_prt"
done

#RUN 16country Single#########################################################################################################################################################################################################################################################################################################################################################################################################################
# # PROBLEM_GROUP="FRED_16country_monthly"
# # "FRED_16country_quarterly"
# # "OECD_16country_monthly"
# # "OECD_16country_quarterly"

# ${CMD} experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce,mark_with_cc" problem="${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,${PROBLEM_GROUP}/industry_eu_deu,${PROBLEM_GROUP}/industry_eu_esp,${PROBLEM_GROUP}/industry_eu_est,${PROBLEM_GROUP}/industry_eu_fin,${PROBLEM_GROUP}/industry_eu_fra,${PROBLEM_GROUP}/industry_eu_grc,${PROBLEM_GROUP}/industry_eu_irl,${PROBLEM_GROUP}/industry_eu_ita,${PROBLEM_GROUP}/industry_eu_ltu,${PROBLEM_GROUP}/industry_eu_lux,${PROBLEM_GROUP}/industry_eu_nld,${PROBLEM_GROUP}/industry_eu_prt,${PROBLEM_GROUP}/industry_eu_svk,${PROBLEM_GROUP}/industry_eu_svn"

#RUN 16country Group#########################################################################################################################################################################################################################################################################################################################################################################################################################
PROBLEM_GROUPS=(
  #"FRED_16country_monthly"
  "FRED_16country_quarterly"
  "OECD_16country_monthly"
  "OECD_16country_quarterly"
)

for PROBLEM_GROUP in "${PROBLEM_GROUPS[@]}"; do
  echo "Running PROBLEM_GROUP=${PROBLEM_GROUP}"

  ${CMD} experiment="INDUSTRY_RECOMMENDER" solver="hc_predictor_ce" problem="${PROBLEM_GROUP}/industry_eu_aut,${PROBLEM_GROUP}/industry_eu_bel,/${PROBLEM_GROUP}/industry_eu_deu,/${PROBLEM_GROUP}/industry_eu_fin,/${PROBLEM_GROUP}/industry_eu_fra,/${PROBLEM_GROUP}/industry_eu_ita,/${PROBLEM_GROUP}/industry_eu_lux,/${PROBLEM_GROUP}/industry_eu_nld,/${PROBLEM_GROUP}/industry_eu_prt"
done