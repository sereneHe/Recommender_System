#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config"

# HDL (mg/dL)、LDL (mg/dL)、TRIG (mg/dL)
# ${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_microbiome_metabolome" solver.recalculate_dag=true
# feature_select=none
${CMD} experiment="CODIET_CE_RECOMMENDER" solver="hc_predictor_ce" problem="codiet_hdl_compact" solver.recalculate_dag=true
