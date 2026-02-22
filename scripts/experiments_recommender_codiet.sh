#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config-cluster"


${CMD} experiment="CODIET_RECOMMENDER" solver="mark_with_cc, mark" solver.rho0=1 problem.target="'GLU (mg/dL)', 'HDL (mg/dL)', 'LDL (mg/dL)', 'TRIG (mg/dL)', 'HbA1c (%)', 'Systolic Blood Pressure (mm Hg)', 'Diastolic Blood Pressure (mm Hg)', 'CRP (mg/dL)', 'whtr(waist-height_ratio)'"

