#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config-cluster"


#${CMD} experiment="CODIET_RECOMMENDER3" solver="mark" solver.N_SELECT_FEATURES="range(2,101)"  problem.target="'GLU (mg/dL)', 'HDL (mg/dL)', 'LDL (mg/dL)', 'TRIG (mg/dL)', 'HbA1c (%)', 'Systolic Blood Pressure (mm Hg)', 'Diastolic Blood Pressure (mm Hg)', 'CRP (mg/dL)', 'whtr(waist-height_ratio)'"

${CMD} experiment="CODIET_RECOMMENDER4" solver="mark_with_cc" solver.rho0="0.1,1,0.01" solver.rho_mult="1.5,2" solver.n_outer="1,3,6,8,10,12,15,20"  problem="codiet, codiet_diast, codiet_glu, codiet_hba, codiet_hdl, codiet_ldl, codiet_syst, codiet_trig, codiet_whtr"


