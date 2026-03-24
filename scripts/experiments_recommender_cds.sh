#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 run_experiments.py --multirun --config-name=config-cluster"


${CMD} experiment="CDS_RECOMMENDER" solver="mark_with_cc, mark" problem=cds problem.target="'8B69AP', '06DABK', 'EFAGG9', '2H6677', 'FH49GG', '8D8575', 'GG6EBT', 'DD359M', 'FF667M', '0H99B7', '2H66B7', '8A87AG', 'NN2A8G', '6A516F'"

# '05ABBF',