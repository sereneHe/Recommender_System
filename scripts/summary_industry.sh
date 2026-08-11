#!/bin/sh

cd "$(dirname "$0")/.." || exit 1

python3 multirun_summary_mlflow.py \
  --group FRED_16country_monthly \
  --reports-dir reports/FRED/FRED_16country_monthly \
  --source multirun \
  --solver hc_predictor_ce,mark_with_cc,mark,hc_predictor_ci,hc_predictor\
  --split-settings

python3 constraints_count.py \