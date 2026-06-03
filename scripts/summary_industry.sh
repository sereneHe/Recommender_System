#!/bin/sh

set -e

ROOT="/Users/xiaoyuhe/Recommender_Pavel"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

PROBLEM_GROUPS="${PROBLEM_GROUPS:-FRED_9country_monthly,FRED_9country_quarterly,OECD_9country_monthly,OECD_9country_quarterly}"
SOLVERS="${SOLVERS:-mark_with_cc,hc_predictor_ce}"

REPORTS_ROOT="${REPORTS_ROOT:-${ROOT}/reports}"
EXPERIMENT="${EXPERIMENT:-INDUSTRY_RECOMMENDER}"

OLD_IFS="$IFS"
IFS=','

for PROBLEM_GROUP in $PROBLEM_GROUPS; do
  for SOLVER in $SOLVERS; do
    IFS="$OLD_IFS"

    case "$PROBLEM_GROUP" in
      FRED_*)
        REPORTS_DIR="${REPORTS_ROOT}/FRED"
        ;;
      OECD_*)
        REPORTS_DIR="${REPORTS_ROOT}/OECD"
        ;;
      *)
        REPORTS_DIR="${REPORTS_ROOT}"
        ;;
    esac

    OUTPUT_NAME="${PROBLEM_GROUP}_${SOLVER}"

    echo "Running group=${PROBLEM_GROUP}, solver=${SOLVER}"

    "${PYTHON_EXEC}" scripts/multirun_summary_mlflow.py \
      --group "${PROBLEM_GROUP}" \
      --reports-dir "${REPORTS_DIR}" \
      --experiment "${EXPERIMENT}" \
      --solver "${SOLVER}" \
      --output-name "${OUTPUT_NAME}"

    IFS=','
  done
done

IFS="$OLD_IFS"