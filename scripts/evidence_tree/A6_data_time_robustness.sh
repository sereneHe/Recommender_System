#!/usr/bin/env bash
# Evidence-tree axis A6: data representation, time structure, robustness.
#
# Nodes: A6.lag, A6.trend, A6.regime, A6.huber.  Runs on FRED monthly industry
# data with expanding-window time-series CV.
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a6_fred_v1 \
#   qsub -l walltime=24:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A6_data_time_robustness.sh,EVIDENCE_BATCH_ID=et_a6_fred_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A6"
EV_SCOPE="FRED"
# PROBLEMS and the prefix must be set BEFORE sourcing _common.sh, otherwise the
# synthetic_er default wins and this cohort silently runs synthetic data.
PROBLEMS="${PROBLEMS:-FRED_16country_monthly/industry_eu_ita}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-ET_A6_FRED}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A6" "data / time / robustness (batch=${EVIDENCE_BATCH_ID})"

FRED_COMMON=(
  "solver.time_limit=${TIME_LIMIT}"
  "solver.n_runs=${N_RUNS}"
  "solver.n_outer=${N_OUTER}"
  "solver.n_inner=${N_INNER}"
  "solver.hidden_dim=${HIDDEN_DIM}"
  "solver.depth=${DEPTH}"
  "solver.dropout=${DROPOUT}"
  "solver.learning_rate=${LR}"
  "solver.weight_decay=${WEIGHT_DECAY}"
  "solver.grad_clip_norm=${GRAD_CLIP_NORM}"
  "solver.use_validation=true"
  "solver.restore_best_validation_model=true"
  "solver.feature_selector=none"
  "solver.dag_fit_scope=inner_train"
  "solver.recalculate_dag=true"
  "solver.w_matrix_space=model_standardized"
  "solver.constraint_audit_enabled=true"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_stochastic_constrained_optimizer=false"
  "solver.ce_constraint_backend=alm_pbm"
  "solver.ce_pbm_backend=stochastic_pbm"
  "solver.cv_strategy=time_series"
  "solver.cv_time_test_size=${CV_TIME_TEST_SIZE:-12}"
  "solver.cv_time_gap=${CV_TIME_GAP:-0}"
  "solver.validation_split_strategy=time"
  "solver.prediction_loss=mse"
  "++problem.evidence_batch_id=${EVIDENCE_BATCH_ID}"
)

# Shared reference: default data representation, MSE loss.
run_arm "ref" "fred_ref" "${FRED_COMMON[@]}"

run_arm "A6.lag" "feature_lag_3" "${FRED_COMMON[@]}" \
  "+problem.feature_lag=${FEATURE_LAG:-3}"

run_arm "A6.trend" "add_time_trend" "${FRED_COMMON[@]}" \
  "+problem.add_time_trend=true"

run_arm "A6.regime" "regime_break_2020" "${FRED_COMMON[@]}" \
  "+problem.regime_break_date=${REGIME_BREAK_DATE:-2020-03-01}"

run_arm "A6.huber" "huber_loss" "${FRED_COMMON[@]}" \
  "solver.prediction_loss=huber" "solver.huber_delta=${HUBER_DELTA:-1.0}"

echo "=== A6 data/time robustness complete: batch=${EVIDENCE_BATCH_ID} ==="
