#!/usr/bin/env bash
# Evidence-tree axis A3: CI/CE statistic and uncertainty.
#
# Nodes: A3.covariance, A3.window_SE, A3.HAC_SE, A3.sign_flip_filter.
# References: ref_ce (CE, window SE, filter off).
#
# sign-flip contrast: both arms use the SAME dependent-constraint set and
# differ ONLY in ce_window_filter_enabled (false vs true).  The previous
# version left the default (true) on both arms, so it had no contrast.
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a3_er_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A3_ci_statistic.sh,EVIDENCE_BATCH_ID=et_a3_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A3"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A3" "CI/CE statistic (batch=${EVIDENCE_BATCH_ID})"

# ref_ce IS the window-SE arm (EV_CE_BASE sets ce_se_method=window), so it
# doubles as the A3.window_SE reference; no separate window arm is run.
run_arm "ref" "ref_ce" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_window_filter_enabled=false"

# A3.covariance: residual covariance instead of standardized partial correlation.
run_arm "A3.covariance" "ce_covariance" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_window_filter_enabled=false" \
  "solver.ce_statistic_kind=covariance"

# A3.HAC_SE: one single window-vs-HAC contrast (ref = ref_ce = window).
run_arm "A3.HAC_SE" "ce_hac_se" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_window_filter_enabled=false" \
  "solver.ce_se_method=hac" "solver.ce_hac_max_lag=6"

# A3.sign_flip_filter: dependent constraints ON in both arms; only the filter
# flips.  This gives the node a real reference/treatment pair.
run_arm "A3.sign_flip_filter" "dep_filter_off" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ci_add_collider_conditional_dependence=true" \
  "solver.ci_dependent_statistic=signed" \
  "solver.ce_window_filter_enabled=false"

run_arm "A3.sign_flip_filter" "dep_filter_on" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ci_add_collider_conditional_dependence=true" \
  "solver.ci_dependent_statistic=signed" \
  "solver.ce_window_filter_enabled=true" "solver.ce_window_max_sign_flip_rate=0.30"

echo "=== A3 CI/CE statistic complete: batch=${EVIDENCE_BATCH_ID} ==="
