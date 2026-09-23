#!/usr/bin/env bash
# Evidence-tree axis A2: constraint embedding.
#
# Nodes: A2.W_global, A2.W_target_residual, A2.W_mask, A2.W_bias_calibration,
# A2.balanced_batch, A2.pruning.
# References: ref_nn (no constraint), ref_ce (CE only).
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a2_er_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A2_constraint_embedding.sh,EVIDENCE_BATCH_ID=et_a2_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A2"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A2" "constraint embedding (batch=${EVIDENCE_BATCH_ID})"

run_ref_nn

run_arm "A2.W_global" "w_global" "${EV_COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=legacy_global" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "A2.W_target_residual" "w_target_residual" "${EV_COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=target_residual" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "A2.W_mask" "w_mask" "${EV_COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=legacy_global" \
  "solver.w_prediction_dependent_mask=true" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_arm "A2.W_bias_calibration" "w_bias_calibration" "${EV_COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.w_bias_calibration=true" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

run_ref_ce

run_arm "A2.balanced_batch" "ce_balanced_batch" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_use_balanced_batches=true"

run_arm "A2.pruning" "ce_pruning" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ci_prune_redundant=true" "solver.ci_max_dsep_separator_size=3"

echo "=== A2 constraint embedding complete: batch=${EVIDENCE_BATCH_ID} ==="
