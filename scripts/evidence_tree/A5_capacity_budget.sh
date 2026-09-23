#!/usr/bin/env bash
# Evidence-tree axis A5: model capacity & training budget.
#
# Nodes: A5.hidden_depth, A5.lr_wd, A5.n_outer_inner.
# Reference: ref_nn (default capacity, 10 x 100 updates).
#
# Budget confound note: 5 x 50 changes the gradient-update total AND the dual
# update count AND wall-clock.  We therefore also run a gradient-matched arm
# (20 x 50 = 1000 updates, same as 10 x 100) so the budget effect is not
# conflated with "less training".  A wall-clock-matched arm needs the lockfile
# runner and is not included here.
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a5_er_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A5_capacity_budget.sh,EVIDENCE_BATCH_ID=et_a5_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A5"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A5" "capacity & budget (batch=${EVIDENCE_BATCH_ID})"

run_ref_nn

run_arm "A5.hidden_depth" "hicap_64x3" "${EV_COMMON[@]}" \
  "solver.hidden_dim=64" "solver.depth=3" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

run_arm "A5.lr_wd" "lr0p05_wd0p10" "${EV_COMMON[@]}" \
  "solver.learning_rate=0.05" "solver.weight_decay=0.10" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

# Budget: total updates halved (500) vs the 1000-update reference.
run_arm "A5.n_outer_inner" "updates_half_5x50" "${EV_COMMON[@]}" \
  "solver.n_outer=5" "solver.n_inner=50" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

# Budget: same total updates (1000) as the reference, different split.
run_arm "A5.n_outer_inner" "updates_matched_20x50" "${EV_COMMON[@]}" \
  "solver.n_outer=20" "solver.n_inner=50" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"

echo "=== A5 capacity & budget complete: batch=${EVIDENCE_BATCH_ID} ==="
