#!/usr/bin/env bash
# Evidence-tree axis B0: comparable classical baselines.
#
# Nodes: B0.mark_10, B0.mark_100, B0.mark_cc_10, B0.mark_cc_100,
# B0.hc_nn_no_constraint, B0.hc_w_only, B0.hc_ce_only, B0.hc_w_ce.
#
# The tree-count node (A5.1 / B0) is only meaningful if the arm names make the
# tree budget explicit AND the contrast changes only the tree count:
#   mark_10 / mark_100         : n_estimators 10 vs 100 (n_outer equal)
#   mark_cc_10 / mark_cc_100   : n_estimators 10 vs 100 at a FIXED n_outer
# The legacy pre-fix mark_cc_100 (which also changed n_outer) is diagnostic only
# and cannot support a tree-count or Mark-CC claim.
#
# Submit:
#   EVIDENCE_BATCH_ID=et_b0_er_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/B0_baselines.sh,EVIDENCE_BATCH_ID=et_b0_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="B0"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "B0" "classical baselines (batch=${EVIDENCE_BATCH_ID})"

TREE_SEED=(
  "solver.n_runs=${N_RUNS}"
  "solver.recalculate_dag=false"
  "+solver.cv_strategy=site_gender"
  "+solver.cv_time_test_size=null"
  "+solver.cv_time_gap=0"
)

# Tree-count fairness rule: the Mark-CC contrast must vary ONLY the tree count.
# mark_with_cc fits n_outer graph refits x n_estimators trees, so the OLD arms
# (mark_cc_10 = n_outer 1, mark_cc_100 = n_outer 10) confounded tree count with
# the graph-refit budget and MUST NOT be used for a tree-count or Mark-CC claim.
# The fair contrast below holds n_outer equal and changes n_estimators only.
# (Total trees per arm = n_outer x n_estimators; if a total-budget contrast is
# wanted instead, set CC_N_OUTER=1 and keep n_estimators 10 vs 100.)
CC_N_OUTER="${CC_N_OUTER:-${N_OUTER}}"

# --- Mark, explicit tree budgets ---
EV_SOLVER="mark"
run_arm "B0.mark_10" "mark_10" "${TREE_SEED[@]}" "solver.n_estimators=10"
run_arm "B0.mark_100" "mark_100" "${TREE_SEED[@]}" "solver.n_estimators=100"

# --- Mark-CC: SAME n_outer for both arms; only n_estimators changes ---------
EV_SOLVER="mark_with_cc"
run_arm "B0.mark_cc_10" "mark_cc_10" "${TREE_SEED[@]}" \
  "solver.n_outer=${CC_N_OUTER}" "solver.n_estimators=10" "+solver.w_matrix_space=raw_sem"
run_arm "B0.mark_cc_100" "mark_cc_100" "${TREE_SEED[@]}" \
  "solver.n_outer=${CC_N_OUTER}" "solver.n_estimators=100" "+solver.w_matrix_space=raw_sem"

# --- HC neural baselines (matched to the CE arms' network) ---
EV_SOLVER="hc_predictor_ce"
run_arm "B0.hc_nn_no_constraint" "hc_nn_no_constraint" "${EV_COMMON[@]}" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"
run_arm "B0.hc_w_only" "hc_w_only" "${EV_COMMON[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" \
  "solver.use_ci_penalty=false" "solver.w_constraint_mode=legacy_global" \
  "solver.constraint_audit_oracle=synthetic_linear_sem"

# Canonical CE baselines.  These are owned by B0 so A2 does not rerun the
# same CE-only or W+CE treatments as embedding experiments.
run_arm "B0.hc_ce_only" "hc_ce_only" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}"
run_arm "B0.hc_w_ce" "hc_w_ce" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.use_w_constraints=true" "solver.w_constraint_mode=legacy_global"

echo "=== B0 baselines complete: batch=${EVIDENCE_BATCH_ID} ==="
