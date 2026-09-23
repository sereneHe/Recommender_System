#!/usr/bin/env bash
# Evidence-tree axis A4: graph estimation & causal prior.
#
# Nodes: A4.edge_penalty, A4.parents_limit, A4.clique_cap, A4.mip_time,
# A4.mip_gap.  (A4.stability_selection / A4.non_gaussian_moments remain
# literature nodes.)
#
# Fixes applied here:
#   - time limit and MIP gap are SEPARATE contrasts (never changed together);
#   - every arm freezes weights_bound / lambda1 / lambda2 / nonzero_threshold /
#     dag_solver_backend so Big-M and regularisation cannot leak into the
#     comparison;
#   - parents_limit is swept over 3/4/5 (a single cap can exclude the true DAG);
#   - the clique arm turns the clique constraints ON explicitly and records the
#     resolved backend/clique diagnostics.
#
# MILP-bound: estimate ~15 min/run on synthetic.
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a4_er_v1 \
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A4_graph_prior.sh,EVIDENCE_BATCH_ID=et_a4_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A4"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A4" "graph estimation & causal prior (batch=${EVIDENCE_BATCH_ID})"

DAG_FIXED=(
  "solver.recalculate_dag=true"
  "solver.dag_solver_backend=milp"
  "solver.weights_bound=1.0"
  "solver.lambda1=0.01"
  "solver.lambda2=0.01"
  "solver.nonzero_threshold=0.001"
  "solver.loss_type=l2" "solver.reg_type=l2" "solver.a_reg_type=l1"
  "solver.constraints_mode=weights"
)
# CE_DAG ends with constraint_audit_oracle=none so it is NOT overwritten by
# EV_CE_BASE's synthetic_linear_sem (a real ER graph is estimated here, so the
# true-SEM oracle does not apply).
CE_DAG=("${EV_CE_BASE[@]}" "solver.constrained=true" "solver.constraint_audit_oracle=none")

# --- shared MIP control: one arm serves both the time and the gap contrast ---
run_arm "A4.mip_control" "mip_control" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.time_limit=120" "solver.target_mip_gap=0.01"

# --- A4.mip_time: only the time limit changes (gap fixed at 0.01) ---
run_arm "A4.mip_time" "mip_time_1800" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.time_limit=1800" "solver.target_mip_gap=0.01"

# --- A4.mip_gap: only the gap changes (time fixed at 120) ---
run_arm "A4.mip_gap" "mip_gap_1e4" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.time_limit=120" "solver.target_mip_gap=0.0001"

# --- A4.edge_penalty ---
run_arm "A4.edge_penalty" "edge_penalty_0" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.edge_penalty=0.0"
run_arm "A4.edge_penalty" "edge_penalty_0p05" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.edge_penalty=0.05"

# --- A4.parents_limit: sweep 3/4/5 against an uncapped reference ---
run_arm "A4.parents_limit" "parents_none" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.max_parents=null"
run_arm "A4.parents_limit" "parents_3" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.max_parents=3"
run_arm "A4.parents_limit" "parents_4" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.max_parents=4"
run_arm "A4.parents_limit" "parents_5" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.max_parents=5"

# --- A4.clique_cap: explicit clique solver + cap ---
run_arm "A4.clique_cap" "clique_off" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.enable_clique_constraints=false"
run_arm "A4.clique_cap" "clique_on" "${EV_COMMON[@]}" "${DAG_FIXED[@]}" "${CE_DAG[@]}" \
  "solver.dag_solver_backend=milp_clique" "solver.enable_clique_constraints=true" \
  "solver.max_clique_size=5"

echo "=== A4 graph prior complete: batch=${EVIDENCE_BATCH_ID} ==="
