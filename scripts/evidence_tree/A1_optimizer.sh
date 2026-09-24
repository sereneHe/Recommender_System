#!/usr/bin/env bash
# Evidence-tree axis A1: optimizer / constraint-embedding algorithm.
#
# Nodes: A1.ALM_ALL, A1.PBM_ALL, A1.HYBRID_ALM_PBM, A1.STOCHASTIC_PBM,
# A1.SCO_LAYER.
#
# Routing facts this script respects:
#   alm_pbm  routes INDEPENDENT constraints to ALM and DEPENDENT ones to PBM;
#            with independent-only constraints it is equivalent to ALM.
#   alm_all  every constraint to ALM.
#   pbm_all  every constraint to PBM (use this for PBM / SPBM arms).
#   stochastic_pbm is the local SPBM dual/penalty update used with pbm_all.
#   the SCO layer is a separate optimization-layer arm, not a pure-SPBM arm.
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a1_er_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A1_optimizer.sh,EVIDENCE_BATCH_ID=et_a1_er_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A1"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A1" "optimizer backends (batch=${EVIDENCE_BATCH_ID})"

# Shared CE reference is alm_pbm (independent-only -> ALM today), so
# A1.ALM_ALL is mostly a routing sanity check rather than an optimizer contrast.
run_ref_ce

run_arm "A1.ALM_ALL" "ce_alm_all" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_constraint_backend=alm_all"

# Deterministic PBM backend (EV_CE_BASE already picks stochastic_pbm, so
# without this override PBM_ALL and STOCHASTIC_PBM would be identical runs).
# humancompatible_pbm only accepts ci_pbm_penalty_update in {const,dimin,dimin_dual};
# the repo default is "adapt" and would raise ValueError, so pin it explicitly.
run_arm "A1.PBM_ALL" "ce_pbm_all" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_constraint_backend=pbm_all" "solver.ce_pbm_backend=humancompatible_pbm" \
  "solver.ci_pbm_penalty_update=const"

run_arm "A1.STOCHASTIC_PBM" "ce_spbm_all" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_constraint_backend=pbm_all" "solver.ce_pbm_backend=stochastic_pbm"

run_arm "A1.SCO_LAYER" "ce_sco" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" \
  "solver.ce_constraint_backend=pbm_all" "solver.ce_pbm_backend=stochastic_pbm" \
  "solver.use_stochastic_constrained_optimizer=true"

echo "=== A1 optimizer complete: batch=${EVIDENCE_BATCH_ID} ==="
