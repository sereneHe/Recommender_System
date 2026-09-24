#!/usr/bin/env bash
# Evidence-tree axis A8: NN-favourable synthetic mechanisms.
#
# (Numbered A8 because A7 is CoDiet's type-matched-constraint axis in this
# tree; keeping them separate prevents the two from being merged.)
#
# Tests whether the NN's inductive bias (smooth continuous functions, deep
# composition, high-dimensional shared representation) beats a fixed XGB
# reference on data whose TRUE conditional mean is smooth/continuous rather
# than the small linear ER used elsewhere.  Its own axis/scopes guarantee that
# "NN bias", "CE adds information" and "graph estimation is correct" are never
# conflated.
#
# Mechanisms (MECHANISM env selects one; each has its own problem config and
# its own scope so results are never mixed with linear ER):
#   smooth_additive   -> synthetic_smooth_er.yaml        p=20  n=10000
#   compositional     -> synthetic_compositional_er.yaml p=20  n=10000
#   highdim_smooth    -> synthetic_highdim.yaml          p=50  n=10000
#   periodic          -> synthetic_periodic.yaml         p=20  n=10000
#   temporal_smooth   -> synthetic_temporal.yaml         p=20  n=10000
#
# Arms (same graph/noise seed, split and target; true-DAG constraints only):
#   nn0            HC-NN, no W, no CE          (NN baseline)
#   nn_ce_true     HC-NN + true-DAG CE         (does CE add information?)
#   nn_w_true      HC-NN + true-DAG W          (separate W from CE)
#   nn_w_ce_true   HC-NN + true-DAG W + CE     (complementarity)
#   xgb100         XGB, fixed 100 rounds       (fair XGB reference)
#
# Submit:
#   EVIDENCE_BATCH_ID=et_a8_smooth_er_v1 MECHANISM=smooth_additive \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/A8_nn_favorable_synthetic.sh,EVIDENCE_BATCH_ID=et_a8_smooth_er_v1,MECHANISM=smooth_additive \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="A8"
MECHANISM="${MECHANISM:-smooth_additive}"

case "${MECHANISM}" in
  smooth_additive)   EV_SCOPE="synthetic/SmoothER";        PROBLEMS="synthetic_smooth_er";        N_NODES=20; EXPECTED_EDGES=30 ;;
  compositional)     EV_SCOPE="synthetic/CompositionalER"; PROBLEMS="synthetic_compositional_er"; N_NODES=20; EXPECTED_EDGES=30 ;;
  highdim_smooth)    EV_SCOPE="synthetic/HighDim";         PROBLEMS="synthetic_highdim";          N_NODES=50; EXPECTED_EDGES=120 ;;
  periodic)          EV_SCOPE="synthetic/Periodic";        PROBLEMS="synthetic_periodic";         N_NODES=20; EXPECTED_EDGES=30 ;;
  temporal_smooth)   EV_SCOPE="synthetic/Temporal";        PROBLEMS="synthetic_temporal";         N_NODES=20; EXPECTED_EDGES=30 ;;
  *) echo "Unknown MECHANISM=${MECHANISM} (expected smooth_additive|compositional|highdim_smooth|periodic|temporal_smooth)." >&2; exit 2 ;;
esac
export PROBLEMS N_NODES EXPECTED_EDGES

# A8 is a larger-sample regime; do not inherit the small-ER 1000-sample default.
N_SAMPLES="${N_SAMPLES:-10000}"
export N_SAMPLES
# A8 uses its own seeds; the registry pins the exact SmoothER table.
export GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44 45 46}"
export NOISE_SEEDS="${NOISE_SEEDS:-101 102 103}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A8" "NN-favourable mechanism=${MECHANISM} (batch=${EVIDENCE_BATCH_ID})"

# XGB reference uses the same 100-round budget as B0's mark_100.
XGB_SEED=(
  "solver.n_runs=${N_RUNS}"
  "solver.recalculate_dag=false"
  "+solver.cv_strategy=site_gender"
  "+solver.cv_time_test_size=null"
  "+solver.cv_time_gap=0"
)

# Oracle audit (stage A).  Static mechanisms use the true generator oracle
# E[Y|X] via synthetic_utils.structural_conditional_mean.  The temporal
# mechanism cannot be reconstructed from a static feature matrix, so it runs
# without the oracle until lag-aware handling lands (see estimator guidance).
if [[ "${MECHANISM}" == "temporal_smooth" ]]; then
  ORACLE_OVERRIDES=("solver.constraint_audit_oracle=none")
else
  ORACLE_OVERRIDES=(
    "solver.constraint_audit_oracle=synthetic_generator_oracle"
    "+solver.constraint_audit_oracle_mechanism=${MECHANISM}"
  )
fi

# --- NN baseline (no constraint) -------------------------------------------
run_arm "A8" "nn0" "${EV_COMMON[@]}" "${ORACLE_OVERRIDES[@]}" \
  "solver.constrained=false" "solver.use_w_constraints=false" \
  "solver.use_ci_penalty=false"

# --- true-DAG CE only -------------------------------------------------------
run_arm "A8" "nn_ce_true" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" "${ORACLE_OVERRIDES[@]}"

# --- true-DAG W only --------------------------------------------------------
run_arm "A8" "nn_w_true" "${EV_COMMON[@]}" "${ORACLE_OVERRIDES[@]}" \
  "solver.constrained=true" "solver.use_w_constraints=true" "solver.use_ci_penalty=false" \
  "solver.w_constraint_mode=legacy_global"

# --- true-DAG W + CE --------------------------------------------------------
run_arm "A8" "nn_w_ce_true" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" "${ORACLE_OVERRIDES[@]}" \
  "solver.use_w_constraints=true" "solver.w_constraint_mode=legacy_global"

# --- XGB reference (fixed 100 rounds) --------------------------------------
EV_SOLVER="mark"
run_arm "A8" "xgb100" "${XGB_SEED[@]}" "solver.n_estimators=100"

echo "=== A8 ${MECHANISM} complete: batch=${EVIDENCE_BATCH_ID} ==="
