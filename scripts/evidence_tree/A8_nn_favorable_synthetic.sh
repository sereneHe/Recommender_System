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
A8_STAGE="${A8_STAGE:-smoke}"
# Arms selected for this stage.  smoke keeps the full 5-arm set; pilot/confirm/
# locked run the CE primary question (nn0, nn_ce_true) plus the XGB baseline.
A8_ARMS="${A8_ARMS:-nn0,nn_ce_true,nn_w_true,nn_w_ce_true,xgb100}"

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
# Seeds come from the stage policy (scripts/test/_a8_stage.sh), the single
# source of truth shared with experiment_registry.yaml.  A hardcoded fallback
# here duplicated a stale table, so a direct invocation of this script could
# silently run a different unit set than the registered protocol.
if [[ -z "${GRAPH_SEEDS:-}" || -z "${NOISE_SEEDS:-}" ]]; then
  # shellcheck source=/dev/null
  source "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../test" && pwd)/_a8_stage.sh"
fi
export GRAPH_SEEDS NOISE_SEEDS

# Temporal cannot be judged by the CE question until a lag-aware oracle exists.
if [[ "${MECHANISM}" == "temporal_smooth" ]]; then
  A8_ARMS="nn0,xgb100"
fi

arm_enabled() { [[ ",${A8_ARMS}," == *",$1,"* ]]; }

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
announce_stage "A8" "NN-favourable mechanism=${MECHANISM} stage=${A8_STAGE} arms=${A8_ARMS} (batch=${EVIDENCE_BATCH_ID})"

UNITS=$(( ${#PLAN_SEEDS[@]} * ${#EV_NOISE_SEEDS[@]} ))
echo "[A8] stage=${A8_STAGE} mechanism=${MECHANISM} graph_seeds=${GRAPH_SEEDS} noise_seeds=${NOISE_SEEDS} units=${UNITS} arms=${A8_ARMS}"
echo "[A8] n_samples=${N_SAMPLES} n_nodes=${N_NODES} expected_edges=${EXPECTED_EDGES}"

# L0 dry-run: print the resolved plan and exit WITHOUT running any Hydra job, so
# the L0 preflight can verify config correctness with zero compute.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for arm in nn0 nn_ce_true nn_w_true nn_w_ce_true xgb100; do
    arm_enabled "${arm}" && echo "DRY_RUN arm=${arm} enabled"
  done
  echo "DRY_RUN: no Hydra jobs launched."
  exit 0
fi

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
if arm_enabled nn0; then
  run_arm "A8" "nn0" "${EV_COMMON[@]}" "${ORACLE_OVERRIDES[@]}" \
    "solver.constrained=false" "solver.use_w_constraints=false" \
    "solver.use_ci_penalty=false"
fi

# --- true-DAG CE only (primary question: does CE help the same NN?) ---------
if arm_enabled nn_ce_true; then
  run_arm "A8" "nn_ce_true" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" "${ORACLE_OVERRIDES[@]}"
fi

# --- true-DAG W only (secondary; not in pilot/confirm main screen) ----------
if arm_enabled nn_w_true; then
  run_arm "A8" "nn_w_true" "${EV_COMMON[@]}" "${ORACLE_OVERRIDES[@]}" \
    "solver.constrained=true" "solver.use_w_constraints=true" "solver.use_ci_penalty=false" \
    "solver.w_constraint_mode=legacy_global"
fi

# --- true-DAG W + CE (secondary) --------------------------------------------
if arm_enabled nn_w_ce_true; then
  run_arm "A8" "nn_w_ce_true" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}" "${ORACLE_OVERRIDES[@]}" \
    "solver.use_w_constraints=true" "solver.w_constraint_mode=legacy_global"
fi

# --- XGB reference (fixed 100 rounds; horizontal baseline only) -------------
if arm_enabled xgb100; then
  EV_SOLVER="mark"
  run_arm "A8" "xgb100" "${XGB_SEED[@]}" "solver.n_estimators=100"
fi

echo "=== A8 ${MECHANISM} complete: batch=${EVIDENCE_BATCH_ID} ==="
