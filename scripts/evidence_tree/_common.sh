#!/usr/bin/env bash
# Shared helpers for the evidence-tree validation scripts
# scripts/evidence_tree/<axis>_*.sh.
#
# One script per evidence-tree axis (A1..A6, B0, G0, F0).  This file freezes
# the nuisance settings, fixes the experiment naming (ET_<axis>_ER_<arm>), and
# writes the explicit evidence_node / evidence_arm / evidence_scope fields into
# the problem config so the Evidence Builder can match runs without relying on
# the experiment-name string.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../causal_predictor_plan" && pwd)/_common.sh"

# --- cohort identity (set by the axis script before sourcing) ---------------
EV_AXIS="${EV_AXIS:-A0}"
EV_SCOPE="${EV_SCOPE:-synthetic/ER}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-ET_${EV_AXIS}_ER}"
# A cohort id is REQUIRED and must be identical for every arm of one
# submission; never regenerate a timestamp per arm.
EVIDENCE_BATCH_ID="${EVIDENCE_BATCH_ID:-}"
if [[ -z "${EVIDENCE_BATCH_ID}" ]]; then
  die "Set EVIDENCE_BATCH_ID explicitly (e.g. EVIDENCE_BATCH_ID=et_${EV_AXIS,,}_er_v1)."
fi

# All arms in one evidence batch share this cache root.  The estimator still
# keys each fold cache by its training rows and solver settings; the shared
# root makes the resulting fold-W identity auditable and reusable by paired
# arms without mixing independent evidence batches.
export HC_CE_W_CACHE_DIR="${HC_CE_W_CACHE_DIR:-${REPO_ROOT}/results/evidence_w_cache/${EVIDENCE_BATCH_ID}}"

GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
SEEDS="${GRAPH_SEEDS}"
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"
read -r -a EV_NOISE_SEEDS <<< "${NOISE_SEEDS//,/ }"

PROBLEMS="${PROBLEMS:-synthetic_er}"
TIME_LIMIT="${TIME_LIMIT:-120}"
N_RUNS="${N_RUNS:-5}"
N_OUTER="${N_OUTER:-10}"
N_INNER="${N_INNER:-100}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"

# Evidence labels are data contracts, not cosmetic names.  Refuse a run when
# the problem selector does not belong to the scope advertised by the axis;
# otherwise a synthetic SF or FRED run could silently be recorded as ER.
case "${EV_SCOPE}" in
  synthetic/ER)
    [[ "${PROBLEMS}" == "synthetic_er" ]] || die "EV_SCOPE=synthetic/ER requires PROBLEMS=synthetic_er (got ${PROBLEMS})."
    ;;
  synthetic/SF)
    [[ "${PROBLEMS}" == "synthetic_sf" ]] || die "EV_SCOPE=synthetic/SF requires PROBLEMS=synthetic_sf (got ${PROBLEMS})."
    ;;
  synthetic/SmoothER|synthetic/CompositionalER|synthetic/HighDim|synthetic/Periodic|synthetic/Temporal)
    # A7 NN-favourable mechanisms.  Each has its own problem config and MUST NOT
    # be recorded as plain ER, or its evidence would be mixed with linear ER.
    [[ "${PROBLEMS}" == synthetic_* ]] || die "EV_SCOPE=${EV_SCOPE} requires a synthetic_* problem selector (got ${PROBLEMS})."
    ;;
  FRED)
    [[ "${PROBLEMS}" == FRED_* ]] || die "EV_SCOPE=FRED requires a FRED_* problem selector (got ${PROBLEMS})."
    ;;
  CoDiet)
    [[ "${PROBLEMS}" == codiet* ]] || die "EV_SCOPE=CoDiet requires a codiet* problem selector (got ${PROBLEMS})."
    ;;
esac

# Frozen nuisances: identical for every arm within a cohort.
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEPTH="${DEPTH:-2}"
DROPOUT="${DROPOUT:-0.15}"
LR="${LR:-0.25}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.25}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"

export HC_WEIBULL_GAUSSIANIZE=0
export HC_CE_BD_MCMC="${HC_CE_BD_MCMC:-0}"

if [[ ${#EV_NOISE_SEEDS[@]} -eq 0 ]]; then
  die "NOISE_SEEDS must contain at least one integer."
fi

EV_COMMON=(
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
  "solver.recalculate_dag=false"
  "solver.w_matrix_space=raw_sem"
  "solver.constraint_audit_enabled=true"
  "solver.ci_target_related_only=true"
  "solver.ci_target_constraint_role=endpoint"
  "solver.ce_use_balanced_batches=false"
  "solver.ce_batch_size=128"
  "solver.use_stochastic_constrained_optimizer=false"
  "solver.ce_constraint_backend=alm_pbm"
  "solver.ce_pbm_backend=stochastic_pbm"
  "++problem.n_samples=${N_SAMPLES}"
  "++problem.n_nodes=${N_NODES}"
  "++problem.expected_edges=${EXPECTED_EDGES}"
  "++problem.evidence_batch_id=${EVIDENCE_BATCH_ID}"
  "++problem.sem_type=gauss"
)

EV_CE_BASE=(
  "solver.constrained=true"
  "solver.use_w_constraints=false"
  "solver.use_ci_penalty=true"
  "solver.ci_penalty_kind=conditional_expectation"
  "solver.ce_statistic_kind=partial_correlation"
  "solver.ce_statistic_shrinkage=0.05"
  "solver.ce_residualize_method=linear"
  "solver.ce_se_method=window"
  "solver.ce_cross_window_n_windows=5"
  "solver.ce_tolerance_sd_multiplier=1.96"
  "solver.ci_add_dsep_independence=true"
  "solver.ci_add_collider_marginal_independence=false"
  "solver.ci_add_collider_conditional_dependence=false"
  "solver.ci_add_shielded_collider_dependence=false"
  "solver.constraint_audit_oracle=synthetic_linear_sem"
)

# run_arm <evidence_node> <arm_label> [overrides...]
#   experiment name  : ET_<axis>_ER_<arm label>
#   evidence fields  : evidence_node/arm/scope written into the problem config
run_arm() {
  local node="$1"
  local label="$2"
  shift 2
  local graph_seed noise_seed model_seed
  for graph_seed in "${PLAN_SEEDS[@]}"; do
    [[ "${graph_seed}" =~ ^[0-9]+$ ]] || die "Invalid GRAPH_SEEDS value ${graph_seed}."
    for noise_seed in "${EV_NOISE_SEEDS[@]}"; do
      [[ "${noise_seed}" =~ ^[0-9]+$ ]] || die "Invalid NOISE_SEEDS value ${noise_seed}."
      model_seed="$((graph_seed * 100000 + noise_seed))"
      export HC_SPBM_RANDOM_SEED="${model_seed}"
      local -a evidence_fields=(
        "++problem.evidence_node=${node}"
        "++problem.evidence_arm=${label}"
        "++problem.evidence_scope=${EV_SCOPE}"
      )
      local -a seed_overrides=(
        "solver.random_state=${model_seed}"
        "++problem.seed=${graph_seed}"
        "++problem.graph_seed=${graph_seed}"
        "++problem.noise_seed=${noise_seed}"
      )
      # Tree solvers (mark / mark_with_cc) do not declare cv_random_state or
      # validation_random_state, so add BOTH with '+' and never pass the plain
      # key first (a plain override on a missing key aborts the Hydra run).
      case "${EV_SOLVER:-hc_predictor_ce}" in
        mark|mark_with_cc)
          seed_overrides+=(
            "+solver.cv_random_state=$((model_seed + 10000))"
            "+solver.validation_random_state=$((model_seed + 20000))"
          )
          ;;
        *)
          seed_overrides+=(
            "solver.cv_random_state=$((model_seed + 10000))"
            "solver.validation_random_state=$((model_seed + 20000))"
          )
          ;;
      esac
      if (( ${#PLAN_EXTRA_OVERRIDES[@]} > 0 )); then
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "${EV_SOLVER:-hc_predictor_ce}" "${PROBLEMS}" "${seed_overrides[@]}" "$@" \
          "${evidence_fields[@]}" "${PLAN_EXTRA_OVERRIDES[@]}"
      else
        run_hydra "${EXPERIMENT_PREFIX}_${label}_graph${graph_seed}_noise${noise_seed}" \
          "${EV_SOLVER:-hc_predictor_ce}" "${PROBLEMS}" "${seed_overrides[@]}" "$@" "${evidence_fields[@]}"
      fi
    done
  done
}

# Shared references.  Their evidence_node is "ref" so every comparison can
# resolve them as "EV:ref:<arm>".
run_ref_nn() {
  run_arm "ref" "ref_nn" "${EV_COMMON[@]}" \
    "solver.constrained=false" "solver.use_w_constraints=false" \
    "solver.use_ci_penalty=false" "solver.constraint_audit_oracle=none"
}

run_ref_ce() {
  run_arm "ref" "ref_ce" "${EV_COMMON[@]}" "${EV_CE_BASE[@]}"
}
