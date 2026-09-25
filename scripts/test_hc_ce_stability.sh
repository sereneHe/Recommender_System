#!/bin/sh

# Staged stability study for HC-CE / SPBM on monthly industry data.
#
# Usage:
#   sh scripts/test_hc_ce_stability.sh
#   HC_CE_STAGE=attribution SELECTED_WEIGHT_DECAY=0.05 \
#     sh scripts/test_hc_ce_stability.sh
#   PROBLEM=FRED_16country_monthly/industry_eu_lux \
#     sh scripts/test_hc_ce_stability.sh
#
# The script deliberately does not set solver.random_state,
# solver.cv_random_state, solver.validation_random_state, or problem.seed.
#
# Run stages in order.  Do not start the attribution stage until the
# weight-decay stage has selected its value from outer-fold cv_errors.yaml.
# This keeps the DAG/MILP workload small while Gurobi is being repaired.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CMD="${PYTHON_BIN} run_experiments.py --multirun --config-name=config"

# LTU is the target used in the prior HC-CE stability run.  Override PROBLEM
# only after the winner is confirmed here.
PROBLEM="${PROBLEM:-FRED_16country_monthly/industry_eu_ltu}"
SOLVER="${SOLVER:-hc_predictor_ce}"
EXPERIMENT="${EXPERIMENT:-SPBM_CE_RECOMMENDER}"
TIME_LIMIT="${TIME_LIMIT:-120}"
HC_CE_STAGE="${HC_CE_STAGE:-weight_decay}"

# The old script was called SPBM but accidentally inherited alm_pbm.  Make the
# backend explicit so the experiment name, optimizer, and artifact agree.
CE_BACKEND="${CE_BACKEND:-pbm_all}"

# All stages share the best currently supported stability controls.  W remains
# recalculated because industry_eu has no persisted non-zero W_est.  The CV
# code now estimates W only from each outer training fold and reuses it within
# that fold; do not override this with recalculate_dag=false for industry data.
COMMON_OVERRIDES="
  solver.time_limit=${TIME_LIMIT}
  solver.learning_rate=0.01
  solver.early_stopping_patience=2
  solver.ce_constraint_backend=${CE_BACKEND}
  solver.recalculate_dag=true
  solver.use_w_constraints=false
"

STABLE_OPTIMIZER_OVERRIDES="
  solver.ce_use_balanced_batches=false
  solver.use_stochastic_constrained_optimizer=false
"

case "${HC_CE_STAGE}" in
  weight_decay)
    # Stage 1: keep lr=.01 and all stability switches fixed; scan only decay.
    # The .25 point is the old default, so this also supplies a matched
    # baseline without a separate lr=.25 sweep.
    ${CMD} experiment="${EXPERIMENT}" solver="${SOLVER}" problem="${PROBLEM}" \
      ${COMMON_OVERRIDES} \
      ${STABLE_OPTIMIZER_OVERRIDES} \
      solver.weight_decay=0.01,0.05,0.25 \
      "$@"
    ;;

  attribution)
    : "${SELECTED_WEIGHT_DECAY:?Run HC_CE_STAGE=weight_decay first, then set SELECTED_WEIGHT_DECAY to its winning value.}"
    # Stage 2: lr stays .01.  This 2x2 ablation attributes any stability change
    # to balanced constraint batches and/or SCO, after regularization has been
    # selected.  Compare mean and std of outer test_errs, not inner validation.
    ${CMD} experiment="${EXPERIMENT}" solver="${SOLVER}" problem="${PROBLEM}" \
      ${COMMON_OVERRIDES} \
      solver.weight_decay="${SELECTED_WEIGHT_DECAY}" \
      solver.ce_use_balanced_batches=true,false \
      solver.use_stochastic_constrained_optimizer=true,false \
      "$@"
    ;;

  *)
    echo "Unknown HC_CE_STAGE=${HC_CE_STAGE}; use weight_decay or attribution." >&2
    exit 2
    ;;
esac
