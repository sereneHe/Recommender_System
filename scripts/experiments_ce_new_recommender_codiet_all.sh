#!/usr/bin/env bash
# Combined launcher for the CoDiet CE-new screens.
#
# CoDiet runs in its own job because it is by far the heaviest workload: 9
# targets x (hc_predictor + mark + mark_with_cc + no-constraint + W-only) with
# MILP DAG fitting (time_limit=1800) and birth-death posterior filtering.
# Split it across PBS jobs with CODIET_PROBLEMS / SEEDS if a single job risks the
# walltime limit.
#
# Run with the PBS wrapper:
#   qsub -v EXPERIMENT_SCRIPT=scripts/experiments_ce_new_recommender_codiet_all.sh \
#     cluster_computing/run_metacentrum.pbs
#
# Subset example:
#   qsub -v EXPERIMENT_SCRIPT=scripts/experiments_ce_new_recommender_codiet_all.sh,\
#CODIET_PROBLEMS=codiet,codiet_glu,SEEDS=42 \
#     cluster_computing/run_metacentrum.pbs

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_CODIET_BASELINE="${RUN_CODIET_BASELINE:-1}"
RUN_CODIET_CMI="${RUN_CODIET_CMI:-1}"

SEEDS="${SEEDS:-42 43 44}"
CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND:-alm_pbm}"
HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND:-alm}"
DRY_RUN="${DRY_RUN:-0}"

declare -a EXTRA_OVERRIDES=()
EXTRA_OVERRIDE_COUNT="$#"
if (( $# > 0 )); then
  EXTRA_OVERRIDES=("$@")
fi

require_bool() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || {
    echo "${name} must be 0 or 1, got ${value}." >&2
    exit 2
  }
}

for toggle in RUN_CODIET_BASELINE RUN_CODIET_CMI; do
  require_bool "${toggle}" "${!toggle}"
done

run_stage() {
  local label="$1"
  shift
  echo
  echo "================================================================================"
  echo "=== ${label} ==="
  echo "================================================================================"
  "$@"
}

run_child() {
  local script="$1"
  shift
  if (( EXTRA_OVERRIDE_COUNT > 0 )); then
    "$@" "${script}" "${EXTRA_OVERRIDES[@]}"
  else
    "$@" "${script}"
  fi
}

# PROBLEMS is forwarded to the child through the environment when set.
PROBLEMS_EXPORT=()
if [[ -n "${CODIET_PROBLEMS:-}" ]]; then
  PROBLEMS_EXPORT=(PROBLEMS="${CODIET_PROBLEMS}")
fi

echo "=== CE-new CoDiet launcher ==="
echo "repo=${REPO_ROOT} seeds=${SEEDS} problems=${CODIET_PROBLEMS:-<default 9 targets>} dry_run=${DRY_RUN}"

if [[ "${RUN_CODIET_BASELINE}" == "1" ]]; then
  run_stage "CoDiet: HC/Mark/Mark-CC, no-constraint, and W-only controls" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_codiet.sh" env \
      SEEDS="${SEEDS}" CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      HC_BASELINE_BACKEND="${HC_BASELINE_BACKEND}" INCLUDE_CLASSICAL_BASELINES=1 \
      STAGE=baseline DRY_RUN="${DRY_RUN}" ${PROBLEMS_EXPORT[@]+"${PROBLEMS_EXPORT[@]}"} \
      bash
fi

if [[ "${RUN_CODIET_CMI}" == "1" ]]; then
  run_stage "CoDiet: matched W+CMI arm" \
    run_child "${SCRIPT_DIR}/experiments_ce_new_recommender_codiet.sh" env \
      SEEDS="${SEEDS}" CE_CONSTRAINT_BACKEND="${CE_CONSTRAINT_BACKEND}" \
      STAGE=discrete_cmi DRY_RUN="${DRY_RUN}" ${PROBLEMS_EXPORT[@]+"${PROBLEMS_EXPORT[@]}"} \
      bash
fi

echo "=== CE-new CoDiet launcher complete ==="
