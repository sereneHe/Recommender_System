#!/usr/bin/env bash
# Shared launcher helpers for the staged causal-predictor experiments.
# This file is sourced by 01...08 scripts; do not submit it directly.

set -euo pipefail

PLAN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PLAN_SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

source "${REPO_ROOT}/scripts/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CONFIG_NAME="${CONFIG_NAME:-config}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-42 43 44}"
# The PBS launcher supplies hydra.run.dir/hydra.sweep.dir as positional
# overrides. Preserve them for every child Hydra run, after stage defaults.
PLAN_EXTRA_OVERRIDES=("$@")

# Accept either space-separated or comma-separated seeds without turning them
# into a Hydra Cartesian product.
read -r -a PLAN_SEEDS <<< "${SEEDS//,/ }"
if [[ ${#PLAN_SEEDS[@]} -eq 0 ]]; then
  echo "SEEDS must contain at least one integer." >&2
  exit 2
fi

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "Set ${name} before running this stage."
}

print_command() {
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
}

run_hydra() {
  local experiment="$1"
  local solver="$2"
  local problem="$3"
  shift 3

  local -a command=(
    "${PYTHON_BIN}" run_experiments.py --multirun
    "--config-name=${CONFIG_NAME}"
    "experiment=${experiment}"
    "solver=${solver}"
    "problem=${problem}"
    "$@"
  )
  print_command "${command[@]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${command[@]}"
  fi
}

run_seeded() {
  local label="$1"
  local experiment_prefix="$2"
  local solver="$3"
  local problem="$4"
  local include_problem_seed="$5"
  shift 5

  local seed
  for seed in "${PLAN_SEEDS[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed ${seed}."
    export HC_SPBM_RANDOM_SEED="${seed}"
    if [[ -n "${HC_CE_BD_SEED_BASE:-}" ]]; then
      [[ "${HC_CE_BD_SEED_BASE}" =~ ^[0-9]+$ ]] || die "Invalid HC_CE_BD_SEED_BASE=${HC_CE_BD_SEED_BASE}."
      export HC_CE_BD_SEED="$((HC_CE_BD_SEED_BASE + seed))"
    fi
    local -a seed_overrides=(
      "solver.random_state=${seed}"
      "solver.cv_random_state=$((seed + 10000))"
      "solver.validation_random_state=$((seed + 20000))"
    )
    if [[ "${include_problem_seed}" == "true" ]]; then
      # ``problem=synthetic_er,synthetic_sf`` is a Hydra config-group sweep.
      # For a sweep, the selected problem config is composed after overrides
      # are parsed, so a normal ``problem.seed=...`` update is rejected as a
      # struct key.  ``++`` works for both sweep and single-problem runs: it
      # updates the existing seed when present and appends it otherwise.
      seed_overrides+=("++problem.seed=${seed}")
    fi
    if (( ${#PLAN_EXTRA_OVERRIDES[@]} > 0 )); then
      run_hydra "${experiment_prefix}_${label}_seed${seed}" "${solver}" "${problem}" \
        "${seed_overrides[@]}" "$@" "${PLAN_EXTRA_OVERRIDES[@]}"
    else
      run_hydra "${experiment_prefix}_${label}_seed${seed}" "${solver}" "${problem}" \
        "${seed_overrides[@]}" "$@"
    fi
  done
}

announce_stage() {
  local phase="$1"
  local description="$2"
  echo
  echo "=== ${phase}: ${description} ==="
  echo "repo=${REPO_ROOT} seeds=${PLAN_SEEDS[*]} dry_run=${DRY_RUN}"
}
