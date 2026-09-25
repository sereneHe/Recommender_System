#!/usr/bin/env bash

# Reproduce the four notebook baselines on the Industry data with
# recalculate_dag=false.  The default targets are the three distinct target
# records represented by the best historical Industry runs (SVN, DEU, ITA).
# Override PROBLEMS to run the full 16-country set.
#
# Set REPLAY_TOP3=0 for one four-arm comparison.  The default runs the three
# historical candidate profiles so that a one-off result can be separated from
# a configuration that is stable across seeds/targets.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HC_WEIBULL_GAUSSIANIZE="${HC_WEIBULL_GAUSSIANIZE:-0}"
export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"

source "${SCRIPT_DIR}/python_runtime.sh"
project_python_require "${REPO_ROOT}"
CONFIG_NAME="${CONFIG_NAME:-config}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-REPLAY_INDUSTRY_BASELINES}"
GROUP="${GROUP:-FRED_16country_monthly}"
PROBLEMS="${PROBLEMS:-${GROUP}/industry_eu_svn,${GROUP}/industry_eu_deu,${GROUP}/industry_eu_ita}"
SEEDS="${SEEDS:-42 43 44}"
REPLAY_TOP3="${REPLAY_TOP3:-1}"
PROFILE="${PROFILE:-historical_best}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
N_RUNS="${N_RUNS:-5}"
EXTRA_OVERRIDES=("$@")

# The three lowest Industry records are mlruns/43/eccf... (.41898),
# mlruns/43/3adb... (.44116), and mlruns/43/30ab... (.46502); all three use
# lr=.25, wd=.25, hidden_dim=32, depth=2, n_outer=10, n_inner=100.  Thus
# there are not three distinct top settings to pretend to recover.  The two
# additional profiles below are the actual low-LR ALM/SPBM sweeps in
# exdbn_dnn.o23759276 and exdbn_dnn.o23759275, included to test stability.
PROFILE_NAMES=(historical_best low_lr_alm low_lr_spbm)

profile_values() {
  case "$1" in
    historical_best) printf '0.25 0.25 32 2 10 100' ;;
    low_lr_alm)      printf '0.01 0.25 32 2 10 100' ;;
    low_lr_spbm)     printf '0.01 0.01 32 2 10 100' ;;
    *)
      echo "Unknown PROFILE=$1 (use historical_best, low_lr_alm, or low_lr_spbm)." >&2
      exit 2
      ;;
  esac
}

if [[ "${REPLAY_TOP3}" == "1" ]]; then
  PROFILES=("${PROFILE_NAMES[@]}")
else
  PROFILES=("${PROFILE}")
fi

read -r -a SEED_LIST <<< "${SEEDS//,/ }"
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS must contain at least one integer." >&2
  exit 2
fi

run_hydra() {
  local method="$1"
  local solver="$2"
  local backend="$3"
  local profile="$4"
  local seed="$5"
  shift 5

  local experiment="${EXPERIMENT_PREFIX}_${profile}_${method}_seed${seed}"
  echo
  echo "=== ${experiment} ==="
  echo "Industry run: recalculate_dag=false, problems=${PROBLEMS}"

  local -a command=(
    "${PYTHON_BIN}" run_experiments.py --multirun
    "--config-name=${CONFIG_NAME}"
    "experiment=${experiment}"
    "solver=${solver}"
    "problem=${PROBLEMS}"
  )
  if (( $# > 0 )); then
    command+=("$@")
  fi
  if (( ${#EXTRA_OVERRIDES[@]} > 0 )); then
    command+=("${EXTRA_OVERRIDES[@]}")
  fi
  HC_CONSTRAINT_BACKEND="${backend}" "${command[@]}"
}

for profile in "${PROFILES[@]}"; do
  read -r lr wd hidden depth n_outer n_inner <<< "$(profile_values "${profile}")"
  for seed in "${SEED_LIST[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }

    common_hc=(
      "solver.n_runs=${N_RUNS}"
      "solver.learning_rate=${lr}"
      "solver.weight_decay=${wd}"
      "solver.hidden_dim=${hidden}"
      "solver.depth=${depth}"
      "solver.n_outer=${n_outer}"
      "solver.n_inner=${n_inner}"
      "solver.time_limit=${TIME_LIMIT}"
      "solver.recalculate_dag=false"
      "solver.random_state=${seed}"
      "solver.cv_random_state=$((seed + 10000))"
      "solver.validation_random_state=$((seed + 20000))"
      "solver.constrained=true"
      "solver.use_w_constraints=true"
      "solver.prediction_loss=mse"
      "solver.cv_strategy=site_gender"
      "solver.cv_time_test_size=null"
      "solver.cv_time_gap=0"
    )
    run_hydra alm hc_predictor alm "${profile}" "${seed}" "${common_hc[@]}"
    run_hydra spbm hc_predictor spbm "${profile}" "${seed}" "${common_hc[@]}"

    common_tree=(
      "solver.n_runs=${N_RUNS}"
      "solver.recalculate_dag=false"
      "+solver.cv_strategy=site_gender"
      "+solver.cv_time_test_size=null"
      "+solver.cv_time_gap=0"
      "solver.random_state=${seed}"
    )
    run_hydra mark mark alm "${profile}" "${seed}" "${common_tree[@]}"
    run_hydra mark_cc mark_with_cc alm "${profile}" "${seed}" \
      "solver.n_outer=${n_outer}" "solver.time_limit=${TIME_LIMIT}" "${common_tree[@]}"
  done
done

echo
echo "Industry replay completed."
echo "Problems: ${PROBLEMS}"
echo "Profiles: ${PROFILES[*]}"
echo "Methods: ALM, SPBM, MARK, MARK-CC"
