#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_TO_RUN="${SCRIPT_TO_RUN:-${ROOT_DIR}/scripts/NCD_analysis_incremental_features.sh}"
DEGREE_VALUES="${DEGREE_VALUES:-1,2,3,4,5,6,7,8}"
DEGREE_ENV_NAME="${DEGREE_ENV_NAME:-N_SELECT_FEATURES}"
REPEATS="${REPEATS:-3}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/reports/results}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
CSV_OUT="${CSV_OUT:-${RESULTS_DIR}/runningtime_$(date +%Y%m%d_%H%M%S).csv}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

if [[ ! -x "${SCRIPT_TO_RUN}" ]]; then
  echo "[ERR] script not executable or missing: ${SCRIPT_TO_RUN}" >&2
  exit 1
fi

if ! [[ "${REPEATS}" =~ ^[0-9]+$ ]] || [[ "${REPEATS}" -lt 1 ]]; then
  echo "[ERR] REPEATS must be a positive integer, got: ${REPEATS}" >&2
  exit 1
fi

IFS=',' read -r -a DEG_ARR <<< "${DEGREE_VALUES}"
if [[ "${#DEG_ARR[@]}" -eq 0 ]]; then
  echo "[ERR] DEGREE_VALUES is empty" >&2
  exit 1
fi

if [[ ! -f "${CSV_OUT}" ]]; then
  echo "timestamp,degree_env,degree,repeat,runtime_seconds,exit_code,script,log_file" > "${CSV_OUT}"
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] SCRIPT_TO_RUN=${SCRIPT_TO_RUN}"
echo "[INFO] DEGREE_ENV_NAME=${DEGREE_ENV_NAME}"
echo "[INFO] DEGREE_VALUES=${DEGREE_VALUES}"
echo "[INFO] REPEATS=${REPEATS}"
echo "[INFO] CSV_OUT=${CSV_OUT}"

for raw_degree in "${DEG_ARR[@]}"; do
  degree="$(echo "${raw_degree}" | xargs)"
  if [[ -z "${degree}" ]]; then
    continue
  fi

  for ((rep=1; rep<=REPEATS; rep++)); do
    ts="$(date +%Y-%m-%dT%H:%M:%S%z)"
    stamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_DIR}/runningtime_${DEGREE_ENV_NAME}_${degree}_r${rep}_${stamp}.log"

    echo "[RUN] ${DEGREE_ENV_NAME}=${degree} repeat=${rep}/${REPEATS}"

    start_epoch="$(date +%s)"
    set +e
    env "${DEGREE_ENV_NAME}=${degree}" "${SCRIPT_TO_RUN}" > "${log_file}" 2>&1
    exit_code="$?"
    set -e
    end_epoch="$(date +%s)"

    runtime="$((end_epoch - start_epoch))"
    echo "${ts},${DEGREE_ENV_NAME},${degree},${rep},${runtime},${exit_code},${SCRIPT_TO_RUN},${log_file}" >> "${CSV_OUT}"

    if [[ "${exit_code}" -eq 0 ]]; then
      echo "[OK] degree=${degree} repeat=${rep} runtime=${runtime}s"
    else
      echo "[WARN] degree=${degree} repeat=${rep} failed exit=${exit_code} runtime=${runtime}s log=${log_file}" >&2
    fi
  done
done

echo "[DONE] runtime sweep completed"
echo "[DONE] csv: ${CSV_OUT}"
