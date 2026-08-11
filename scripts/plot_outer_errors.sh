#!/bin/sh
set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/Users/xiaoyuhe/Recommender_Pavel}"
PYTHON="${PYTHON:-python3}"

RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/multirun/2026-07-01/11-19-21/0}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_ROOT}/mlruns/44/ed0bd195634e4d60bf365723144eb698/artifacts}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/reports/CODIET}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-HDL_train_valid_test_vs_outer_mean_sd}"
PLOT_TITLE="${PLOT_TITLE:-HDL: train, validation, and final test error vs outer}"

cd "${PROJECT_ROOT}"

"${PYTHON}" "plot_outer_errors.py" --run-dir "${RUN_DIR}" --artifact-dir "${ARTIFACT_DIR}" --output-dir "${OUTPUT_DIR}" --output-prefix "${OUTPUT_PREFIX}" --title "${PLOT_TITLE}"
