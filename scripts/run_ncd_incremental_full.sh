#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

export MODEL_NAME="${MODEL_NAME:-DT}"
export N_RUNS="${N_RUNS:-150}"
export N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}"
export N_EXPERIMENTS="${N_EXPERIMENTS:-20}"
export MAX_WORKERS="${MAX_WORKERS:-1}"
export DO_PERMUTE="${DO_PERMUTE:-False}"
export USE_LIPIDOMICS="${USE_LIPIDOMICS:-True}"
export TARGET_COLUMNS="${TARGET_COLUMNS:-GLU (mg/dL),HDL (mg/dL),LDL (mg/dL),TRIG (mg/dL),HbA1c (%),Systolic Blood Pressure (mm Hg),Diastolic Blood Pressure (mm Hg),CRP (mg/dL),whtr(waist-height_ratio)}"

mkdir -p "${ROOT_DIR}/reports/results/pkl_outs"
exec ./scripts/NCD_analysis_incremental_features.sh

