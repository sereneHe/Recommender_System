#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

TARGETS=(
  "GLU (mg/dL)"
  "HDL (mg/dL)"
  "LDL (mg/dL)"
  "TRIG (mg/dL)"
  "HbA1c (%)"
  "Systolic Blood Pressure (mm Hg)"
  "Diastolic Blood Pressure (mm Hg)"
  "CRP (mg/dL)"
)

MODELS=(
  "REG"
  "DT"
  "RF"
  "KerREG"
  "GP"
  "XGB"
)

N_RUNS="${N_RUNS:-1}"
N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}"

for target in "${TARGETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== Running model=${model} target=${target} n_runs=${N_RUNS} n_select_features=${N_SELECT_FEATURES}"
    PYTHONPATH="${ROOT_DIR}/src" python -m hei_project.train \
      solver.model_name="${model}" \
      solver.n_runs="${N_RUNS}" \
      solver.n_select_features="${N_SELECT_FEATURES}" \
      targets.columns="[\"${target}\"]"
  done
done
