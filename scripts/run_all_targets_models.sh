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

N_RUNS="${N_RUNS:-150}"
N_SELECT_FEATURES="${N_SELECT_FEATURES:-2}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

is_completed() {
  local model="$1"
  local target="$2"
  PYTHONPATH="${ROOT_DIR}/src" python - <<'PY' "${ROOT_DIR}" "${model}" "${target}" "${N_RUNS}" "${N_SELECT_FEATURES}"
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

root_dir = Path(sys.argv[1])
model = sys.argv[2]
target = sys.argv[3]
n_runs = sys.argv[4]
n_select_features = sys.argv[5]

tracking_uri = (root_dir / "mlruns").resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

target_value = f"['{target}']"
for exp in client.search_experiments():
    if exp.name != "codiet_recommender":
        continue
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and params.model_name = '{model}'",
        max_results=5000,
    )
    for run in runs:
        if (
            run.data.params.get("targets") == target_value
            and run.data.params.get("n_runs") == n_runs
            and run.data.params.get("n_select_features") == n_select_features
        ):
            raise SystemExit(0)

raise SystemExit(1)
PY
}

for target in "${TARGETS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed "${model}" "${target}"; then
      echo "=== Skipping completed model=${model} target=${target}"
      continue
    fi
    echo "=== Running model=${model} target=${target} n_runs=${N_RUNS} n_select_features=${N_SELECT_FEATURES}"
    PYTHONPATH="${ROOT_DIR}/src" python -m hei_project.train \
      solver.model_name="${model}" \
      solver.n_runs="${N_RUNS}" \
      solver.n_select_features="${N_SELECT_FEATURES}" \
      targets.columns="[\"${target}\"]"
  done
done
