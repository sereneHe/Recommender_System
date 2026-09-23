#!/usr/bin/env bash
# P1 step 1: audit conditional-independence statistics against the true
# synthetic graph BEFORE any nonlinear CI statistic is put into training.
#
# Runs the audit for each (graph_type, sem_type) pair and writes one summary
# CSV per pair plus a per-triple CSV.  Use the summary to decide which
# statistic, if any, is worth a training arm on that generator.
#
# Usage:
#   bash scripts/causal_predictor_plan/10_audit_nonlinear_ci.sh
#   GRAPH_TYPES="ER" SEM_TYPES="nonlinear" GRAPH_SEEDS="42 43 44" \
#     bash scripts/causal_predictor_plan/10_audit_nonlinear_ci.sh

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GRAPH_TYPES="${GRAPH_TYPES:-ER SF}"
SEM_TYPES="${SEM_TYPES:-gauss nonlinear}"
GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44}"
NOISE_SEEDS="${NOISE_SEEDS:-101}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
MAX_COND="${MAX_COND:-2}"
MAX_TRIPLES="${MAX_TRIPLES:-250}"
TARGET_FPR="${TARGET_FPR:-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ci_audit}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "=== CI statistic audit ==="
echo "graph_types=${GRAPH_TYPES} sem_types=${SEM_TYPES} graph_seeds=${GRAPH_SEEDS} out=${OUTPUT_DIR}"

for graph_type in ${GRAPH_TYPES}; do
  for sem_type in ${SEM_TYPES}; do
    out="${OUTPUT_DIR}/ci_audit_${graph_type}_${sem_type}.csv"
    echo
    echo "--- ${graph_type} / ${sem_type} -> ${out} ---"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN: would run audit for ${graph_type}/${sem_type}"
      continue
    fi
    "${PYTHON_BIN}" scripts/causal_predictor_plan/audit_nonlinear_ci.py \
      --graph-type "${graph_type}" \
      --sem-type "${sem_type}" \
      --graph-seeds ${GRAPH_SEEDS} \
      --noise-seeds ${NOISE_SEEDS} \
      --n-samples "${N_SAMPLES}" \
      --n-nodes "${N_NODES}" \
      --expected-edges "${EXPECTED_EDGES}" \
      --max-cond "${MAX_COND}" \
      --max-triples "${MAX_TRIPLES}" \
      --target-fpr "${TARGET_FPR}" \
      --output "${out}"
  done
done

echo
echo "=== CI statistic audit complete ==="
