#!/usr/bin/env bash
# A3 audit: compare linear partial correlation with residual-HSIC and soft-CMI
# on the true ER/SF nonlinear SEM.  This is an oracle/statistic audit only; it
# must not be reported as a trained KCI/HSIC predictor until a differentiable
# training backend exists.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/python_runtime.sh
project_python_require "${ROOT}"
GRAPH_TYPES="${GRAPH_TYPES:-ER SF}"
GRAPH_SEEDS="${GRAPH_SEEDS:-42 43 44 45 46}"
NOISE_SEEDS="${NOISE_SEEDS:-101 102}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/evidence_ci/${EVIDENCE_BATCH_ID:-a3_nonlinear_ci}}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_NODES="${N_NODES:-10}"
EXPECTED_EDGES="${EXPECTED_EDGES:-15}"
MAX_COND="${MAX_COND:-2}"
MAX_TRIPLES="${MAX_TRIPLES:-250}"
TARGET_FPR="${TARGET_FPR:-0.05}"

mkdir -p "${OUTPUT_DIR}"
for graph_type in ${GRAPH_TYPES}; do
  out="${OUTPUT_DIR}/${graph_type}_nonlinear_summary.csv"
  echo "[A3] ${graph_type}/nonlinear -> ${out}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi
  "${PYTHON_BIN}" scripts/causal_predictor_plan/audit_nonlinear_ci.py \
    --graph-type "${graph_type}" --sem-type nonlinear \
    --n-samples "${N_SAMPLES}" --n-nodes "${N_NODES}" \
    --expected-edges "${EXPECTED_EDGES}" --max-cond "${MAX_COND}" \
    --max-triples "${MAX_TRIPLES}" \
    --graph-seeds ${GRAPH_SEEDS} --noise-seeds ${NOISE_SEEDS} \
    --target-fpr "${TARGET_FPR}" --output "${out}"
done
echo "A3 audit complete.  It tests statistic quality, not KCI/HSIC training." 
