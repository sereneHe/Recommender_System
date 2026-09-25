#!/usr/bin/env bash
# Internal helper; submit one of 03--10_fred_*.sh instead.

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

: "${PROBLEMS:?PROBLEMS must be set to two comma-separated FRED problems}"
: "${GRAPH_SEEDS:=42 43 44}"
: "${NOISE_SEEDS:=101}"
: "${FRED_TIME_LIMIT:=120}"

PROBLEMS="${PROBLEMS}" \
GRAPH_SEEDS="${GRAPH_SEEDS}" \
NOISE_SEEDS="${NOISE_SEEDS}" \
TIME_LIMIT="${FRED_TIME_LIMIT}" \
EVIDENCE_BATCH_ID=et_a6_fred_v2 \
  bash "${ROOT}/scripts/evidence_tree/A6_data_time_robustness.sh"
