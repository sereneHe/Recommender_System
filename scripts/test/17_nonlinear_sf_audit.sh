#!/usr/bin/env bash
# Task 17 — nonlinear ER + SF CI-statistic audit (statistic quality only).
#
# Split out of task 11 so nonlinear/sem-type audits never share an allocation
# (or a batch id) with FRED / CoDiet / noise families.  This audits whether the
# linear partial-correlation / residual-HSIC / soft-CMI statistics behave on a
# known nonlinear SEM; it does NOT train a KCI/HSIC predictor.
#
# Still NOT covered here (method not implemented): A3.KCI_HSIC training backend,
# A4.stability_selection (classical B-bootstrap), A4.non_gaussian_moments.
#
# Submit:
#   qsub -l walltime=12:00:00 \
#     -v EXPERIMENT_SCRIPT=scripts/test/17_nonlinear_sf_audit.sh \
#     cluster_computing/run_metacentrum.pbs
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
DRY_RUN="${DRY_RUN:-0}"
BATCH="${EVIDENCE_BATCH_ID:-et_a3_nonlinear_sf_v1}"
: "${GRAPH_SEEDS:=42 43 44 45 46}"
: "${NOISE_SEEDS:=101 102}"

echo "### nonlinear ER + SF CI audit  batch=${BATCH}  graph_seeds=${GRAPH_SEEDS} noise_seeds=${NOISE_SEEDS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: would run A3_nonlinear_ci_audit.sh GRAPH_TYPES='ER SF'"
  exit 0
fi
EVIDENCE_BATCH_ID="${BATCH}" GRAPH_TYPES="ER SF" \
  GRAPH_SEEDS="${GRAPH_SEEDS}" NOISE_SEEDS="${NOISE_SEEDS}" \
  OUTPUT_DIR="reports/evidence_ci/${BATCH}" \
  bash scripts/evidence_tree/A3_nonlinear_ci_audit.sh

echo "[NOT IMPLEMENTED] A3.KCI_HSIC training backend / A4.stability_selection / A4.non_gaussian_moments: no method yet."
