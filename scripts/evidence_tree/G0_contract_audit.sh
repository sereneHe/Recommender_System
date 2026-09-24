#!/usr/bin/env bash
# Evidence-tree axis G0: fair-comparison contract audit (NO training).
#
# Nodes: G0.1_metric_schema_valid, G0.2_complete_artifact,
# G0.3_exact_split_hash, G0.4_nuisance_config_hash, G0.5_fold_W_cache_hash,
# G0.6_frozen_selection_receipt.
#
# This is a HARD gate, not a field report.  It rescans, rebuilds the evidence
# index, and then verifies the per-PAIR receipts: the exact split must match on
# every paired comparison, the nuisance config must match, the fold-W cache hash
# must match wherever W is a shared nuisance, the artifact must be complete, and
# no duplicate/excluded pairing may exist inside a frozen cohort.  It exits
# non-zero when any receipt is missing or unequal, so an incomplete batch cannot
# be reported as clean evidence.
#
# Run this ONLY after the result sync has completed, otherwise the index is
# stale and the audit is meaningless.
#
# Submit (cheap; can also run locally):
#   EVIDENCE_BATCH_ID=et_g0_v1 \
#   qsub -v EXPERIMENT_SCRIPT=scripts/evidence_tree/G0_contract_audit.sh,EVIDENCE_BATCH_ID=et_g0_v1 \
#     cluster_computing/run_metacentrum.pbs

EV_AXIS="G0"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../causal_predictor_plan" && pwd)/_common.sh"
announce_stage "G0" "fair-comparison contract audit (no training)"

"${PYTHON_BIN}" scripts/scan_progress_tree.py
"${PYTHON_BIN}" scripts/build_evidence_index.py

# G0.1-G0.5 + unique pairing.  Exits non-zero on any failing contract.
if ! "${PYTHON_BIN}" scripts/evidence_tree/g0_check.py \
      --index reports/evidence_index.csv \
      --registry experiment_registry.yaml \
      --receipt-out reports/gates/G0_H1.json; then
  echo "G0 CONTRACT AUDIT FAILED: the fair-comparison contract is not satisfied." >&2
  echo "Do NOT treat this batch as clean evidence; fix the receipts and rerun." >&2
  exit 1
fi

echo "=== G0 contract audit complete: batch=${EVIDENCE_BATCH_ID} ==="
