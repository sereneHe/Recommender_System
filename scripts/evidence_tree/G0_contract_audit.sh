#!/usr/bin/env bash
# Evidence-tree axis G0: fair-comparison contract audit (NO training).
#
# Nodes: G0.1_metric_schema_valid, G0.2_complete_artifact,
# G0.3_exact_split_hash, G0.4_nuisance_config_hash, G0.5_fold_W_cache_hash,
# G0.6_frozen_selection_receipt.
#
# This script runs the scanner and then reports which contract fields are
# present per run.  It never trains.  G0.3 / G0.5 / G0.6 are not implemented
# yet (exact split hash, fold-W cache hash, frozen selection receipt); the
# script reports them as missing rather than faking them.
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

"${PYTHON_BIN}" - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("reports/progress/runs_metrics.csv")
df = pd.read_csv(p, low_memory=False)
required = {
    "G0.1_metric_schema_valid": ["metric_validity", "nmse", "n_folds_nmse"],
    "G0.2_complete_artifact": ["has_valid_artifact", "is_canonical"],
    "G0.3_exact_split_hash": ["split_hash"],
    "G0.4_nuisance_config_hash": ["nuisance_hash", "resolved_config_hash"],
    "G0.5_fold_W_cache_hash": ["fold_w_cache_hash"],
    "G0.6_frozen_selection_receipt": ["frozen_selection_receipt"],
}
print("contract fields present per node:")
for node, cols in required.items():
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    print(f"  {node}: present={present} missing={missing}")
PY

echo "=== G0 contract audit complete: batch=${EVIDENCE_BATCH_ID} ==="
