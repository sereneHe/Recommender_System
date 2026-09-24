#!/usr/bin/env bash
# Strict dashboard refresh order for the recommender-pavel progress tree.
#
#   0) sync evidence from the MetaCentrum server (tarkil)   [optional, non-fatal]
#   1) completed-job check        (PBS/multirun completeness)
#   2) metric integrity / scale   -> reports/progress/metric_integrity.md   (gate)
#   3) scan metacentrum_runs      -> reports/progress/runs_metrics.csv
#   4) build evidence pairs       -> reports/evidence_pairs.csv
#   5) strict evidence index      -> reports/evidence_index.csv + nodes + exclusions
#   6) progress tree / dashboard  -> reports/progress/progress_tree.md
#
# If step 2 fails, the dashboard must show `invalid` and NOT emit verdicts.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/python_runtime.sh
project_python_require "$(pwd)"
PY="${PYTHON_BIN}"

# One refresh = one sync receipt.  G0 only consumes the receipt carrying THIS
# id, so a stale mirror can never be audited as if it were fresh.
REFRESH_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SYNC_REFRESH_ID="${REFRESH_ID}"

if [ "${SKIP_SYNC:-0}" != "1" ] && [ -f scripts/sync_from_server.sh ]; then
  echo "[0/6] sync from server ..."
  bash scripts/sync_from_server.sh || {
    echo "[refresh] result sync failed; refusing to audit or publish conclusions" >&2
    exit 1
  }
else
  echo "[0/6] local-only mirror (SKIP_SYNC=1); writing local sync receipt ..."
  "$PY" - "${REFRESH_ID}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "scripts/evidence_tree")
from g0_check import manifest_hash
refresh_id = sys.argv[1]
digest, count, total = manifest_hash("metacentrum_runs")
receipt = {
    "schema_version": 1,
    "origin": "local",
    "status": "ok",
    "refresh_id": refresh_id,
    "synced_at_utc": datetime.now(timezone.utc).isoformat(),
    "server_host": None,
    "server_path": None,
    "local_dir": "metacentrum_runs",
    "file_count": count,
    "total_bytes": total,
    "manifest_hash": digest,
}
Path("reports/sync_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[sync] local receipt files={count} manifest={digest[:12]}")
PY
fi

echo "[1/6] completed-job check ..."
find metacentrum_runs -name "cv_errors.yaml" -o -name "cv_fold_metrics.csv" | wc -l | xargs echo "  metric artifacts:"

echo "[2/6] metric integrity / scale check ..."
if ! "$PY" scripts/check_metric_integrity.py; then
  echo "[refresh] METRIC INTEGRITY FAILED -> dashboard status = invalid (no verdicts)" >&2
  exit 1
fi

echo "[3/6] scan metacentrum_runs ..."
"$PY" scripts/scan_progress_tree.py

echo "[4/7] build evidence pairs + strict index ..."
"$PY" scripts/build_evidence_index.py
echo "[5/7] G0 contract audit ..."
G0_STATUS=passed
if ! "$PY" scripts/evidence_tree/g0_check.py \
      --index reports/evidence_index.csv \
      --registry experiment_registry.yaml \
      --receipt-out reports/gates/G0_H1.json \
      --sync-receipt reports/sync_receipt.json \
      --expect-refresh-id "${REFRESH_ID}" \
      --mirror-dir metacentrum_runs; then
  G0_STATUS=failed
fi
echo "$G0_STATUS" > reports/gates/G0_H1.status
echo "[6/8] build evidence nodes ..."
"$PY" scripts/build_evidence_nodes.py
"$PY" scripts/render_evidence_tree.py

echo "[7/8] pipeline state machine ..."
"$PY" scripts/evidence_tree/pipeline_state.py

echo "[8/8] progress tree -> reports/progress/progress_tree.md  ·  dashboard -> reports/progress/dashboard.html"
if [ "$G0_STATUS" != "passed" ]; then
  echo "[refresh] evidence retained and dashboard updated with G0=${G0_STATUS}; no strict conclusion published" >&2
  exit 1
fi
echo "[refresh] done; G0 passed."
