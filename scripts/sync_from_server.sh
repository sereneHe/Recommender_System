#!/usr/bin/env bash
# Sync experiment evidence from the MetaCentrum server into the local mirror.
# Only evidence artifacts are pulled (yaml/csv/md); never the whole tree.
#
#   SERVER_HOST=tarkil SERVER_PATH=/storage/brno2/home/hexiaoyu/Recommender_Pavel/metacentrum_runs \
#     bash scripts/sync_from_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER_HOST="${SERVER_HOST:-tarkil}"
SERVER_PATH="${SERVER_PATH:-/storage/brno2/home/hexiaoyu/Recommender_Pavel/metacentrum_runs}"
LOCAL_DIR="${LOCAL_DIR:-metacentrum_runs}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15}"

mkdir -p "$LOCAL_DIR"

echo "[sync] $SERVER_HOST:$SERVER_PATH -> $LOCAL_DIR"
if rsync -az --partial \
    -e "ssh $SSH_OPTS" \
    --include='*/' --include='*.yaml' --include='*.csv' --include='*.md' --exclude='*' \
    "$SERVER_HOST:$SERVER_PATH/" "$LOCAL_DIR/"; then
  echo "[sync] ok  files=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')"
else
  echo "[sync] FAILED (server unreachable?) — keeping existing local mirror" >&2
  exit 2
fi

# Sync receipt: G0 may only audit a mirror whose freshness is provable.  The
# receipt binds the refresh id, the remote snapshot time, and a hash of the
# pulled file manifest (relative path + size + mtime per file).
"${PYTHON_BIN:-python3}" - "$LOCAL_DIR" "${SYNC_REFRESH_ID:-}" "${SERVER_HOST}" "${SERVER_PATH}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

local_dir, refresh_id, host, server_path = sys.argv[1:5]
entries = []
total_bytes = 0
for p in sorted(Path(local_dir).rglob("*")):
    if p.is_file():
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append(f"{p.relative_to(local_dir)} {st.st_size} {int(st.st_mtime)}")
        total_bytes += st.st_size
manifest_hash = hashlib.sha256("\n".join(entries).encode()).hexdigest()
receipt = {
    "schema_version": 1,
    "origin": "remote",
    "status": "ok",
    "refresh_id": refresh_id or None,
    "synced_at_utc": datetime.now(timezone.utc).isoformat(),
    "server_host": host,
    "server_path": server_path,
    "local_dir": local_dir,
    "file_count": len(entries),
    "total_bytes": total_bytes,
    "manifest_hash": manifest_hash,
}
out = Path("reports/sync_receipt.json")
out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[sync] receipt {out} files={len(entries)} manifest={manifest_hash[:12]}")
PY
