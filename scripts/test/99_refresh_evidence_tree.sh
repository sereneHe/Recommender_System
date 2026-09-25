#!/usr/bin/env bash
# Run only after all experiment shards have completed and their artifacts are
# present in metacentrum_runs/.  This is bookkeeping/audit, not a model run.

set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "${ROOT}"
SKIP_SYNC=1 bash scripts/refresh_progress_tree.sh
