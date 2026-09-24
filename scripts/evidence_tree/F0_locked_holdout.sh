#!/usr/bin/env bash
# Evidence-tree axis F0: frozen selection + locked holdout.
#
# F0 is a guarded gate.  It runs only when a schema-v2 frozen-selection receipt
# validates, the runner is a plain reviewed executable, and every holdout seed
# lies inside the reserved table frozen in experiment_registry.yaml.  A real F0
# runner must:
#   1. refit the frozen configuration exactly once on the reserved holdout;
#   2. report raw MSE, fold-normalised MSE, MAE, per-target results,
#      constraint counts/violations, runtime and failure rate.
# F0 does NOT pick a winner: the selection receipt is created by a human after
# reviewing the H1/validation results.
#
# This script refuses to run until those conditions hold, so an accidental
# submission cannot be mistaken for a final held-out result.

EV_AXIS="F0"
EV_SCOPE="synthetic/ER"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

RECEIPT="${FROZEN_RECEIPT:-reports/frozen_selection/${FROZEN_SCOPE:-synthetic_ER}.yaml}"
if [[ ! -f "${RECEIPT}" ]]; then
  echo "F0 blocked: missing frozen selection receipt ${RECEIPT}" >&2
  exit 2
fi
# Receipt must be schema v2, hash-intact, validation-only, and bound to the
# current registry and reserved seed table.
"${PYTHON_BIN}" scripts/evidence_tree/f0_guard.py check-receipt --receipt "${RECEIPT}" || exit 2

# The runner must be a plain executable file, never an arbitrary shell string.
if [[ -z "${LOCKED_HOLDOUT_COMMAND:-}" ]]; then
  echo "F0 blocked: set LOCKED_HOLDOUT_COMMAND to a reviewed, independent holdout runner script." >&2
  exit 2
fi
"${PYTHON_BIN}" scripts/evidence_tree/f0_guard.py check-command --command "${LOCKED_HOLDOUT_COMMAND}" || exit 2

# Holdout seeds default to the reserved table; any explicit override is checked
# against it, so a caller cannot shop for a luckier seed set.
F0_SEED_LINES="$("${PYTHON_BIN}" - "${RECEIPT}" <<'PY'
import sys, yaml
from pathlib import Path
r = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
t = r["reserved_holdout_seed_table"]
print(" ".join(str(x) for x in t["graph_seeds"]))
print(" ".join(str(x) for x in t["noise_seeds"]))
PY
)"
F0_GRAPH_DEFAULT="$(printf '%s\n' "${F0_SEED_LINES}" | sed -n 1p)"
F0_NOISE_DEFAULT="$(printf '%s\n' "${F0_SEED_LINES}" | sed -n 2p)"
HOLDOUT_GRAPH_SEEDS="${HOLDOUT_GRAPH_SEEDS:-${F0_GRAPH_DEFAULT}}"
HOLDOUT_NOISE_SEEDS="${HOLDOUT_NOISE_SEEDS:-${F0_NOISE_DEFAULT}}"
"${PYTHON_BIN}" scripts/evidence_tree/f0_guard.py check-seeds \
  --graph "${HOLDOUT_GRAPH_SEEDS}" --noise "${HOLDOUT_NOISE_SEEDS}" || exit 2

export HOLDOUT_GRAPH_SEEDS HOLDOUT_NOISE_SEEDS FROZEN_RECEIPT="${RECEIPT}"
read -r -a HOLD_CMD <<< "${LOCKED_HOLDOUT_COMMAND}"
echo "Executing reviewed locked-holdout runner ${HOLD_CMD[0]} from frozen receipt ${RECEIPT}."
echo "Reserved holdout seeds: graph=[${HOLDOUT_GRAPH_SEEDS}] noise=[${HOLDOUT_NOISE_SEEDS}]"
"${HOLD_CMD[@]}"
HOLD_EXIT=$?
if [[ ${HOLD_EXIT} -ne 0 ]]; then
  echo "F0 blocked: holdout runner exited ${HOLD_EXIT}; no holdout receipt written." >&2
  exit ${HOLD_EXIT}
fi

# The runner succeeded on the full reserved table.  Bind the outcome to the
# cohort + frozen receipt so the dashboard can show "holdout confirmed" with
# provenance instead of a bare claim.
"${PYTHON_BIN}" - "${RECEIPT}" "${EVIDENCE_BATCH_ID}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

receipt_path, cohort = Path(sys.argv[1]), sys.argv[2]
receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8")) or {}
core = {k: v for k, v in receipt.items() if k != "receipt_hash"}
payload = {
    "schema_version": 1,
    "gate": "F0",
    "cohort_id": cohort,
    "scope": receipt.get("scope"),
    "selected_config_hash": receipt.get("selected_config_hash"),
    "frozen_receipt_hash": receipt.get("receipt_hash"),
    "frozen_receipt_name": receipt_path.name,
    "reserved_holdout_seed_table": receipt.get("reserved_holdout_seed_table"),
    "registry_hash": receipt.get("registry_hash"),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["holdout_receipt_hash"] = hashlib.sha256(encoded).hexdigest()
out = Path("reports/holdout_receipts") / f"{cohort}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote holdout receipt {out} (hash={payload['holdout_receipt_hash']})")
PY
echo "=== F0 locked holdout complete: batch=${EVIDENCE_BATCH_ID} ==="
