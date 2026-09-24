#!/usr/bin/env bash
# E2E smoke test for the closed-loop evidence pipeline.
#
# It proves the pipeline is wired end-to-end WITHOUT submitting a PBS job,
# reserving a real cohort, or writing any H1 result:
#
#   1. protocol loads and validates;
#   2. `submit_cohort.sh --dry-run --remote` prints the plan only (no reserve,
#      no manifest, no qsub);
#   3. the remote registry root exists and its cohort-subdir mkdir is atomic
#      (success / duplicate / unreachable-fail-closed) using a clearly-marked
#      throwaway probe that is removed immediately;
#   4. no cohort was reserved locally or remotely by this run;
#   5. a fresh sync receipt passes and a stale / mismatched one fails G0;
#   6. the dashboard API payload and the frontend state machine cover all
#      pipeline states;
#   7. .gitignore ignores runtime products but keeps the frozen receipts.
#
# Usage: bash scripts/evidence_tree/smoke_e2e.sh
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

source scripts/python_runtime.sh
project_python_require "${PROJ_DIR}" || { echo "SMOKE FAIL: no project python"; exit 1; }

FAILED=0
ok()   { echo "  ok   $*"; }
bad()  { echo "  FAIL $*" >&2; FAILED=1; }

echo "== 1/7 protocol =="
if "${PYTHON_BIN}" scripts/experiment_registry.py validate >/tmp/smoke_reg.txt 2>&1; then
  ok "registry valid: $(cat /tmp/smoke_reg.txt)"
else
  bad "registry did not validate"; cat /tmp/smoke_reg.txt >&2
fi

echo "== 2/7 submit dry-run (plan only) =="
PLAN="$(bash scripts/evidence_tree/submit_cohort.sh --hyp H1 \
        --script scripts/evidence_tree/H1_mechanism.sh --dry-run --remote 2>&1)"
for needle in "independent units : 20" "qsub -l walltime" \
              "no local cohort dir, no manifest, no scheduler job" \
              "atomic mkdir; re-submitting the same id"; do
  if grep -qF "${needle}" <<<"${PLAN}"; then ok "plan contains: ${needle}"
  else bad "plan missing: ${needle}"; fi
done

echo "== 3/7 remote registry root + atomic mkdir =="
REGDIR="${SERVER_REGISTRY_DIR:-/storage/brno2/home/hexiaoyu/Recommender_Pavel/reports/cohort_registry}"
PROBE="${REGDIR}/_smoke_probe_$$"
if ssh -o BatchMode=yes -o ConnectTimeout=15 "${SERVER_HOST:-tarkil}" "test -d '${REGDIR}'"; then
  ok "remote registry root exists: ${REGDIR}"
else
  bad "remote registry root missing: ${REGDIR} (init it before submitting H1)"
fi
ssh -o BatchMode=yes -o ConnectTimeout=15 "${SERVER_HOST:-tarkil}" "mkdir '${PROBE}'" 2>/dev/null \
  && ok "atomic mkdir succeeded" || bad "atomic mkdir failed"
if ssh -o BatchMode=yes -o ConnectTimeout=15 "${SERVER_HOST:-tarkil}" "mkdir '${PROBE}'" 2>/dev/null; then
  bad "duplicate mkdir unexpectedly succeeded"
else
  ok "duplicate mkdir rejected"
fi
ssh -o BatchMode=yes -o ConnectTimeout=15 "${SERVER_HOST:-tarkil}" "rmdir '${PROBE}'" 2>/dev/null \
  && ok "probe cleaned up" || bad "probe cleanup failed"
if ssh -o BatchMode=yes -o ConnectTimeout=30 no-such-host.invalid "mkdir /tmp/nope" 2>/dev/null; then
  bad "unreachable host unexpectedly succeeded"
else
  ok "unreachable host fail-closed"
fi

echo "== 4/7 no cohort reserved by this run =="
REMOTE_N="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${SERVER_HOST:-tarkil}" \
            "ls -A '${REGDIR}' 2>/dev/null | wc -l")"
if [[ "${REMOTE_N}" == "0" ]]; then ok "remote registry empty (no H1 cohort)"
else bad "remote registry has ${REMOTE_N} entries"; fi
if grep -q "smoke_probe" <<<"$(ls reports/cohorts 2>/dev/null || true)"; then
  bad "smoke probe leaked a local manifest"
else ok "no smoke manifest written"; fi

echo "== 5/7 sync receipt freshness =="
"${PYTHON_BIN}" - <<'PY'
import json, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "scripts/evidence_tree")
import g0_check

def rec(**over):
    p = Path(tempfile.mkdtemp()) / "sync.json"
    payload = {"schema_version": 1, "origin": "remote", "status": "ok",
               "refresh_id": "r1", "synced_at_utc": datetime.now(timezone.utc).isoformat(),
               "manifest_hash": "x", "file_count": 0}
    payload.update(over)
    p.write_text(json.dumps(payload))
    return p

checks = []
checks.append(("fresh passes", g0_check.validate_sync_receipt(rec(), "r1")[0] == []))
checks.append(("stale rejected", g0_check.validate_sync_receipt(
    rec(synced_at_utc="2000-01-01T00:00:00+00:00"), max_age_sec=60)[0] != []))
checks.append(("wrong refresh_id rejected", g0_check.validate_sync_receipt(rec(), "OTHER")[0] != []))
checks.append(("failed status rejected", g0_check.validate_sync_receipt(rec(status="failed"))[0] != []))
for name, good in checks:
    print(("  ok   " if good else "  FAIL ") + name)
sys.exit(0 if all(g for _, g in checks) else 1)
PY
[[ $? -eq 0 ]] || bad "sync freshness checks failed"

echo "== 6/7 API payload + frontend state machine =="
"${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import serve_evidence_dashboard as dash
s = dash.state()
need_api = {"pipeline", "gates", "sync", "cohort_registry", "cohort_manifests",
            "holdout_receipts", "frozen_selection", "nodes", "comparisons",
            "exclusions", "pairs"}
missing = sorted(need_api - set(s))
print(("  ok   " if not missing else "  FAIL ") + f"api keys (missing={missing})")
html = Path("reports/progress/dashboard.html").read_text(encoding="utf-8")
states = ["registered","submitted","running","synced","gate_failed","validated",
          "frozen","holdout_running","holdout_confirmed"]
miss = [x for x in states if x not in html]
print(("  ok   " if not miss else "  FAIL ") + f"frontend states (missing={miss})")
print(("  ok   " if "renderGateFailures" in html else "  FAIL ") + "gate-failure drill-down")
sys.exit(0 if not missing and not miss and "renderGateFailures" in html else 1)
PY
[[ $? -eq 0 ]] || bad "API/frontend checks failed"

echo "== 7/7 version-control boundary =="
check_ignored() {  # <path> <expected: ignored|tracked>
  if git check-ignore -q "$1"; then
    [[ "$2" == "ignored" ]] && ok "ignored: $1" || bad "should be tracked but ignored: $1"
  else
    [[ "$2" == "tracked" ]] && ok "tracked: $1" || bad "should be ignored but tracked: $1"
  fi
}
check_ignored "reports/evidence_index.csv" ignored
check_ignored "reports/evidence_pairs.csv" ignored
check_ignored "reports/evidence_nodes.csv" ignored
check_ignored "reports/gates/G0_H1.json" ignored
check_ignored "reports/cohorts/et_h1.yaml" ignored
check_ignored "reports/cohort_registry/et_h1/record.json" ignored
check_ignored "reports/holdout_receipts/et_f0.json" ignored
check_ignored "reports/sync_receipt.json" ignored
check_ignored "reports/frozen_selection/synthetic_ER.yaml" tracked

echo
if [[ "${FAILED}" == "0" ]]; then
  echo "=== E2E SMOKE PASS (no cohort reserved, no job submitted) ==="
  exit 0
fi
echo "=== E2E SMOKE FAILED ===" >&2
exit 1
