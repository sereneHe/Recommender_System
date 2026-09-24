#!/usr/bin/env bash
# Single submission entry point for an evidence cohort.
#
# Instead of running the loose .sh scripts by hand, this wrapper performs the
# whole chain in order and refuses to proceed if any step is not clean:
#
#   1. validate experiment_registry.yaml;
#   2. mint a cohort id EXACTLY ONCE in the atomic cohort registry
#      (local mkdir + server mkdir; a second submission with the same id fails);
#   3. write a cohort manifest (registry hash, seed table hash, script, commit,
#      attempt number, canonical status);
#   4. submit the PBS job and record its job id in the manifest.
#
# Retry semantics: never patch an old cohort.  Retire it
# (experiment_registry.py retire) and submit a NEW cohort that reruns the full
# seed table; pass --supersedes <old> to link them.
#
# Usage:
#   scripts/evidence_tree/submit_cohort.sh --hyp H1 --script scripts/evidence_tree/H1_mechanism.sh
#   scripts/evidence_tree/submit_cohort.sh --hyp H1 --script ... --cohort et_h1_er_v1
#   scripts/evidence_tree/submit_cohort.sh --hyp H1 --script ... --dry-run
#   scripts/evidence_tree/submit_cohort.sh --hyp H1 --script ... --cohort et_h1_er_v2 --supersedes et_h1_er_v1
#   Flags: --remote (default) reserves on the authoritative server first;
#          --local-only skips the server (tests / offline dry runs).
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

HYP=""
COHORT=""
EXPERIMENT_SCRIPT=""
WALLTIME="12:00:00"
PBS_TEMPLATE="cluster_computing/run_metacentrum.pbs"
DRY_RUN=0
SUBMITTER="${USER:-unknown}"
REMOTE=1
SUPERSEDES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hyp) HYP="$2"; shift 2 ;;
    --cohort) COHORT="$2"; shift 2 ;;
    --script) EXPERIMENT_SCRIPT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --pbs) PBS_TEMPLATE="$2"; shift 2 ;;
    --submitter) SUBMITTER="$2"; shift 2 ;;
    --supersedes) SUPERSEDES="$2"; shift 2 ;;
    --remote) REMOTE=1; shift ;;
    --local-only) REMOTE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${HYP}" && -n "${EXPERIMENT_SCRIPT}" ]] || {
  echo "usage: $0 --hyp <H1|F0> --script <path> [--cohort <id>] [--dry-run]" >&2
  exit 2
}
[[ -f "${EXPERIMENT_SCRIPT}" ]] || { echo "script not found: ${EXPERIMENT_SCRIPT}" >&2; exit 2; }

source "${PROJ_DIR}/scripts/python_runtime.sh"
project_python_require "${PROJ_DIR}"

echo "### 1/4 validate registry"
"${PYTHON_BIN}" scripts/experiment_registry.py validate

# Default cohort id: hypothesis + UTC timestamp.  Never regenerated per arm.
if [[ -z "${COHORT}" ]]; then
  COHORT="et_$(echo "${HYP}" | tr '[:upper:]' '[:lower:]')_$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "### DRY RUN — plan only: NO cohort reservation, NO manifest, NO qsub"
  "${PYTHON_BIN}" - "${HYP}" "${COHORT}" "${EXPERIMENT_SCRIPT}" "${WALLTIME}" "${PBS_TEMPLATE}" "${SUBMITTER}" <<'PY'
import hashlib, sys
from pathlib import Path
import yaml

hyp_name, cohort, script, walltime, pbs, submitter = sys.argv[1:7]
root = Path(".").resolve()
reg_path = root / "experiment_registry.yaml"
reg = yaml.safe_load(reg_path.read_text()) or {}
hyp = reg["hypotheses"][hyp_name]
sys.path.insert(0, str(root / "scripts"))
from experiment_registry import (effective_seed_table, seed_table_hash,
                                 seed_table_units, normalize_seed_table,
                                 SERVER_HOST, SERVER_REGISTRY_DIR)

table = normalize_seed_table(effective_seed_table(hyp))
units = seed_table_units(table)
arms = [a for a in (hyp.get("reference_arm"), hyp.get("candidate_arm")) if a]
cmds = units * max(len(arms), 1)
declared = hyp.get("seed_table_hash")
computed = seed_table_hash(effective_seed_table(hyp))

print(f"hypothesis        : {hyp['id']}  ({hyp.get('title','')})")
print(f"cohort id         : {cohort}")
print(f"registry version  : {reg.get('registry_version')}")
print(f"registry hash     : {hashlib.sha256(reg_path.read_bytes()).hexdigest()[:16]}")
print(f"seed-table hash   : {computed}")
print(f"hash matches decl  : {computed == declared}")
print(f"independent units : {units}   (graph {len(table['graph_seeds'])} x noise {len(table['noise_seeds'])})")
print(f"arms              : {arms}")
print(f"total commands    : {cmds}   ({len(arms)} arms x {units} units)")
print(f"split / fold-W    : {hyp.get('split_contract')} / {hyp.get('fold_w_contract')}")
print(f"canonical attempt  : 1  (immutable; a retry must be a NEW cohort, never a reuse)")
print(f"remote reserve dir : {SERVER_HOST}:{SERVER_REGISTRY_DIR.rstrip('/')}/{cohort}")
print(f"                     (atomic mkdir; re-submitting the same id -> RuntimeError, fail-closed)")
print(f"cluster python     : /storage/praha1/home/hexiaoyu/codiet311_linux/bin/python  (selected by run_metacentrum.pbs)")
print(f"PBS command        : qsub -l walltime={walltime} \\")
print(f"                       -v EXPERIMENT_SCRIPT={script},EVIDENCE_BATCH_ID={cohort},HYPOTHESIS={hyp_name} \\")
print(f"                       {pbs}")
print("dry-run guarantees  : no local cohort dir, no manifest, no scheduler job")
PY
  exit 0
fi

echo "### 2/4 reserve cohort ${COHORT} (atomic mkdir; duplicates fail)"
REMOTE_FLAG="--remote"
if [[ "${REMOTE}" == "0" ]]; then
  REMOTE_FLAG="--local-only"
fi
"${PYTHON_BIN}" scripts/experiment_registry.py reserve \
  --hyp "${HYP}" --cohort "${COHORT}" --submitter "${SUBMITTER}" ${REMOTE_FLAG}

MANIFEST_DIR="reports/cohorts"
MANIFEST="${MANIFEST_DIR}/${COHORT}.yaml"
mkdir -p "${MANIFEST_DIR}"

echo "### 3/4 write cohort manifest ${MANIFEST}"
"${PYTHON_BIN}" - "${HYP}" "${COHORT}" "${EXPERIMENT_SCRIPT}" "${MANIFEST}" "${WALLTIME}" "${PBS_TEMPLATE}" "${SUBMITTER}" "${SUPERSEDES:-}" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

hyp_name, cohort, script, manifest_path, walltime, pbs, submitter, supersedes = sys.argv[1:9]
root = Path(".").resolve()
registry_path = root / "experiment_registry.yaml"
registry = yaml.safe_load(registry_path.read_text()) or {}
hyp = registry["hypotheses"][hyp_name]
sys.path.insert(0, str(root / "scripts"))
from experiment_registry import effective_seed_table, seed_table_hash, seed_table_units

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"

manifest = {
    "cohort_id": cohort,
    "hypothesis": hyp_name,
    "hypothesis_id": hyp["id"],
    "scope": hyp["scope"],
    "experiment_script": script,
    "walltime": walltime,
    "pbs_template": pbs,
    "seed_table": effective_seed_table(hyp),
    "seed_table_hash": seed_table_hash(effective_seed_table(hyp)),
    "expected_units": seed_table_units(effective_seed_table(hyp)),
    "fold_w_contract": hyp["fold_w_contract"],
    "split_contract": hyp["split_contract"],
    "registry_version": registry.get("registry_version"),
    "registry_hash": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    "git_commit": commit,
    "submitter": submitter,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "pbs_job_id": None,
    # Canonical-attempt declaration: this manifest IS attempt 1 of an immutable
    # cohort.  A retry never edits this file; it retires this cohort and mints
    # a new one that reruns the FULL seed table (see --supersedes).
    "attempt": 1,
    "status": "active",
    "supersedes": supersedes or None,
}
Path(manifest_path).write_text(yaml.safe_dump(manifest, sort_keys=False))
print(f"wrote {manifest_path}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "### 4/4 dry-run: skipping retire + qsub"
  echo "cohort=${COHORT} manifest=${MANIFEST}"
  exit 0
fi

# A retry retires the old cohort FIRST so it can never re-enter strict
# inference, even if the new submission later fails.
if [[ -n "${SUPERSEDES}" ]]; then
  echo "### retiring superseded cohort ${SUPERSEDES}"
  "${PYTHON_BIN}" scripts/experiment_registry.py retire \
    --cohort "${SUPERSEDES}" --reason "superseded by ${COHORT}" \
    --superseded-by "${COHORT}"
fi

echo "### 4/4 submit PBS"
if ! JOB_ID="$(qsub -l "walltime=${WALLTIME}" \
  -v "EXPERIMENT_SCRIPT=${EXPERIMENT_SCRIPT},EVIDENCE_BATCH_ID=${COHORT},HYPOTHESIS=${HYP}" \
  "${PBS_TEMPLATE}" 2>&1)"; then
  echo "qsub FAILED: ${JOB_ID}" >&2
  echo "retiring cohort ${COHORT} so a partial submission never lingers as active" >&2
  "${PYTHON_BIN}" scripts/experiment_registry.py retire \
    --cohort "${COHORT}" --reason "qsub failed; no job submitted" 2>/dev/null || true
  exit 1
fi
echo "qsub job id: ${JOB_ID}"

"${PYTHON_BIN}" - "${MANIFEST}" "${JOB_ID}" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
m = yaml.safe_load(p.read_text()) or {}
m["pbs_job_id"] = sys.argv[2].strip()
p.write_text(yaml.safe_dump(m, sort_keys=False))
print(f"recorded pbs_job_id={m['pbs_job_id']} in {p}")
PY
echo "=== submitted cohort ${COHORT} (job ${JOB_ID}) ==="
