#!/usr/bin/env python3
"""Loader / validator / cohort registrar for `experiment_registry.yaml`.

The registry is the single source of truth for:
  * each hypothesis' reference/candidate arm pair,
  * scope, primary/effect metric and the pre-declared practical threshold,
  * the minimum number of independent units and the unit definition,
  * the exact seed table and its hash,
  * the split / fold-W contracts.

`build_evidence_index.py` imports `load`, `validate`, `effective_seed_table`
and `seed_table_units` from this module.

Cohort identity is minted through an ATOMIC directory registry
(`reports/cohort_registry/<cohort_id>/`).  Creating the directory IS the
reservation: `mkdir` is atomic on the local filesystem and (over ssh) on the
single authoritative server, so two concurrent submissions for the same id
cannot both succeed.  A plain local JSONL append is NOT sufficient (two
machines can append the same id) and is not used.

Retry semantics: an interrupted H1 attempt is never patched.  The old cohort is
marked `diagnostic_ineligible` via `retire_cohort` and the retry is a brand-new
immutable cohort that reruns the full seed table.

CLI:
    python scripts/experiment_registry.py validate
    python scripts/experiment_registry.py hash H1
    python scripts/experiment_registry.py reserve --hyp H1 --cohort et_h1_er_v1 --remote
    python scripts/experiment_registry.py retire --cohort et_h1_er_v1 --reason interrupted --superseded-by et_h1_er_v2
    python scripts/experiment_registry.py show H1
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiment_registry.yaml"
# Atomic directory registry: one subdirectory per cohort id.
COHORT_REGISTRY_DIR = ROOT / "reports" / "cohort_registry"
# Cohort manifests (written by submit_cohort.sh) carry the pre-declared
# canonical status: active | diagnostic_ineligible.
COHORT_MANIFESTS_DIR = ROOT / "reports" / "cohorts"

SCHEMA_VERSION = 1
VALID_CONTRACTS = {"same", "none", "candidate", "both"}
REQUIRED_HYP_KEYS = (
    "id", "scope", "comparison_type", "primary_metric", "effect_metric",
    "practical_threshold", "min_independent_units", "unit_definition",
    "split_contract", "fold_w_contract",
)

SERVER_HOST = os.environ.get("SERVER_HOST", "tarkil")
SERVER_PROJECT = os.environ.get(
    "SERVER_PROJECT", "/storage/brno2/home/hexiaoyu/Recommender_Pavel")
SERVER_REGISTRY_DIR = os.environ.get(
    "SERVER_REGISTRY_DIR", f"{SERVER_PROJECT}/reports/cohort_registry")
SSH_OPTS = os.environ.get("SSH_OPTS", "-o BatchMode=yes -o ConnectTimeout=15")


def seed_table_hash(table: dict) -> str:
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load(path: str | Path = REGISTRY_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def effective_seed_table(hyp: dict) -> dict:
    """The seed table a hypothesis actually draws from."""
    return hyp.get("seed_table") or hyp.get("reserved_holdout_seed_table") or {}


def seed_table_units(table: dict) -> int:
    if not table:
        return 0
    return len(table.get("graph_seeds") or []) * len(table.get("noise_seeds") or [])


def normalize_seed_table(seed_table: dict) -> dict:
    """Validate a seed table; preserve the frozen order (never re-sort)."""
    if not isinstance(seed_table, dict):
        raise ValueError("seed table must be a mapping")
    graph = [int(x) for x in (seed_table.get("graph_seeds") or [])]
    noise = [int(x) for x in (seed_table.get("noise_seeds") or [])]
    if not graph or not noise:
        raise ValueError("seed table must list at least one graph seed and one noise seed")
    if len(set(graph)) != len(graph) or len(set(noise)) != len(noise):
        raise ValueError("seed table contains duplicate seeds")
    return {"graph_seeds": graph, "noise_seeds": noise}


def seed_combinations(seed_table: dict) -> list[tuple[int, int]]:
    st = normalize_seed_table(seed_table)
    return [(g, n) for g in st["graph_seeds"] for n in st["noise_seeds"]]


def hypothesis(reg: dict, name: str) -> dict:
    hyps = reg.get("hypotheses") or {}
    if name not in hyps:
        raise KeyError(f"unknown hypothesis {name!r}; known: {sorted(hyps)}")
    return hyps[name]


def validate(reg: dict) -> list[str]:
    """Return a list of problems; empty means the registry is usable."""
    problems: list[str] = []
    try:
        if int(reg.get("schema_version", 0)) != SCHEMA_VERSION:
            problems.append(f"schema_version must be {SCHEMA_VERSION}")
    except Exception:
        problems.append("schema_version is not an integer")

    hyps = reg.get("hypotheses") or {}
    if not hyps:
        problems.append("registry defines no hypotheses")

    for key, h in hyps.items():
        if not isinstance(h, dict):
            problems.append(f"{key}: hypothesis is not a mapping")
            continue
        for k in REQUIRED_HYP_KEYS:
            if k not in h:
                problems.append(f"{key}: missing required key '{k}'")
        if h.get("split_contract") not in VALID_CONTRACTS:
            problems.append(f"{key}: invalid split_contract {h.get('split_contract')!r}")
        if h.get("fold_w_contract") not in VALID_CONTRACTS:
            problems.append(f"{key}: invalid fold_w_contract {h.get('fold_w_contract')!r}")
        try:
            pt = float(h["practical_threshold"])
            if not (0.0 <= pt <= 1.0):
                problems.append(f"{key}: practical_threshold out of [0,1]")
        except Exception:
            problems.append(f"{key}: practical_threshold is not a number")
        try:
            if int(h["min_independent_units"]) < 1:
                problems.append(f"{key}: min_independent_units must be >= 1")
        except Exception:
            problems.append(f"{key}: min_independent_units is not an integer")
        table = effective_seed_table(h)
        if not table:
            problems.append(f"{key}: no seed_table / reserved_holdout_seed_table")
        else:
            try:
                units = seed_table_units(table)
            except Exception:
                problems.append(f"{key}: malformed seed table")
                continue
            if units < int(h.get("min_independent_units") or 0):
                problems.append(
                    f"{key}: seed table provides {units} units but "
                    f"min_independent_units is {h.get('min_independent_units')}")
            declared = h.get("seed_table_hash") or h.get("reserved_holdout_seed_table_hash")
            if declared and seed_table_hash(table) != declared:
                problems.append(
                    f"{key}: seed-table hash mismatch "
                    f"(declared {str(declared)[:12]}, computed {seed_table_hash(table)[:12]})"
                )
        reserved = h.get("reserved_holdout_seed_table")
        if reserved is not None:
            if seed_table_units(reserved) < int(h.get("min_independent_units") or 0):
                problems.append(f"{key}: reserved holdout table smaller than min units")
            claimed = h.get("reserved_holdout_seed_table_hash")
            if claimed and seed_table_hash(reserved) != claimed:
                problems.append(f"{key}: reserved_holdout_seed_table_hash mismatch")
    # H1 and F0 must not share any seed unit.
    h1 = hyps.get("H1")
    f0 = hyps.get("F0")
    if (isinstance(h1, dict) and isinstance(f0, dict)
            and f0.get("reserved_holdout_seed_table")):
        try:
            overlap = (set(seed_combinations(effective_seed_table(h1)))
                       & set(seed_combinations(f0["reserved_holdout_seed_table"])))
            if overlap:
                problems.append(f"H1 and F0 share holdout seed units: {sorted(overlap)}")
        except ValueError:
            pass
    return problems


def _record_path(registry_dir: Path | str, cohort_id: str) -> Path:
    return Path(registry_dir) / cohort_id / "record.json"


def _ssh_mkdir(host: str, remote_dir: str, ssh_opts: str,
               _runner=None) -> tuple[bool, str]:
    """Atomically `mkdir <remote_dir>` on the server.

    Returns (reserved, message).  rc==0 means this caller won the reservation.
    rc==1 with the directory present means a duplicate id.  Anything else
    (e.g. ssh rc==255) is fail-closed: the caller must NOT proceed.
    """
    run = _runner or (lambda cmd: subprocess.run(
        cmd, capture_output=True, text=True, timeout=60))
    cmd = ["ssh", *shlex.split(ssh_opts), host, "mkdir", remote_dir]
    try:
        r = run(cmd)
    except Exception as exc:
        return False, f"ssh failed: {exc}"
    if r.returncode == 0:
        return True, "reserved on server"
    probe = ["ssh", *shlex.split(ssh_opts), host, "test", "-d", remote_dir]
    try:
        p = run(probe)
    except Exception as exc:
        return False, f"ssh probe failed: {exc}"
    if p.returncode == 0:
        return False, "duplicate cohort id on server"
    return False, f"remote mkdir failed (rc={r.returncode}): {(r.stderr or '').strip()}"


def reserve_cohort(
    hyp_name: str,
    cohort_id: str,
    submitter: str = "unknown",
    registry_path: Path | str = REGISTRY_PATH,
    registry_dir: Path | str = COHORT_REGISTRY_DIR,
    remote: bool = False,
    server_host: str | None = None,
    server_registry_dir: str | None = None,
    ssh_opts: str | None = None,
    _ssh_runner=None,
) -> dict:
    """Mint a cohort id exactly once.

    The reservation IS the atomic `mkdir`: locally always, plus on the single
    authoritative server when ``remote=True`` (checked first; a remote failure
    is fail-closed).  A duplicate id raises ``RuntimeError``; the caller must
    mint a new immutable cohort instead of reusing the id.
    """
    reg = load(registry_path)
    hyp = hypothesis(reg, hyp_name)
    cohort_id = str(cohort_id).strip()
    if not cohort_id or "/" in cohort_id or cohort_id in (".", ".."):
        raise ValueError(f"invalid cohort id: {cohort_id!r}")
    if remote:
        host = server_host or SERVER_HOST
        remote_dir = f"{(server_registry_dir or SERVER_REGISTRY_DIR).rstrip('/')}/{cohort_id}"
        ok, msg = _ssh_mkdir(host, remote_dir, ssh_opts or SSH_OPTS,
                             _runner=_ssh_runner)
        if not ok:
            raise RuntimeError(f"remote reserve failed for {cohort_id!r}: {msg}")
    path = Path(registry_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        (path / cohort_id).mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"cohort id {cohort_id!r} is already registered; mint a new id "
            f"(a retry must be a new immutable cohort, never a reuse)")
    record = {
        "cohort_id": cohort_id,
        "hypothesis": hyp_name,
        "hypothesis_id": hyp["id"],
        "scope": hyp["scope"],
        "seed_table_hash": seed_table_hash(effective_seed_table(hyp)),
        "expected_units": seed_table_units(effective_seed_table(hyp)),
        "registry_version": reg.get("registry_version"),
        "status": "active",
        "submitter": submitter,
        "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _record_path(path, cohort_id).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def retire_cohort(
    cohort_id: str,
    reason: str,
    superseded_by: str | None = None,
    registry_dir: Path | str = COHORT_REGISTRY_DIR,
) -> dict:
    """Mark a cohort `diagnostic_ineligible` (interrupted / superseded).

    The old cohort keeps its artifacts for diagnosis but can never enter strict
    inference again.  A retry must be a new cohort rerunning the FULL seed table.
    """
    rp = _record_path(registry_dir, cohort_id)
    if not rp.exists():
        raise KeyError(f"unknown cohort id: {cohort_id!r}")
    record = json.loads(rp.read_text(encoding="utf-8"))
    record["status"] = "diagnostic_ineligible"
    record["ineligible_reason"] = reason
    record["superseded_by"] = superseded_by
    record["retired_at_utc"] = datetime.now(timezone.utc).isoformat()
    rp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
    manifest = Path(COHORT_MANIFESTS_DIR) / f"{cohort_id}.yaml"
    if manifest.exists():
        try:
            import yaml as _yaml
            m = _yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            m["status"] = "diagnostic_ineligible"
            m["ineligible_reason"] = reason
            m["superseded_by"] = superseded_by
            manifest.write_text(_yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
        except Exception:
            pass
    return record


def cohort_status(cohort_id: str,
                  registry_dir: Path | str = COHORT_REGISTRY_DIR) -> str:
    """active | diagnostic_ineligible | unregistered (no record = legacy)."""
    rp = _record_path(registry_dir, str(cohort_id))
    if not rp.exists():
        return "unregistered"
    try:
        return json.loads(rp.read_text(encoding="utf-8")).get("status", "active")
    except Exception:
        return "unregistered"


def diagnostic_patterns(reg: dict) -> list[str]:
    """Cohort-id glob patterns that are diagnostic-only (never strict)."""
    out = []
    for rec in reg.get("diagnostic_records") or []:
        if isinstance(rec, dict) and rec.get("strict_inference_allowed") is False:
            out.append(str(rec.get("id", "")))
    return [p for p in out if p]


def is_cohort_eligible_for_strict(cohort_id: str, reg: dict,
                                  registry_dir: Path | str = COHORT_REGISTRY_DIR) -> bool:
    """A cohort enters strict (frozen-protocol) inference only when it has an
    ACTIVE reservation and matches no diagnostic-only pattern."""
    cid = str(cohort_id)
    if cohort_status(cid, registry_dir) != "active":
        return False
    return not any(fnmatch.fnmatch(cid, p) for p in diagnostic_patterns(reg))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate")

    p_hash = sub.add_parser("hash")
    p_hash.add_argument("hyp")

    p_res = sub.add_parser("reserve")
    p_res.add_argument("--hyp", required=True)
    p_res.add_argument("--cohort", required=True)
    p_res.add_argument("--submitter", default="unknown")
    p_res.add_argument("--registry-dir", default=str(COHORT_REGISTRY_DIR))
    p_res.add_argument("--remote", action="store_true",
                       help="reserve atomically on the authoritative server first")
    p_res.add_argument("--local-only", action="store_true",
                       help="skip the server even if --remote defaults change")
    p_res.add_argument("--server-host", default=None)
    p_res.add_argument("--server-registry-dir", default=None)

    p_ret = sub.add_parser("retire")
    p_ret.add_argument("--cohort", required=True)
    p_ret.add_argument("--reason", required=True)
    p_ret.add_argument("--superseded-by", default=None)
    p_ret.add_argument("--registry-dir", default=str(COHORT_REGISTRY_DIR))

    p_show = sub.add_parser("show")
    p_show.add_argument("hyp")

    args = parser.parse_args(argv)
    if args.cmd == "validate":
        problems = validate(load())
        if problems:
            for p in problems:
                print(f"INVALID: {p}", file=sys.stderr)
            return 1
        print("experiment_registry: valid")
        return 0
    if args.cmd == "hash":
        hyp = hypothesis(load(), args.hyp)
        print(f"seed_table_hash={seed_table_hash(effective_seed_table(hyp))}")
        if hyp.get("reserved_holdout_seed_table"):
            print("reserved_holdout_seed_table_hash="
                  f"{seed_table_hash(hyp['reserved_holdout_seed_table'])}")
        return 0
    if args.cmd == "reserve":
        rec = reserve_cohort(args.hyp, args.cohort, args.submitter,
                             registry_dir=args.registry_dir,
                             remote=args.remote and not args.local_only,
                             server_host=args.server_host,
                             server_registry_dir=args.server_registry_dir)
        print(json.dumps(rec, sort_keys=True))
        return 0
    if args.cmd == "retire":
        rec = retire_cohort(args.cohort, args.reason, args.superseded_by,
                            registry_dir=args.registry_dir)
        print(json.dumps(rec, sort_keys=True))
        return 0
    if args.cmd == "show":
        print(json.dumps(hypothesis(load(), args.hyp), indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
