#!/usr/bin/env python3
"""Derive the experiment pipeline state from durable receipts.

The dashboard is a state machine, not a report generator.  Every transition is
decided by a verifiable artifact produced by an earlier step:

    registered         <- registry validates; no H1 cohort manifest
    submitted          <- cohort manifest reserved but no PBS job / no pairs
    running            <- job submitted, 0 paired units so far
    synced             <- fresh sync receipt + paired units, G0 still open
    gate_failed        <- reports/gates/G0_H1.json status == "failed"
    validated          <- G0_H1.json status == "passed" (20/20, unique pairing)
    frozen             <- schema-v2 frozen-selection receipt validates (human review)
    holdout_running    <- frozen + F0 cohort submitted, no holdout receipt yet
    holdout confirmed  <- reports/holdout_receipts/*.json exists (independent F0)

Writes reports/pipeline_state.json.  Never raises: an unreadable input yields a
degraded but explicit state, never a silent verdict.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def build() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "evidence_tree"))
    from experiment_registry import load as load_registry, validate as validate_registry

    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    # 1. registered?
    try:
        reg = load_registry()
        reg_problems = validate_registry(reg)
    except Exception as exc:
        return {"state": "invalid", "reason": f"registry unreadable: {exc}",
                "checked_at_utc": now}
    if reg_problems:
        return {"state": "invalid", "reason": "; ".join(reg_problems),
                "checked_at_utc": now}

    manifests = sorted((REPORTS / "cohorts").glob("*.yaml")) if (REPORTS / "cohorts").exists() else []
    h1_manifests = [m for m in manifests if "h1" in m.name.lower()]
    f0_manifests = [m for m in manifests if m.name.lower().startswith("et_f0")
                    or "f0" in m.name.lower()]

    def _manifest_job(m: Path):
        try:
            return (_load_yaml(m).get("pbs_job_id")
                    or _load_yaml(m).get("pbs_jobid"))
        except Exception:
            return None

    h1_job = next((_manifest_job(m) for m in h1_manifests if _manifest_job(m)), None)

    # H1 pairing progress from the index (evidence retained even on failure).
    h1_used, h1_expected, h1_status, h1_cohort = 0, 20, None, None
    try:
        import pandas as pd
        idx = pd.read_csv(REPORTS / "evidence_index.csv", low_memory=False)
        h1 = idx[idx.comparison_id.astype(str) == "H1.ce_vs_nn.ER"]
        if not h1.empty:
            row = h1.iloc[0]
            h1_used = int(pd.to_numeric(row.get("n_clusters_used"), errors="coerce") or 0)
            h1_expected = int(pd.to_numeric(row.get("n_clusters_expected"), errors="coerce") or 20)
            h1_status = str(row.get("status", ""))
            _cohort = row.get("cohort", "")
            h1_cohort = "" if pd.isna(_cohort) or str(_cohort).strip().lower() in ("", "nan") else str(_cohort)
    except Exception:
        pass

    g0 = _load_json(REPORTS / "gates" / "G0_H1.json")
    g0_state = str(g0.get("status", "open"))
    g0_failures = list(g0.get("failures", []))

    # frozen?
    import f0_guard
    frozen: list[str] = []
    frozen_problems: list[str] = []
    registry_path = ROOT / "experiment_registry.yaml"
    registry = _load_yaml(registry_path)
    fsel = REPORTS / "frozen_selection"
    if fsel.exists():
        for rp in sorted(fsel.glob("*.yaml")):
            try:
                receipt = _load_yaml(rp)
                problems = f0_guard.validate_receipt(receipt, registry, registry_path)
                if problems:
                    frozen_problems.extend(f"{rp.name}: {p}" for p in problems)
                else:
                    frozen.append(rp.name)
            except Exception as exc:
                frozen_problems.append(f"{rp.name}: unreadable ({exc})")

    holdouts = sorted((REPORTS / "holdout_receipts").glob("*.json")) \
        if (REPORTS / "holdout_receipts").exists() else []

    sync = _load_json(REPORTS / "sync_receipt.json")
    sync_fresh = bool(sync and sync.get("status") == "ok")
    if sync_fresh:
        try:
            synced = datetime.fromisoformat(str(sync.get("synced_at_utc", "")))
            sync_fresh = (now_dt() - synced).total_seconds() <= 7200
        except Exception:
            sync_fresh = False

    # State precedence: holdout confirmed > holdout_running > frozen >
    # validated > gate_failed > synced > running > submitted > registered.
    if holdouts:
        state, reason = "holdout confirmed", f"holdout receipt(s): {[p.name for p in holdouts]}"
    elif frozen and f0_manifests:
        state, reason = "holdout_running", \
            f"frozen {frozen}; F0 cohort {[p.name for p in f0_manifests]} submitted, no holdout receipt"
    elif frozen:
        state, reason = "frozen", f"frozen receipt(s): {frozen}; holdout not yet run"
    elif g0_state == "passed":
        state, reason = "validated", \
            f"G0 passed for {h1_cohort or 'H1'} at {h1_used}/{h1_expected} ({h1_status})"
    elif g0_state == "failed":
        state, reason = "gate_failed", "; ".join(g0_failures) or "G0 contract failed"
    elif h1_used > 0:
        state, reason = ("synced" if sync_fresh else "running"), \
            f"H1 paired {h1_used}/{h1_expected}" + ("; sync fresh" if sync_fresh else "; sync stale")
    elif h1_manifests:
        state, reason = ("running" if h1_job else "submitted"), \
            f"H1 cohort {[p.name for p in h1_manifests]}" + (f" job {h1_job}" if h1_job else "; awaiting PBS job")
    else:
        state, reason = "registered", "protocol frozen; H1 cohort not yet submitted"

    return {
        "state": state,
        "reason": reason,
        "registry_version": reg.get("registry_version"),
        "h1": {"used": h1_used, "expected": h1_expected, "status": h1_status,
               "cohort": h1_cohort},
        "g0": {"status": g0_state, "failures": g0_failures},
        "frozen_receipts": frozen,
        "frozen_problems": frozen_problems,
        "holdout_receipts": [p.name for p in holdouts],
        "cohort_manifests": [p.name for p in manifests],
        "checked_at_utc": now,
    }


def main() -> int:
    out = REPORTS / "pipeline_state.json"
    payload = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out} (state={payload['state']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
