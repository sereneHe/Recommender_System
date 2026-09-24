#!/usr/bin/env python3
"""Hard fair-comparison contract audit for the evidence index.

This is the logic behind ``G0_contract_audit.sh``.  It is a pure function of the
built ``reports/evidence_index.csv`` plus the frozen ``experiment_registry.yaml``
so it can be unit-tested without running the scanner.

It exits non-zero (via ``main``) when ANY of the following fails for a frozen
comparison:

* G0.1  the metric schema columns exist;
* G0.2  the artifact is complete (expected == used units, no missing arm);
* G0.3  every paired comparison shares the exact split receipt;
* G0.4  every paired comparison shares the nuisance configuration;
* G0.5  every W-applicable pair shares the fold-W cache receipt;
* G0.6  a frozen-selection receipt exists (checked by the shell);
*       no duplicate/excluded pairing exists within the frozen cohort.

Legacy comparisons are reported but not gated: they are diagnostic records and
must not be presented as strict evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def as_bool(series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def _present(series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return ~text.isin(("", "nan", "none", "nat"))


def manifest_hash(mirror_dir: str | Path) -> tuple[str, int, int]:
    """Hash the pulled-mirror file manifest (relpath + size + mtime per file).

    MUST match the manifest computation in scripts/sync_from_server.sh and the
    SKIP_SYNC branch of scripts/refresh_progress_tree.sh.
    """
    entries = []
    total_bytes = 0
    root = Path(mirror_dir)
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append(f"{p.relative_to(root)} {st.st_size} {int(st.st_mtime)}")
                total_bytes += st.st_size
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return digest, len(entries), total_bytes


def validate_sync_receipt(receipt_path: str | Path,
                          expect_refresh_id: str | None = None,
                          max_age_sec: int = 7200,
                          mirror_dir: str | Path | None = None) -> tuple[list[str], dict]:
    """Check that G0 audits a fresh, complete sync.  Returns (problems, receipt).

    A missing, failed, expired, refresh-id-mismatched, or manifest-mismatched
    receipt is a hard G0 failure: auditing a stale mirror would let an
    incomplete batch look like clean evidence.
    """
    problems: list[str] = []
    rp = Path(receipt_path)
    if not rp.exists():
        return ([f"sync_freshness: sync receipt missing: {rp} "
                  "(run the refresh sync step first)"], {})
    try:
        receipt = json.loads(rp.read_text(encoding="utf-8"))
    except Exception as exc:
        return ([f"sync_freshness: sync receipt unreadable: {exc}"], {})
    if receipt.get("status") != "ok":
        problems.append(f"sync_freshness: sync status is {receipt.get('status')!r}, not ok")
    if expect_refresh_id and receipt.get("refresh_id") != expect_refresh_id:
        problems.append(
            "sync_freshness: receipt refresh_id "
            f"{receipt.get('refresh_id')!r} != current refresh {expect_refresh_id!r} "
            "(G0 only consumes the receipt of THIS refresh)")
    try:
        synced = datetime.fromisoformat(str(receipt.get("synced_at_utc", "")))
        age = (datetime.now(timezone.utc) - synced).total_seconds()
        if age < 0 or age > max_age_sec:
            problems.append(
                f"sync_freshness: receipt age {age:.0f}s exceeds {max_age_sec}s; re-sync")
    except Exception:
        problems.append("sync_freshness: receipt has no parseable synced_at_utc")
    if mirror_dir is not None:
        digest, count, _ = manifest_hash(mirror_dir)
        if receipt.get("manifest_hash") and receipt["manifest_hash"] != digest:
            problems.append(
                f"sync_freshness: mirror changed after sync "
                f"(receipt files={receipt.get('file_count')}, now files={count}); re-sync")
    return problems, receipt


def audit_index(idx: pd.DataFrame, registry: dict) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)`` for the frozen comparisons in the index."""
    failures: list[str] = []
    notes: list[str] = []

    required = {"comparison_id", "n_clusters_used", "n_clusters_expected",
                "rel_improvement", "ci_lo", "ci_hi", "status"}
    missing = sorted(required - set(idx.columns))
    if missing:
        failures.append(f"G0.1_metric_schema_valid: index lacks columns {missing}")
        return failures, notes
    notes.append("G0.1_metric_schema_valid: required metric columns present")

    hyps = registry.get("hypotheses") or {}
    frozen_ids = {
        h["id"]: name for name, h in hyps.items()
        if h.get("comparison_type") != "holdout" and not h.get("allow_diagnostic_only")
    }

    used = pd.to_numeric(idx.get("n_clusters_used"), errors="coerce").fillna(0)
    paired = idx[used > 0]

    if paired.empty:
        failures.append("no paired comparison exists; split/W receipts cannot be verified")
        return failures, notes

    for cid in sorted(set(idx.comparison_id.astype(str)) & set(frozen_ids)):
        name = frozen_ids[cid]
        h = hyps[name]
        sub = idx[idx.comparison_id.astype(str) == cid]
        sub_paired = paired[paired.comparison_id.astype(str) == cid]
        min_units = int(h["min_independent_units"])

        if sub_paired.empty:
            failures.append(f"G0.2_complete_artifact: {cid} has no paired units")
            continue

        row = sub_paired.iloc[0]
        used_n = int(pd.to_numeric(row.get("n_clusters_used"), errors="coerce") or 0)
        exp_n = int(pd.to_numeric(row.get("n_clusters_expected"), errors="coerce") or 0)
        excluded = int(pd.to_numeric(row.get("n_excluded"), errors="coerce") or 0)
        if used_n < min_units or exp_n < min_units:
            failures.append(
                f"G0.2_complete_artifact: {cid} has {used_n}/{exp_n} units but the "
                f"protocol requires {min_units}"
            )
        elif used_n != exp_n:
            failures.append(
                f"G0.2_complete_artifact: {cid} expected {exp_n} units but paired {used_n}"
            )
        else:
            notes.append(f"G0.2_complete_artifact: {cid} complete at {used_n} units")

        if excluded > 0:
            failures.append(
                f"unique_pairing: {cid} has {excluded} duplicate/excluded pairings"
            )

        # G0.3 exact split + receipt presence.
        if "split_hash_equal" not in sub_paired.columns:
            failures.append(f"G0.3_exact_split_hash: {cid} lacks split_hash_equal")
        elif not as_bool(sub_paired["split_hash_equal"]).all():
            failures.append(f"G0.3_exact_split_hash: split mismatch on {cid}")
        elif "split_receipts_present" in sub_paired.columns and not as_bool(
            sub_paired["split_receipts_present"]
        ).all():
            failures.append(f"G0.3_exact_split_hash: {cid} is missing a split receipt")
        else:
            notes.append(f"G0.3_exact_split_hash: {cid} shares the exact split")

        # G0.4 nuisance.
        if "nuisance_hash_equal" not in sub_paired.columns:
            failures.append(f"G0.4_nuisance_config_hash: {cid} lacks nuisance_hash_equal")
        elif not as_bool(sub_paired["nuisance_hash_equal"]).all():
            failures.append(f"G0.4_nuisance_config_hash: nuisance mismatch on {cid}")
        else:
            notes.append(f"G0.4_nuisance_config_hash: {cid} shares the nuisance config")

        # G0.5 fold-W receipt, only where W is a shared nuisance.
        if "fold_w_cache_hash_equal" not in sub_paired.columns:
            failures.append(f"G0.5_fold_W_cache_hash: {cid} lacks fold_w_cache_hash_equal")
        else:
            applicable = (
                as_bool(sub_paired["fold_w_cache_applicable"])
                if "fold_w_cache_applicable" in sub_paired.columns
                else pd.Series(True, index=sub_paired.index)
            )
            bad = sub_paired.loc[applicable & ~as_bool(sub_paired["fold_w_cache_hash_equal"])]
            if not bad.empty:
                failures.append(f"G0.5_fold_W_cache_hash: W-cache mismatch on {cid}")
            elif "fold_w_receipts_present" in sub_paired.columns and not as_bool(
                sub_paired["fold_w_receipts_present"]
            ).all():
                failures.append(f"G0.5_fold_W_cache_hash: {cid} is missing a fold-W receipt")
            else:
                notes.append(
                    f"G0.5_fold_W_cache_hash: {cid} W-applicable pairs share the W cache"
                )

    return failures, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(ROOT / "reports" / "evidence_index.csv"))
    parser.add_argument("--registry", default=str(ROOT / "experiment_registry.yaml"))
    parser.add_argument("--receipt-out", default=str(ROOT / "reports" / "gates" / "G0_H1.json"))
    parser.add_argument("--sync-receipt", default=str(ROOT / "reports" / "sync_receipt.json"))
    parser.add_argument("--expect-refresh-id", default=None,
                        help="only accept the sync receipt written by THIS refresh")
    parser.add_argument("--max-age-sec", type=int, default=7200)
    parser.add_argument("--mirror-dir", default=str(ROOT / "metacentrum_runs"))
    parser.add_argument("--skip-sync-check", action="store_true",
                        help="audit without a sync receipt (unit tests only; never in refresh)")
    args = parser.parse_args(argv)

    import yaml

    sync_problems: list[str] = []
    sync_receipt: dict = {}
    if not args.skip_sync_check:
        sync_problems, sync_receipt = validate_sync_receipt(
            args.sync_receipt, args.expect_refresh_id, args.max_age_sec,
            args.mirror_dir)

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"G0 CONTRACT AUDIT FAILED: {idx_path} missing", file=sys.stderr)
        _write_receipt(Path(args.receipt_out), "open", [f"evidence index missing: {idx_path}"], None, None)
        return 1
    idx = pd.read_csv(idx_path, low_memory=False)
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8")) or {}
    failures, notes = audit_index(idx, registry)
    failures = sync_problems + failures
    for n in notes:
        print(n)
    h1 = idx[idx.comparison_id.astype(str) == "H1.ce_vs_nn.ER"] if "comparison_id" in idx else idx.iloc[0:0]
    h1_used = 0
    h1_expected = 0
    h1_cohort = ""
    if not h1.empty:
        row = h1.iloc[0]
        h1_used = int(pd.to_numeric(row.get("n_clusters_used"), errors="coerce") or 0)
        h1_expected = int(pd.to_numeric(row.get("n_clusters_expected"), errors="coerce") or 0)
        _cohort = row.get("cohort", "")
        h1_cohort = "" if pd.isna(_cohort) or str(_cohort).strip().lower() in ("", "nan") else str(_cohort)
    state = "passed" if not failures else ("open" if h1_used == 0 else "failed")
    _write_receipt(
        Path(args.receipt_out), state, failures, h1_cohort or None,
        {"used": h1_used, "expected": h1_expected},
        notes=notes, index_path=idx_path,
        sync={"refresh_id": sync_receipt.get("refresh_id"),
              "synced_at_utc": sync_receipt.get("synced_at_utc"),
              "origin": sync_receipt.get("origin"),
              "manifest_hash": (sync_receipt.get("manifest_hash") or "")[:12]},
    )
    if failures:
        print("G0 CONTRACT AUDIT FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("G0 contract audit passed")
    return 0


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + atomic rename so a concurrent reader never sees
    a half-written receipt (two refreshes racing must not corrupt the JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_receipt(path: Path, state: str, failures: list[str], cohort: str | None,
                   units: dict | None, notes: list[str] | None = None,
                   index_path: Path | None = None,
                   sync: dict | None = None) -> None:
    payload = {
        "schema_version": 1,
        "gate": "G0",
        "target_hypothesis": "H1.ce_vs_nn.ER",
        "status": state,
        "cohort_id": cohort,
        "units": units or {"used": 0, "expected": 20},
        "failures": failures,
        "notes": notes or [],
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()
        if index_path and index_path.exists() else None,
        "sync": sync or {},
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
