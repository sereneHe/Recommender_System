#!/usr/bin/env python3
"""Audit whether legacy cohorts are strict single-variable contrasts.

The H1 protocol requires the reference and candidate arms to differ ONLY in the
declared treatment (the CE penalty).  A legacy pair that also changes an
uncontrolled key (dropout, graph estimation, optimizer, seed handling, ...) is
not a strict single-variable contrast and must be rerun in the new H1 cohort
rather than promoted into a confirmatory claim.

Because the evidence builder stratifies pairs by the nuisance hash, a pair that
forms inside a common nuisance stratum is single-variable BY CONSTRUCTION; a
comparison that produces no common stratum (or is excluded as
``invalid_configuration``) changed more than the treatment.

Usage:
    python scripts/evidence_tree/audit_legacy_single_variable.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def audit(index: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in index.itertuples():
        cid = str(r.comparison_id)
        used = int(pd.to_numeric(getattr(r, "n_clusters_used", 0), errors="coerce") or 0)
        excl = int(pd.to_numeric(getattr(r, "n_excluded", 0), errors="coerce") or 0)
        nuis = str(getattr(r, "nuisance_hash_equal", "")).lower() == "true"
        cfg_bad = 0
        if not exclusions.empty:
            sel = exclusions[exclusions.comparison_id.astype(str) == cid]
            cfg_bad = int(sel.reason.astype(str).str.startswith("invalid_configuration").sum())
        if used > 0 and nuis and cfg_bad == 0:
            verdict = "strict_single_variable"
        elif used > 0:
            verdict = "pairs_but_unverified"
        else:
            verdict = "not_single_variable_or_missing"
        rows.append({
            "comparison_id": cid,
            "n_clusters_used": used,
            "n_excluded": excl,
            "invalid_configuration": cfg_bad,
            "nuisance_hash_equal": nuis,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(ROOT / "reports" / "evidence_index.csv"))
    parser.add_argument("--exclusions", default=str(ROOT / "reports" / "evidence_exclusions.csv"))
    args = parser.parse_args(argv)

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"missing {idx_path}; run the refresh first", file=sys.stderr)
        return 1
    index = pd.read_csv(idx_path, low_memory=False)
    excl_path = Path(args.exclusions)
    if excl_path.exists() and excl_path.stat().st_size > 0:
        try:
            exclusions = pd.read_csv(excl_path, low_memory=False)
        except pd.errors.EmptyDataError:
            exclusions = pd.DataFrame()
    else:
        exclusions = pd.DataFrame()

    report = audit(index, exclusions)
    legacy = report[report.comparison_id.str.contains(r"priority|pilot|repair|ce_vs_nn\.ER", regex=True)]
    print(report.to_string(index=False))
    print()
    rerun = legacy[legacy.verdict != "strict_single_variable"]
    if rerun.empty:
        print("All inspected legacy contrasts are strict single-variable.")
    else:
        print("Rerun these in the new H1 cohort (not strict single-variable):")
        for cid in rerun.comparison_id:
            print(f"  - {cid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
