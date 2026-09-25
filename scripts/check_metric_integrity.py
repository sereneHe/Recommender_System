#!/usr/bin/env python3
"""Metric integrity / scale check (dashboard gate step 2).

Verifies the fix for the double-normalization bug and emits the dashboard
header fields.  If any check fails, the dashboard must show `invalid` instead
of producing support/contradict verdicts.

Writes reports/progress/metric_integrity.md
Exit code 0 = OK, 1 = integrity failure.
"""
from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "progress"
runs = pd.read_csv(OUT / "runs_metrics.csv")

valid = runs[runs.metric_validity == "valid"]
legacy = runs[runs.metric_validity == "legacy_unverified"]
missing = runs[runs.metric_validity == "missing"]

# 1) valid rows must NOT equal the double-normalized legacy value when a
#    normalizer is present (that would mean the bug is back).
bad_scale = []
for _, r in valid.sample(min(60, len(valid)), random_state=0).iterrows():
    d = ROOT / r["run_dir"]
    fm = d / "cv_fold_metrics.csv"
    ce = d / "cv_errors.yaml"
    if not (fm.exists() and ce.exists()):
        continue
    a = pd.read_csv(fm)["test_nmse"].mean()
    b = yaml.safe_load(ce.read_text())["test_errs"]
    b = sum(b) / len(b)
    if abs(a - b) > 1e-9:  # cv_fold_metrics must equal cv_errors (both normalized once)
        bad_scale.append((r["run_dir"], a, b))
    if a < 1e-4:  # implausible for these datasets -> double normalization
        bad_scale.append((r["run_dir"], a, "too_small"))

# 2) dashboard header fields
receipt_root = ROOT / "reports" / "frozen_selection"
receipt_files = sorted(receipt_root.glob("*.yaml")) if receipt_root.exists() else []
valid_receipts = []
for receipt_path in receipt_files:
    try:
        receipt = yaml.safe_load(receipt_path.read_text()) or {}
        claimed = receipt.get("receipt_hash")
        payload = dict(receipt)
        payload.pop("receipt_hash", None)
        expected = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (receipt.get("schema_version") == 1
                and receipt.get("selected_config_hash") in (receipt.get("candidate_config_hashes") or [])
                and receipt.get("holdout_metrics_present") is False
                and claimed == expected):
            valid_receipts.append(receipt_path)
    except Exception:
        continue
frozen = bool(valid_receipts)
locked_holdout = False  # F0 runner has not emitted a locked-holdout receipt
n_pairs_used = n_pairs_expected = 0
try:
    idx = pd.read_csv(ROOT / "reports" / "evidence_index.csv")
    n_pairs_used = int(idx["n_clusters_used"].sum())
    n_pairs_expected = int(idx["n_clusters_expected"].sum())
except Exception:
    pass

lines = ["# Metric integrity / 指标完整性", ""]
lines.append("| field | value |")
lines.append("|---|---|")
lines.append(f"| valid runs | {len(valid)} |")
lines.append(f"| invalid runs (legacy_unverified + missing) | {len(legacy) + len(missing)} |")
lines.append(f"| pair coverage | {n_pairs_used}/{n_pairs_expected} |")
lines.append(f"| frozen config | {frozen} |")
lines.append(f"| locked holdout | {locked_holdout} |")
lines.append(f"| valid selection receipts | {len(valid_receipts)} |")
lines.append(f"| scopes with valid evidence | {', '.join(sorted(valid['problem'].dropna().unique()))} |")
lines.append(f"| scale check failures | {len(bad_scale)} |")
lines.append("")
if bad_scale:
    lines.append("## FAILURES")
    for x in bad_scale[:20]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("STATUS = invalid")
else:
    lines.append("STATUS = valid")
(OUT / "metric_integrity.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
sys.exit(1 if bad_scale else 0)
