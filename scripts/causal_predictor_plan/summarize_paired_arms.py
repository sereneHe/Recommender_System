#!/usr/bin/env python3
"""Paired arm comparison for CE-new runs.

Scans Hydra run directories, extracts the per-fold ``test_nmse`` (or
``train_nmse``) for two named arms, pairs them by the run context (the
experiment name with the arm label removed, i.e. same graph/noise/seed and
problem), and reports the paired delta with mean, standard deviation and a
confidence interval.

Only a paired delta between arms that share the same fitting procedure is
interpretable.  For example, comparing ``mark`` (10 trees, no y scaling) with
``mark_with_cc`` (100 trees, standardized y) is *not* a W-constraint ablation;
use M0 (rho0=0) vs M1 (rho0>0) for that.

Usage:
  python summarize_paired_arms.py --root multirun \
    --arm-a m0_xgb100_no_w --arm-b m1_xgb100_w
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import yaml


def _read_yaml(path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _t_critical(df, confidence=0.95):
    # Two-sided t critical value; small lookup, no scipy dependency.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    }
    if df <= 0:
        return float("nan")
    if df in table:
        return table[df]
    if df >= 30:
        return 1.96
    # linear interpolation between the nearest tabulated degrees of freedom
    keys = sorted(table)
    lower = max(k for k in keys if k <= df)
    upper = min((k for k in keys if k >= df), default=30)
    if upper == lower:
        return table[lower]
    frac = (df - lower) / (upper - lower)
    return table[lower] + frac * (table[upper] - table[lower])


def _arm_of(experiment, arm_a, arm_b):
    if arm_a in experiment:
        return "A"
    if arm_b in experiment:
        return "B"
    return None


def _context_of(experiment, arm_a, arm_b):
    context = experiment.replace(arm_a, "ARM").replace(arm_b, "ARM")
    return context


def summarize(root, arm_a, arm_b, metric, output, confidence):
    root = Path(root).expanduser().resolve()
    rows = {}
    for config_path in sorted(root.rglob("config.yaml")):
        if ".hydra" in config_path.parts:
            continue
        run_dir = config_path.parent
        metrics_path = run_dir / "cv_fold_metrics.csv"
        if not metrics_path.exists():
            continue
        config = _read_yaml(config_path)
        experiment = str(config.get("experiment", "")) if isinstance(config, dict) else ""
        arm = _arm_of(experiment, arm_a, arm_b)
        if arm is None:
            continue
        context = _context_of(experiment, arm_a, arm_b)
        try:
            metrics = pd.read_csv(metrics_path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if metric not in metrics:
            continue
        values = pd.to_numeric(metrics[metric], errors="coerce").dropna().tolist()
        if not values:
            continue
        record = rows.setdefault(context, {"A": None, "B": None, "fold_A": [], "fold_B": []})
        record[arm] = sum(values) / len(values)
        record[f"fold_{arm}"] = values

    paired = []
    for context, record in rows.items():
        if record["A"] is None or record["B"] is None:
            continue
        paired.append(
            {
                "context": context,
                "arm_a": record["A"],
                "arm_b": record["B"],
                "delta_b_minus_a": record["B"] - record["A"],
            }
        )
    frame = pd.DataFrame(paired)
    output = Path(output).expanduser().resolve() if output else root / "paired_arm_delta.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = {"n_pairs": len(frame), "arm_a": arm_a, "arm_b": arm_b, "metric": metric}
    if not frame.empty:
        deltas = frame["delta_b_minus_a"]
        n = len(deltas)
        mean = float(deltas.mean())
        std = float(deltas.std(ddof=1)) if n > 1 else float("nan")
        sem = std / math.sqrt(n) if n > 1 else float("nan")
        tcrit = _t_critical(n - 1, confidence)
        summary.update(
            {
                "mean_arm_a": float(frame["arm_a"].mean()),
                "mean_arm_b": float(frame["arm_b"].mean()),
                "mean_delta": mean,
                "std_delta": std,
                "ci_low": mean - tcrit * sem,
                "ci_high": mean + tcrit * sem,
                "win_rate_b_lower": float((deltas < 0).mean()),
            }
        )
        # fold-level paired deltas when both arms have matching fold counts
        fold_deltas = []
        for _, row in frame.iterrows():
            fa = rows[row["context"]]["fold_A"]
            fb = rows[row["context"]]["fold_B"]
            if len(fa) == len(fb):
                fold_deltas.extend([b - a for a, b in zip(fa, fb)])
        if fold_deltas:
            fd = pd.Series(fold_deltas)
            summary["n_fold_pairs"] = len(fd)
            summary["mean_fold_delta"] = float(fd.mean())
            summary["std_fold_delta"] = float(fd.std(ddof=1)) if len(fd) > 1 else float("nan")
    return frame, summary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root with per-run Hydra output folders.")
    parser.add_argument("--arm-a", required=True, help="Substring identifying arm A (reference).")
    parser.add_argument("--arm-b", required=True, help="Substring identifying arm B (treatment).")
    parser.add_argument("--metric", default="test_nmse", help="Metric column (test_nmse or train_nmse).")
    parser.add_argument("--output", help="Output CSV path (default: <root>/paired_arm_delta.csv).")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for the interval.")
    args = parser.parse_args()
    frame, summary, output = summarize(
        args.root, args.arm_a, args.arm_b, args.metric, args.output, args.confidence
    )
    print(f"Paired delta: {args.arm_b} - {args.arm_a}  (metric={args.metric})")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Per-pair table: {output}")


if __name__ == "__main__":
    main()
