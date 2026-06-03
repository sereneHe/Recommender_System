#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/Users/xiaoyuhe/Recommender_Pavel")
PROBLEM_GROUP = os.environ.get("PROBLEM_GROUPS", "FRED_16country_monthly").split(",")[0].strip()
REPORT_FAMILY = PROBLEM_GROUP.split("_", 1)[0]
BASE_DIR = ROOT / "reports" / REPORT_FAMILY

PROBLEM = os.environ.get("PROBLEM", f"{PROBLEM_GROUP}/industry_eu_fin").split(",")[0].strip()
TARGET = os.environ.get("TARGET", PROBLEM.rsplit("_", 1)[-1].upper())

DEFAULT_OUTPUT_DIR = ROOT / "reports" / "plots" / PROBLEM_GROUP
OUTPUT_DIR = Path(os.environ.get("PLOT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
OUTPUT_PLOT = OUTPUT_DIR / f"{TARGET}_test_mean.png"
OUTPUT_TABLE = OUTPUT_DIR / f"{TARGET}_test_mean_summary.csv"

SOLVER_FILTER = os.environ.get("SOLVERS", os.environ.get("solver", "hc_predictor_ce,hc_predictor,mark_with_cc,mark"))
REQUESTED_SOLVERS = {item.strip() for item in SOLVER_FILTER.split(",") if item.strip()}

METHOD_SOURCES = [
    ("hc_predictor", f"{PROBLEM_GROUP}_hc_predictor.csv"),
    ("hc_predictor_ci", f"{PROBLEM_GROUP}_hc_predictor_ci.csv"),
    ("mark", f"{PROBLEM_GROUP}_mark.csv"),
    ("mark_with_cc", f"{PROBLEM_GROUP}_mark_with_cc.csv"),
    ("hc_predictor_ce_ori", f"{PROBLEM_GROUP}_hc_predictor_ce_ori.csv"),
    ("hc_predictor_ce_ceThreshold_0.05", f"{PROBLEM_GROUP}_hc_predictor_ce_ceThreshold_0.05.csv"),
    ("hc_predictor_ce_milpLambda_0.05", f"{PROBLEM_GROUP}_hc_predictor_ce_milpLambda_0.05.csv"),
    ("hc_predictor_ce_shielded_collider", f"{PROBLEM_GROUP}_hc_predictor_ce_shielded_collider.csv"),
    ("hc_predictor_ce_shielded_collider_limit", f"{PROBLEM_GROUP}_hc_predictor_ce_shielded_collider_limit.csv"),
]

METHOD_LABELS = {
    "hc_predictor": "HC-Predictor",
    "hc_predictor_ci": "HC-Predictor-CI (correlation)",
    "mark": "MARK",
    "mark_with_cc": "MARK with CC",
    "hc_predictor_ce_ori": "HC-Predictor-CE (dense DAG)",
    "hc_predictor_ce_ceThreshold_0.05": "HC-Predictor-CE (ceThreshold=0.05)",
    "hc_predictor_ce_milpLambda_0.05": "HC-Predictor-CE (milpLambda=0.05)",
    "hc_predictor_ce_shielded_collider": "HC-Predictor-CE (shielded collider)",
    "hc_predictor_ce_shielded_collider_limit": "HC-Predictor-CE (shielded collider limit)",
}

METHOD_COLORS = {
    "hc_predictor": "#4C97D8",
    "hc_predictor_ci": "#E15759",
    "mark": "#6B6B6B",
    "mark_with_cc": "#F28E2B",
}

CE_BLUE_PALETTE = {
    "hc_predictor_ce_shielded_collider": "#1F4E8C",
    "hc_predictor_ce_shielded_collider_limit": "#3A78B8",
    "hc_predictor_ce_ceThreshold_0.05": "#5FA2D9",
    "hc_predictor_ce_ori": "#8EC1E8",
    "hc_predictor_ce_milpLambda_0.05": "#B7D5F0",
}


def method_requested(method: str) -> bool:
    if method.startswith("hc_predictor_ce"):
        return "hc_predictor_ce" in REQUESTED_SOLVERS or method in REQUESTED_SOLVERS
    return method in REQUESTED_SOLVERS


def load_fin_summary() -> pd.DataFrame:
    rows = []
    skipped_missing = []
    skipped_target = []
    for method, filename in METHOD_SOURCES:
        if not method_requested(method):
            continue
        path = BASE_DIR / filename
        if not path.exists():
            skipped_missing.append(method)
            continue
        df = pd.read_csv(path)
        df = df[df["target"] == TARGET].copy()
        if df.empty:
            skipped_target.append(method)
            continue
        row = df.iloc[0]
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "test_mean": float(row["test_mean"]),
                "train_mean": float(row["train_mean"]) if "train_mean" in row else float("nan"),
                "runtime": float(row["runtime"]) if "runtime" in row else float("nan"),
                "selected_features": row.get("selected features", ""),
            }
        )
    if not rows:
        raise SystemExit(f"No {TARGET} rows found under {BASE_DIR} for solvers={sorted(REQUESTED_SOLVERS)}")
    if skipped_missing:
        print(f"Skipped missing method files: {', '.join(skipped_missing)}")
    if skipped_target:
        print(f"Skipped methods without {TARGET}: {', '.join(skipped_target)}")
    summary = pd.DataFrame(rows).sort_values("test_mean", ascending=True).reset_index(drop=True)
    return summary


def plot_fin(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_height = max(4.0, 0.52 * len(summary) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = list(range(len(summary)))
    colors = []
    for method in summary["method"]:
        if method in CE_BLUE_PALETTE:
            colors.append(CE_BLUE_PALETTE[method])
        else:
            colors.append(METHOD_COLORS.get(method, "#4C97D8"))

    ax.barh(
        y_positions,
        summary["test_mean"],
        color=colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.6,
    )

    for y, mean in zip(y_positions, summary["test_mean"]):
        ax.text(mean + 0.03, y, f"{mean:.3f}", va="center", ha="left", fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["method_label"], fontsize=12)
    ax.set_xlabel("test_mean", fontsize=16)
    ax.set_title(f"{PROBLEM_GROUP}: {TARGET} comparison", fontsize=18, pad=14)
    ax.set_xlim(0.8, 1.8)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_fin_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_TABLE, index=False)
    plot_fin(summary)
    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved plot: {OUTPUT_PLOT}")
    print(f"Problem: {PROBLEM}")
    print(f"Target: {TARGET}")
    print(f"Solver filter: {', '.join(sorted(REQUESTED_SOLVERS))}")
    print("Methods plotted:")
    for method_label in summary["method_label"]:
        print(f"  - {method_label}")


if __name__ == "__main__":
    main()
