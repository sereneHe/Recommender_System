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
OUTPUT_DIR = Path(os.environ.get("PLOT_OUTPUT_DIR", str(ROOT / "reports" / "plots" / PROBLEM_GROUP)))
OUTPUT_PLOT = OUTPUT_DIR / f"{PROBLEM_GROUP}_test_mean.png"
OUTPUT_TABLE = OUTPUT_DIR / f"{PROBLEM_GROUP}_test_mean.csv"

PREFIX = f"{PROBLEM_GROUP}_"
TARGET_FILTER_TEXT = os.environ.get("TARGET_FILTER", "").strip()
TARGET_FILTER = {item.strip().upper() for item in TARGET_FILTER_TEXT.split(",") if item.strip()} or None
SOLVER_FILTER = os.environ.get("SOLVERS", os.environ.get("solver", "hc_predictor_ce,hc_predictor,mark_with_cc,mark"))
REQUESTED_SOLVERS = {item.strip() for item in SOLVER_FILTER.split(",") if item.strip()}

METHOD_LABELS = {
    "hc_predictor": "HC-Predictor",
    "hc_predictor_ci": "HC-Predictor-CI (correlation)",
    "mark": "MARK",
    "hc_predictor_ce_ori": "HC-Predictor-CE (dense DAG)",
    "hc_predictor_ce_ceThreshold_0.05": "HC-Predictor-CE (ceThreshold=0.05)",
    "hc_predictor_ce_milpLambda_0.05": "HC-Predictor-CE (milpLambda=0.05)",
    "hc_predictor_ce_shielded_collider": "HC-Predictor-CE (shielded collider)",
    "hc_predictor_ce_shielded_collider_limit": "HC-Predictor-CE (shielded collider limit)",
    "mark_with_cc": "MARK with CC",
}

METHOD_COLORS = {
    "hc_predictor": "#4C97D8",
    "hc_predictor_ci": "#E15759",
    "mark": "#6B6B6B",
    "mark_with_cc": "#F28E2B",
}


def infer_method_name(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(PREFIX):
        return ""
    return stem[len(PREFIX) :]


def is_direct_method_file(path: Path) -> bool:
    method = infer_method_name(path)
    if not method:
        return False
    # Exclude comparison / pivot / summary tables that already contain multiple methods.
    forbidden_tokens = [
        "comparison",
        "pivot",
        "best",
        "_vs_",
        "new_",
    ]
    return not any(token in method for token in forbidden_tokens)


def method_requested(method: str) -> bool:
    if method.startswith("hc_predictor_ce"):
        return "hc_predictor_ce" in REQUESTED_SOLVERS or method in REQUESTED_SOLVERS
    return method in REQUESTED_SOLVERS


def load_summary() -> pd.DataFrame:
    rows = []
    for path in sorted(BASE_DIR.glob(f"{PROBLEM_GROUP}_*.csv")):
        if not is_direct_method_file(path):
            continue
        method = infer_method_name(path)
        if not method_requested(method):
            continue
        df = pd.read_csv(path)
        required_cols = {"target", "test_mean"}
        if not required_cols.issubset(df.columns):
            continue

        if TARGET_FILTER:
            df = df[df["target"].isin(TARGET_FILTER)].copy()
            if df.empty:
                continue

        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_targets": int(df["target"].nunique()),
                "test_mean_mean": float(df["test_mean"].mean()),
                "test_mean_sd": float(df["test_mean"].std(ddof=1)),
                "train_mean_mean": float(df["train_mean"].mean()) if "train_mean" in df.columns else float("nan"),
                "train_mean_sd": float(df["train_mean"].std(ddof=1)) if "train_mean" in df.columns else float("nan"),
                "runtime_mean": float(df["runtime"].mean()) if "runtime" in df.columns else float("nan"),
                "runtime_sd": float(df["runtime"].std(ddof=1)) if "runtime" in df.columns else float("nan"),
                "source_file": path.name,
                "targets": ", ".join(sorted(df["target"].astype(str).unique())),
            }
        )

    if not rows:
        raise SystemExit(f"No direct method CSVs found in {BASE_DIR} for {PROBLEM_GROUP}")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["test_mean_mean", "method_label"], ascending=[True, True]).reset_index(drop=True)
    return summary


def plot_summary(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_height = max(4.0, 0.45 * len(summary) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = list(range(len(summary)))
    ce_rows = [method for method in summary["method"] if method.startswith("hc_predictor_ce")]
    ce_shades = ["#1F4E8C", "#3A78B8", "#5FA2D9", "#8EC1E8", "#B7D5F0", "#D5E6F5"]
    ce_palette = {
        method: ce_shades[min(i, len(ce_shades) - 1)]
        for i, method in enumerate(ce_rows)
    }
    colors = []
    for method in summary["method"]:
        if method in ce_palette:
            colors.append(ce_palette[method])
        else:
            colors.append(METHOD_COLORS.get(method, "#4C97D8"))

    ax.barh(
        y_positions,
        summary["test_mean_mean"],
        xerr=summary["test_mean_sd"],
        color=colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )

    for y, mean, sd in zip(y_positions, summary["test_mean_mean"], summary["test_mean_sd"]):
        ax.text(
            mean + 0.23,
            y,
            f"{mean:.3f} ± {sd:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["method_label"], fontsize=12)
    ax.set_xlabel("test_mean", fontsize=16)
    ax.set_title(f"{PROBLEM_GROUP}: test_mean", fontsize=18, pad=14)
    ax.set_xlim(0.8, 1.8)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()

    # Keep all methods visible and avoid a crowded legend; labels are already on the axis.
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_TABLE, index=False)
    plot_summary(summary)
    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved plot: {OUTPUT_PLOT}")
    if TARGET_FILTER:
        print(f"Targets used: {', '.join(sorted(TARGET_FILTER))}")
    print("Methods plotted:")
    for method_label in summary["method_label"]:
        print(f"  - {method_label}")


if __name__ == "__main__":
    main()
