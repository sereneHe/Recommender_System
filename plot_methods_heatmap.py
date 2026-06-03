#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/xiaoyuhe/Recommender_Pavel")
PROBLEM_GROUP = os.environ.get("PROBLEM_GROUPS", "FRED_16country_monthly").split(",")[0].strip()
REPORT_FAMILY = PROBLEM_GROUP.split("_", 1)[0]
BASE_DIR = ROOT / "reports" / REPORT_FAMILY
OUTPUT_DIR = Path(os.environ.get("PLOT_OUTPUT_DIR", str(ROOT / "reports" / "plots" / PROBLEM_GROUP)))
OUTPUT_PLOT = OUTPUT_DIR / f"{PROBLEM_GROUP}_all_methods_heatmap.png"
OUTPUT_TABLE = OUTPUT_DIR / f"{PROBLEM_GROUP}_all_methods_heatmap.csv"
OUTPUT_BEST = OUTPUT_DIR / f"{PROBLEM_GROUP}_best_method_by_country_sorted.csv"
SOLVER_FILTER = os.environ.get("SOLVERS", os.environ.get("solver", "hc_predictor_ce,hc_predictor,mark_with_cc,mark"))
REQUESTED_SOLVERS = {item.strip() for item in SOLVER_FILTER.split(",") if item.strip()}

DEFAULT_TARGETS = [
    "AUT", "BEL", "DEU", "ESP", "EST", "FIN", "FRA", "GRC",
    "IRL", "ITA", "LTU", "LUX", "NLD", "PRT", "SVK", "SVN",
]
TARGET_FILTER_TEXT = os.environ.get("TARGET_FILTER", "").strip()
TARGETS = [item.strip().upper() for item in TARGET_FILTER_TEXT.split(",") if item.strip()] or DEFAULT_TARGETS

METHOD_FILES = [
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
    "hc_predictor_ci": "HC-Predictor-CI",
    "mark": "MARK",
    "mark_with_cc": "MARK with CC",
    "hc_predictor_ce_ori": "HC-Predictor-CE (dense DAG)",
    "hc_predictor_ce_ceThreshold_0.05": "HC-Predictor-CE (ceThreshold=0.05)",
    "hc_predictor_ce_milpLambda_0.05": "HC-Predictor-CE (milpLambda=0.05)",
    "hc_predictor_ce_shielded_collider": "HC-Predictor-CE (shielded collider)",
    "hc_predictor_ce_shielded_collider_limit": "HC-Predictor-CE (shielded collider limit)",
}


def method_requested(method: str) -> bool:
    if method.startswith("hc_predictor_ce"):
        return "hc_predictor_ce" in REQUESTED_SOLVERS or method in REQUESTED_SOLVERS
    return method in REQUESTED_SOLVERS


def load_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method, filename in METHOD_FILES:
        if not method_requested(method):
            continue
        path = BASE_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "target" not in df.columns or "test_mean" not in df.columns:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "target": row["target"],
                    "test_mean": float(row["test_mean"]),
                }
            )

    if not rows:
        raise SystemExit(f"No method CSVs found in {BASE_DIR} for {PROBLEM_GROUP}")

    long_df = pd.DataFrame(rows)
    target_order = [target for target in TARGETS if target in set(long_df["target"])]
    pivot = (
        long_df.pivot_table(index="method", columns="target", values="test_mean", aggfunc="mean")
        .reindex(columns=target_order)
    )

    # Sort methods by overall mean, ignoring missing values.
    method_order = (
        long_df.groupby("method")["test_mean"].mean().sort_values(ascending=True).index.tolist()
    )
    pivot = pivot.reindex(method_order)

    label_map = long_df.drop_duplicates("method").set_index("method")["method_label"].to_dict()
    pivot.index = [label_map.get(method, method) for method in pivot.index]
    return long_df, pivot


def plot_heatmap(pivot: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    fig_width = max(12.0, 0.75 * len(pivot.columns) + 4.0)
    fig_height = max(5.5, 0.42 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.8, vmax=1.8)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xlabel("target", fontsize=14)
    ax.set_ylabel("method", fontsize=14)
    ax.set_title(f"{PROBLEM_GROUP}: test_mean by country and method", fontsize=17, pad=14)

    # Cell annotations.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            color = "black" if value < 1.35 else "white"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("test_mean", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_best(long_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for target, group in long_df.groupby("target"):
        best = group.sort_values("test_mean", ascending=True).iloc[0]
        best_rows.append(
            {
                "target": target,
                "best_method": best["method"],
                "best_method_label": best["method_label"],
                "best_test_mean": best["test_mean"],
            }
        )
    best_df = (
        pd.DataFrame(best_rows)
        .sort_values(["best_test_mean", "target"], ascending=[True, True])
        .reset_index(drop=True)
    )
    best_df.insert(0, "rank", range(1, len(best_df) + 1))
    best_df.to_csv(OUTPUT_BEST, index=False)
    return best_df


def main() -> None:
    long_df, pivot = load_matrix()
    pivot.to_csv(OUTPUT_TABLE)
    best_df = save_best(long_df)
    plot_heatmap(pivot)
    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved best table: {OUTPUT_BEST}")
    print(f"Saved plot: {OUTPUT_PLOT}")
    print("Best methods per target:")
    for _, row in best_df.iterrows():
        print(f"  - {row['target']}: {row['best_method_label']} ({row['best_test_mean']:.3f})")


if __name__ == "__main__":
    main()
