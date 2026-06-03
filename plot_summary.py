#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORTS_DIR = Path("/Users/xiaoyuhe/Recommender_Pavel/reports/OECD")
OUTPUT_TABLE = REPORTS_DIR / "OECD_summary.csv"
OUTPUT_PLOT = REPORTS_DIR / "OECD_test_mean.png"

METHODS = ["hc_predictor", "hc_predictor_ci", "hc_predictor_ce", "mark"]
METHOD_LABELS = {
    "hc_predictor": "HC-Predictor",
    "hc_predictor_ci": "HC-Predictor-CI",
    "hc_predictor_ce": "HC-Predictor-CE",
    "mark": "MARK",
}
METHOD_COLORS = {
    "hc_predictor": "#4C97D8",
    "hc_predictor_ci": "#E15759",
    "hc_predictor_ce": "#59A14F",
    "mark": "#6B6B6B",
}

GROUP_ORDER = [
    "OECD_9country_quarterly",
    "OECD_9country_monthly",
    "OECD_16country_quarterly",
    "OECD_16country_monthly",
]
GROUP_LABELS = {
    "OECD_9country_quarterly": "9-country Q",
    "OECD_9country_monthly": "9-country M",
    "OECD_16country_quarterly": "16-country Q",
    "OECD_16country_monthly": "16-country M",
}


def load_summary() -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        for method in METHODS:
            path = REPORTS_DIR / f"{group}_{method}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            rows.append(
                {
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_targets": len(df),
                    "test_mean_mean": df["test_mean"].mean(),
                    "test_mean_sd": df["test_mean"].std(ddof=1),
                    "train_mean_mean": df["train_mean"].mean(),
                    "train_mean_sd": df["train_mean"].std(ddof=1),
                    "runtime_mean": df["runtime"].mean(),
                    "runtime_sd": df["runtime"].std(ddof=1),
                }
            )
    if not rows:
        raise SystemExit(f"No OECD summary CSVs found in {REPORTS_DIR}")
    return pd.DataFrame(rows)


def plot_test_mean(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    x_positions = list(range(len(GROUP_ORDER)))

    for method in METHODS:
        method_df = (
            summary[summary["method"] == method]
            .set_index("group")
            .reindex(GROUP_ORDER)
            .reset_index()
        )
        ax.errorbar(
            x_positions,
            method_df["test_mean_mean"],
            yerr=method_df["test_mean_sd"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker="o",
            markersize=5,
            linewidth=1.5,
            capsize=4,
            capthick=1.2,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=12)
    ax.set_ylabel("test_mean", fontsize=18)
    ax.set_xlabel("OECD dataset groups", fontsize=20)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(frameon=False, fontsize=13, loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    summary.to_csv(OUTPUT_TABLE, index=False)
    plot_test_mean(summary)
    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
