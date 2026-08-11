#!/usr/bin/env python3
"""Plot train/validation outer-loop errors and final test-error reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "multirun" / "2026-07-01" / "11-19-21" / "0"
DEFAULT_ARTIFACT_DIR = (
    ROOT
    / "mlruns"
    / "44"
    / "ed0bd195634e4d60bf365723144eb698"
    / "artifacts"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "CODIET"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot train_err and valid_error mean±sd vs outer, with final "
            "test_err mean±sd as a horizontal reference."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Hydra run directory containing cv_validation_history.yaml.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="MLflow artifact directory containing cv_errors.yaml.",
    )
    parser.add_argument(
        "--validation-history",
        type=Path,
        default=None,
        help="Explicit path to cv_validation_history.yaml.",
    )
    parser.add_argument(
        "--cv-errors",
        type=Path,
        default=None,
        help="Explicit path to cv_errors.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the plot and CSV are saved.",
    )
    parser.add_argument(
        "--output-prefix",
        default="HDL_train_valid_test_vs_outer_mean_sd",
        help="Output filename prefix, without extension.",
    )
    parser.add_argument(
        "--title",
        default="error vs outer_iteration",
        help="Plot title.",
    )
    return parser.parse_args()


def load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_outer_summary(histories: list[dict], test_errs: list[float]) -> pd.DataFrame:
    rows = []
    for fold_item in histories:
        fold = int(fold_item["fold"])
        for history_item in fold_item.get("history", []):
            rows.append(
                {
                    "fold": fold,
                    "outer": int(history_item["outer"]),
                    "train_err": float(history_item["train_loss"]),
                    "valid_error": float(history_item["val_loss"]),
                }
            )

    if not rows:
        raise ValueError("No outer-loop validation history found.")

    frame = pd.DataFrame(rows).sort_values(["fold", "outer"])
    summary = frame.groupby("outer", as_index=False).agg(
        train_mean=("train_err", "mean"),
        train_sd=("train_err", "std"),
        valid_mean=("valid_error", "mean"),
        valid_sd=("valid_error", "std"),
    )

    test_array = np.asarray(test_errs, dtype=float)
    summary["test_mean"] = float(np.nanmean(test_array)) if len(test_array) else np.nan
    summary["test_sd"] = (
        float(np.nanstd(test_array, ddof=1)) if len(test_array) > 1 else 0.0
    )
    return summary


def plot_summary(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = summary["outer"].to_numpy()

    train_mean = summary["train_mean"].to_numpy()
    train_sd = summary["train_sd"].fillna(0.0).to_numpy()
    valid_mean = summary["valid_mean"].to_numpy()
    valid_sd = summary["valid_sd"].fillna(0.0).to_numpy()
    test_mean = float(summary["test_mean"].iloc[0])
    test_sd = float(summary["test_sd"].iloc[0])

    ax.plot(
        x,
        train_mean,
        color="#1f77b4",
        marker="o",
        linewidth=2.7,
        label="train err mean",
    )
    ax.fill_between(
        x,
        train_mean - train_sd,
        train_mean + train_sd,
        color="#1f77b4",
        alpha=0.15,
        linewidth=0,
        label="train ± sd",
    )

    ax.plot(
        x,
        valid_mean,
        color="#d62728",
        marker="s",
        linewidth=2.7,
        label="valid error mean",
    )
    ax.fill_between(
        x,
        valid_mean - valid_sd,
        valid_mean + valid_sd,
        color="#d62728",
        alpha=0.13,
        linewidth=0,
        label="valid ± sd",
    )

    if np.isfinite(test_mean):
        ax.axhline(
            test_mean,
            color="#2ca02c",
            linestyle="--",
            linewidth=2.6,
            label=f"final test err mean = {test_mean:.3f}",
        )
        ax.axhspan(
            test_mean - test_sd,
            test_mean + test_sd,
            color="#2ca02c",
            alpha=0.12,
            linewidth=0,
            label="final test ± sd",
        )

    ax.set_title(title)
    ax.set_xlabel("outer")
    ax.set_ylabel("error")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_path = args.validation_history or args.run_dir / "cv_validation_history.yaml"
    errors_path = args.cv_errors or args.artifact_dir / "cv_errors.yaml"

    histories = load_yaml(history_path)
    errors = load_yaml(errors_path)
    test_errs = errors.get("test_errs", [])

    summary = build_outer_summary(histories, test_errs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    png_path = args.output_dir / f"{args.output_prefix}.png"

    summary.to_csv(csv_path, index=False)
    plot_summary(summary, png_path, args.title)

    print(f"Saved plot: {png_path}")
    print(f"Saved table: {csv_path}")
    print(
        "Note: test_err is recorded only as final CV test error, so it is "
        "drawn as a horizontal mean±sd reference."
    )


if __name__ == "__main__":
    main()
