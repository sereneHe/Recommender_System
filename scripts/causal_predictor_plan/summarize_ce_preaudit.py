#!/usr/bin/env python3
"""Summarize CE pre-audit artifacts without applying automatic Go/No-Go gates."""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def _read_yaml(path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _dataset_key(config):
    problem = config.get("problem", {}) if isinstance(config, dict) else {}
    name = str(problem.get("name", "unknown"))
    target = str(problem.get("target", "unknown"))
    data_path = str(problem.get("data_path", ""))
    graph_type = str(problem.get("graph_type", ""))
    if graph_type:
        return f"{name}_{graph_type}:{target}"
    if data_path:
        return f"{Path(data_path).name}:{target}"
    return f"{name}:{target}"


def _arm_key(config):
    solver = config.get("solver", {}) if isinstance(config, dict) else {}
    constrained = bool(solver.get("constrained", True))
    use_ci = bool(solver.get("use_ci_penalty", False))
    use_w = bool(solver.get("use_w_constraints", False))
    if not constrained and not use_ci:
        arm = "dnn_only"
    elif use_ci:
        arm = "ce_active"
    else:
        arm = "other_constrained"
    return f"{arm};W={int(use_w)};CI={solver.get('ci_penalty_kind', 'unknown')}"


def _constraint_key(row):
    return "|".join(
        [
            str(row.get("relation", "")),
            str(row.get("x_name", "")),
            str(row.get("y_name", "")),
            str(row.get("z_names", "")),
            str(row.get("source", "")),
        ]
    )


def _spearman(left, right):
    pairs = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pairs) < 3 or pairs["left"].nunique() < 2 or pairs["right"].nunique() < 2:
        return None
    return float(pairs["left"].rank().corr(pairs["right"].rank()))


def summarize(root, output_dir):
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else root
    run_rows = []
    stability_rows = []

    for metadata_path in sorted(root.rglob("constraint_metadata.csv")):
        run_dir = metadata_path.parent
        stats_path = run_dir / "constraint_stat_audit.csv"
        metrics_path = run_dir / "cv_fold_metrics.csv"
        config_path = run_dir / "config.yaml"
        if not stats_path.exists() or not config_path.exists():
            continue
        config = _read_yaml(config_path)
        solver = config.get("solver", {}) if isinstance(config, dict) else {}
        dataset = _dataset_key(config)
        arm = _arm_key(config)
        run_id = str(run_dir)
        seed = solver.get("random_state")
        try:
            metadata = pd.read_csv(metadata_path).fillna("")
            stats = pd.read_csv(stats_path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if metadata.empty:
            constraint_counts = pd.Series(dtype=float)
            keys = set()
        else:
            metadata["constraint_key"] = metadata.apply(_constraint_key, axis=1)
            constraint_counts = metadata.groupby("fold").size()
            keys = set(metadata["constraint_key"].tolist())

        train_stats = stats[stats.get("evaluation_split", "") == "outer_train"].copy()
        if not train_stats.empty:
            if "constraint_key" not in train_stats:
                train_stats["constraint_key"] = train_stats.apply(_constraint_key, axis=1)
            raw_stat = pd.to_numeric(train_stats.get("observed_y_statistic"), errors="coerce")
            train_stats["observed_y_statistic"] = raw_stat
            group_snr = []
            group_sign_flip = []
            for _, group in train_stats.groupby("constraint_key"):
                values = group["observed_y_statistic"].dropna().to_numpy(dtype=float)
                if len(values) >= 2:
                    scale = float(values.std(ddof=0))
                    group_snr.append(abs(float(values.mean())) / scale if scale > 1e-12 else None)
                    nonzero = values[abs(values) > 1e-8]
                    if len(nonzero):
                        group_sign_flip.append(
                            min(int((nonzero > 0).sum()), int((nonzero < 0).sum())) / len(nonzero)
                        )
            median_fold_snr = pd.Series([v for v in group_snr if v is not None]).median() if group_snr else None
            median_fold_flip = pd.Series(group_sign_flip).median() if group_sign_flip else None
            window_snr = pd.to_numeric(
                train_stats.get("window_abs_mean_over_spread"), errors="coerce"
            ).median()
            window_sign_flip = pd.to_numeric(
                train_stats.get("window_sign_flip_rate"), errors="coerce"
            ).median()
            mean_train_violation = pd.to_numeric(
                train_stats.get("prediction_violation"), errors="coerce"
            ).mean()
        else:
            median_fold_snr = median_fold_flip = window_snr = window_sign_flip = None
            mean_train_violation = None

        mean_test_nmse = None
        spearman = None
        if metrics_path.exists():
            try:
                metrics = pd.read_csv(metrics_path)
                mean_test_nmse = pd.to_numeric(metrics.get("test_nmse"), errors="coerce").mean()
                if not stats.empty and not metrics.empty:
                    test_stats = stats[stats.get("evaluation_split", "") == "outer_test"].copy()
                    if not test_stats.empty:
                        test_stats["prediction_violation"] = pd.to_numeric(
                            test_stats.get("prediction_violation"), errors="coerce"
                        )
                        violation_by_fold = test_stats.groupby("fold")["prediction_violation"].mean()
                        joined = metrics.set_index("fold")["test_nmse"].to_frame().join(
                            violation_by_fold.rename("prediction_violation"), how="inner"
                        )
                        spearman = _spearman(
                            joined["prediction_violation"],
                            pd.to_numeric(joined["test_nmse"], errors="coerce"),
                        )
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                pass

        run_rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "seed": seed,
                "run_dir": run_id,
                "n_constraints_mean_per_fold": float(constraint_counts.mean()) if len(constraint_counts) else 0.0,
                "n_constraints_min_per_fold": int(constraint_counts.min()) if len(constraint_counts) else 0,
                "n_constraints_max_per_fold": int(constraint_counts.max()) if len(constraint_counts) else 0,
                "n_unique_constraints": len(keys),
                "median_fold_abs_mean_over_sd": median_fold_snr,
                "median_fold_sign_flip_rate": median_fold_flip,
                "median_window_abs_mean_over_spread": window_snr,
                "median_window_sign_flip_rate_descriptive": window_sign_flip,
                "mean_train_prediction_violation": mean_train_violation,
                "mean_test_nmse": mean_test_nmse,
                "spearman_test_violation_vs_nmse_descriptive": spearman,
            }
        )
        for key in keys:
            stability_rows.append(
                {"dataset": dataset, "arm": arm, "run_id": run_id, "seed": seed, "constraint_key": key}
            )

    runs = pd.DataFrame(run_rows)
    stability = pd.DataFrame(stability_rows)
    if not stability.empty:
        run_totals = runs.groupby(["dataset", "arm"])["run_dir"].nunique().rename("n_runs")
        support = (
            stability.groupby(["dataset", "arm", "constraint_key"])["run_id"]
            .nunique()
            .rename("n_runs_with_constraint")
            .reset_index()
            .join(run_totals, on=["dataset", "arm"])
        )
        support["run_support_rate"] = support["n_runs_with_constraint"] / support["n_runs"]
        stability = support.sort_values(["dataset", "arm", "run_support_rate"], ascending=[True, True, False])

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "ce_preaudit_run_summary.csv"
    stability_path = output_dir / "ce_preaudit_constraint_stability.csv"
    runs.to_csv(runs_path, index=False)
    stability.to_csv(stability_path, index=False)
    paired_path = output_dir / "ce_preaudit_paired_comparison.csv"
    if not runs.empty:
        paired_input = runs.copy()
        paired_input["arm_short"] = paired_input["arm"].str.split(";").str[0]
        paired = paired_input.pivot_table(
            index=["dataset", "seed"],
            columns="arm_short",
            values="mean_test_nmse",
            aggfunc="mean",
        ).reset_index()
        if "dnn_only" in paired and "ce_active" in paired:
            paired = paired.dropna(subset=["dnn_only", "ce_active"]).copy()
            paired["delta_ce_minus_dnn_nmse"] = paired["ce_active"] - paired["dnn_only"]
            paired["ce_lite_lower_nmse"] = paired["delta_ce_minus_dnn_nmse"] < 0
            aggregate = (
                paired.groupby("dataset")
                .agg(
                    n_paired_seeds=("seed", "nunique"),
                    ce_lite_win_rate=("ce_lite_lower_nmse", "mean"),
                    mean_delta_nmse=("delta_ce_minus_dnn_nmse", "mean"),
                    median_delta_nmse=("delta_ce_minus_dnn_nmse", "median"),
                )
                .reset_index()
            )
            paired = paired.merge(aggregate, on="dataset", how="left")
        paired.to_csv(paired_path, index=False)
    else:
        pd.DataFrame().to_csv(paired_path, index=False)
    return runs_path, stability_path, paired_path, len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing per-run Hydra output folders.")
    parser.add_argument("--output-dir", help="Defaults to --root.")
    args = parser.parse_args()
    runs_path, stability_path, paired_path, n_runs = summarize(args.root, args.output_dir)
    print(f"Summarized {n_runs} run folders.")
    print(f"Run summary: {runs_path}")
    print(f"Constraint stability: {stability_path}")
    print(f"Paired DNN-only vs W+CE-Lite: {paired_path}")
    print("All SNR/correlation values are descriptive; this script applies no Go/No-Go thresholds.")


if __name__ == "__main__":
    main()
