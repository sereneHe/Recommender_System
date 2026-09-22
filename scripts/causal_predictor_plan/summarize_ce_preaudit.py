#!/usr/bin/env python3
"""Summarize CE pre-audit artifacts and emit a three-stage Go/No-Go gate.

The gate mirrors the corrected pre-screening design: it decides only on
constraint quality (A: how many d-separation constraints survive and how
reliably they recur across seeds; B: how well the statistic is estimated
across time windows/folds) plus a small paired screening comparison
(C: DNN-only vs W+CE-Lite).  The violation-vs-test-error correlation is
*not* part of the gate; it is reported as descriptive only because it is
confounded by baseline fit quality, small fold counts, and wrong-DAG
reverse causality.

Stage C uses the same folds that produced the screening metrics, so the
gate is a screening filter, not a final held-out verdict.  Thresholds are
CLI-configurable; defaults follow the plan (>=3 constraints, >=60% seed
support, SNR >= 2.0, sign-flip <= 30%, CE-Lite win rate >= 60%).
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_THRESHOLDS = {
    "min_constraints": 3,
    "min_support_rate": 0.60,
    "min_window_snr": 2.0,
    "max_sign_flip_rate": 0.30,
    "min_win_rate": 0.60,
    "min_paired_seeds": 2,
}


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


def _arm_short(arm):
    return str(arm).split(";")[0]


def _median_or_none(series):
    if series is None:
        return None
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _apply_gate(runs, stability, paired, thresholds):
    """Return one Go/No-Go screening row per dataset.

    A: constraint quantity + cross-seed support.
    B: cross-window (or cross-fold fallback) SNR + sign-flip rate.
    C: paired DNN-only vs W+CE-Lite win rate over paired seeds.
    """
    if runs.empty:
        return pd.DataFrame()

    work = runs.copy()
    work["arm_short"] = work["arm"].map(_arm_short)
    ce_runs = work[work["arm_short"] == "ce_active"]
    if ce_runs.empty:
        ce_runs = work

    stability_work = stability.copy()
    if not stability_work.empty:
        stability_work["arm_short"] = stability_work["arm"].map(_arm_short)
        stability_ce = stability_work[stability_work["arm_short"] == "ce_active"]
        if stability_ce.empty:
            stability_ce = stability_work
    else:
        stability_ce = stability_work

    paired_agg = {}
    if isinstance(paired, pd.DataFrame) and not paired.empty and "dataset" in paired:
        if "ce_lite_win_rate" in paired and "n_paired_seeds" in paired:
            for _, row in paired.drop_duplicates("dataset").iterrows():
                paired_agg[row["dataset"]] = (
                    row.get("ce_lite_win_rate"),
                    row.get("n_paired_seeds"),
                )

    gate_rows = []
    for dataset in sorted(set(work["dataset"])):
        dataset_runs = work[work["dataset"] == dataset]
        dataset_ce_runs = ce_runs[ce_runs["dataset"] == dataset]

        # Stage A: cross-seed constraint quantity and support.
        if not stability_ce.empty:
            keys = stability_ce[stability_ce["dataset"] == dataset]
        else:
            keys = stability_ce
        n_union = int(keys["constraint_key"].nunique()) if not keys.empty else 0
        if not keys.empty:
            support = keys.drop_duplicates("constraint_key")["run_support_rate"]
            support_fraction = float((support >= thresholds["min_support_rate"]).mean())
        else:
            support_fraction = 0.0
        stage_a_pass = (
            n_union >= thresholds["min_constraints"]
            and support_fraction >= thresholds["min_support_rate"]
        )

        # Stage B: statistic estimability. Prefer explicit windows; fall back
        # to cross-fold spread when windows are unavailable (e.g. HAC-only
        # monthly data with too few rows for five >=100-row windows).
        snr_window = _median_or_none(dataset_ce_runs.get("median_window_abs_mean_over_spread"))
        flip_window = _median_or_none(dataset_ce_runs.get("median_window_sign_flip_rate_descriptive"))
        if snr_window is not None:
            snr_value, snr_source = snr_window, "window"
        else:
            snr_value, snr_source = _median_or_none(dataset_ce_runs.get("median_fold_abs_mean_over_sd")), "fold"
        if flip_window is not None:
            flip_value, flip_source = flip_window, "window"
        else:
            flip_value, flip_source = _median_or_none(dataset_ce_runs.get("median_fold_sign_flip_rate")), "fold"
        snr_ok = snr_value is not None and snr_value >= thresholds["min_window_snr"]
        flip_ok = flip_value is None or flip_value <= thresholds["max_sign_flip_rate"]
        stage_b_pass = bool(snr_ok and flip_ok)

        # Stage C: paired screening comparison.
        win_rate, n_paired = paired_agg.get(dataset, (None, None))
        win_rate = None if win_rate is None or pd.isna(win_rate) else float(win_rate)
        n_paired = None if n_paired is None or pd.isna(n_paired) else int(n_paired)
        stage_c_pass = (
            win_rate is not None
            and n_paired is not None
            and n_paired >= thresholds["min_paired_seeds"]
            and win_rate >= thresholds["min_win_rate"]
        )

        reasons = []
        if not stage_a_pass:
            reasons.append("A:constraints/support")
        if not stage_b_pass:
            reasons.append("B:snr/sign_flip")
        if not stage_c_pass:
            reasons.append("C:paired_win_rate")
        # The pre-audit gate decides Go/No-Go from constraint quality alone
        # (A and B) because it runs before the paired CE-Lite comparison.
        preaudit_pass = bool(stage_a_pass and stage_b_pass)
        gate_rows.append(
            {
                "dataset": dataset,
                "n_runs": int(dataset_runs["run_dir"].nunique()),
                "n_ce_runs": int(dataset_ce_runs["run_dir"].nunique()) if not dataset_ce_runs.empty else 0,
                "stage_a_n_constraints_union": n_union,
                "stage_a_support_fraction": support_fraction,
                "stage_a_pass": bool(stage_a_pass),
                "stage_b_snr": snr_value,
                "stage_b_snr_source": snr_source,
                "stage_b_sign_flip_rate": flip_value,
                "stage_b_sign_flip_source": flip_source,
                "stage_b_pass": bool(stage_b_pass),
                "stage_c_win_rate": win_rate,
                "stage_c_n_paired_seeds": n_paired,
                "stage_c_pass": bool(stage_c_pass),
                "preaudit_decision": "Go" if preaudit_pass else "No-Go",
                "decision": "Go" if (stage_a_pass and stage_b_pass and stage_c_pass) else "No-Go",
                "failed_stages": ";".join(reasons) if reasons else "none",
            }
        )
    return pd.DataFrame(gate_rows)


def summarize(root, output_dir, thresholds=None):
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
    paired = pd.DataFrame()
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

    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    gate = _apply_gate(runs, stability, paired, thresholds)
    gate_path = output_dir / "ce_preaudit_gate.csv"
    gate.to_csv(gate_path, index=False)
    return runs_path, stability_path, paired_path, gate_path, len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing per-run Hydra output folders.")
    parser.add_argument("--output-dir", help="Defaults to --root.")
    parser.add_argument(
        "--min-constraints",
        type=int,
        default=DEFAULT_THRESHOLDS["min_constraints"],
        help="Stage A: minimum d-separation constraints in the cross-seed union.",
    )
    parser.add_argument(
        "--min-support-rate",
        type=float,
        default=DEFAULT_THRESHOLDS["min_support_rate"],
        help="Stage A: minimum fraction of constraints recurring across seeds.",
    )
    parser.add_argument(
        "--min-window-snr",
        type=float,
        default=DEFAULT_THRESHOLDS["min_window_snr"],
        help="Stage B: minimum |mean|/spread (window, else fold).",
    )
    parser.add_argument(
        "--max-sign-flip-rate",
        type=float,
        default=DEFAULT_THRESHOLDS["max_sign_flip_rate"],
        help="Stage B: maximum cross-window (else fold) sign-flip rate.",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=DEFAULT_THRESHOLDS["min_win_rate"],
        help="Stage C: minimum paired CE-Lite win rate over DNN-only.",
    )
    parser.add_argument(
        "--min-paired-seeds",
        type=int,
        default=DEFAULT_THRESHOLDS["min_paired_seeds"],
        help="Stage C: minimum paired seeds before the win rate is trusted.",
    )
    parser.add_argument(
        "--stage",
        choices=("preaudit", "full"),
        default="full",
        help=(
            "Which decision to headline: 'preaudit' uses stage A+B only "
            "(constraint quality, run before the paired comparison); 'full' "
            "also requires stage C (paired CE-Lite win rate)."
        ),
    )
    args = parser.parse_args()
    thresholds = {
        "min_constraints": args.min_constraints,
        "min_support_rate": args.min_support_rate,
        "min_window_snr": args.min_window_snr,
        "max_sign_flip_rate": args.max_sign_flip_rate,
        "min_win_rate": args.min_win_rate,
        "min_paired_seeds": args.min_paired_seeds,
    }
    runs_path, stability_path, paired_path, gate_path, n_runs = summarize(
        args.root, args.output_dir, thresholds=thresholds
    )
    print(f"Summarized {n_runs} run folders.")
    print(f"Run summary: {runs_path}")
    print(f"Constraint stability: {stability_path}")
    print(f"Paired DNN-only vs W+CE-Lite: {paired_path}")
    print(f"Three-stage Go/No-Go gate: {gate_path}")
    print(
        "Gate thresholds: "
        f"A: >= {thresholds['min_constraints']} constraints and "
        f">= {thresholds['min_support_rate']:.0%} seed support; "
        f"B: SNR >= {thresholds['min_window_snr']} and sign-flip <= "
        f"{thresholds['max_sign_flip_rate']:.0%}; "
        f"C: win rate >= {thresholds['min_win_rate']:.0%} over "
        f">= {thresholds['min_paired_seeds']} paired seeds."
    )
    try:
        gate = pd.read_csv(gate_path)
    except (OSError, pd.errors.EmptyDataError):
        gate = pd.DataFrame()
    if not gate.empty:
        column = "preaudit_decision" if args.stage == "preaudit" else "decision"
        if column in gate:
            counts = gate[column].value_counts().to_dict()
            go = ", ".join(
                f"{row.dataset}" for _, row in gate.iterrows() if row[column] == "Go"
            )
            print(
                f"[{args.stage}] decision column '{column}': "
                f"{counts}. Go datasets: {go if go else '(none)'}"
            )
    print(
        "The gate is a screening filter on the same folds; it is not a final "
        "held-out verdict. Violation-vs-error correlations remain descriptive."
    )


if __name__ == "__main__":
    main()
