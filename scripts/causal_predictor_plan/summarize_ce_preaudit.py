#!/usr/bin/env python3
"""Summarize CE pre-audit artifacts and emit a staged screening report.

The gate mirrors the corrected pre-screening design: it decides only on
constraint quality (A: how many independent d-separation constraints survive
and how reliably they recur across seeds; B applies only to directional
dependence constraints, whose target is nonzero) plus a paired screening
comparison (C: W+CE-Lite vs W-only). The violation-vs-test-error correlation is
*not* part of the gate; it is reported as descriptive only because it is
confounded by baseline fit quality, small fold counts, and wrong-DAG
reverse causality.

Independent constraints target zero: their |mean|/spread ratio and sign flips
are descriptive and must not be used as a Go/No-Go gate.  Instead Stage B
requires that the observed independence statistic stays within its SE
tolerance often enough (null compatibility); directional SNR/sign-flip gates
apply only when dependence constraints are present. Stage C uses the same
folds that produced the screening metrics, so the gate is a screening filter,
not a final held-out verdict. Defaults require >=3 constraints, >=60% seed
support, >=80% null compatibility, and >=60% paired wins over W-only.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_THRESHOLDS = {
    "min_constraints": 3,
    "min_support_rate": 0.60,
    "min_null_compatibility": 0.80,
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
    solver_name = str(solver.get("name", ""))
    constrained = bool(solver.get("constrained", True))
    use_ci = bool(solver.get("use_ci_penalty", False))
    use_w = bool(solver.get("use_w_constraints", False))
    if solver_name == "mark":
        arm = "mark"
    elif not constrained and not use_ci:
        arm = "dnn_only"
    elif use_ci:
        arm = "ce_active"
    elif use_w:
        arm = "w_only"
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
        if "ce_lite_win_rate_vs_w_only" in paired and "n_paired_seeds_vs_w_only" in paired:
            for _, row in paired.drop_duplicates("dataset").iterrows():
                paired_agg[row["dataset"]] = (
                    row.get("ce_lite_win_rate_vs_w_only"),
                    row.get("n_paired_seeds_vs_w_only"),
                )

    gate_rows = []
    for dataset in sorted(set(work["dataset"])):
        dataset_runs = work[work["dataset"] == dataset]
        dataset_ce_runs = ce_runs[ce_runs["dataset"] == dataset]

        # Stage A: count/support only d-separation independence constraints.
        if not stability_ce.empty:
            keys = stability_ce[stability_ce["dataset"] == dataset]
        else:
            keys = stability_ce
        if not keys.empty:
            keys = keys[keys["constraint_key"].astype(str).str.startswith("independent|")]
        n_union = int(keys["constraint_key"].nunique()) if not keys.empty else 0
        if not keys.empty:
            support = keys.drop_duplicates("constraint_key")["seed_support_rate"]
            support_fraction = float((support >= thresholds["min_support_rate"]).mean())
        else:
            support_fraction = 0.0
        stage_a_pass = (
            n_union >= thresholds["min_constraints"]
            and support_fraction >= thresholds["min_support_rate"]
        )

        # Stage B has two applicable sub-criteria. For an independence null the
        # statistic is expected to be near zero, so |mean|/spread and sign flips
        # are not valid; instead require that the observed statistic is
        # compatible with the null (within its SE tolerance) often enough.
        # Directional SNR/sign-flip gates apply only when dependence
        # constraints are present.
        n_dependent = (
            int(
                pd.to_numeric(
                    dataset_ce_runs.get("n_dependent_constraints"), errors="coerce"
                ).fillna(0).gt(0).sum()
            )
            if not dataset_ce_runs.empty
            else 0
        )
        n_independent_median = _median_or_none(
            dataset_ce_runs.get("n_independent_constraints")
        )
        n_dependent_median = _median_or_none(
            dataset_ce_runs.get("n_dependent_constraints")
        )
        null_compatibility_median = _median_or_none(
            dataset_ce_runs.get("observed_independence_tolerance_compatibility_rate")
        )
        has_independence = n_union > 0
        has_dependence = n_dependent > 0

        null_ok = True
        if has_independence:
            null_ok = (
                null_compatibility_median is not None
                and null_compatibility_median >= thresholds["min_null_compatibility"]
            )

        snr_value = None
        flip_value = None
        snr_source = "not_applicable_no_dependent_constraints"
        flip_source = "not_applicable_no_dependent_constraints"
        directional_ok = True
        if has_dependence:
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
            directional_ok = bool(snr_ok and flip_ok)

        stage_b_pass = bool(null_ok and directional_ok)
        if has_independence and has_dependence:
            stage_b_status = "independence_null_compatibility+directional_dependence"
        elif has_independence:
            stage_b_status = "independence_null_compatibility"
        elif has_dependence:
            stage_b_status = "directional_dependence"
        else:
            stage_b_status = "no_applicable_stage_b_criterion"

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
        # Pre-audit only advances stable independence sets to a paired screen.
        # Stage C, compared with W-only, is required for an efficacy decision.
        preaudit_pass = bool(stage_a_pass and stage_b_pass)
        gate_rows.append(
            {
                "dataset": dataset,
                "n_runs": int(dataset_runs["run_dir"].nunique()),
                "n_ce_runs": int(dataset_ce_runs["run_dir"].nunique()) if not dataset_ce_runs.empty else 0,
                "stage_a_n_constraints_union": n_union,
                "stage_a_support_fraction": support_fraction,
                "stage_a_pass": bool(stage_a_pass),
                "median_independent_constraints_per_run": n_independent_median,
                "median_dependent_constraints_per_run": n_dependent_median,
                "median_observed_independence_tolerance_compatibility_rate": null_compatibility_median,
                "stage_b_null_compatibility_pass": bool(null_ok),
                "stage_b_snr": snr_value,
                "stage_b_snr_source": snr_source,
                "stage_b_sign_flip_rate": flip_value,
                "stage_b_sign_flip_source": flip_source,
                "stage_b_status": stage_b_status,
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


def summarize(root, output_dir, thresholds=None, experiment_prefix=None, planned_runs=None):
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else root
    run_rows = []
    stability_rows = []
    coverage = {"included": 0, "excluded_by_prefix": 0, "missing_metrics": 0}
    missing_metrics_rows = []

    for config_path in sorted(root.rglob("config.yaml")):
        run_dir = config_path.parent
        metadata_path = run_dir / "constraint_metadata.csv"
        stats_path = run_dir / "constraint_stat_audit.csv"
        metrics_path = run_dir / "cv_fold_metrics.csv"
        config = _read_yaml(config_path)
        experiment_name = str(config.get("experiment", "")) if isinstance(config, dict) else ""
        if experiment_prefix and not experiment_name.startswith(experiment_prefix):
            coverage["excluded_by_prefix"] += 1
            continue
        if not metrics_path.exists():
            # A run that never wrote fold metrics crashed or hit the walltime
            # limit.  Keep it visible instead of silently dropping it, or the
            # paired comparison becomes a survivors-only view.
            coverage["missing_metrics"] += 1
            missing_metrics_rows.append(
                {
                    "run_dir": str(run_dir),
                    "experiment": experiment_name,
                    "has_metadata": metadata_path.exists(),
                    "has_stat_audit": stats_path.exists(),
                }
            )
            continue
        coverage["included"] += 1
        solver = config.get("solver", {}) if isinstance(config, dict) else {}
        dataset = _dataset_key(config)
        arm = _arm_key(config)
        run_id = str(run_dir)
        seed = solver.get("random_state")
        try:
            metadata = pd.read_csv(metadata_path).fillna("") if metadata_path.exists() else pd.DataFrame()
            stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if metadata.empty:
            constraint_counts = pd.Series(dtype=float)
            keys = set()
            relation_counts = {"independent": 0, "dependent": 0}
        else:
            metadata["constraint_key"] = metadata.apply(_constraint_key, axis=1)
            constraint_counts = metadata.groupby("fold").size()
            keys = set(metadata["constraint_key"].tolist())
            unique_metadata = metadata.drop_duplicates("constraint_key")
            relation_counts = unique_metadata["relation"].astype(str).value_counts().to_dict()

        train_stats = (
            stats[stats["evaluation_split"] == "outer_train"].copy()
            if "evaluation_split" in stats
            else stats.copy()
        )
        if not train_stats.empty:
            if "constraint_key" not in train_stats:
                train_stats["constraint_key"] = train_stats.apply(_constraint_key, axis=1)
            raw_stat = pd.to_numeric(train_stats.get("observed_y_statistic"), errors="coerce")
            train_stats["observed_y_statistic"] = raw_stat
            if "relation" in train_stats:
                dependent_stats = train_stats[train_stats["relation"].astype(str) == "dependent"]
                independent_stats = train_stats[train_stats["relation"].astype(str) == "independent"]
            else:
                dependent_stats = train_stats.iloc[0:0]
                independent_stats = train_stats
            group_snr = []
            group_sign_flip = []
            for _, group in dependent_stats.groupby("constraint_key"):
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
            window_snr = _median_or_none(
                dependent_stats.get("window_abs_mean_over_spread")
            )
            window_sign_flip = _median_or_none(
                dependent_stats.get("window_sign_flip_rate")
            )
            null_violation = (
                pd.to_numeric(independent_stats["observed_violation"], errors="coerce").dropna()
                if "observed_violation" in independent_stats
                else pd.Series(dtype=float)
            )
            null_compatibility_rate = (
                float((null_violation <= 1e-12).mean()) if len(null_violation) else None
            )
            median_abs_observed_independence = (
                pd.to_numeric(
                    independent_stats["observed_y_statistic"], errors="coerce"
                ).abs().median()
                if "observed_y_statistic" in independent_stats
                else None
            )
            mean_train_violation = (
                pd.to_numeric(train_stats["prediction_violation"], errors="coerce").mean()
                if "prediction_violation" in train_stats
                else None
            )
        else:
            median_fold_snr = median_fold_flip = window_snr = window_sign_flip = None
            null_compatibility_rate = median_abs_observed_independence = None
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
                "n_independent_constraints": int(relation_counts.get("independent", 0)),
                "n_dependent_constraints": int(relation_counts.get("dependent", 0)),
                "observed_independence_tolerance_compatibility_rate": null_compatibility_rate,
                "median_abs_observed_independence_statistic": median_abs_observed_independence,
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
        seed_totals = runs.groupby(["dataset", "arm"])["seed"].nunique().rename("n_seeds")
        support = (
            stability.groupby(["dataset", "arm", "constraint_key"])["seed"]
            .nunique()
            .rename("n_seeds_with_constraint")
            .reset_index()
            .join(seed_totals, on=["dataset", "arm"])
        )
        support["seed_support_rate"] = support["n_seeds_with_constraint"] / support["n_seeds"]
        stability = support.sort_values(["dataset", "arm", "seed_support_rate"], ascending=[True, True, False])

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
        aggregates = []
        for comparator, suffix in (("w_only", "w_only"), ("dnn_only", "dnn_only")):
            if comparator not in paired or "ce_active" not in paired:
                continue
            delta_col = f"delta_ce_minus_{suffix}_nmse"
            win_col = f"ce_lite_lower_nmse_vs_{suffix}"
            paired[delta_col] = paired["ce_active"] - paired[comparator]
            paired[win_col] = paired[delta_col] < 0
            valid = paired.dropna(subset=[comparator, "ce_active"])
            if valid.empty:
                continue
            aggregate = (
                valid.groupby("dataset")
                .agg(
                    **{
                        f"n_paired_seeds_vs_{suffix}": ("seed", "nunique"),
                        f"ce_lite_win_rate_vs_{suffix}": (win_col, "mean"),
                        f"mean_delta_nmse_vs_{suffix}": (delta_col, "mean"),
                        f"median_delta_nmse_vs_{suffix}": (delta_col, "median"),
                    }
                )
                .reset_index()
            )
            aggregates.append(aggregate)
        if aggregates:
            aggregate = aggregates[0]
            for other in aggregates[1:]:
                aggregate = aggregate.merge(other, on="dataset", how="outer")
            paired = paired.merge(aggregate, on="dataset", how="left")
            # Keep the legacy field names as aliases for the actual CE
            # increment over W-only, never over a no-constraint baseline.
            if "ce_lite_win_rate_vs_w_only" in paired:
                paired["ce_lite_win_rate"] = paired["ce_lite_win_rate_vs_w_only"]
                paired["n_paired_seeds"] = paired["n_paired_seeds_vs_w_only"]
    paired.to_csv(paired_path, index=False)

    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    gate = _apply_gate(runs, stability, paired, thresholds)
    gate_path = output_dir / "ce_preaudit_gate.csv"
    gate.to_csv(gate_path, index=False)

    coverage_rows = [
        {
            "experiment_prefix": experiment_prefix,
            "n_config_dirs": coverage["included"]
            + coverage["excluded_by_prefix"]
            + coverage["missing_metrics"],
            "n_included": coverage["included"],
            "n_excluded_by_prefix": coverage["excluded_by_prefix"],
            "n_missing_metrics": coverage["missing_metrics"],
            "planned_runs": planned_runs,
            "n_failed_or_timeout": (
                planned_runs - coverage["included"] if planned_runs is not None else None
            ),
        }
    ]
    coverage_path = output_dir / "ce_preaudit_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    missing_path = output_dir / "ce_preaudit_missing_runs.csv"
    pd.DataFrame(missing_metrics_rows).to_csv(missing_path, index=False)
    return runs_path, stability_path, paired_path, gate_path, len(runs), coverage_path, missing_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing per-run Hydra output folders.")
    parser.add_argument("--output-dir", help="Defaults to --root.")
    parser.add_argument(
        "--experiment-prefix",
        help="Only summarize runs whose composed config experiment name starts with this prefix.",
    )
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
        "--min-null-compatibility",
        type=float,
        default=DEFAULT_THRESHOLDS["min_null_compatibility"],
        help=(
            "Stage B: minimum fraction of independence constraints whose "
            "observed statistic is within its SE tolerance (null compatibility)."
        ),
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
        help="Stage C: minimum paired CE-Lite win rate over W-only.",
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
            "Which decision to headline: 'preaudit' advances stable constraint "
            "sets to a paired screen; 'full' also requires stage C (CE-Lite "
            "improvement over W-only)."
        ),
    )
    parser.add_argument(
        "--planned-runs",
        type=int,
        default=None,
        help=(
            "Optional number of runs that were supposed to complete. When set, "
            "the coverage table reports n_failed_or_timeout = planned - included."
        ),
    )
    args = parser.parse_args()
    thresholds = {
        "min_constraints": args.min_constraints,
        "min_support_rate": args.min_support_rate,
        "min_null_compatibility": args.min_null_compatibility,
        "min_window_snr": args.min_window_snr,
        "max_sign_flip_rate": args.max_sign_flip_rate,
        "min_win_rate": args.min_win_rate,
        "min_paired_seeds": args.min_paired_seeds,
    }
    (
        runs_path,
        stability_path,
        paired_path,
        gate_path,
        n_runs,
        coverage_path,
        missing_path,
    ) = summarize(
        args.root,
        args.output_dir,
        thresholds=thresholds,
        experiment_prefix=args.experiment_prefix,
        planned_runs=args.planned_runs,
    )
    print(f"Summarized {n_runs} run folders.")
    print(f"Run summary: {runs_path}")
    print(f"Constraint stability: {stability_path}")
    print(f"Paired W-only / DNN-only comparisons: {paired_path}")
    print(f"Three-stage Go/No-Go gate: {gate_path}")
    print(f"Coverage / failure accounting: {coverage_path}")
    print(f"Runs missing fold metrics: {missing_path}")
    print(
        "Gate thresholds: "
        f"A: >= {thresholds['min_constraints']} independent constraints and "
        f">= {thresholds['min_support_rate']:.0%} seed support; "
        f"B: independence null compatibility >= "
        f"{thresholds['min_null_compatibility']:.0%}"
        " (plus directional SNR/sign-flip when dependence constraints exist); "
        f"C: CE-Lite win rate >= {thresholds['min_win_rate']:.0%} over W-only "
        f"with >= {thresholds['min_paired_seeds']} paired seeds."
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
