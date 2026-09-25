#!/usr/bin/env python3
"""Summarise the Phase-1 synthetic oracle screen.

This is deliberately independent from MLflow: a PBS job can write its own
partial report beside the copied ``multirun`` files, and all finished jobs can
later be scanned together for the final paired report.  The report is a
diagnostic, not an automatic pass/fail test.  In particular, a near-zero
linear residual covariance is meaningful for the Gaussian SEM screen, but it
is not evidence that the same statistic is valid for arbitrary real data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import networkx as nx
import pandas as pd
import yaml


ARM_LABELS = {
    "s0_no_constraint": "S0 — MSE-only DNN",
    "s1_true_w": "S1 — true-W moment penalty",
    "s2_true_independent": "S2 — true-DAG independent CE",
    "s3_true_independent_dependent_exploratory": "S3 — terminal-target dependent negative control",
    "s4_estimated_independent": "S4 — estimated-DAG independent CE",
    "s4_estimated_independent_reference": "S4R — estimated-DAG independent CE reference",
    "s5_posterior_stable_independent": "S5 — posterior-stable independent CE",
    "s6_pbm_zero_tolerance_independent": "S6 — PBM zero-tolerance independent CE",
    "s7_pbm_soft_tolerance_independent": "S7 — PBM tolerance soft independent CE",
    "s8_all_dsep_unpruned": "S8 — all d-separators unpruned",
    "s9_all_dsep_pruned": "S9 — all d-separators subset-pruned",
    "s0_dense_reference": "D0 — dense estimated-DAG reference",
    "s1_dense_time_limit": "D1 — longer MILP time limit",
    "s2_dense_big_m": "D2 — big-M/weight-bound sensitivity",
    "s3_dense_l1": "D3 — historical l1 regularisation",
    "s4_dense_release_threshold": "D4 — larger released-edge threshold",
    "s5_dense_pruned_ci": "D5 — minimal-set CI pruning",
    "s6_dense_posterior_stable": "D6 — posterior-stable CI graph",
    "s7_dense_edge_penalty": "D7 — explicit edge-cardinality penalty",
    "s8_dense_max_parents": "D8 — maximum-parent constraint",
    "s9_dense_edge_penalty_max_parents": "D9 — edge penalty plus parent cap",
    "s10_dense_clique_cap": "D10 — skeleton clique-cap lazy cuts",
    "s11_edge_penalty_001": "D11 — edge penalty 0.01",
    "s12_edge_penalty_003": "D12 — edge penalty 0.03",
    "s13_edge_penalty_010": "D13 — edge penalty 0.10",
    "s14_edge_penalty_030": "D14 — edge penalty 0.30",
    "s15_max_parents_006": "D15 — maximum-parent cap 6",
    "s16_edge_penalty_003_max_parents_004": "D16 — edge penalty 0.03 plus parent cap 4",
    "s17_edge_penalty_003_max_parents_006": "D17 — edge penalty 0.03 plus parent cap 6",
}
REQUIRED_ARMS = ("s0_no_constraint", "s1_true_w", "s2_true_independent", "s4_estimated_independent")
BASELINE_ARM = "s0_no_constraint"


def required_arms_for_prefix(prefix: str) -> tuple[str, ...]:
    """Return the control arms required by the selected Phase-1 screen."""
    if prefix.startswith("PLAN01D"):
        return (
            "s0_dense_reference",
            "s1_dense_time_limit",
            "s2_dense_big_m",
            "s3_dense_l1",
            "s4_dense_release_threshold",
            "s5_dense_pruned_ci",
            "s6_dense_posterior_stable",
            "s7_dense_edge_penalty",
            "s8_dense_max_parents",
            "s9_dense_edge_penalty_max_parents",
            "s10_dense_clique_cap",
        )
    if prefix.startswith("PLAN01A"):
        return (
            "s0_no_constraint",
            "s1_true_w",
            "s2_true_independent",
            "s4_estimated_independent_reference",
            "s5_posterior_stable_independent",
            "s6_pbm_zero_tolerance_independent",
            "s7_pbm_soft_tolerance_independent",
            "s8_all_dsep_unpruned",
            "s9_all_dsep_pruned",
        )
    return REQUIRED_ARMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired Phase-1 synthetic-oracle summary tables."
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=None,
        help="Directory to scan recursively for Hydra run directories. Repeat for multiple PBS jobs (default: multirun).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1",
        help="Directory for CSV and Markdown summaries (default: results/phase1).",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="PLAN01_SYNTHETIC",
        help="Only scan configs whose experiment starts with this prefix.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of deterministic bootstrap samples for paired NMSE intervals.",
    )
    return parser.parse_args()


def nested_value(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    series = series[np.isfinite(series)]
    return float(series.mean()) if len(series) else float("nan")


def finite_std(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    series = series[np.isfinite(series)]
    return float(series.std(ddof=1)) if len(series) > 1 else float("nan")


def finite_count(values: Any) -> int:
    series = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(series).sum())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def parse_experiment(experiment: str, prefix: str) -> tuple[str, float, float]:
    """Extract the deliberately explicit arm/graph/noise suffix from the launcher."""
    tail = experiment[len(prefix) :].lstrip("_")
    match = re.fullmatch(
        r"(?P<arm>s\d+_.+)_graph(?P<graph>-?\d+)_noise(?P<noise>-?\d+)",
        tail,
    )
    if match is None:
        return tail or "unparsed", float("nan"), float("nan")
    return (
        match.group("arm"),
        float(match.group("graph")),
        float(match.group("noise")),
    )


def compact_signature(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    """Fingerprint the generator's true W without loading its CSV format."""
    if not path.exists():
        return ""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def true_w_structure_metrics(path: Path) -> dict[str, float]:
    """Return structural facts needed to audit a synthetic-graph prior.

    A parent cap or clique cap is only a fair stabilization intervention when
    it does not exclude the known synthetic DAG.  These are facts about the
    *true* graph, not quality measures of the estimated graph.
    """
    unavailable = {
        "true_w_edges": float("nan"),
        "true_w_max_in_degree": float("nan"),
        "true_w_max_skeleton_clique": float("nan"),
    }
    if not path.exists():
        return unavailable
    try:
        matrix = np.asarray(np.loadtxt(path, delimiter=","), dtype=float)
    except (OSError, ValueError):
        return unavailable
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return unavailable
    active = np.abs(matrix) > 0
    np.fill_diagonal(active, False)
    skeleton = np.logical_or(active, active.T)
    graph = nx.from_numpy_array(skeleton.astype(np.uint8), create_using=nx.Graph)
    max_clique = max((len(clique) for clique in nx.find_cliques(graph)), default=0)
    return {
        "true_w_edges": float(active.sum()),
        "true_w_max_in_degree": float(active.sum(axis=0).max(initial=0)),
        "true_w_max_skeleton_clique": float(max_clique),
    }


def configured_bool(value: Any) -> bool:
    """Read YAML/Hydra booleans without treating the string ``false`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def phase1_status(arm: str, n_ci_rows: int, n_dependent_rows: int, n_w_rows: int) -> str:
    """State precisely what a row can and cannot establish.

    This prevents an absent graph-derived CI from being mistaken for a
    satisfied constraint, and prevents the legacy first-moment W penalty from
    being labelled a full structural-equation test.
    """
    if arm == "s0_no_constraint":
        return "baseline_no_graph_derived_constraint"
    if arm == "s1_true_w":
        return (
            "w_diagnostics_recorded_legacy_mean_moment_not_full_structural_test"
            if n_w_rows
            else "missing_w_audit"
        )
    if arm == "s2_true_independent":
        return (
            "inactive_no_target_endpoint_dsep_constraint"
            if not n_ci_rows
            else "observed_oracle_prediction_CI_statistics_recorded"
        )
    if arm.startswith("s3_"):
        return (
            "inactive_no_target_endpoint_dependent_constraint"
            if not n_dependent_rows
            else "exploratory_dependent_CI_diagnostic_not_hard_causal_constraint"
        )
    if arm in {"s4_estimated_independent", "s4_estimated_independent_reference"}:
        return (
            "inactive_estimated_DAG_has_no_target_endpoint_dsep_constraint"
            if not n_ci_rows
            else "estimated_DAG_observed_oracle_prediction_CI_statistics_recorded"
        )
    if arm == "s5_posterior_stable_independent":
        return (
            "inactive_posterior_stable_DAG_has_no_target_endpoint_dsep_constraint"
            if not n_ci_rows
            else "posterior_stable_independent_CI_statistics_recorded"
        )
    if arm in {"s6_pbm_zero_tolerance_independent", "s7_pbm_soft_tolerance_independent"}:
        return (
            "inactive_PBM_DAG_has_no_target_endpoint_dsep_constraint"
            if not n_ci_rows
            else "PBM_independent_CI_statistics_recorded"
        )
    if arm in {"s8_all_dsep_unpruned", "s9_all_dsep_pruned"}:
        return (
            "inactive_all_dsep_DAG_has_no_target_endpoint_dsep_constraint"
            if not n_ci_rows
            else "all_dsep_independent_CI_statistics_recorded"
        )
    if arm in {
        "s0_dense_reference",
        "s1_dense_time_limit",
        "s2_dense_big_m",
        "s3_dense_l1",
        "s4_dense_release_threshold",
        "s5_dense_pruned_ci",
        "s6_dense_posterior_stable",
        "s7_dense_edge_penalty",
        "s8_dense_max_parents",
        "s9_dense_edge_penalty_max_parents",
        "s10_dense_clique_cap",
        "s11_edge_penalty_001",
        "s12_edge_penalty_003",
        "s13_edge_penalty_010",
        "s14_edge_penalty_030",
        "s15_max_parents_006",
        "s16_edge_penalty_003_max_parents_004",
        "s17_edge_penalty_003_max_parents_006",
    }:
        return "dense_graph_stability_screen"
    return "unrecognised_phase1_arm"


def count_metrics(path: Path) -> dict[str, Any]:
    data = read_csv(path)
    if data.empty:
        return {
            "constraint_backend": "",
            "independent_constraints_mean": float("nan"),
            "dependent_constraints_mean": float("nan"),
            "alm_constraints_mean": float("nan"),
            "pbm_constraints_mean": float("nan"),
            "w_constraints_mean": float("nan"),
            "active_w_edges_mean": float("nan"),
            "total_constraints_mean": float("nan"),
            "active_constraint_folds": 0,
            "active_independent_ci_folds": 0,
            "constraint_count_folds": 0,
            "w_est_hashes": "",
            "w_cache_hits": 0,
            "w_solver_statuses": "",
            "w_solver_runtime_seconds_mean": float("nan"),
            "w_solver_mip_gap_mean": float("nan"),
            "w_solver_sol_count_mean": float("nan"),
            "w_solver_cycle_lazy_cuts_mean": float("nan"),
            "w_solver_clique_lazy_cuts_mean": float("nan"),
            "w_solver_selected_edges_mean": float("nan"),
            "w_solver_max_skeleton_clique_mean": float("nan"),
            "w_solver_edge_penalty_applied_mean": float("nan"),
            "w_solver_max_parents_applied_mean": float("nan"),
            "w_solver_true_w_edges_mean": float("nan"),
            "w_solver_estimated_w_edges_mean": float("nan"),
            "w_solver_true_edge_tp_mean": float("nan"),
            "w_solver_false_positive_edges_mean": float("nan"),
            "w_solver_false_negative_edges_mean": float("nan"),
            "w_solver_edge_precision_mean": float("nan"),
            "w_solver_edge_recall_mean": float("nan"),
            "w_solver_edge_f1_mean": float("nan"),
            "w_solver_shd_mean": float("nan"),
            "w_solver_skeleton_jaccard_mean": float("nan"),
        }
    if "stage" in data.columns:
        folds = data.loc[data["stage"].astype(str) == "cv_fold"].copy()
        data = folds if not folds.empty else data
    out: dict[str, Any] = {
        "constraint_backend": "",
        "active_constraint_folds": 0,
        "constraint_count_folds": int(len(data)),
    }
    if "backend" in data:
        choices = sorted({str(value) for value in data["backend"].dropna()})
        out["constraint_backend"] = "|".join(choices)
    for column in (
        "independent_constraints",
        "dependent_constraints",
        "alm_constraints",
        "pbm_constraints",
        "w_constraints",
        "active_w_edges",
        "total_constraints",
        "w_solver_runtime_seconds",
        "w_solver_mip_gap",
        "w_solver_sol_count",
        "w_solver_cycle_lazy_cuts",
        "w_solver_clique_lazy_cuts",
        "w_solver_selected_edges",
        "w_solver_max_skeleton_clique",
        "w_solver_edge_penalty_applied",
        "w_solver_max_parents_applied",
        "w_solver_true_w_edges",
        "w_solver_estimated_w_edges",
        "w_solver_true_edge_tp",
        "w_solver_false_positive_edges",
        "w_solver_false_negative_edges",
        "w_solver_edge_precision",
        "w_solver_edge_recall",
        "w_solver_edge_f1",
        "w_solver_shd",
        "w_solver_skeleton_jaccard",
    ):
        values = data[column] if column in data else []
        out[f"{column}_mean"] = finite_mean(values)
        out[f"{column}_sd"] = finite_std(values)
        out[f"{column}_min"] = (
            float(pd.to_numeric(values, errors="coerce").min())
            if len(values) and np.isfinite(pd.to_numeric(values, errors="coerce")).any()
            else float("nan")
        )
        out[f"{column}_max"] = (
            float(pd.to_numeric(values, errors="coerce").max())
            if len(values) and np.isfinite(pd.to_numeric(values, errors="coerce")).any()
            else float("nan")
        )
    total = pd.to_numeric(data.get("total_constraints", pd.Series(dtype=float)), errors="coerce")
    out["active_constraint_folds"] = int((total > 0).sum())
    independent = pd.to_numeric(
        data.get("independent_constraints", pd.Series(dtype=float)), errors="coerce"
    )
    out["active_independent_ci_folds"] = int((independent > 0).sum())
    if "w_est_sha256" in data:
        hashes = sorted({str(value) for value in data["w_est_sha256"].dropna() if str(value)})
        out["w_est_hashes"] = "|".join(hashes)
    else:
        out["w_est_hashes"] = ""
    if "w_cache_hit" in data:
        out["w_cache_hits"] = int(
            pd.to_numeric(data["w_cache_hit"], errors="coerce").fillna(0).sum()
        )
    if "w_solver_status" in data:
        statuses = sorted(
            {
                str(value)
                for value in data["w_solver_status"].dropna()
                if str(value)
            }
        )
        out["w_solver_statuses"] = "|".join(statuses)
    else:
        out["w_solver_statuses"] = ""
    return out


def relation_metrics(data: pd.DataFrame, relation: str) -> dict[str, Any]:
    prefix = relation
    subset = data.loc[data.get("relation", pd.Series(dtype=str)).astype(str) == relation].copy()
    out: dict[str, Any] = {f"{prefix}_audit_rows": int(len(subset))}
    for response, source in (
        ("observed", "observed_y"),
        ("oracle", "oracle"),
        ("prediction", "prediction"),
    ):
        statistic = pd.to_numeric(subset.get(f"{source}_statistic", pd.Series(dtype=float)), errors="coerce")
        violation = pd.to_numeric(subset.get(f"{source}_violation", pd.Series(dtype=float)), errors="coerce")
        enforced = pd.to_numeric(subset.get(f"{source}_enforced_statistic", pd.Series(dtype=float)), errors="coerce")
        out[f"{prefix}_{response}_mean_T"] = finite_mean(statistic)
        out[f"{prefix}_{response}_mean_abs_T"] = finite_mean(np.abs(statistic))
        out[f"{prefix}_{response}_mean_enforced_T"] = finite_mean(enforced)
        out[f"{prefix}_{response}_mean_violation"] = finite_mean(violation)
    return out


def ci_audit_metrics(path: Path) -> dict[str, Any]:
    data = read_csv(path)
    if data.empty:
        data = pd.DataFrame(columns=["relation"])
    out: dict[str, Any] = {"ci_audit_rows": int(len(data))}
    out.update(relation_metrics(data, "independent"))
    out.update(relation_metrics(data, "dependent"))
    return out


def w_audit_metrics(path: Path) -> dict[str, Any]:
    data = read_csv(path)
    if data.empty:
        return {"w_audit_rows": 0}
    # L2 and target-equation values repeat once per W coordinate.  Reduce to
    # one row per held-out fold before averaging across the run.
    group_columns = [column for column in ("fold", "stage") if column in data]
    if group_columns:
        data = data.groupby(group_columns, dropna=False).first().reset_index()
    out: dict[str, Any] = {"w_audit_rows": int(len(data))}
    for response in ("observed", "oracle", "prediction"):
        out[f"w_{response}_mean_moment_l2"] = finite_mean(data.get(f"{response}_l2", []))
        out[f"w_{response}_target_equation_mean"] = finite_mean(
            data.get(f"{response}_target_equation_mean", [])
        )
        out[f"w_{response}_target_equation_rmse"] = finite_mean(
            data.get(f"{response}_target_equation_rmse", [])
        )
    return out


def load_completed_run(config_path: Path, prefix: str) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        return None, {"run_dir": str(config_path.parent), "reason": f"unreadable_config: {exc}"}
    experiment = str(config.get("experiment", ""))
    if not experiment.startswith(prefix):
        return None, None
    cv_path = config_path.parent / "cv_errors.yaml"
    if not cv_path.exists():
        return None, {"run_dir": str(config_path.parent), "reason": "missing_cv_errors.yaml"}
    try:
        cv_errors = yaml.safe_load(cv_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        return None, {"run_dir": str(config_path.parent), "reason": f"unreadable_cv_errors: {exc}"}
    test_errors = cv_errors.get("test_errs", [])
    if not isinstance(test_errors, list) or not finite_count(test_errors):
        return None, {"run_dir": str(config_path.parent), "reason": "cv_errors_has_no_numeric_test_errs"}

    arm, graph_from_name, noise_from_name = parse_experiment(experiment, prefix)
    graph_seed = numeric(nested_value(config, "problem", "graph_seed"))
    noise_seed = numeric(nested_value(config, "problem", "noise_seed"))
    if not np.isfinite(graph_seed):
        graph_seed = graph_from_name
    if not np.isfinite(noise_seed):
        noise_seed = noise_from_name
    solver = nested_value(config, "solver", default={})
    problem = nested_value(config, "problem", default={})
    if not isinstance(solver, dict):
        solver = {}
    if not isinstance(problem, dict):
        problem = {}

    training_settings = {
        key: solver.get(key)
        for key in (
            "n_runs",
            "n_outer",
            "n_inner",
            "hidden_dim",
            "depth",
            "learning_rate",
            "weight_decay",
            "ce_batch_size",
            "ce_use_balanced_batches",
            "dag_fit_scope",
            "validation_fraction",
            "validation_split_strategy",
        )
    }
    # Seeds intentionally change *between* graph/noise realizations but must
    # be identical between arms within a realization. They therefore do not
    # belong to the across-realization training-budget consistency signature.
    seed_settings = {
        key: solver.get(key)
        for key in ("random_state", "cv_random_state", "validation_random_state")
    }
    constraint_settings = {
        key: solver.get(key)
        for key in (
            "constrained",
            "recalculate_dag",
            "time_limit",
            "target_mip_gap",
            "gurobi_seed",
            "gurobi_threads",
            "use_w_constraints",
            "use_ci_penalty",
            "ci_penalty_kind",
            "ce_constraint_backend",
            "ci_target_related_only",
            "ci_target_constraint_role",
            "ci_add_dsep_independence",
            "ci_add_collider_marginal_independence",
            "ci_add_collider_conditional_dependence",
            "ci_add_shielded_collider_dependence",
            "ci_dependent_statistic",
            "ci_dependent_margin",
            "ce_independence_tolerance",
            "ci_prune_redundant",
            "ci_dsep_all_separators",
            "w_matrix_space",
            "reg_type",
            "lambda1",
            "weights_bound",
            "nonzero_threshold",
            "edge_penalty",
            "max_parents",
            "enable_clique_constraints",
            "max_clique_size",
            "clique_callback_mode",
            "clique_cut_formulation",
            "clique_top_k",
            "max_clique_cuts_per_callback",
            "deduplicate_clique_cuts",
        )
    }
    independent_ci_generation_settings = {
        key: solver.get(key)
        for key in (
            "ci_penalty_kind",
            "ce_constraint_backend",
            "ci_threshold",
            "ci_skip_if_direct_edge",
            "ci_target_related_only",
            "ci_target_constraint_role",
            "ci_add_dsep_independence",
            "ci_add_collider_marginal_independence",
            "ci_add_collider_conditional_dependence",
            "ci_add_shielded_collider_dependence",
            "ci_max_dsep_separator_size",
            "ce_independence_tolerance",
            "ci_prune_redundant",
            "ci_dsep_all_separators",
            "edge_penalty",
            "max_parents",
        )
    }
    counts = count_metrics(config_path.parent / "constraint_counts.csv")
    ci_audit = ci_audit_metrics(config_path.parent / "constraint_stat_audit.csv")
    w_audit = w_audit_metrics(config_path.parent / "w_constraint_audit.csv")
    true_w_path = config_path.parent / "W_true.csv"
    true_structure = true_w_structure_metrics(true_w_path)
    max_parents = numeric(solver.get("max_parents"))
    clique_cap = numeric(solver.get("max_clique_size"))
    parent_cap_excludes_truth = bool(
        np.isfinite(max_parents)
        and np.isfinite(true_structure["true_w_max_in_degree"])
        and true_structure["true_w_max_in_degree"] > max_parents
    )
    clique_cap_excludes_truth = bool(
        configured_bool(solver.get("enable_clique_constraints", False))
        and np.isfinite(clique_cap)
        and np.isfinite(true_structure["true_w_max_skeleton_clique"])
        and true_structure["true_w_max_skeleton_clique"] > clique_cap
    )
    exclusion_reasons = []
    if parent_cap_excludes_truth:
        exclusion_reasons.append("max_parents")
    if clique_cap_excludes_truth:
        exclusion_reasons.append("max_clique_size")
    row: dict[str, Any] = {
        "run_dir": str(config_path.parent.resolve()),
        "run_mtime": float(config_path.stat().st_mtime),
        "experiment": experiment,
        "arm": arm,
        "arm_label": ARM_LABELS.get(arm, arm),
        "graph_type": str(problem.get("graph_type", "")),
        "graph_seed": graph_seed,
        "noise_seed": noise_seed,
        "model_seed": numeric(solver.get("random_state")),
        "test_nmse_mean": finite_mean(test_errors),
        "test_nmse_fold_sd": finite_std(test_errors),
        "test_nmse_folds": finite_count(test_errors),
        "training_settings_signature": compact_signature(training_settings),
        "seed_settings_signature": compact_signature(seed_settings),
        "constraint_settings_signature": compact_signature(constraint_settings),
        "independent_ci_generation_signature": compact_signature(
            independent_ci_generation_settings
        ),
        "w_est_hashes": counts.get("w_est_hashes", ""),
        "true_w_sha256": file_sha256(true_w_path),
        **true_structure,
        "structural_prior_excludes_true_graph": bool(exclusion_reasons),
        "structural_prior_exclusion_reasons": "|".join(exclusion_reasons),
        **{f"setting_{key}": value for key, value in training_settings.items()},
        **{f"constraint_setting_{key}": value for key, value in constraint_settings.items()},
        **counts,
        **ci_audit,
        **w_audit,
    }
    row["phase1_scope_status"] = phase1_status(
        arm,
        int(row["ci_audit_rows"]),
        int(row["dependent_audit_rows"]),
        int(row["w_audit_rows"]),
    )
    return row, None


def scan_runs(roots: list[Path], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            skipped.append({"run_dir": str(root), "reason": "scan_root_does_not_exist"})
            continue
        for path in root.rglob("config.yaml"):
            if ".hydra" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            row, skipped_row = load_completed_run(path, prefix)
            if row is not None:
                rows.append(row)
            elif skipped_row is not None:
                skipped.append(skipped_row)
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def latest_unique_runs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["graph_type", "arm", "graph_seed", "noise_seed"]
    ordered = raw.sort_values([*keys, "run_mtime", "run_dir"], kind="stable")
    newest = ordered.drop_duplicates(keys, keep="last").copy()
    duplicate = ordered.loc[~ordered.index.isin(newest.index)].copy()
    return newest.sort_values(keys, kind="stable").reset_index(drop=True), duplicate.reset_index(drop=True)


def bootstrap_mean_interval(values: pd.Series, samples: int, seed: int) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_deltas(details: pd.DataFrame) -> pd.DataFrame:
    key = ["graph_type", "graph_seed", "noise_seed"]
    baseline_arm = (
        "s0_dense_reference"
        if "s0_dense_reference" in set(details.get("arm", pd.Series(dtype=str)).astype(str))
        else BASELINE_ARM
    )
    baseline = details.loc[details["arm"] == baseline_arm, [*key, "test_nmse_mean"]].rename(
        columns={"test_nmse_mean": "s0_test_nmse_mean"}
    )
    paired = details.merge(baseline, how="left", on=key)
    paired["paired_to_s0"] = np.isfinite(pd.to_numeric(paired["s0_test_nmse_mean"], errors="coerce"))
    paired["test_nmse_delta_vs_s0"] = paired["test_nmse_mean"] - paired["s0_test_nmse_mean"]
    return paired


def one_value(values: pd.Series) -> Any:
    clean = [value for value in values.dropna().unique().tolist()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return "MIXED"


def grouped_summary(details: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    paired = paired_deltas(details)
    rows: list[dict[str, Any]] = []
    group_columns = ["graph_type", "arm", "arm_label"]
    for group_index, ((graph_type, arm, label), group) in enumerate(
        paired.groupby(group_columns, dropna=False, sort=True)
    ):
        nmse_low, nmse_high = bootstrap_mean_interval(
            group["test_nmse_mean"], bootstrap_samples, 1000 + group_index
        )
        paired_group = group.loc[group["paired_to_s0"]].copy()
        delta_low, delta_high = bootstrap_mean_interval(
            paired_group["test_nmse_delta_vs_s0"], bootstrap_samples, 2000 + group_index
        )
        row: dict[str, Any] = {
            "graph_type": graph_type,
            "arm": arm,
            "arm_label": label,
            "n_completed_runs": int(len(group)),
            "n_graph_seeds": int(group["graph_seed"].nunique(dropna=True)),
            "n_noise_seeds": int(group["noise_seed"].nunique(dropna=True)),
            "n_paired_to_s0": int(len(paired_group)),
            "n_missing_s0_pair": int(len(group) - len(paired_group)),
            "test_nmse_mean": finite_mean(group["test_nmse_mean"]),
            "test_nmse_run_sd": finite_std(group["test_nmse_mean"]),
            "test_nmse_bootstrap_ci_low": nmse_low,
            "test_nmse_bootstrap_ci_high": nmse_high,
            "test_nmse_delta_vs_s0_mean": finite_mean(paired_group["test_nmse_delta_vs_s0"]),
            "test_nmse_delta_vs_s0_bootstrap_ci_low": delta_low,
            "test_nmse_delta_vs_s0_bootstrap_ci_high": delta_high,
            "n_unique_training_settings": int(group["training_settings_signature"].nunique()),
            "training_settings_consistent": bool(group["training_settings_signature"].nunique() == 1),
            "n_unique_constraint_settings": int(group["constraint_settings_signature"].nunique()),
            "constraint_settings_consistent": bool(group["constraint_settings_signature"].nunique() == 1),
            "phase1_scope_status": "|".join(sorted(group["phase1_scope_status"].dropna().unique())),
        }
        for column in (
            "setting_n_runs",
            "setting_n_outer",
            "setting_n_inner",
            "setting_hidden_dim",
            "setting_depth",
            "setting_learning_rate",
            "setting_weight_decay",
            "setting_ce_batch_size",
            "setting_ce_use_balanced_batches",
            "setting_dag_fit_scope",
            "constraint_setting_recalculate_dag",
            "constraint_setting_time_limit",
            "constraint_setting_target_mip_gap",
            "constraint_setting_gurobi_seed",
            "constraint_setting_gurobi_threads",
            "constraint_setting_use_w_constraints",
            "constraint_setting_use_ci_penalty",
            "constraint_setting_ci_penalty_kind",
            "constraint_setting_ce_constraint_backend",
            "constraint_setting_ci_target_constraint_role",
            "constraint_setting_ci_add_dsep_independence",
            "constraint_setting_ci_add_collider_conditional_dependence",
            "constraint_setting_ci_dependent_statistic",
            "constraint_setting_ce_independence_tolerance",
            "constraint_setting_ci_prune_redundant",
            "constraint_setting_ci_dsep_all_separators",
            "constraint_setting_w_matrix_space",
            "constraint_setting_reg_type",
            "constraint_setting_lambda1",
            "constraint_setting_weights_bound",
            "constraint_setting_nonzero_threshold",
            "constraint_setting_edge_penalty",
            "constraint_setting_max_parents",
            "constraint_setting_enable_clique_constraints",
            "constraint_setting_max_clique_size",
            "constraint_setting_clique_callback_mode",
            "constraint_setting_clique_cut_formulation",
            "constraint_setting_clique_top_k",
            "constraint_setting_max_clique_cuts_per_callback",
            "constraint_setting_deduplicate_clique_cuts",
            "w_est_hashes",
            "w_solver_statuses",
        ):
            row[column] = one_value(group[column]) if column in group else ""
        for column in (
            "independent_constraints_mean",
            "independent_constraints_sd",
            "independent_constraints_min",
            "independent_constraints_max",
            "dependent_constraints_mean",
            "dependent_constraints_sd",
            "dependent_constraints_min",
            "dependent_constraints_max",
            "alm_constraints_mean",
            "pbm_constraints_mean",
            "w_constraints_mean",
            "active_w_edges_mean",
            "active_w_edges_sd",
            "active_w_edges_min",
            "active_w_edges_max",
            "constraint_setting_time_limit",
            "constraint_setting_weights_bound",
            "constraint_setting_nonzero_threshold",
            "constraint_setting_reg_type",
            "constraint_setting_lambda1",
            "constraint_setting_edge_penalty",
            "constraint_setting_max_parents",
            "true_w_edges",
            "true_w_max_in_degree",
            "true_w_max_skeleton_clique",
            "total_constraints_mean",
            "total_constraints_sd",
            "total_constraints_min",
            "total_constraints_max",
            "active_constraint_folds",
            "active_independent_ci_folds",
            "constraint_count_folds",
            "w_cache_hits",
            "w_solver_runtime_seconds_mean",
            "w_solver_runtime_seconds_sd",
            "w_solver_runtime_seconds_min",
            "w_solver_runtime_seconds_max",
            "w_solver_mip_gap_mean",
            "w_solver_mip_gap_sd",
            "w_solver_mip_gap_min",
            "w_solver_mip_gap_max",
            "w_solver_sol_count_mean",
            "w_solver_sol_count_sd",
            "w_solver_sol_count_min",
            "w_solver_sol_count_max",
            "w_solver_cycle_lazy_cuts_mean",
            "w_solver_cycle_lazy_cuts_sd",
            "w_solver_cycle_lazy_cuts_min",
            "w_solver_cycle_lazy_cuts_max",
            "w_solver_clique_lazy_cuts_mean",
            "w_solver_clique_lazy_cuts_sd",
            "w_solver_clique_lazy_cuts_min",
            "w_solver_clique_lazy_cuts_max",
            "w_solver_selected_edges_mean",
            "w_solver_selected_edges_sd",
            "w_solver_selected_edges_min",
            "w_solver_selected_edges_max",
            "w_solver_max_skeleton_clique_mean",
            "w_solver_max_skeleton_clique_sd",
            "w_solver_max_skeleton_clique_min",
            "w_solver_max_skeleton_clique_max",
            "w_solver_edge_penalty_applied_mean",
            "w_solver_max_parents_applied_mean",
            "w_solver_true_w_edges_mean",
            "w_solver_true_w_edges_sd",
            "w_solver_true_w_edges_min",
            "w_solver_true_w_edges_max",
            "w_solver_estimated_w_edges_mean",
            "w_solver_estimated_w_edges_sd",
            "w_solver_estimated_w_edges_min",
            "w_solver_estimated_w_edges_max",
            "w_solver_true_edge_tp_mean",
            "w_solver_false_positive_edges_mean",
            "w_solver_false_negative_edges_mean",
            "w_solver_edge_precision_mean",
            "w_solver_edge_recall_mean",
            "w_solver_edge_f1_mean",
            "w_solver_shd_mean",
            "w_solver_skeleton_jaccard_mean",
            "ci_audit_rows",
            "independent_audit_rows",
            "dependent_audit_rows",
            "independent_observed_mean_T",
            "independent_observed_mean_abs_T",
            "independent_oracle_mean_T",
            "independent_oracle_mean_abs_T",
            "independent_prediction_mean_T",
            "independent_prediction_mean_abs_T",
            "independent_observed_mean_violation",
            "independent_oracle_mean_violation",
            "independent_prediction_mean_violation",
            "dependent_observed_mean_enforced_T",
            "dependent_oracle_mean_enforced_T",
            "dependent_prediction_mean_enforced_T",
            "dependent_observed_mean_violation",
            "dependent_oracle_mean_violation",
            "dependent_prediction_mean_violation",
            "w_observed_mean_moment_l2",
            "w_oracle_mean_moment_l2",
            "w_prediction_mean_moment_l2",
            "w_observed_target_equation_rmse",
            "w_oracle_target_equation_rmse",
            "w_prediction_target_equation_rmse",
        ):
            row[column] = finite_mean(group[column]) if column in group else float("nan")
        row["n_structural_prior_excludes_true_graph"] = int(
            group.get(
                "structural_prior_excludes_true_graph",
                pd.Series(False, index=group.index),
            ).fillna(False).astype(bool).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["graph_type", "arm"], kind="stable")


def missing_pairs(
    details: pd.DataFrame,
    required_arms: tuple[str, ...] = REQUIRED_ARMS,
) -> pd.DataFrame:
    observed_arms = set(details["arm"].astype(str))
    required = [*required_arms]
    if any(arm.startswith("s3_") for arm in observed_arms):
        required.extend(sorted(arm for arm in observed_arms if arm.startswith("s3_")))
    rows: list[dict[str, Any]] = []
    for (graph_type, graph_seed, noise_seed), group in details.groupby(
        ["graph_type", "graph_seed", "noise_seed"], dropna=False
    ):
        present = set(group["arm"].astype(str))
        missing = [arm for arm in required if arm not in present]
        rows.append(
            {
                "graph_type": graph_type,
                "graph_seed": graph_seed,
                "noise_seed": noise_seed,
                "present_arms": "|".join(sorted(present)),
                "missing_required_arms": "|".join(missing),
                "complete_required_pair": not missing,
            }
        )
    return pd.DataFrame(rows).sort_values(["graph_type", "graph_seed", "noise_seed"], kind="stable")


def pair_consistency(
    details: pd.DataFrame,
    missing: pd.DataFrame,
    independent_reference_arms: tuple[str, str] = (
        "s2_true_independent",
        "s4_estimated_independent",
    ),
    estimated_w_arms: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Verify the controls that make Phase-1 comparisons interpretable.

    The true-DAG/estimated-DAG reference arms intentionally differ in graph
    source, but use the same independent
    CI *builder* settings. All arms should use the same graph/noise realization,
    NN seed and update budget. For the estimated-DAG arms, the recorded W hash
    must also agree; otherwise a backend comparison is confounded by a second
    graph solve.
    """
    rows: list[dict[str, Any]] = []
    keys = ["graph_type", "graph_seed", "noise_seed"]
    completed = {
        (row.graph_type, row.graph_seed, row.noise_seed): row
        for row in missing.itertuples(index=False)
    }
    for key, group in details.groupby(keys, dropna=False, sort=True):
        graph_type, graph_seed, noise_seed = key
        present = set(group["arm"].astype(str))
        missing_row = completed.get(key)
        independent_ci = group.loc[group["arm"].isin(independent_reference_arms)]
        recalc = group.get(
            "constraint_setting_recalculate_dag", pd.Series(False, index=group.index)
        ).astype(str).str.lower()
        true_w_group = group.loc[recalc.isin({"false", "0", "no"})]
        hashes = [value for value in true_w_group["true_w_sha256"].dropna().astype(str) if value]
        same_true_w = bool(hashes) and len(set(hashes)) == 1
        estimated_w_group = group.loc[group["arm"].isin(estimated_w_arms)]
        estimated_w_hashes = [
            value
            for value in estimated_w_group.get("w_est_hashes", pd.Series(dtype=str)).dropna().astype(str)
            if value
        ]
        same_estimated_w = (
            bool(estimated_w_arms)
            and len(estimated_w_group) == len(estimated_w_arms)
            and len(estimated_w_hashes) == len(estimated_w_arms)
            and len(set(estimated_w_hashes)) == 1
        )
        rows.append(
            {
                "graph_type": graph_type,
                "graph_seed": graph_seed,
                "noise_seed": noise_seed,
                "present_arms": "|".join(sorted(present)),
                "complete_required_pair": (
                    bool(missing_row.complete_required_pair)
                    if missing_row is not None
                    else False
                ),
                "shared_training_budget": bool(
                    group["training_settings_signature"].nunique(dropna=False) == 1
                ),
                "shared_model_seed": bool(group["model_seed"].nunique(dropna=False) == 1),
                "same_true_W": same_true_w,
                "same_estimated_W": same_estimated_w,
                "same_independent_CI_generation": (
                    bool(len(independent_ci) == 2)
                    and independent_ci["independent_ci_generation_signature"].nunique(dropna=False) == 1
                ),
                "pair_status": (
                    "ready_for_paired_comparison"
                    if (
                        missing_row is not None
                        and bool(missing_row.complete_required_pair)
                        and group["training_settings_signature"].nunique(dropna=False) == 1
                        and group["model_seed"].nunique(dropna=False) == 1
                        and (len(true_w_group) < 2 or same_true_w)
                        and (not estimated_w_arms or same_estimated_w)
                        and len(independent_ci) == 2
                        and independent_ci["independent_ci_generation_signature"].nunique(dropna=False) == 1
                    )
                    else "incomplete_or_control_mismatch"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable")


def add_pair_controls(summary: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return summary
    aggregates = (
        pairs.groupby("graph_type", dropna=False)
        .agg(
            n_phase1_realizations=("graph_seed", "size"),
            complete_required_pairs=("complete_required_pair", "sum"),
            shared_training_budget_pairs=("shared_training_budget", "sum"),
            shared_model_seed_pairs=("shared_model_seed", "sum"),
            same_true_W_pairs=("same_true_W", "sum"),
            same_estimated_W_pairs=("same_estimated_W", "sum"),
            same_independent_CI_generation_pairs=("same_independent_CI_generation", "sum"),
            ready_for_paired_comparison_pairs=(
                "pair_status",
                lambda values: int((values == "ready_for_paired_comparison").sum()),
            ),
        )
        .reset_index()
    )
    return summary.merge(aggregates, how="left", on="graph_type")


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    view = data.loc[:, [column for column in columns if column in data]].copy()
    if view.empty:
        return "_No completed Phase-1 runs were found._\n"
    for column in view:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.5g}")
    names = list(view.columns)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("|", "/") for column in names) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, summary: pd.DataFrame, missing: pd.DataFrame) -> None:
    text = """# Phase 1 synthetic-oracle summary

This report compares arms only within the same graph seed and innovation-noise
seed.  `test_nmse_delta_vs_s0` is arm minus S0, so a negative value favours the
arm on predictive NMSE.  The bootstrap interval is descriptive and does not
make cross-validation folds independent observations.

For CI rows, `T` is the configured conditional-expectation statistic (linear
residual covariance in this Gaussian SEM screen).  The three T columns are for
observed `Y`, the analytic `E[Y|X]` oracle, and `Y_hat`, respectively.  A row
with no CI audit is *inactive*, not proof that a constraint is satisfied.

The S1 W penalty is the existing standardized first-moment condition.  Its
raw target-equation RMSE is a separate diagnostic, not a claim that the loss
enforces the full structural equation.  `w_solver_edge_*` compares the
fold-local estimated support with `W_true.csv`; precision/recall/F1 are
directed-edge metrics, and `w_solver_shd` counts one operation per mismatched
unordered node pair (extra, missing, or reversed edge).  S3 remains
exploratory.

## Grouped summary

"""
    text += markdown_table(
        summary,
        [
            "graph_type",
            "arm",
            "n_completed_runs",
            "n_paired_to_s0",
            "ready_for_paired_comparison_pairs",
            "test_nmse_mean",
            "test_nmse_delta_vs_s0_mean",
            "test_nmse_delta_vs_s0_bootstrap_ci_low",
            "test_nmse_delta_vs_s0_bootstrap_ci_high",
            "independent_constraints_mean",
            "independent_constraints_sd",
            "independent_constraints_min",
            "independent_constraints_max",
            "dependent_constraints_mean",
            "dependent_constraints_sd",
            "active_w_edges_mean",
            "active_w_edges_sd",
            "active_w_edges_min",
            "active_w_edges_max",
            "true_w_edges",
            "true_w_max_in_degree",
            "true_w_max_skeleton_clique",
            "n_structural_prior_excludes_true_graph",
            "total_constraints_mean",
            "total_constraints_sd",
            "total_constraints_min",
            "total_constraints_max",
            "w_solver_statuses",
            "w_solver_runtime_seconds_mean",
            "w_solver_mip_gap_mean",
            "w_solver_clique_lazy_cuts_mean",
            "w_solver_selected_edges_mean",
            "w_solver_max_skeleton_clique_mean",
            "w_solver_edge_penalty_applied_mean",
            "w_solver_max_parents_applied_mean",
            "w_solver_true_w_edges_mean",
            "w_solver_estimated_w_edges_mean",
            "w_solver_edge_precision_mean",
            "w_solver_edge_recall_mean",
            "w_solver_edge_f1_mean",
            "w_solver_shd_mean",
            "w_solver_skeleton_jaccard_mean",
            "ci_audit_rows",
            "independent_observed_mean_abs_T",
            "independent_oracle_mean_abs_T",
            "independent_prediction_mean_abs_T",
            "phase1_scope_status",
        ],
    )
    text += "\n## Required-arm pairing check\n\n"
    text += markdown_table(
        missing,
        [
            "graph_type",
            "graph_seed",
            "noise_seed",
            "complete_required_pair",
            "missing_required_arms",
        ],
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    scan_roots = [Path(path) for path in (args.scan_root or ["multirun"])]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, skipped = scan_runs(scan_roots, args.experiment_prefix)
    if raw.empty:
        skipped.to_csv(output_dir / "phase1_skipped_runs.csv", index=False)
        raise SystemExit(
            "No completed Phase-1 runs found. Check --scan-root and --experiment-prefix; "
            f"details written to {output_dir / 'phase1_skipped_runs.csv'}."
        )
    details, duplicates = latest_unique_runs(raw)
    required_arms = required_arms_for_prefix(args.experiment_prefix)
    if args.experiment_prefix.startswith("PLAN01A"):
        independent_reference_arms = (
            "s2_true_independent",
            "s4_estimated_independent_reference",
        )
    elif args.experiment_prefix.startswith("PLAN01D"):
        # Both arms use the same CI builder; only the MILP control differs.
        independent_reference_arms = (
            "s0_dense_reference",
            "s1_dense_time_limit",
        )
    else:
        independent_reference_arms = ("s2_true_independent", "s4_estimated_independent")
    estimated_w_arms = (
        (
            "s4_estimated_independent_reference",
            "s5_posterior_stable_independent",
            "s6_pbm_zero_tolerance_independent",
            "s7_pbm_soft_tolerance_independent",
            "s8_all_dsep_unpruned",
            "s9_all_dsep_pruned",
        )
        if args.experiment_prefix.startswith("PLAN01A")
        else ()
    )
    missing = missing_pairs(details, required_arms=required_arms)
    pairs = pair_consistency(
        details,
        missing,
        independent_reference_arms=independent_reference_arms,
        estimated_w_arms=estimated_w_arms,
    )
    summary = add_pair_controls(
        grouped_summary(details, max(100, args.bootstrap_samples)), pairs
    )
    paired_deltas(details).sort_values(
        ["graph_type", "arm", "graph_seed", "noise_seed"], kind="stable"
    ).to_csv(output_dir / "phase1_run_summary.csv", index=False)
    summary.to_csv(output_dir / "phase1_summary.csv", index=False)
    missing.to_csv(output_dir / "phase1_missing_pairs.csv", index=False)
    pairs.to_csv(output_dir / "phase1_pair_consistency.csv", index=False)
    duplicates.to_csv(output_dir / "phase1_duplicate_runs_ignored.csv", index=False)
    skipped.to_csv(output_dir / "phase1_skipped_runs.csv", index=False)
    write_markdown(output_dir / "phase1_summary.md", summary, missing)
    print(f"Wrote Phase-1 summaries for {len(details)} unique completed runs to {output_dir}")
    if not missing.empty and not bool(missing["complete_required_pair"].all()):
        print("WARNING: required-arm pairs are incomplete; see phase1_missing_pairs.csv.")


if __name__ == "__main__":
    main()
