#!/usr/bin/env python3
"""Scan metacentrum_runs evidence -> reports/progress tables (v2, metric-fixed).

METRIC FIX (critical):
  recommender_utils.py already divides each fold by its normalizer and writes
  both cv_fold_metrics.csv:test_nmse and cv_errors.yaml:test_errs (identical).
  Therefore:
    * prefer cv_fold_metrics.csv:test_nmse
    * else use cv_errors.yaml:test_errs DIRECTLY
    * NEVER divide by cv_score_normalizers.yaml again  (that was the bug)

  metric_validity:
    valid                    -> cv_fold_metrics.csv:test_nmse present
    legacy_unverified        -> only cv_errors.yaml (schema not verifiable)
    invalid_double_normalized-> value computed as test_errs/normalizer (old bug)
    missing                  -> no metric

Strict evidence builders must read only metric_validity == "valid".

Outputs (reports/progress/):
  runs_metrics.csv   one row per (experiment, run_id) with provenance + validity
  arm_summary.csv    grouped by experiment
  phase_summary.csv  grouped by inferred phase
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "metacentrum_runs"
OUT = ROOT / "reports" / "progress"
OUT.mkdir(parents=True, exist_ok=True)


def load_yaml(p: Path):
    try:
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def sha(obj, n=16):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def file_fingerprint(adir: Path) -> str:
    """Fingerprint result content, not its mirror path.

    The same server run can exist once under MLflow artifacts and once under
    a copied Hydra multirun directory.  A path-based identity would count it
    twice; config-only identity would incorrectly merge independent reruns.
    """
    h = hashlib.sha256()
    for name in ("config.yaml", "cv_fold_metrics.csv", "cv_errors.yaml",
                 "cv_score_normalizers.yaml"):
        p = adir / name
        h.update(name.encode())
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:24]


def execution_cohort(adir: Path) -> str:
    """Return the server-job directory containing an artifact.

    This is a fallback cohort key for legacy runs that predate the explicit
    ``problem.evidence_batch_id`` field.  It prevents the evidence builder
    from averaging a rerun made on a later code/configuration snapshot with an
    older run that happens to share graph and noise seeds.
    """
    parts = adir.relative_to(ROOT).parts
    try:
        return parts[parts.index("metacentrum_runs") + 1]
    except (ValueError, IndexError):
        return "local_or_unknown"


def iter_run_dirs():
    seen = set()

    def fresh(adir):
        key = str(adir)
        if key in seen:
            return False
        seen.add(key)
        return True

    # layout A: mlruns/<exp_id>/<run_id>/artifacts/config.yaml
    for cfg in MR.rglob("artifacts/config.yaml"):
        adir = cfg.parent
        if not fresh(adir):
            continue
        yield adir, adir.parent.parent.name, adir.parent.name
    # layout B: multirun/**/<run>/config.yaml  (+ cv_errors.yaml / cv_fold_metrics.csv)
    for ce in MR.rglob("cv_errors.yaml"):
        d = ce.parent
        if d.name == "artifacts" or not (d / "config.yaml").exists():
            continue
        if not fresh(d):
            continue
        yield d, d.name, d.name


def read_metric(adir: Path):
    """Return (nmse, folds, validity). NO double normalization."""
    fm = adir / "cv_fold_metrics.csv"
    if fm.exists():
        try:
            d = pd.read_csv(fm)
            if "test_nmse" in d.columns:
                v = pd.to_numeric(d["test_nmse"], errors="coerce").dropna()
                if len(v):
                    return float(v.mean()), v.tolist(), "valid"
        except Exception:
            pass
    ce = load_yaml(adir / "cv_errors.yaml") or {}
    test = [float(x) for x in (ce.get("test_errs") or []) if isinstance(x, (int, float))]
    if test:
        return float(mean(test)), test, "legacy_unverified"
    return None, [], "missing"


def legacy_double(adir: Path):
    """The old (buggy) value: test_errs / normalizer. Kept only for the flag."""
    ce = load_yaml(adir / "cv_errors.yaml") or {}
    nz = load_yaml(adir / "cv_score_normalizers.yaml") or {}
    test = ce.get("test_errs") or []
    norm = nz.get("values") or []
    if test and norm and len(test) == len(norm):
        vals = [t / n for t, n in zip(test, norm) if n]
        if vals:
            return float(mean(vals))
    return None


def graph_metrics(adir: Path):
    p = adir / "constraint_counts.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    out = {}
    for col in ("w_solver_edge_f1", "w_solver_shd", "w_solver_edge_precision",
                "w_solver_edge_recall", "w_solver_skeleton_jaccard",
                "w_solver_true_w_edges", "w_solver_estimated_w_edges",
                "independent_constraints", "dependent_constraints", "total_constraints"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                out[col] = float(v.mean())
    out["_n_folds"] = int(len(df))
    return out


def infer_phase(exp: str) -> str:
    e = exp.upper()
    for k in ("PLAN01", "PLAN02", "PLAN03", "PLAN04", "PLAN05", "PLAN06", "PLAN07", "PLAN08", "PLAN09"):
        if e.startswith(k):
            return "P" + k[-1]
    if e.startswith("PLAN_CE_NEW_ER"):
        return "ER"
    if e.startswith("PLAN_CE_NEW_CODIET") or e.startswith("PLAN_CE_NEW_INDUSTRY"):
        return "CE_NEW"
    if e.startswith("PLAN_CE_NEW_SF"):
        return "SF"
    if e.startswith("PLAN_EXDBN_ISSUE"):
        return "EXDBN"
    if e.startswith("REPLAY_"):
        return "REPLAY"
    return "OTHER"


def main():
    rows = []
    for adir, exp_id, run_id in iter_run_dirs():
        cfg = load_yaml(adir / "config.yaml") or {}
        solver = cfg.get("solver", {}) or {}
        problem = cfg.get("problem", {}) or {}
        exp = str(cfg.get("experiment") or exp_id)
        nmse, folds, validity = read_metric(adir)
        gm = graph_metrics(adir)
        feats = problem.get("features") or []
        rows.append({
            "phase": infer_phase(exp), "experiment": exp, "exp_id": exp_id, "run_id": run_id,
            "solver": solver.get("name"), "problem": problem.get("name"),
            "graph_type": problem.get("graph_type"), "sem_type": problem.get("sem_type"),
            "target": problem.get("target"), "seed": problem.get("seed"),
            "graph_seed": problem.get("graph_seed"), "noise_seed": problem.get("noise_seed"),
            "n_samples": problem.get("n_samples"), "n_nodes": problem.get("n_nodes"),
            "expected_edges": problem.get("expected_edges"),
            "evidence_batch_id": problem.get("evidence_batch_id"),
            "evidence_node": problem.get("evidence_node"),
            "evidence_arm": problem.get("evidence_arm"),
            "evidence_scope": problem.get("evidence_scope"),
            "execution_cohort": execution_cohort(adir),
            "cv_strategy": solver.get("cv_strategy"), "n_outer": solver.get("n_outer"),
            "n_inner": solver.get("n_inner"), "n_estimators": solver.get("n_estimators"),
            "custom_objective": solver.get("custom_objective"),
            "recalculate_dag": solver.get("recalculate_dag"),
            "constraints_mode": solver.get("constraints_mode"),
            "ci_penalty_kind": solver.get("ci_penalty_kind"),
            "ci_add_shielded_collider_dependence": solver.get("ci_add_shielded_collider_dependence"),
            "ce_use_balanced_batches": solver.get("ce_use_balanced_batches"),
            "ci_prune_redundant": solver.get("ci_prune_redundant"),
            "w_constraint_mode": solver.get("w_constraint_mode"),
            "w_prediction_dependent_mask": solver.get("w_prediction_dependent_mask"),
            "w_bias_calibration": solver.get("w_bias_calibration"),
            "ce_se_method": solver.get("ce_se_method"),
            "ce_tolerance_sd_multiplier": solver.get("ce_tolerance_sd_multiplier"),
            "ce_window_max_sign_flip_rate": solver.get("ce_window_max_sign_flip_rate"),
            "ce_residualize_method": solver.get("ce_residualize_method"),
            "ce_statistic_shrinkage": solver.get("ce_statistic_shrinkage"),
            "edge_penalty": solver.get("edge_penalty"),
            "max_parents": solver.get("max_parents"),
            "enable_clique_constraints": solver.get("enable_clique_constraints"),
            "max_clique_size": solver.get("max_clique_size"),
            "target_mip_gap": solver.get("target_mip_gap"),
            "time_limit": solver.get("time_limit"),
            "hidden_dim": solver.get("hidden_dim"),
            "depth": solver.get("depth"),
            "dropout": solver.get("dropout"),
            "learning_rate": solver.get("learning_rate"),
            "weight_decay": solver.get("weight_decay"),
            "grad_clip_norm": solver.get("grad_clip_norm"),
            "use_validation": solver.get("use_validation"),
            "restore_best_validation_model": solver.get("restore_best_validation_model"),
            "ci_add_dsep_independence": solver.get("ci_add_dsep_independence"),
            "constrained": solver.get("constrained"),
            "huber_delta": solver.get("huber_delta"),
            "dropout": solver.get("dropout"),
            "grad_clip_norm": solver.get("grad_clip_norm"),
            "early_stopping_patience": solver.get("early_stopping_patience"),
            "ce_statistic_kind": solver.get("ce_statistic_kind"),
            "w_matrix_space": solver.get("w_matrix_space"),
            "use_ci_penalty": solver.get("use_ci_penalty"),
            "use_w_constraints": solver.get("use_w_constraints"),
            "ce_constraint_backend": solver.get("ce_constraint_backend"),
            "ce_pbm_backend": solver.get("ce_pbm_backend"),
            "stochastic_constrained": solver.get("use_stochastic_constrained_optimizer"),
            "dag_fit_scope": solver.get("dag_fit_scope"),
            "nmse": nmse, "n_folds_nmse": len(folds), "metric_validity": validity,
            "nmse_legacy_double": legacy_double(adir),
            "n_folds_graph": gm.get("_n_folds"),
            "edge_f1": gm.get("w_solver_edge_f1"), "shd": gm.get("w_solver_shd"),
            "edge_precision": gm.get("w_solver_edge_precision"),
            "edge_recall": gm.get("w_solver_edge_recall"),
            "skeleton_jaccard": gm.get("w_solver_skeleton_jaccard"),
            "true_w_edges": gm.get("w_solver_true_w_edges"),
            "est_w_edges": gm.get("w_solver_estimated_w_edges"),
            "indep_constraints": gm.get("independent_constraints"),
            "dep_constraints": gm.get("dependent_constraints"),
            "total_constraints": gm.get("total_constraints"),
            "features_hash": sha(sorted(map(str, feats))),
            "resolved_config_hash": sha(cfg),
            "artifact_fingerprint": file_fingerprint(adir),
            "layout": "multirun" if "/multirun/" in str(adir) else "mlruns",
            "has_cv_fold_metrics": (adir / "cv_fold_metrics.csv").exists(),
            "has_cv_errors": (adir / "cv_errors.yaml").exists(),
            "has_constraint_counts": (adir / "constraint_counts.csv").exists(),
            "job_id": str(adir.relative_to(ROOT)).split("/")[0],
            "run_dir": str(adir.relative_to(ROOT)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs found")
        return
    # Artifact completeness (used by the G0 gates).  A run is "complete" only if
    # it has a config, a metric source, and at least one graph fold row.
    df["artifact_complete"] = (
        df["has_cv_fold_metrics"].fillna(False)
        & df["has_cv_errors"].fillna(False)
        & (pd.to_numeric(df["n_folds_graph"], errors="coerce").fillna(0) > 0)
    )
    # Canonicalize mirror duplicates by a LOGICAL run key (not the artifact
    # fingerprint, which differs between an MLflow legacy copy and the multirun
    # copy that carries cv_fold_metrics.csv).  Prefer the valid multirun copy.
    df["_job"] = df["run_dir"].astype(str).str.split("/").str[1]
    df["logical_key"] = (
        df["experiment"].astype("object").where(df["experiment"].notna(), "na").astype(str) + "|"
        + df["graph_seed"].astype("object").where(df["graph_seed"].notna(), "na").astype(str) + "|"
        + df["noise_seed"].astype("object").where(df["noise_seed"].notna(), "na").astype(str) + "|"
        + df["target"].astype("object").where(df["target"].notna(), "na").astype(str) + "|"
        + df["seed"].astype("object").where(df["seed"].notna(), "na").astype(str) + "|"
        + df["evidence_batch_id"].astype("object").where(df["evidence_batch_id"].notna(), "na").astype(str) + "|"
        + df["resolved_config_hash"].astype("object").where(df["resolved_config_hash"].notna(), "na").astype(str) + "|"
        + df["_job"].astype("object").where(df["_job"].notna(), "na").astype(str)
    )
    df["has_valid_artifact"] = df["metric_validity"].eq("valid")
    df["is_canonical"] = True
    for _, group in df.groupby("logical_key", dropna=False):
        if len(group) <= 1:
            continue
        order = group.sort_values(
            ["has_valid_artifact", "layout", "run_dir"],
            ascending=[False, False, True],
        )
        df.loc[group.index, "is_canonical"] = False
        df.loc[order.index[0], "is_canonical"] = True
    df["duplicate_group_size"] = df.groupby("logical_key", dropna=False)["run_id"].transform("size")
    df = df.sort_values(["phase", "experiment", "target", "seed", "is_canonical"], ascending=[True, True, True, True, False]).reset_index(drop=True)
    df.to_csv(OUT / "runs_metrics.csv", index=False)

    report_df = df[df["is_canonical"]].copy()
    agg = (report_df.groupby(["phase", "experiment", "solver", "problem", "graph_type"], dropna=False)
             .agg(n_runs=("run_id", "count"), n_valid=("metric_validity", lambda s: (s == "valid").sum()),
                  n_targets=("target", pd.Series.nunique),
                  nmse_valid=("nmse", lambda s: s[df.loc[s.index, "metric_validity"] == "valid"].mean()),
                  nmse_legacy=("nmse_legacy_double", "mean"),
                  f1_mean=("edge_f1", "mean"), shd_mean=("shd", "mean"))
             .reset_index().sort_values(["phase", "experiment"]))
    agg.to_csv(OUT / "arm_summary.csv", index=False)

    ph = (report_df.groupby(["phase"], dropna=False)
            .agg(n_experiments=("experiment", pd.Series.nunique), n_runs=("run_id", "count"),
                 n_valid=("metric_validity", lambda s: (s == "valid").sum()))
            .reset_index().sort_values("phase"))
    ph.to_csv(OUT / "phase_summary.csv", index=False)

    print(f"rows={len(df)} canonical={len(report_df)} duplicates={len(df)-len(report_df)}  "
          f"valid={int((report_df.metric_validity=='valid').sum())}  "
          f"legacy_unverified={int((report_df.metric_validity=='legacy_unverified').sum())}  "
          f"missing={int((report_df.metric_validity=='missing').sum())}")
    print(ph.to_string(index=False))


if __name__ == "__main__":
    main()
