#!/usr/bin/env python3
"""Strict evidence builder v3 (nuisance-stratified, replicate-aware).

Fixes over v2
-------------
* nuisance_config_hash is now BLACKLIST-based: full resolved config minus the
  declared treatment keys minus `experiment`.  Any uncontrolled difference
  (dropout, grad_clip_norm, data fingerprint, graph-estimation keys, ...)
  therefore breaks the pair instead of being silently ignored.
* runs are stratified by nuisance_config_hash; candidate/reference are paired
  ONLY inside the same stratum, by cluster (graph x noise / target x seed).
* each explicit evidence batch (or legacy PBS-job cohort) is analysed
  separately.  A later rerun of the same graph/noise seed is never averaged
  into an earlier code/configuration snapshot; bootstrap resamples CLUSTERS,
  never folds or reruns.
* per-comparison `require` predicates drop known-bad arms (e.g. e6 runs with
  use_w_constraints=false from the old bug).
* any result with < MIN_CLUSTERS independent units is suffixed `_low_power`.

Outputs: reports/evidence_pairs.csv, evidence_index.csv, evidence_exclusions.csv
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "progress"
PAIRS = ROOT / "reports" / "evidence_pairs.csv"
INDEX = ROOT / "reports" / "evidence_index.csv"
EXCL = ROOT / "reports" / "evidence_exclusions.csv"
POOLED = ROOT / "reports" / "evidence_pooled.csv"
RNG = np.random.default_rng(20260923)
N_BOOT = 20000
PRACTICAL = 0.05
EQUIV_MARGIN = 0.05
MIN_CLUSTERS = 5

GRAPH_SUF = __import__("re").compile(r"^(?P<base>.+?)_graph(?P<g>\d+)_noise(?P<n>\d+)$")
SEED_SUF = __import__("re").compile(r"^(?P<base>.+?)_seed(?P<s>\d+)$")
CONFIG_CACHE: dict[str, dict] = {}


def load_cfg(run_dir: str) -> dict:
    if run_dir not in CONFIG_CACHE:
        try:
            CONFIG_CACHE[run_dir] = yaml.safe_load((ROOT / run_dir / "config.yaml").read_text()) or {}
        except Exception:
            CONFIG_CACHE[run_dir] = {}
    return CONFIG_CACHE[run_dir]


def flatten(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list):
        out[p] = json.dumps(o, default=str)
    else:
        out[p] = o
    return out


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# Blacklist: keys REMOVED before hashing the nuisance config, per treatment.
TREATMENT_PREFIXES = {
    "ce": ["solver.ce_", "solver.ci_", "solver.constrained", "solver.constraint_audit",
           "solver.use_ci_penalty", "solver.w_matrix_space", "solver.weights_bound",
           "solver.use_w_constraints", "solver.w_constraint_mode", "solver.w_prediction",
           "solver.rho0", "solver.rho_mult", "solver.lambda0", "solver.lambda1", "solver.lambda2",
           "solver.lambda_update_rate"],
    "w": ["solver.use_w_constraints", "solver.w_constraint_mode", "solver.w_prediction",
          "solver.weights_bound", "solver.constraints_mode", "solver.constrained",
          "solver.constraint_audit", "solver.w_matrix_space", "solver.rho0", "solver.rho_mult"],
    "trees": ["solver.n_estimators"],
    "none": [],
    "end_to_end": [],
}


def nuisance_hash(cfg: dict, treat: str) -> str:
    flat = flatten(cfg)
    drops = ["experiment"] + TREATMENT_PREFIXES.get(treat, [])
    sel = {k: v for k, v in flat.items() if not any(k == d or k.startswith(d) for d in drops)}
    return sha(sel)


def parse_arm(exp: str):
    m = GRAPH_SUF.match(exp)
    if m:
        return m["base"], int(m["g"]), int(m["n"]), None
    m = SEED_SUF.match(exp)
    if m:
        return m["base"], None, None, int(m["s"])
    return exp, None, None, None


def scope_of(r):
    if r["problem"] == "synthetic":
        return f"synthetic/{r['graph_type']}"
    if r["problem"] == "industry_eu":
        return "FRED"
    if r["problem"] == "codiet":
        return "CoDiet"
    return str(r["problem"])


def cluster_of(r):
    if r["problem"] == "synthetic":
        return f"{r['graph_seed']}|{r['noise_seed']}"
    seed = r["s"] if pd.notna(r["s"]) else r["seed"]
    return f"{r['target']}|{seed}"


def boot_ci(d):
    n = len(d)
    if n < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    return tuple(np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))


def req(**kw):
    """require predicate over the runs dataframe."""
    def f(d):
        m = pd.Series(True, index=d.index)
        for k, v in kw.items():
            m &= d[k].eq(v)
        return m
    return f


# id, root, scope, ref_arm, cand_arm, type, treatment, candidate require-predicate
COMPARISONS = [
    ("C0b.ce_vs_nn.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e0_no_constraint",
     "PLAN_CE_NEW_ER_e2_pure_ce_upgraded", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("C0b.ce_vs_nn.SF", "C0b", "synthetic/SF", "PLAN_CE_NEW_SF_e0_no_constraint",
     "PLAN_CE_NEW_SF_e2_pure_ce_upgraded", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("C0b.w_vs_nn.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e0_no_constraint",
     "PLAN_CE_NEW_ER_e1_true_w", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=False)),
    ("C0b.wce_vs_w.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e1_true_w",
     "PLAN_CE_NEW_ER_e3_w_plus_ce_upgraded", "mechanism", "ce",
     req(use_w_constraints=True, use_ci_penalty=True)),
    ("C0b.wce_vs_ce.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e2_pure_ce_upgraded",
     "PLAN_CE_NEW_ER_e3_w_plus_ce_upgraded", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=True)),
    ("C0b.w_ablation.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_m0_xgb100_no_w",
     "PLAN_CE_NEW_ER_m1_xgb100_w", "mechanism", "w", None),
    # ---- B0 baseline rail ----
    ("B0.tree_count.ER", "B0", "synthetic/ER", "PLAN_CE_NEW_ER_b1_mark",
     "PLAN_CE_NEW_ER_a1_mark_100", "end_to_end", "trees", None),
    ("B0.cc_vs_mark100.ER", "B0", "synthetic/ER", "PLAN_CE_NEW_ER_a1_mark_100",
     "PLAN_CE_NEW_ER_b2_mark_with_cc", "end_to_end", "none", None),
    # ---- nonlinear boundary ----
    ("P1.3.nonlin_ce.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_b2nl_mark_with_cc",
     "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce", "end_to_end", "end_to_end",
     req(use_ci_penalty=True)),
    ("P1.3.nonlin_wce.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_b2nl_mark_with_cc",
     "PLAN_CE_NEW_ER_e6_nonlinear_w_plus_ce", "end_to_end", "end_to_end",
     req(use_w_constraints=True, use_ci_penalty=True)),
    # Direct nonlinear mechanism contrasts.  These use the same NN rather
    # than an out-of-cohort Mark-CC baseline, so they are the relevant test of
    # whether the CE statistics add value under misspecification.
    ("P1.3.nonlin_ce_vs_nn.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_e7_nonlinear_no_constraint",
     "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.quantile_vs_linear.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce",
     "PLAN_CE_NEW_ER_e5_nonlinear_quantile_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.nonlin_wce_vs_ce.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce",
     "PLAN_CE_NEW_ER_e6_nonlinear_w_plus_ce", "mechanism", "w",
     req(use_ci_penalty=True, use_w_constraints=True)),
    # ---- product (provisional) ----
    ("C0a.hcce_vs_markcc.ER", "C0a", "synthetic/ER", "PLAN_CE_NEW_ER_b2_mark_with_cc",
     "PLAN_CE_NEW_ER_e2_pure_ce_upgraded", "end_to_end", "end_to_end",
     req(use_ci_penalty=True, use_w_constraints=False)),
    # ---- Clean configuration-repair batch ----
    ("C0b.ce_vs_nn.repair_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_lin_nn",
     "PLAN_EVIDENCE_REPAIR_ER_lin_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("C0b.w_vs_nn.repair_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_lin_nn",
     "PLAN_EVIDENCE_REPAIR_ER_lin_w", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=False)),
    ("C0b.wce_vs_ce.repair_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_lin_ce",
     "PLAN_EVIDENCE_REPAIR_ER_lin_w_ce", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=True)),
    ("P1.3.nonlin_ce_vs_nn.repair_ER", "P1.3", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_nonlin_nn",
     "PLAN_EVIDENCE_REPAIR_ER_nonlin_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.quantile_vs_linear.repair_ER", "P1.3", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_nonlin_ce",
     "PLAN_EVIDENCE_REPAIR_ER_nonlin_quantile_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.nonlin_wce_vs_ce.repair_ER", "P1.3", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_nonlin_ce",
     "PLAN_EVIDENCE_REPAIR_ER_nonlin_w_ce", "mechanism", "w",
     req(use_ci_penalty=True, use_w_constraints=True)),
    # ---- Powered CE confirmation, then explicitly gated optimizer sweep ----
    ("C0b.ce_vs_nn.priority_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_nn",
     "PLAN_EVIDENCE_PRIORITY_ER_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.alm_vs_pbm.priority_ER", "A1", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_opt_alm_all",
     "PLAN_EVIDENCE_PRIORITY_ER_opt_pbm_all", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.pbm_vs_spbm.priority_ER", "A1", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_opt_pbm_all",
     "PLAN_EVIDENCE_PRIORITY_ER_opt_spbm_all", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.spbm_vs_sco.priority_ER", "A1", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_opt_spbm_all",
     "PLAN_EVIDENCE_PRIORITY_ER_sco_spbm_all", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),

    # ---- Field-matched EV:<node>:<arm> comparisons --------------------------
    # These match the explicit evidence_node/evidence_arm fields written by
    # scripts/evidence_tree/*.sh, so the builder does not depend on the
    # experiment-name string.
    # A1 optimizer backends (ref_ce is alm_pbm = ALM under independent-only).
    ("A1.ALM_ALL.er", "A1", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A1.ALM_ALL:ce_alm_all", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.PBM_ALL.er", "A1", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A1.PBM_ALL:ce_pbm_all", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.STOCHASTIC_PBM.er", "A1", "synthetic/ER", "EV:A1.PBM_ALL:ce_pbm_all",
     "EV:A1.STOCHASTIC_PBM:ce_spbm_all", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.SCO_LAYER.er", "A1", "synthetic/ER", "EV:A1.STOCHASTIC_PBM:ce_spbm_all",
     "EV:A1.SCO_LAYER:ce_sco", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # A2 constraint embedding.
    ("A2.W_global.er", "A2", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A2.W_global:w_global", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
    ("A2.W_target_residual.er", "A2", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A2.W_target_residual:w_target_residual", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
    ("A2.W_mask.er", "A2", "synthetic/ER", "EV:A2.W_global:w_global",
     "EV:A2.W_mask:w_mask", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
    ("A2.W_bias_calibration.er", "A2", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A2.W_bias_calibration:w_bias_calibration", "mechanism", "w", req(use_w_constraints=False, use_ci_penalty=False)),
    ("A2.balanced_batch.er", "A2", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A2.balanced_batch:ce_balanced_batch", "mechanism", "none", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A2.pruning.er", "A2", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A2.pruning:ce_pruning", "mechanism", "none", req(use_ci_penalty=True, use_w_constraints=False)),
    # A3 CI/CE statistic.
    ("A3.covariance.er", "A3", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A3.covariance:ce_covariance", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # Single window-vs-HAC contrast (ref_ce is the window arm); the previous
    # two reverse-direction entries double-counted the same comparison.
    ("A3.window_vs_hac.er", "A3", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A3.HAC_SE:ce_hac_se", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A3.sign_flip_filter.er", "A3", "synthetic/ER", "EV:A3.sign_flip_filter:dep_filter_off",
     "EV:A3.sign_flip_filter:dep_filter_on", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # A4 graph estimation & causal prior.  One shared MIP control arm serves
    # both the time and the gap contrast.
    ("A4.mip_time.er", "A4", "synthetic/ER", "EV:A4.mip_control:mip_control",
     "EV:A4.mip_time:mip_time_1800", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.mip_gap.er", "A4", "synthetic/ER", "EV:A4.mip_control:mip_control",
     "EV:A4.mip_gap:mip_gap_1e4", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.edge_penalty.er", "A4", "synthetic/ER", "EV:A4.edge_penalty:edge_penalty_0",
     "EV:A4.edge_penalty:edge_penalty_0p05", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.parents_limit_3.er", "A4", "synthetic/ER", "EV:A4.parents_limit:parents_none",
     "EV:A4.parents_limit:parents_3", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.parents_limit_4.er", "A4", "synthetic/ER", "EV:A4.parents_limit:parents_none",
     "EV:A4.parents_limit:parents_4", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.parents_limit_5.er", "A4", "synthetic/ER", "EV:A4.parents_limit:parents_none",
     "EV:A4.parents_limit:parents_5", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.clique_cap.er", "A4", "synthetic/ER", "EV:A4.clique_cap:clique_off",
     "EV:A4.clique_cap:clique_on", "mechanism", "graph", req(recalculate_dag=True)),
    # A5 capacity & budget.
    ("A5.hidden_depth.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.hidden_depth:hicap_64x3", "mechanism", "none", None),
    ("A5.lr_wd.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.lr_wd:lr0p05_wd0p10", "mechanism", "none", None),
    ("A5.n_outer_inner.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.n_outer_inner:updates_half_5x50", "mechanism", "budget", None),
    ("A5.n_outer_inner_matched.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.n_outer_inner:updates_matched_20x50", "mechanism", "budget", None),
    # A6 data / time / robustness (FRED).
    ("A6.lag.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.lag:feature_lag_3", "mechanism", "data", None),
    ("A6.trend.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.trend:add_time_trend", "mechanism", "data", None),
    ("A6.regime.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.regime:regime_break_2020", "mechanism", "data", None),
    ("A6.huber.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.huber:huber_loss", "mechanism", "loss", None),
    # B0 classical baselines.
    ("B0.mark_100.er", "B0", "synthetic/ER", "EV:B0.mark_10:mark_10",
     "EV:B0.mark_100:mark_100", "end_to_end", "trees", None),
    ("B0.mark_cc_100.er", "B0", "synthetic/ER", "EV:B0.mark_cc_10:mark_cc_10",
     "EV:B0.mark_cc_100:mark_cc_100", "end_to_end", "trees", None),
    ("B0.hc_nn.er", "B0", "synthetic/ER", "EV:B0.mark_100:mark_100",
     "EV:B0.hc_nn_no_constraint:hc_nn_no_constraint", "end_to_end", "none", None),
    ("B0.hc_w_only.er", "B0", "synthetic/ER", "EV:B0.hc_nn_no_constraint:hc_nn_no_constraint",
     "EV:B0.hc_w_only:hc_w_only", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
]


def cohort_of(r) -> str:
    """Explicit batch id wins; otherwise use the containing PBS-job mirror.

    Distinct PBS jobs often contain the same synthetic graph/noise seeds but
    were made after a code or configuration change.  They are replications for
    diagnosis, not independent observations to be averaged together.
    """
    batch = getattr(r, "evidence_batch_id", None)
    if pd.notna(batch) and str(batch).strip():
        return str(batch).strip()
    legacy = getattr(r, "execution_cohort", None)
    if pd.notna(legacy) and str(legacy).strip():
        return str(legacy).strip()
    parts = Path(str(r.run_dir)).parts
    try:
        return parts[parts.index("metacentrum_runs") + 1]
    except (ValueError, IndexError):
        return "legacy_unknown_cohort"


def main():
    df = pd.read_csv(OUT / "runs_metrics.csv")
    if "is_canonical" in df.columns:
        df = df[df.is_canonical.fillna(True).astype(bool)].copy()
    df["arm"], df["g"], df["n"], df["s"] = zip(*df["experiment"].map(parse_arm))
    df["scope"] = df.apply(scope_of, axis=1)
    df["cluster"] = df.apply(cluster_of, axis=1)
    val = df[df.metric_validity == "valid"].copy()
    # The scanner added these fields in v4.  Keep a fallback so an older
    # runs_metrics.csv can still be inspected before a refresh.
    if "evidence_batch_id" not in val:
        val["evidence_batch_id"] = np.nan
    if "execution_cohort" not in val:
        val["execution_cohort"] = np.nan
    for col in ("evidence_node", "evidence_arm", "evidence_scope"):
        if col not in val:
            val[col] = np.nan
    val["cohort"] = val.apply(cohort_of, axis=1)

    now = datetime.now().isoformat(timespec="seconds")
    try:
        git_commit = (ROOT / ".git" / "HEAD").read_text().strip()[:40]
    except Exception:
        git_commit = ""

    pair_rows, idx_rows, excl_rows = [], [], []

    def _select(sub, token):
        """Resolve an arm token.

        ``EV:<node>:<arm>`` matches the explicit evidence_node/evidence_arm
        fields written into the problem config, so the builder no longer
        depends on the experiment-name string.  Any other token is a legacy
        experiment-name prefix match.
        """
        if token.startswith("EV:"):
            _, node, arm = token.split(":", 2)
            return sub[
                (sub.evidence_node.astype(str) == node)
                & (sub.evidence_arm.astype(str) == arm)
            ]
        return sub[sub.arm == token]

    for cid, root, scope, ref_arm, cand_arm, ctype, treat, require in COMPARISONS:
        sub = val[val.scope == scope]
        ref_all = _select(sub, ref_arm)
        cand_all = _select(sub, cand_arm)
        if require is not None and len(cand_all):
            keep = require(cand_all)
            # Preserve an auditable record of a misconfigured arm instead of
            # silently filtering it.  This is how the historical E6 W=false
            # artifact remains visible while being excluded from inference.
            for bad in cand_all[~keep].itertuples():
                excl_rows.append(dict(
                    comparison_id=cid, cohort=cohort_of(bad), cluster=bad.cluster,
                    reason="invalid_configuration:treatment_requirement",
                    candidate_run_id=bad.run_id, candidate_run_dir=bad.run_dir,
                    candidate_config_hash=bad.resolved_config_hash,
                    updated_at=now,
                ))
            cand_all = cand_all[keep]
        if ref_all.empty or cand_all.empty:
            idx_rows.append(dict(comparison_id=cid, root=root, scope=scope, reference=ref_arm,
                                 candidate=cand_arm, comparison_type=ctype,
                                 n_clusters_expected=0, n_clusters_used=0, n_excluded=0,
                                 power="low", rel_improvement=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 statistical_win=False, practical_win=False,
                                 status="out_of_scope", note="missing valid arm", updated_at=now))
            continue

        # ---- cohort + nuisance stratification ----
        # A cohort is one explicit evidence batch (new scripts) or one PBS job
        # mirror (legacy artifacts).  Never average an old and a new config
        # simply because they used the same synthetic seeds.
        def stratify(rows):
            out = {}
            for r in rows.itertuples():
                nh = nuisance_hash(load_cfg(r.run_dir), treat) if ctype == "mechanism" else "n/a"
                out.setdefault((r.cohort, nh, r.cluster), []).append(r)
            return out

        rs, cs = stratify(ref_all), stratify(cand_all)
        cohorts = sorted({k[0] for k in rs} & {k[0] for k in cs})
        if not cohorts:
            idx_rows.append(dict(
                comparison_id=cid, root=root, scope=scope, cohort="", reference=ref_arm,
                candidate=cand_arm, comparison_type=ctype, treatment=treat,
                n_clusters_expected=0, n_clusters_used=0, n_excluded=0, power="low",
                rel_improvement=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                statistical_win=False, practical_win=False, status="invalid_pairing",
                note="no common cohort after candidate requirements", updated_at=now))
            continue

        for cohort in cohorts:
            # Matching configurations must appear in both treatment arms.
            strata = sorted(
                {k[1] for k in rs if k[0] == cohort}
                & {k[1] for k in cs if k[0] == cohort}
            )
            cluster_vals: dict[str, list[float]] = {}
            excluded = 0
            expected_clusters = set()
            ref_ch = cand_ch = ""
            nh_used = "n/a"
            nh_equal = False
            for nh in strata:
                ref_clusters = {k[2] for k in rs if k[0] == cohort and k[1] == nh}
                cand_clusters = {k[2] for k in cs if k[0] == cohort and k[1] == nh}
                expected_clusters |= ref_clusters & cand_clusters
                for cl in sorted(ref_clusters & cand_clusters):
                    R, C = rs.get((cohort, nh, cl), []), cs.get((cohort, nh, cl), [])
                    # More than one record with a cluster/nuisance/cohort key
                    # is a duplicate execution, not an extra independent unit.
                    if len(R) != 1 or len(C) != 1:
                        excluded += max(len(R), len(C))
                        excl_rows.append(dict(
                            comparison_id=cid, cohort=cohort, cluster=cl,
                            reason="invalid_pairing:duplicate_within_cohort",
                            reference_run_dirs=";".join(str(x.run_dir) for x in R),
                            candidate_run_dirs=";".join(str(x.run_dir) for x in C),
                            nuisance_hash=nh, updated_at=now))
                        continue
                    rr, cc = R[0], C[0]
                    rv, cv = float(rr.nmse), float(cc.nmse)
                    if not np.isfinite(rv) or not np.isfinite(cv) or rv == 0:
                        excluded += 1
                        excl_rows.append(dict(
                            comparison_id=cid, cohort=cohort, cluster=cl,
                            reason="invalid_metric:nonfinite_or_zero",
                            reference_run_id=rr.run_id, candidate_run_id=cc.run_id,
                            reference_run_dir=rr.run_dir, candidate_run_dir=cc.run_dir,
                            nuisance_hash=nh, updated_at=now))
                        continue
                    rel = (rv - cv) / rv
                    cluster_vals.setdefault(cl, []).append(rel)
                    ref_ch = str(rr.resolved_config_hash)
                    cand_ch = str(cc.resolved_config_hash)
                    nh_used = nh
                    nh_equal = True
                    pair_rows.append(dict(
                        comparison_id=cid, node_id=cid, scope_id=scope, root=root,
                        cohort=cohort,
                        dataset_id=(load_cfg(rr.run_dir).get("problem", {}) or {}).get("name"),
                        sem_type=(load_cfg(rr.run_dir).get("problem", {}) or {}).get("sem_type"),
                        target_id=(load_cfg(rr.run_dir).get("problem", {}) or {}).get("target"),
                        cluster=cl, graph_seed=cl.split("|")[0] if "|" in cl else cl,
                        noise_seed=cl.split("|")[1] if "|" in cl else "",
                        reference_arm=ref_arm, candidate_arm=cand_arm,
                        reference_run_id=rr.run_id, candidate_run_id=cc.run_id,
                        reference_run_dir=rr.run_dir, candidate_run_dir=cc.run_dir,
                        reference_config_hash=rr.resolved_config_hash,
                        candidate_config_hash=cc.resolved_config_hash,
                        reference_artifact_fingerprint=rr.artifact_fingerprint,
                        candidate_artifact_fingerprint=cc.artifact_fingerprint,
                        nuisance_hash=nh, nmse_ref=rv, nmse_cand=cv,
                        relative_improvement=rel, metric_validity="valid",
                        git_commit=git_commit, timestamp=now,
                    ))

            # Bootstrap only independent graph x noise (or target x seed)
            # clusters within this one execution cohort.
            d = np.asarray([np.mean(v) for v in cluster_vals.values()], dtype=float)
            n_clusters = len(d)
            lo, hi = boot_ci(d)
            rel_mean = float(np.mean(d)) if n_clusters else np.nan
            stat_win = bool(n_clusters > 0 and lo > 0)
            prac_win = bool(n_clusters > 0 and lo > 0 and rel_mean >= PRACTICAL)
            power = "low" if n_clusters < MIN_CLUSTERS else "ok"
            if n_clusters == 0:
                status = "invalid_pairing" if excluded or expected_clusters else "out_of_scope"
            elif prac_win:
                # RELAXED ACCEPTANCE: a single-variable controlled comparison is
                # enough to support a claim, even below MIN_CLUSTERS.  Low power
                # is recorded separately in the `power` field.
                status = "supported"
            elif hi < 0:
                status = "contradicted"
            elif stat_win:
                status = "below_practical_threshold"
            elif lo > -EQUIV_MARGIN and hi < EQUIV_MARGIN:
                status = "equivalent_within_margin"
            else:
                status = "inconclusive"
            if power == "low" and status in ("supported", "contradicted",
                                             "below_practical_threshold", "equivalent_within_margin"):
                status = status + "_low_power"
            idx_rows.append(dict(
                comparison_id=cid, root=root, scope=scope, cohort=cohort,
                reference=ref_arm, candidate=cand_arm, comparison_type=ctype,
                treatment=treat, n_clusters_expected=len(expected_clusters),
                n_clusters_used=n_clusters, n_excluded=excluded, power=power,
                rel_improvement=rel_mean, ci_lo=lo, ci_hi=hi,
                statistical_win=stat_win, practical_win=prac_win, status=status,
                reference_config_hash=ref_ch, candidate_config_hash=cand_ch,
                nuisance_config_hash=nh_used, nuisance_hash_equal=nh_equal,
                split_hash_equal=False, fold_w_cache_hash_equal=False,
                note=(f"{ctype}; cohort+nuisance stratified; practical>={PRACTICAL:.0%}"),
                updated_at=now))

    pd.DataFrame(pair_rows).to_csv(PAIRS, index=False)
    pd.DataFrame(idx_rows).to_csv(INDEX, index=False)
    pd.DataFrame(excl_rows).to_csv(EXCL, index=False)

    # ---- pooled direction (NON-STRICT): pool every cohort, signal only ----
    pooled_rows = []
    if pair_rows:
        pr = pd.DataFrame(pair_rows)
        for cid, g in pr.groupby("comparison_id"):
            vals = g.groupby("cluster")["relative_improvement"].mean().to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lo, hi = boot_ci(vals)
            mean_rel = float(np.mean(vals))
            if np.isfinite(lo) and lo > 0:
                direction = "positive"
            elif np.isfinite(hi) and hi < 0:
                direction = "negative"
            else:
                direction = "flat"
            pooled_rows.append(dict(
                comparison_id=cid, pooled_units=len(vals),
                pooled_rel=mean_rel, pooled_ci_lo=lo, pooled_ci_hi=hi,
                pooled_direction=direction,
                pooled_practical=bool(np.isfinite(lo) and lo > 0 and mean_rel >= PRACTICAL),
                note="pooled across cohorts; non-strict, signal only",
                updated_at=now))
    pd.DataFrame(pooled_rows).to_csv(POOLED, index=False)
    print(f"wrote {POOLED} ({len(pooled_rows)} pooled comparisons)")
    print(f"wrote {PAIRS} ({len(pair_rows)} pairs)")
    print(f"wrote {INDEX} ({len(idx_rows)} comparisons)")
    print(f"wrote {EXCL} ({len(excl_rows)} exclusions)")
    idx = pd.DataFrame(idx_rows)
    print()
    print(idx[["comparison_id", "n_clusters_used", "n_excluded", "power",
               "rel_improvement", "ci_lo", "ci_hi", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
