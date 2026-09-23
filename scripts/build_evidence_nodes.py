#!/usr/bin/env python3
"""Build reports/evidence_nodes.csv — search-space registry (revised).

Structure (per review):
  S0  search-space registry + runnable anchor
  B0  baseline rail          A1 optimization mechanics
  A2  constraint embedding   A3 CI/CE statistic
  A4  graph estimation & causal prior
  A5  model capacity & training budget
  A6  data representation / time / robustness
  G0  fair-comparison contract    F0 freeze + locked holdout

Counts are NOT additive: each node reports
  unique_runs / valid_runs / paired_units  and  crossed_factor.
Claims are SCOPE-SPECIFIC (status(scope)); literature only yields
literature_hypothesis, never `supported`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
runs = pd.read_csv(REPORTS / "progress" / "runs_metrics.csv")
runs = runs[runs.is_canonical.fillna(True).astype(bool)]
valid = runs[runs.metric_validity == "valid"]
try:
    idx = pd.read_csv(REPORTS / "evidence_index.csv")
except Exception:
    idx = pd.DataFrame(columns=["comparison_id", "status", "scope", "n_clusters_used", "n_clusters_expected"])
try:
    pooled = pd.read_csv(REPORTS / "evidence_pooled.csv")
except Exception:
    pooled = pd.DataFrame(columns=["comparison_id", "pooled_rel", "pooled_direction", "pooled_units"])

HC = lambda d: d.solver.isin(["hc_predictor", "hc_predictor_ce"])
TRUE = lambda d: pd.Series(True, index=d.index)


def N(*conds):
    def f(d):
        m = pd.Series(True, index=d.index)
        for c in conds:
            m &= c(d)
        return m
    return f


def eq(col, val):
    return lambda d: d[col].eq(val)


# id, parent, axis, zh, en, source, crossed, count_fn, comparison_ids
NODES = [
    ("S0", "", "S0", "搜索空间注册表 + 可运行锚点", "Search-space registry + runnable anchor",
     "code", False, TRUE, []),

    # ---------- B0 baseline rail ----------
    ("B0", "S0", "B0", "B0 基线轨（可比较的 fallback）", "Baseline rail (comparable fallback)",
     "evidence", False, N(HC), []),
    ("B0.mark_100", "B0", "B0", "B0.1 Mark-100", "Mark-100", "evidence", False,
     lambda d: d.experiment.str.contains("mark_100", na=False), ["B0.tree_count.ER"]),
    ("B0.mark_cc_total100", "B0", "B0", "B0.2 Mark-CC（总预算 100 树，待模型产物核验）",
     "Mark-CC (total-100 budget; pending model-artifact audit)", "code", False,
     lambda d: d.solver.eq("mark_with_cc"), []),
    ("B0.hc_nn_no_constraint", "B0", "B0", "B0.3 HC-NN 无约束", "HC-NN no-constraint",
     "evidence", False, N(HC, eq("use_ci_penalty", False), eq("use_w_constraints", False)), []),
    ("B0.hc_w_only", "B0", "B0", "B0.4 HC W-only", "HC W-only", "evidence", False,
     N(HC, eq("use_w_constraints", True), eq("use_ci_penalty", False)), []),
    ("B0.hc_ce_only", "B0", "B0", "B0.5 HC CE-only", "HC CE-only", "evidence", False,
     N(HC, eq("use_ci_penalty", True), eq("use_w_constraints", False)),
     ["C0b.ce_vs_nn.ER", "C0b.ce_vs_nn.SF", "C0b.ce_vs_nn.repair_ER", "C0b.ce_vs_nn.priority_ER"]),
    ("B0.hc_w_ce", "B0", "B0", "B0.6 HC W+CE", "HC W+CE", "evidence", False,
     N(HC, eq("use_w_constraints", True), eq("use_ci_penalty", True)),
     ["C0b.wce_vs_w.ER", "C0b.wce_vs_ce.ER", "C0b.wce_vs_ce.repair_ER"]),

    # ---------- A1 optimization mechanics ----------
    ("A1", "S0", "A1", "A1 优化算法空间", "Optimization Mechanics", "code", False, TRUE, []),
    ("A1.ALM_ALL", "A1", "A1", "A1.1 ALM-all（全部 CE 走 ALM）", "ALM-all", "code", False,
     eq("ce_constraint_backend", "alm_all"), ["A1.alm_vs_pbm.priority_ER"]),
    ("A1.PBM_ALL", "A1", "A1", "A1.2 PBM-all（全部 CE 走 PBM）", "PBM-all", "code", False,
     eq("ce_constraint_backend", "pbm_all"), ["A1.alm_vs_pbm.priority_ER", "A1.pbm_vs_spbm.priority_ER"]),
    ("A1.HYBRID_ALM_PBM", "A1", "A1", "A1.3 Hybrid：independent→ALM, dependent→PBM",
     "Hybrid ALM/PBM", "evidence", False, eq("ce_constraint_backend", "alm_pbm"), []),
    ("A1.STOCHASTIC_PBM", "A1", "A1", "A1.4 stochastic-PBM", "Stochastic PBM", "code", False,
     eq("ce_pbm_backend", "stochastic_pbm"), ["A1.pbm_vs_spbm.priority_ER"]),
    ("A1.SCO_LAYER", "A1", "A1", "A1.5 SCO 层（随机约束优化 + proximal/adaptive）",
     "SCO layer", "code", False, eq("stochastic_constrained", True), ["A1.spbm_vs_sco.priority_ER"]),

    # ---------- A2 constraint embedding ----------
    ("A2", "S0", "A2", "A2 约束嵌入方式", "Constraint Embedding Mechanics", "code", False, TRUE, []),
    ("A2.W_only", "A2", "A2", "A2.1 仅 W", "W only", "evidence", False,
     N(eq("use_w_constraints", True), eq("use_ci_penalty", False)), ["C0b.w_vs_nn.ER", "C0b.w_vs_nn.repair_ER"]),
    ("A2.CE_only", "A2", "A2", "A2.2 仅因果约束", "Causal only", "evidence", False,
     N(eq("use_ci_penalty", True), eq("use_w_constraints", False)),
     ["C0b.ce_vs_nn.ER", "C0b.ce_vs_nn.SF", "C0b.ce_vs_nn.repair_ER", "C0b.ce_vs_nn.priority_ER"]),
    ("A2.W_CE", "A2", "A2", "A2.3 W + 因果", "W + causal", "evidence", False,
     N(eq("use_w_constraints", True), eq("use_ci_penalty", True)),
     ["C0b.wce_vs_w.ER", "C0b.wce_vs_ce.ER", "C0b.wce_vs_ce.repair_ER", "P1.3.nonlin_wce_vs_ce.ER"]),
    ("A2.W_global", "A2", "A2", "A2.4 W 变体：legacy global", "W variant: legacy global", "code", False,
     eq("w_constraint_mode", "legacy_global"), ["A2.W_global.shared_ER"]),
    ("A2.W_target_residual", "A2", "A2", "A2.5 W 变体：target residual", "W variant: target residual", "code", False,
     eq("w_constraint_mode", "target_residual"), ["A2.W_target_residual.shared_ER"]),
    ("A2.W_mask", "A2", "A2", "A2.6 W 变体：prediction-dependent mask", "W variant: prediction-dependent mask",
     "code", False, eq("w_prediction_dependent_mask", True), ["A2.W_mask.shared_ER"]),
    ("A2.W_bias_calibration", "A2", "A2", "A2.7 W 变体：bias calibration", "W variant: bias calibration",
     "code", False, eq("w_bias_calibration", True), ["A2.W_bias_calibration.shared_ER"]),
    ("A2.balanced_batch", "A2", "A2", "A2.8 balanced batch（修饰因素）", "balanced batch (modifier)",
     "code", True, eq("ce_use_balanced_batches", True), ["A2.balanced_batch.shared_ER"]),
    ("A2.pruning", "A2", "A2", "A2.9 冗余约束剪枝（修饰因素）", "constraint pruning (modifier)",
     "code", True, eq("ci_prune_redundant", True), ["A2.pruning.shared_ER"]),
    ("A2.W_adds_to_CE", "A2", "A2", "A2.10 W 是否在 CE 之上有增量", "W on top of CE",
     "evidence", False, N(HC, eq("use_w_constraints", True), eq("use_ci_penalty", True)),
     ["C0b.wce_vs_ce.ER", "C0b.wce_vs_ce.repair_ER"]),

    # ---------- A3 CI/CE statistic ----------
    ("A3", "S0", "A3", "A3 CI/CE 统计量", "CI/CE Statistic", "code", False, TRUE, []),
    ("A3.covariance", "A3", "A3", "A3.1 残差协方差（条件期望）", "residual covariance (cond. expectation)",
     "evidence", True, eq("ci_penalty_kind", "conditional_expectation"), ["A3.covariance.shared_ER"]),
    ("A3.partial_correlation", "A3", "A3", "A3.2 标准化偏相关", "standardized partial correlation",
     "evidence", True, eq("ce_statistic_kind", "partial_correlation"),
     ["P1.3.nonlin_ce.ER", "P1.3.nonlin_wce.ER", "P1.3.nonlin_ce_vs_nn.ER", "P1.3.nonlin_ce_vs_nn.repair_ER"]),
    ("A3.discrete_CMI", "A3", "A3", "A3.3 离散 CMI", "discrete CMI", "evidence", True,
     eq("ci_penalty_kind", "discrete_conditional_independence"), []),
    ("A3.quantile_residual", "A3", "A3", "A3.4 分位残差化", "quantile residualization", "code", True,
     eq("ce_residualize_method", "quantile"), ["P1.3.quantile_vs_linear.ER", "P1.3.quantile_vs_linear.repair_ER"]),
    ("A3.window_SE", "A3", "A3", "A3.5 window SE", "window SE", "code", True, eq("ce_se_method", "window"), ["A3.window_SE.shared_ER"]),
    ("A3.HAC_SE", "A3", "A3", "A3.6 HAC SE", "HAC SE", "code", True, eq("ce_se_method", "hac"), ["A3.HAC_SE.shared_ER"]),
    ("A3.sign_flip_filter", "A3", "A3", "A3.7 sign-flip filter", "sign-flip filter", "code", True,
     lambda d: d.ce_window_max_sign_flip_rate.notna() & d.ce_window_max_sign_flip_rate.ne(""), ["A3.sign_flip_filter.shared_ER"]),
    ("A3.KCI_HSIC", "A3", "A3", "A3.8 KCI/HSIC（非线性 CI）", "KCI/HSIC (nonlinear CI)", "literature", True,
     lambda d: pd.Series(False, index=d.index), []),

    # ---------- A4 graph estimation & causal prior ----------
    ("A4", "S0", "A4", "A4 图估计与因果先验", "Graph Estimation & Causal Prior", "code", False, TRUE, []),
    ("A4.edge_penalty", "A4", "A4", "A4.1 edge penalty", "edge penalty", "code", True,
     lambda d: d.edge_penalty.notna() & d.edge_penalty.ne(""), []),
    ("A4.parents_limit", "A4", "A4", "A4.2 max parents", "parents limit", "code", True,
     lambda d: d.max_parents.notna() & d.max_parents.ne(""), []),
    ("A4.clique_cap", "A4", "A4", "A4.3 clique cap", "clique cap", "code", True,
     lambda d: d.enable_clique_constraints.notna() & d.enable_clique_constraints.ne(""), []),
    ("A4.MIP_time_gap", "A4", "A4", "A4.4 MIP time / gap", "MIP time/gap", "code", True,
     lambda d: d.time_limit.notna() & d.time_limit.ne(""), []),
    ("A4.stability_selection", "A4", "A4", "A4.5 stability selection", "stability selection", "literature", True,
     lambda d: pd.Series(False, index=d.index), []),
    ("A4.non_gaussian_moments", "A4", "A4", "A4.6 高阶矩 / 非高斯先验", "higher moments / non-Gaussian",
     "literature", True, lambda d: pd.Series(False, index=d.index), []),

    # ---------- A5 capacity & budget ----------
    ("A5", "S0", "A5", "A5 模型容量与训练预算", "Model Capacity & Training Budget", "code", False, TRUE, []),
    ("A5.tree_count", "A5", "A5", "A5.1 树数 10 → 100", "tree count 10 -> 100", "evidence", True,
     lambda d: d.n_estimators.notna() & d.n_estimators.ne(""), ["B0.tree_count.ER"]),
    ("A5.hidden_depth", "A5", "A5", "A5.2 hidden / depth", "hidden / depth", "code", True,
     lambda d: d.hidden_dim.notna() & d.hidden_dim.ne(""), ["A5.hidden_depth.shared_ER"]),
    ("A5.lr_wd", "A5", "A5", "A5.3 learning rate / weight decay", "lr / weight decay", "code", True,
     lambda d: d.learning_rate.notna() & d.learning_rate.ne(""), ["A5.lr_wd.shared_ER"]),
    ("A5.n_outer_inner", "A5", "A5", "A5.4 n_outer / n_inner", "n_outer / n_inner", "code", True,
     lambda d: d.n_outer.notna() & d.n_outer.ne(""), ["A5.n_outer_inner.shared_ER"]),

    # ---------- A6 data / time / robustness ----------
    ("A6", "S0", "A6", "A6 数据表示、时间结构与鲁棒性", "Data Representation, Time & Robustness",
     "code", False, TRUE, []),
    ("A6.standardization", "A6", "A6", "A6.1 标准化 / Gaussianization", "standardization / Gaussianization",
     "code", True, lambda d: d.w_matrix_space.notna() & d.w_matrix_space.ne(""), []),
    ("A6.lag", "A6", "A6", "A6.2 feature lag", "feature lag", "code", True,
     lambda d: pd.Series(False, index=d.index), []),
    ("A6.trend", "A6", "A6", "A6.3 time trend", "time trend", "code", True,
     lambda d: pd.Series(False, index=d.index), []),
    ("A6.regime", "A6", "A6", "A6.4 regime break", "regime break", "code", True,
     lambda d: pd.Series(False, index=d.index), []),
    ("A6.huber", "A6", "A6", "A6.5 Huber 鲁棒损失", "Huber robust loss", "code", True,
     eq("prediction_loss", "huber"), []),

    # ---------- G0 contract & F0 ----------
    ("G0", "S0", "G0", "G0 公平比较合同", "Fair-comparison contract", "code", False, TRUE, []),
    ("G0.1_metric_schema_valid", "G0", "G0", "G0.1 metric schema valid", "metric schema valid", "code", False,
     lambda d: pd.Series(True, index=d.index), []),
    ("G0.2_complete_artifact", "G0", "G0", "G0.2 complete artifact", "complete artifact", "code", False,
     lambda d: pd.Series(True, index=d.index), []),
    ("G0.3_exact_split_hash", "G0", "G0", "G0.3 exact split hash", "exact split hash", "code", False,
     lambda d: pd.Series(False, index=d.index), []),
    ("G0.4_nuisance_config_hash", "G0", "G0", "G0.4 nuisance config hash", "nuisance config hash", "code", False,
     lambda d: pd.Series(True, index=d.index), []),
    ("G0.5_fold_W_cache_hash", "G0", "G0", "G0.5 fold-W cache hash", "fold-W cache hash", "code", False,
     lambda d: pd.Series(False, index=d.index), []),
    ("G0.6_frozen_selection_receipt", "G0", "G0", "G0.6 frozen selection receipt", "frozen selection receipt",
     "code", False, lambda d: pd.Series(False, index=d.index), []),
    ("F0", "S0", "F0", "F0 冻结配置 + locked holdout", "Freeze + locked holdout", "code", False,
     lambda d: pd.Series(False, index=d.index), []),
]

LIT = {
    "A3.KCI_HSIC": "零偏相关依赖线性/高斯；非线性需核方法 KCI/HSIC (Zhang et al. arXiv:1202.3775)",
    "A4.stability_selection": "子抽样结构选择概率可控制错误选择 (Meinshausen & Bühlmann 2010)",
    "A4.non_gaussian_moments": "非高斯扰动可识别方向 (LiNGAM, Shimizu et al. 2006)",
    "A1.HYBRID_ALM_PBM": "随机/不精确增广拉格朗日 (Li et al. PMLR v130)",
    "A1.STOCHASTIC_PBM": "Adam-SPBM 随机惩罚-障碍 (spbm/Adam_SPBM.pdf)",
    "A1.SCO_LAYER": "SCO/proximal 层与 ALM/PBM 不等价，需独立比较",
}
SCRIPT_ONLY = {"A6.lag", "A6.trend", "A6.regime", "A6.huber", "A3.quantile_residual",
               "A2.W_mask", "A2.W_bias_calibration", "A4.parents_limit"}
REGISTRY_ONLY = {"S0", "B0", "A1", "A2", "A3", "A4", "A5", "A6", "G0"}
LITERATURE_ONLY = {"A3.KCI_HSIC", "A4.stability_selection", "A4.non_gaussian_moments"}
NOT_IMPL_GATES = {"G0.3_exact_split_hash", "G0.5_fold_W_cache_hash", "G0.6_frozen_selection_receipt", "F0"}
OVERRIDE_CLAIM = {"B0.mark_cc_total100": "pending_model_artifact_audit"}


def count(fn, d):
    try:
        r = fn(d)
        return int(r.sum()) if hasattr(r, "sum") else int(r)
    except Exception:
        return 0


def comp_claim(ids):
    if not ids:
        return "-", 0, "missing"
    sub = idx[idx.comparison_id.isin(ids)]
    if len(sub) == 0:
        return "-", 0, "missing"
    # Cohorts are independent execution batches, not additional samples from
    # the same graph/noise seeds.  A node reports the largest coherent cohort
    # rather than summing legacy reruns and accidentally manufacturing power.
    used = int(sub.n_clusters_used.max())
    exp = int(sub.n_clusters_expected.max())
    ev = "missing" if used == 0 else ("complete" if used >= exp else "partial")
    seen = []
    for r in sub.itertuples():
        lab = f"{r.status}({str(r.scope).replace('synthetic/','')})"
        if lab not in seen:
            seen.append(lab)
    return "; ".join(seen), used, ev


rows = []
for nid, parent, axis, zh, en, source, crossed, fn, comps in NODES:
    u = count(fn, runs)
    vr = count(fn, valid)
    claim, paired, ev = comp_claim(comps)
    if source == "literature":
        impl = "hypothesis"
        if claim == "-":
            claim = "literature_hypothesis"
    elif nid in LITERATURE_ONLY:
        impl = "hypothesis"
        claim = "literature_hypothesis" if claim == "-" else claim
    elif nid in REGISTRY_ONLY:
        impl = "registered"
    elif nid in NOT_IMPL_GATES:
        impl = "not_implemented"
    elif u > 0:
        impl = "implemented"
    elif nid in SCRIPT_ONLY or source == "code":
        impl = "implemented"          # code path exists, no runs yet
    else:
        impl = "not_implemented"
    if nid in OVERRIDE_CLAIM:
        claim = OVERRIDE_CLAIM[nid]
    psub = pooled[pooled.comparison_id.isin(comps)].copy() if comps else pooled.iloc[0:0]
    if len(psub):
        psub["_abs"] = pd.to_numeric(psub.pooled_rel, errors="coerce").abs()
        psub = psub.sort_values(["pooled_units", "_abs"], ascending=False)
        top = psub.iloc[0]
        pooled_rel = float(pd.to_numeric(top.pooled_rel, errors="coerce"))
        pooled_dir = str(top.pooled_direction)
        pooled_units = int(pd.to_numeric(top.pooled_units, errors="coerce") or 0)
    else:
        pooled_rel = float("nan"); pooled_dir = "-"; pooled_units = 0
    rows.append(dict(node_id=nid, parent=parent, axis=axis, zh=zh, en=en, source=source,
                     crossed_factor=crossed, implementation=impl, evidence=ev, claim=claim,
                     unique_runs=u, valid_runs=vr, paired_units=paired,
                     pooled_rel=pooled_rel, pooled_direction=pooled_dir, pooled_units=pooled_units,
                     comparison_ids=";".join(comps), literature=LIT.get(nid, "")))

out = pd.DataFrame(rows)
out.to_csv(REPORTS / "evidence_nodes.csv", index=False)
print(f"wrote {REPORTS/'evidence_nodes.csv'} ({len(out)} nodes)")
print(out[["node_id", "axis", "crossed_factor", "implementation", "evidence", "claim",
           "unique_runs", "valid_runs", "paired_units"]].to_string(index=False))
