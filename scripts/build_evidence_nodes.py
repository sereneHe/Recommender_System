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

# Baseline ownership: B0 owns the canonical no-constraint/W/CE/W+CE arms.
# A2 contains only genuinely different embeddings/modifiers; it must not
# register or rerun those baseline arms under another node name.
#
Counts are NOT additive: each node reports
  unique_runs / valid_runs / paired_units  and  crossed_factor.
Claims are SCOPE-SPECIFIC (status(scope)); literature only yields
literature_hypothesis, never `supported`.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd
import yaml

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

receipt_root = ROOT / "reports" / "frozen_selection"
frozen_receipts = []
frozen_receipt_problems: list[str] = []
if receipt_root.exists():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts" / "evidence_tree"))
    import f0_guard

    _registry_path = ROOT / "experiment_registry.yaml"
    try:
        _registry = yaml.safe_load(_registry_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _registry = {}
    # A receipt is only "frozen" when the F0 guard accepts it: schema v2,
    # hash-intact, validation-only, bound to the registry and the reserved
    # holdout seed table.  A rejected receipt is recorded, never silently kept.
    for receipt_path in sorted(receipt_root.glob("*.yaml")):
        try:
            receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8")) or {}
            problems = f0_guard.validate_receipt(receipt, _registry, _registry_path)
            if problems:
                frozen_receipt_problems.extend(f"{receipt_path.name}: {p}" for p in problems)
            else:
                frozen_receipts.append(receipt_path)
        except Exception as exc:
            frozen_receipt_problems.append(f"{receipt_path.name}: unreadable ({exc})")

# Pipeline gate states (dashboard state machine).  Each is a durable receipt
# that decides whether the next step may execute:
#   G0_H1.json            open / running / failed / passed  (pairing contract)
#   frozen_selection/*.yaml                        (human freeze after review)
#   holdout_receipts/*.json                        (independent F0 confirmation)
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

G0_H1_GATE = _load_json(REPORTS / "gates" / "G0_H1.json")
G0_H1_STATE = str(G0_H1_GATE.get("status", "open" if not G0_H1_GATE else "open"))

HOLDOUT_ROOT = REPORTS / "holdout_receipts"
holdout_receipts: list[Path] = []
if HOLDOUT_ROOT.exists():
    holdout_receipts = sorted(HOLDOUT_ROOT.glob("*.json"))


def _strict_cohort_eligible(cohort: str) -> bool:
    """True only for an ACTIVE, registered cohort with no diagnostic pattern.

    Legacy pilot/priority cohorts have no reservation (or are retired), so they
    are marked diagnostic-only in the tree instead of being shown as evidence.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from experiment_registry import (
            COHORT_REGISTRY_DIR, is_cohort_eligible_for_strict, load as _load_reg,
        )
        reg = _load_reg(ROOT / "experiment_registry.yaml")
        return is_cohort_eligible_for_strict(str(cohort), reg, COHORT_REGISTRY_DIR)
    except Exception:
        return False

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


# Data-family guards.  Evidence-tree counts must not silently mix ER, SF,
# FRED, or CoDiet runs just because a solver parameter happens to match.
SYN_ER = lambda d: d.problem.eq("synthetic") & d.graph_type.eq("ER") & d.sem_type.eq("gauss")
SYN_SF = lambda d: d.problem.eq("synthetic") & d.graph_type.eq("SF") & d.sem_type.eq("gauss")
SYN_ER_NONLIN = lambda d: d.problem.eq("synthetic") & d.graph_type.eq("ER") & d.sem_type.eq("nonlinear")
FRED = lambda d: d.problem.eq("industry_eu")


def _has_col(d, col):
    try:
        return col in d.columns
    except Exception:
        return False


def H1_RUNS(d):
    if not _has_col(d, "evidence_node"):
        return pd.Series(False, index=d.index)
    return d["evidence_node"].astype(str).eq("H1")


# id, parent, axis, zh, en, source, crossed, count_fn, comparison_ids
NODES = [
    ("S0", "", "S0", "实验搜索空间与可运行锚点", "Experiment search space and runnable anchor",
     "code", False, TRUE, []),

    # ---------- B0 baseline rail ----------
    ("B0", "S0", "B0", "B0 端到端基线轨（可比较 fallback）", "End-to-end baseline rail (comparable fallback)",
     "evidence", False, N(SYN_ER, HC), []),
    ("B0.mark_100", "B0", "B0", "B0.1 Mark-100（100 棵树）", "Mark-100 (100 trees)", "evidence", False,
     lambda d: SYN_ER(d) & d.experiment.str.contains("mark_100", na=False), ["B0.tree_count.ER", "B0.mark_100.er"]),
    ("B0.mark_cc_total100", "B0", "B0", "B0.2 Mark-CC-100（总预算 100 棵，待产物核验）",
     "Mark-CC-100 (nominal 100-tree budget; artifact audit pending)", "code", False,
     lambda d: SYN_ER(d) & d.solver.eq("mark_with_cc"), ["B0.mark_cc_100.er"]),
    ("B0.hc_nn_no_constraint", "B0", "B0", "B0.3 HC-NN 无约束基线", "HC-NN no-constraint baseline",
     "evidence", False, N(SYN_ER, HC, eq("use_ci_penalty", False), eq("use_w_constraints", False)), ["B0.hc_nn.er"]),
    ("B0.hc_w_only", "B0", "B0", "B0.4 HC-NN + W（基线臂）", "HC-NN + W (baseline arm)", "evidence", False,
     N(SYN_ER, HC, eq("use_w_constraints", True), eq("use_ci_penalty", False)), ["B0.hc_w_only.er"]),
    ("B0.hc_ce_only", "B0", "B0", "B0.5 HC-NN + CE（基线臂）", "HC-NN + CE (baseline arm)", "evidence", False,
     N(SYN_ER, HC, eq("use_ci_penalty", True), eq("use_w_constraints", False)),
     ["B0.hc_ce.er"]),
    ("B0.hc_w_ce", "B0", "B0", "B0.6 HC-NN + W + CE（基线臂）", "HC-NN + W + CE (baseline arm)", "evidence", False,
     N(SYN_ER, HC, eq("use_w_constraints", True), eq("use_ci_penalty", True)),
     ["B0.hc_w_ce.er", "B0.hc_w_ce_vs_ce.er"]),

    # ---------- A1 optimization mechanics ----------
    ("A1", "S0", "A1", "A1 优化器与约束求解后端", "Optimizer and constraint-solver backends", "code", False, TRUE, []),
    ("A1.ALM_ALL", "A1", "A1", "A1.1 CE 全部使用 ALM", "All CE constraints use ALM", "code", False,
     N(SYN_ER, eq("ce_constraint_backend", "alm_all")), ["A1.ALM_ALL.er"]),
    ("A1.PBM_ALL", "A1", "A1", "A1.2 CE 全部使用确定性 PBM", "All CE constraints use deterministic PBM", "code", False,
     N(SYN_ER, eq("ce_constraint_backend", "pbm_all"), eq("ce_pbm_backend", "humancompatible_pbm")), ["A1.PBM_ALL.er"]),
    ("A1.HYBRID_ALM_PBM", "A1", "A1", "A1.3 Hybrid：独立约束→ALM、依赖约束→PBM",
     "Hybrid: independent→ALM, dependent→PBM", "evidence", False,
     lambda d: pd.Series(False, index=d.index), []),
    ("A1.STOCHASTIC_PBM", "A1", "A1", "A1.4 CE 全部使用 SPBM", "All CE constraints use stochastic PBM (SPBM)", "code", False,
     N(SYN_ER, eq("ce_constraint_backend", "pbm_all"), eq("ce_pbm_backend", "stochastic_pbm"), eq("stochastic_constrained", False)), ["A1.STOCHASTIC_PBM.er"]),
    ("A1.SCO_LAYER", "A1", "A1", "A1.5 SPBM + SCO/proximal 训练层",
     "SPBM + SCO/proximal training layer", "code", False, N(SYN_ER, eq("stochastic_constrained", True)), ["A1.SCO_LAYER.er"]),

    # ---------- A2 constraint embedding ----------
    ("A2", "S0", "A2", "A2 约束嵌入与组合方式", "Constraint embedding and combination", "code", False, TRUE, []),
    # W-only, CE-only and W+CE are baseline arms, not separate A2
    # experiments.  Their canonical evidence lives under B0.  Keeping copies
    # here would count the same intervention twice and mix different cohorts.
    ("A2.W_target_residual", "A2", "A2", "A2.1 目标相关残差 W（区别于基线 W）", "Target-related residual W (distinct from baseline W)", "code", False,
     N(SYN_ER, eq("w_constraint_mode", "target_residual")), ["A2.W_target_residual.er"]),
    ("A2.W_mask", "A2", "A2", "A2.2 预测相关 mask W", "Prediction-dependent mask W",
     "code", False, N(SYN_ER, eq("w_prediction_dependent_mask", True)), ["A2.W_mask.er"]),
    ("A2.W_bias_calibration", "A2", "A2", "A2.3 仅 W 偏置校准（不加 W 惩罚）", "W bias-only calibration (no W penalty)",
     "code", False, N(SYN_ER, eq("w_bias_calibration", True)), ["A2.W_bias_calibration.er"]),
    ("A2.balanced_batch", "A2", "A2", "A2.4 平衡批次（CE 采样修饰）", "Balanced batches (CE sampling modifier)",
     "code", True, N(SYN_ER, eq("ce_use_balanced_batches", True)), ["A2.balanced_batch.er"]),
    ("A2.pruning", "A2", "A2", "A2.5 冗余约束剪枝（CE 修饰）", "Redundant-constraint pruning (CE modifier)",
     "code", True, N(SYN_ER, eq("ci_prune_redundant", True)), ["A2.pruning.er"]),
    ("A2.W_adds_to_CE", "A2", "A2", "A2.6 W 在 CE 之上是否提供新增信息", "Does W add information beyond CE?",
     "evidence", False, N(SYN_ER, HC, eq("use_w_constraints", True), eq("use_ci_penalty", True)),
     ["B0.hc_w_ce_vs_ce.er"]),

    # ---------- A3 CI/CE statistic ----------
    ("A3", "S0", "A3", "A3 条件独立统计量与不确定性", "Conditional-independence statistics and uncertainty", "code", False, TRUE, []),
    ("A3.covariance", "A3", "A3", "A3.1 残差协方差 CE 统计量", "Residual-covariance CE statistic",
     "evidence", True, N(SYN_ER, eq("ci_penalty_kind", "conditional_expectation"), eq("ce_statistic_kind", "covariance")), ["A3.covariance.er"]),
    ("A3.partial_correlation", "A3", "A3", "A3.2 标准化偏相关 CE 统计量", "Standardized partial-correlation CE statistic",
     "evidence", True, N(SYN_ER, eq("ce_statistic_kind", "partial_correlation")), []),
    ("A3.pcorr.linear_ER", "A3.partial_correlation", "A3", "A3.2a 线性 ER 数据", "Linear ER data",
     "evidence", True, N(SYN_ER, eq("ce_statistic_kind", "partial_correlation")),
     ["C0b.ce_vs_nn.ER", "C0b.ce_vs_nn.repair_ER", "C0b.ce_vs_nn.priority_ER"]),
    ("A3.pcorr.nonlinear_ER", "A3.partial_correlation", "A3", "A3.2b 非线性 ER 数据", "Nonlinear ER data",
     "evidence", True, N(SYN_ER_NONLIN, eq("ce_statistic_kind", "partial_correlation")),
     ["P1.3.nonlin_ce_vs_nn.ER", "P1.3.nonlin_ce_vs_nn.repair_ER"]),
    ("A3.pcorr.SF", "A3.partial_correlation", "A3", "A3.2c SF 数据", "SF data",
     "evidence", True, N(SYN_SF, eq("ce_statistic_kind", "partial_correlation")), ["C0b.ce_vs_nn.SF"]),
    ("A3.discrete_CMI", "A3", "A3", "A3.3 离散/分箱 CMI", "Discrete/binned CMI", "evidence", True,
     N(SYN_ER, eq("ci_penalty_kind", "discrete_conditional_independence")), []),
    ("A3.quantile_residual", "A3", "A3", "A3.4 分位数残差化", "Quantile residualization", "code", True,
     N(SYN_ER, eq("ce_residualize_method", "quantile")), ["P1.3.quantile_vs_linear.ER", "P1.3.quantile_vs_linear.repair_ER"]),
    ("A3.window_SE", "A3", "A3", "A3.5 跨时间窗 SE", "Cross-window standard error", "code", True, N(SYN_ER, eq("ce_se_method", "window")), ["A3.window_vs_hac.er"]),
    ("A3.HAC_SE", "A3", "A3", "A3.6 HAC/Newey-West SE", "HAC/Newey-West standard error", "code", True, N(SYN_ER, eq("ce_se_method", "hac")), ["A3.window_vs_hac.er"]),
    ("A3.sign_flip_filter", "A3", "A3", "A3.7 有符号依赖的 sign-flip 过滤", "Signed-dependence sign-flip filter", "code", True,
     lambda d: SYN_ER(d) & d.ce_window_filter_enabled.astype(str).str.lower().eq("true"), ["A3.sign_flip_filter.er"]),
    ("A3.KCI_HSIC", "A3", "A3", "A3.8 KCI/HSIC 非线性 CI（待接入训练）", "KCI/HSIC nonlinear CI (not in training yet)", "literature", True,
     lambda d: pd.Series(False, index=d.index), []),

    # ---------- A4 graph estimation & causal prior ----------
    ("A4", "S0", "A4", "A4 图估计与因果先验", "Graph estimation and causal priors", "code", False, TRUE, []),
    ("A4.edge_penalty", "A4", "A4", "A4.1 边数惩罚", "Edge-count penalty", "code", True,
     lambda d: SYN_ER(d) & (pd.to_numeric(d.edge_penalty, errors="coerce").fillna(0) > 0), ["A4.edge_penalty.er"]),
    ("A4.parents_limit", "A4", "A4", "A4.2 最大父节点数限制", "Maximum-parent cap", "code", True,
     lambda d: SYN_ER(d) & (pd.to_numeric(d.max_parents, errors="coerce").fillna(0) > 0),
     ["A4.parents_limit_3.er", "A4.parents_limit_4.er", "A4.parents_limit_5.er"]),
    ("A4.clique_cap", "A4", "A4", "A4.3 Clique cuts/大小上限", "Clique cuts and size cap", "code", True,
     lambda d: SYN_ER(d) & d.enable_clique_constraints.astype(str).str.lower().eq("true"), ["A4.clique_cap.er"]),
    ("A4.MIP_time_gap", "A4", "A4", "A4.4 MIP 时间限制与 gap", "MIP time-limit and gap", "code", True,
     lambda d: SYN_ER(d) & d.recalculate_dag.astype(str).str.lower().eq("true") &
               (pd.to_numeric(d.time_limit, errors="coerce").eq(1800)
                | pd.to_numeric(d.target_mip_gap, errors="coerce").eq(0.01)),
     ["A4.mip_time.er", "A4.mip_gap.er"]),
    ("A4.stability_selection", "A4", "A4", "A4.5 Bootstrap 稳定性选择（待实现）", "Bootstrap stability selection (not implemented)", "literature", True,
     lambda d: pd.Series(False, index=d.index), []),
    ("A4.non_gaussian_moments", "A4", "A4", "A4.6 非高斯高阶矩先验（待实现）", "Non-Gaussian higher-moment prior (not implemented)",
     "literature", True, lambda d: pd.Series(False, index=d.index), []),
    ("A4.graph_quality", "A4", "A4", "A4.7 估计图质量（F1/SHD/边数）", "Estimated-graph quality (F1/SHD/edges)",
     "code", True, lambda d: SYN_ER(d) & pd.to_numeric(d.edge_f1, errors="coerce").notna(), []),
    ("A4.constraint_quality", "A4", "A4", "A4.8 生成约束质量（独立/依赖数量）", "Generated-constraint quality (independent/dependent counts)",
     "code", True, lambda d: SYN_ER(d) & pd.to_numeric(d.indep_constraints, errors="coerce").notna(), []),
    ("A4.solver_stability", "A4", "A4", "A4.9 求解稳定性（状态/gap/时间）", "Solver stability (status/gap/time)",
     "code", True, lambda d: SYN_ER(d) & (pd.to_numeric(d.n_folds_graph, errors="coerce").fillna(0) > 0), []),

    # ---------- A5 capacity & budget ----------
    ("A5", "S0", "A5", "A5 模型容量与训练预算", "Model capacity and training budget", "code", False, TRUE, []),
    ("A5.tree_count", "A5", "A5", "A5.1 Boosting 树预算：10 → 100", "Boosting tree budget: 10 -> 100", "evidence", True,
     lambda d: SYN_ER(d) & d.n_estimators.notna() & d.n_estimators.ne(""), ["B0.tree_count.ER"]),
    ("A5.hidden_depth", "A5", "A5", "A5.2 NN 隐藏层宽度与深度", "NN hidden width and depth", "code", True,
     lambda d: SYN_ER(d) & d.hidden_dim.notna() & d.hidden_dim.ne(""), ["A5.hidden_depth.er"]),
    ("A5.lr_wd", "A5", "A5", "A5.3 NN 学习率与权重衰减", "NN learning rate and weight decay", "code", True,
     lambda d: SYN_ER(d) & d.learning_rate.notna() & d.learning_rate.ne(""), ["A5.lr_wd.er"]),
    ("A5.n_outer_inner", "A5", "A5", "A5.4 外层/内层训练预算", "Outer/inner training budget", "code", True,
     lambda d: SYN_ER(d) & d.n_outer.notna() & d.n_outer.ne(""), ["A5.n_outer_inner.er", "A5.n_outer_inner_matched.er"]),

    # ---------- A6 data / time / robustness ----------
    ("A6", "S0", "A6", "A6 数据表示、时间结构与鲁棒损失", "Data representation, time structure and robust loss",
     "code", False, TRUE, []),
    ("A6.standardization", "A6", "A6", "A6.1 数据尺度：raw / 标准化 / Gaussianization", "Data scale: raw / standardized / Gaussianized",
     "code", True, lambda d: FRED(d) & d.w_matrix_space.notna() & d.w_matrix_space.ne(""), []),
    ("A6.lag", "A6", "A6", "A6.2 预测变量滞后", "Predictor lag", "code", True,
     lambda d: FRED(d) & (pd.to_numeric(d.feature_lag, errors="coerce").fillna(0) > 0), ["A6.lag.fred"]),
    ("A6.trend", "A6", "A6", "A6.3 时间趋势项", "Time-trend feature", "code", True,
     lambda d: FRED(d) & d.add_time_trend.astype(str).str.lower().eq("true"), ["A6.trend.fred"]),
    ("A6.regime", "A6", "A6", "A6.4 Regime 断点处理", "Regime-break handling", "code", True,
     lambda d: FRED(d) & d.regime_break_date.notna() & d.regime_break_date.astype(str).ne(""), ["A6.regime.fred"]),
    ("A6.huber", "A6", "A6", "A6.5 Huber 预测损失", "Huber prediction loss", "code", True,
     lambda d: FRED(d) & d.prediction_loss.eq("huber"), ["A6.huber.fred"]),

    # ---------- A7 mixed/discrete CoDiet ----------
    ("A7", "S0", "A7", "A7 CoDiet 类型匹配约束", "CoDiet type-matched constraints", "code", False,
     lambda d: d.problem.eq("codiet"), []),
    ("A7.codiet_cmi", "A7", "A7", "A7.1 CoDiet 离散 CMI 相对 W-only", "CoDiet discrete CMI versus W-only", "evidence", True,
     lambda d: d.problem.eq("codiet") & d.ci_penalty_kind.eq("discrete_conditional_independence"),
     ["A7.codiet_cmi_vs_w"]),

    # ---------- H1 confirmatory (frozen protocol; gated by G0) ----------
    # The strict H1 conclusion is published ONLY when the G0 gate receipt for
    # H1.ce_vs_nn.ER is "passed" (20/20 units, unique pairing, all contracts).
    # Otherwise the node reports the gate state (open/running/failed), never a
    # method verdict.
    ("H1", "S0", "H1", "H1 CE-only 相对 NN（合成 ER 确认）", "CE-only vs NN confirmatory (synthetic ER)",
     "evidence", False, H1_RUNS, ["H1.ce_vs_nn.ER"]),
    ("H1.ce_vs_nn", "H1", "H1", "H1.1 CE-only vs 匹配 NN（20 graph×noise）", "CE-only vs matched NN (20 graph x noise)",
     "evidence", False, H1_RUNS, ["H1.ce_vs_nn.ER"]),

    # ---------- G0 contract & F0 ----------
    ("G0", "S0", "G0", "G0 公平比较与可追溯性合同", "Fair-comparison and traceability contract", "code", False, TRUE, []),
    ("G0.1_metric_schema_valid", "G0", "G0", "G0.1 指标表结构有效", "Metric schema is valid", "code", False,
     lambda d: d.metric_validity.eq("valid"), []),
    ("G0.2_complete_artifact", "G0", "G0", "G0.2 运行产物完整", "Run artifact is complete", "code", False,
     lambda d: d.artifact_complete.fillna(False).astype(bool), []),
    ("G0.3_exact_split_hash", "G0", "G0", "G0.3 精确 split hash", "Exact split hash", "code", False,
     lambda d: pd.Series(False, index=d.index), []),
    ("G0.4_nuisance_config_hash", "G0", "G0", "G0.4 非处理参数 hash", "Nuisance-configuration hash", "code", False,
     lambda d: pd.Series(False, index=d.index), []),
    ("G0.5_fold_W_cache_hash", "G0", "G0", "G0.5 每折 W-cache hash", "Fold-local W-cache hash", "code", False,
     lambda d: pd.Series(False, index=d.index), []),
    ("G0.6_frozen_selection_receipt", "G0", "G0", "G0.6 冻结选择凭证", "Frozen-selection receipt",
     "code", False, lambda d: pd.Series(False, index=d.index), []),
    ("F0", "S0", "F0", "F0 冻结配置与最终 locked holdout", "Frozen configuration and locked holdout", "code", False,
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
               "G0.6_frozen_selection_receipt",
               "A2.W_mask", "A2.W_bias_calibration", "A4.parents_limit"}
REGISTRY_ONLY = {"S0", "B0", "A1", "A2", "A3", "A4", "A5", "A6", "G0", "H1"}
LITERATURE_ONLY = {"A3.KCI_HSIC", "A4.stability_selection", "A4.non_gaussian_moments"}
NOT_IMPL_GATES = {"F0"}
OVERRIDE_CLAIM = {
    "B0.mark_cc_total100": "open_hard_gate",
    "A1.ALM_ALL": "pending_pairing",
    "A1.PBM_ALL": "pending_pairing",
    "A1.HYBRID_ALM_PBM": "inactive_under_independence_only",
    "A1.STOCHASTIC_PBM": "open",
    "A1.SCO_LAYER": "open",
    "A4.graph_quality": "implemented_unverified",
    "A4.constraint_quality": "implemented_unverified",
    "A4.solver_stability": "implemented_unverified",
    "G0.1_metric_schema_valid": "checker_implemented",
    "G0.2_complete_artifact": "checker_implemented",
}

# Receipt-driven gates.  These nodes are passable: their claim is read from the
# evidence index receipts rather than hard-coded to a permanent failure.  The
# state is one of open (no paired receipt yet) / failed (paired but not all
# receipts match) / passed (every paired receipt matches).
RECEIPT_GATE = {
    "G0.3_exact_split_hash": ("split_hash_equal", "open"),
    "G0.4_nuisance_config_hash": ("nuisance_hash_equal", "open"),
    "G0.5_fold_W_cache_hash": ("fold_w_cache_hash_equal", "open"),
}


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
    # Drop the "planned but not run" out_of_scope label whenever a real verdict
    # exists, so a node never mixes a genuine result with a placeholder spec.
    real = [l for l in seen if not l.startswith("out_of_scope")]
    if real:
        seen = real
    return "; ".join(seen), used, ev


rows = []
for nid, parent, axis, zh, en, source, crossed, fn, comps in NODES:
    diagnostic_only = False
    u = count(fn, runs)
    vr = count(fn, valid)
    claim, paired, ev = comp_claim(comps)
    if nid == "G0.6_frozen_selection_receipt":
        u = vr = len(frozen_receipts)
        paired = 0
        ev = "complete" if frozen_receipts else "missing"
        if frozen_receipts:
            claim = "passed"
        elif frozen_receipt_problems:
            claim = "failed"
        else:
            claim = "open"
    # G0 receipt gates read evidence_index receipts, not run counts.  The gate
    # is passable: it clears only when every comparison that actually formed a
    # pair reports the receipt.  With no pair at all the gate stays OPEN (not
    # yet exercised); with a pair whose receipt is missing/unequal it is FAILED.
    receipt_state = None
    if nid in RECEIPT_GATE:
        col, _fallback = RECEIPT_GATE[nid]
        # Scope the gate to frozen-protocol comparisons once they exist, so a
        # legacy diagnostic pair without receipts can never hold the H1 gate
        # hostage (nor let a legacy pair pass it).  Before any frozen pair
        # exists, report the honest global state.
        try:
            import sys as _sys2
            _sys2.path.insert(0, str(ROOT / "scripts"))
            from experiment_registry import load as _load_reg
            _frozen_ids = {h.get("id") for h in
                           (_load_reg().get("hypotheses") or {}).values() if h.get("id")}
        except Exception:
            _frozen_ids = set()
        _scope = idx[idx.comparison_id.isin(_frozen_ids)] if _frozen_ids else idx
        paired_rows = (_scope[pd.to_numeric(_scope.get("n_clusters_used"), errors="coerce").fillna(0) > 0]
                       if "n_clusters_used" in _scope.columns else _scope.iloc[0:0])
        if col in _scope.columns and len(paired_rows):
            passed = int((paired_rows[col].astype(str).str.lower() == "true").sum())
            u = vr = passed
            receipt_state = "passed" if passed == len(paired_rows) else "failed"
        else:
            u = vr = 0
            receipt_state = "open"
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
    if nid in RECEIPT_GATE:
        claim = receipt_state or RECEIPT_GATE[nid][1]
    elif nid in ("H1", "H1.ce_vs_nn"):
        # H1 publishes a method verdict ONLY through the G0 gate.  Evidence is
        # retained in the index either way, but the tree must not present a
        # strict conclusion while the contract is open or failed.
        if G0_H1_STATE == "passed":
            pass  # keep the index-derived claim (supported/.../inconclusive)
        elif G0_H1_STATE == "failed":
            claim = "failed"
            ev = "partial"
        else:
            # open: nothing paired yet; running: some but not all units paired.
            try:
                _used = int(pd.to_numeric(
                    idx[idx.comparison_id == "H1.ce_vs_nn.ER"]["n_clusters_used"],
                    errors="coerce").fillna(0).max())
            except Exception:
                _used = 0
            claim = "running" if _used > 0 else "open"
            ev = "partial" if _used > 0 else "missing"
    elif nid == "F0":
        if holdout_receipts:
            claim = "passed"
            ev = "complete"
            impl = "implemented"
        elif frozen_receipts:
            claim = "open"
            ev = "partial"
            impl = "implemented"
        elif frozen_receipt_problems:
            claim = "failed"
            ev = "partial"
        else:
            claim = "open"
            ev = "missing"
    elif nid in OVERRIDE_CLAIM:
        claim = OVERRIDE_CLAIM[nid]
    elif (claim and "out_of_scope" in claim and impl == "implemented"
          and "supported" not in claim and "contradicted" not in claim
          and "equivalent" not in claim and "invalid" not in claim):
        # every referenced comparison is a not-yet-run planned spec -> do not
        # report out_of_scope; report implementation / pairing state instead
        claim = "pending_pairing" if axis == "A1" else "implemented_unverified"
    psub = pooled[pooled.comparison_id.isin(comps)].copy() if comps else pooled.iloc[0:0]
    if len(psub):
        # deterministic: the FIRST declared comparison, never the max-effect one
        order = {c: i for i, c in enumerate(comps)}
        psub = psub.assign(_o=psub.comparison_id.map(order)).sort_values("_o")
        top = psub.iloc[0]
        pooled_rel = float(pd.to_numeric(top.pooled_rel, errors="coerce"))
        pooled_dir = str(top.pooled_direction)
        pooled_units = int(pd.to_numeric(top.pooled_units, errors="coerce") or 0)
    else:
        pooled_rel = float("nan"); pooled_dir = "-"; pooled_units = 0

    # #4/#5: keep the cohort identity and a per-cohort (legacy/repair) verdict
    # on every node, instead of collapsing all cohorts into one label.
    sub_idx = idx[idx.comparison_id.isin(comps)] if comps else idx.iloc[0:0]
    # drop planned-but-not-run specs (out_of_scope) from the cohort display
    if len(sub_idx):
        sub_idx = sub_idx[~sub_idx.status.astype(str).str.contains("out_of_scope")]
    if len(sub_idx):
        cohorts = "; ".join(sorted({str(c) for c in sub_idx.cohort
                                    if str(c).strip() and str(c) not in ("nan", "<NA>")}))
        cohort_claims = "; ".join(
            f"{str(r.cohort)[:20] or 'n/a'}={r.status}" for r in sub_idx.itertuples())
        # Mark diagnostic-only when EVERY paired cohort is non-strict-eligible
        # (legacy pilot/priority, retired, or unregistered).  Such a node stays
        # visible for provenance but must never read as strict evidence.
        _paired = sub_idx[pd.to_numeric(sub_idx.get("n_clusters_used"), errors="coerce").fillna(0) > 0]
        _coh = [str(c) for c in _paired.cohort.unique()
                if str(c).strip() and str(c) not in ("nan", "<NA>")]
        if _coh and not any(_strict_cohort_eligible(c) for c in _coh):
            diagnostic_only = True
            cohort_claims = (cohort_claims + " · [diagnostic_only: 不可用于严格推断]").strip(" ·")
    else:
        cohorts = ""; cohort_claims = ""

    # #9: modifier nodes report constraint / SE receipts, not an NMSE claim
    METRIC_NODES = {"A3.window_SE", "A3.HAC_SE", "A3.sign_flip_filter", "A3.quantile_residual",
                    "A2.balanced_batch", "A2.pruning"}
    metrics = ""
    if nid in METRIC_NODES:
        try:
            sel = valid[fn(valid)] if len(valid) else runs.iloc[0:0]
            metrics = (f"n={len(sel)}; "
                       f"mean_indep={pd.to_numeric(sel.indep_constraints, errors='coerce').mean():.1f}; "
                       f"mean_dep={pd.to_numeric(sel.dep_constraints, errors='coerce').mean():.1f}; "
                       f"mean_signflip={pd.to_numeric(sel.ce_window_max_sign_flip_rate, errors='coerce').mean():.2f}")
        except Exception:
            metrics = ""

    rows.append(dict(node_id=nid, parent=parent, axis=axis, zh=zh, en=en, source=source,
                     crossed_factor=crossed, implementation=impl, evidence=ev, claim=claim,
                     unique_runs=u, valid_runs=vr, paired_units=paired,
                     pooled_rel=pooled_rel, pooled_direction=pooled_dir, pooled_units=pooled_units,
                     cohorts=cohorts, cohort_claims=cohort_claims, metrics=metrics,
                     diagnostic_only=diagnostic_only,
                     comparison_ids=";".join(comps), literature=LIT.get(nid, "")))

out = pd.DataFrame(rows)
out.to_csv(REPORTS / "evidence_nodes.csv", index=False)
print(f"wrote {REPORTS/'evidence_nodes.csv'} ({len(out)} nodes)")
print(out[["node_id", "axis", "crossed_factor", "implementation", "evidence", "claim",
           "unique_runs", "valid_runs", "paired_units"]].to_string(index=False))
