#!/usr/bin/env python3
"""Build paired comparisons vs a per-track reference and emit progress_tree.md.

Reads reports/progress/runs_metrics.csv (produced by scan_progress_tree.py).

Decision rule (docs): a candidate is judged by the PAIRED difference of
macro-NMSE against its reference on the same (problem, graph, target, seeds):
  improve  : mean(delta) <= -0.02 and bootstrap 95% CI upper < 0
  regress  : mean(delta) >= +0.02 and bootstrap 95% CI lower > 0
  flat     : otherwise (not distinguishable / below the frozen MDE)

Outputs:
  reports/progress/paired_vs_reference.csv
  reports/progress/progress_tree.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "progress"
RNG = np.random.default_rng(20260923)
MDE = 0.02
N_BOOT = 20000

GRAPH_SUF = re.compile(r"^(?P<base>.+?)_graph(?P<g>\d+)_noise(?P<n>\d+)$")
SEED_SUF = re.compile(r"^(?P<base>.+?)_seed(?P<s>\d+)$")

# phase -> reference arm (experiment name without the seed/graph suffix)
REFERENCE = {
    "ER": "PLAN_CE_NEW_ER_b2_mark_with_cc",          # root reference = Mark-CC
    "OTHER": "PLAN_CE_NEW_SF_b2_mark_with_cc",
    "EXDBN": "PLAN_EXDBN_ISSUE_s0_reference",
    "REPLAY": "REPLAY_SYNTHETIC_BASELINES_historical_best_mark_cc",
    "P3": "PLAN03_INDUSTRY_CONSTRAINTS_i0_dnn_no_constraint",
}
# extra within-track pairs: (reference_arm, candidate_arm)
REFERENCE_EXTRA = [
    ("PLAN_CE_NEW_ER_m0_xgb100_no_w", "PLAN_CE_NEW_ER_m1_xgb100_w"),          # W ablation on mark_with_cc
    ("PLAN_CE_NEW_ER_e0_no_constraint", "PLAN_CE_NEW_ER_e1_true_w"),           # true-W vs no-constraint (linear)
    ("PLAN_CE_NEW_ER_e0_no_constraint", "PLAN_CE_NEW_ER_e2_pure_ce_upgraded"), # pure CE vs no-constraint (linear)
    # nonlinear arms must be compared against the nonlinear Mark-CC baseline
    ("PLAN_CE_NEW_ER_b2nl_mark_with_cc", "PLAN_CE_NEW_ER_e7_nonlinear_no_constraint"),
    ("PLAN_CE_NEW_ER_b2nl_mark_with_cc", "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce"),
    ("PLAN_CE_NEW_ER_b2nl_mark_with_cc", "PLAN_CE_NEW_ER_e5_nonlinear_quantile_ce"),
    ("PLAN_CE_NEW_ER_b2nl_mark_with_cc", "PLAN_CE_NEW_ER_e6_nonlinear_w_plus_ce"),
    ("PLAN_CE_NEW_SF_b2_mark_with_cc", "PLAN_CE_NEW_SF_e0_no_constraint"),
]

PHASE_TITLE = {
    "P1": "P1 合成约束筛查 / Synthetic constraint screen",
    "P2": "P2 Industry HC 调参 / HC tuning",
    "P3": "P3 Industry W/CE/batch 消融 / constraint ablation",
    "P4": "P4 时间/趋势/regime/鲁棒损失 / time robustness",
    "P5": "P5 方法对比 Mark/Mark-CC/HC / method comparison",
    "P6": "P6 CoDiet 类型筛查 / CoDiet screen",
    "P7": "P7 非高斯合成压力测试 / noise robustness",
    "P8": "P8 BD 后验稳定约束 / posterior-stable BD",
    "P9": "P9 CE 预审计门 / CE pre-audit gate",
    "ER": "P9/ER CE-new ER 主筛查（当前活跃线）",
    "CE_NEW": "P9/CE CE-new Industry·CoDiet 预审计",
    "EXDBN": "G1 图形过密诊断 / dense-graph diagnosis",
    "REPLAY": "B  基线复现 / baseline replay",
}


def parse_exp(exp: str):
    m = GRAPH_SUF.match(exp)
    if m:
        return m["base"], int(m["g"]), int(m["n"]), None
    m = SEED_SUF.match(exp)
    if m:
        return m["base"], None, None, int(m["s"])
    return exp, None, None, None


def boot_ci(d: np.ndarray):
    n = len(d)
    if n == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    df = pd.read_csv(OUT / "runs_metrics.csv")
    df["arm"], df["g_seed"], df["n_seed"], df["seed2"] = zip(*df["experiment"].map(parse_exp))
    df["pkey"] = df.apply(
        lambda r: (
            r["problem"], r["graph_type"], r["target"],
            r["g_seed"] if pd.notna(r["g_seed"]) else r["graph_seed"],
            r["n_seed"] if pd.notna(r["n_seed"]) else r["noise_seed"],
            r["seed2"] if pd.notna(r["seed2"]) else r["seed"],
        ),
        axis=1,
    )
    # collapse duplicates: mean macro_nmse per (arm, pkey)
    cell = (
        df.dropna(subset=["macro_nmse"])
        .groupby(["phase", "arm", "pkey"], dropna=False)
        .agg(nmse=("macro_nmse", "mean"), f1=("edge_f1", "mean"), shd=("shd", "mean"))
        .reset_index()
    )

    rows = []
    for phase, ref_arm in REFERENCE.items():
        sub = cell[cell.phase == phase]
        ref = sub[sub.arm == ref_arm].set_index("pkey")["nmse"]
        if ref.empty:
            continue
        for arm, g in sub.groupby("arm"):
            if arm == ref_arm:
                continue
            g = g.set_index("pkey")
            common = g.index.intersection(ref.index)
            if len(common) == 0:
                continue
            d = (g.loc[common, "nmse"] - ref.loc[common]).to_numpy(dtype=float)
            d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            lo, hi = boot_ci(d)
            mean_d = float(np.mean(d))
            if mean_d <= -MDE and hi < 0:
                verdict = "improve"
            elif mean_d >= MDE and lo > 0:
                verdict = "regress"
            else:
                verdict = "flat"
            rows.append(
                {
                    "phase": phase,
                    "reference": ref_arm,
                    "candidate": arm,
                    "n_pairs": len(d),
                    "delta_mean": mean_d,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "win_rate_lower_nmse": float((d < 0).mean()),
                    "verdict": verdict,
                }
            )
    # extra within-track pairs
    for ref_arm, cand_arm in REFERENCE_EXTRA:
        for phase in cell.phase.unique():
            sub = cell[cell.phase == phase]
            ref = sub[sub.arm == ref_arm].set_index("pkey")["nmse"]
            g = sub[sub.arm == cand_arm].set_index("pkey")["nmse"]
            if ref.empty or g.empty:
                continue
            common = g.index.intersection(ref.index)
            d = (g.loc[common] - ref.loc[common]).to_numpy(dtype=float)
            d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            lo, hi = boot_ci(d)
            mean_d = float(np.mean(d))
            verdict = "improve" if (mean_d <= -MDE and hi < 0) else ("regress" if (mean_d >= MDE and lo > 0) else "flat")
            rows.append(
                {
                    "phase": phase, "reference": ref_arm, "candidate": cand_arm,
                    "n_pairs": len(d), "delta_mean": mean_d, "ci_lo": lo, "ci_hi": hi,
                    "win_rate_lower_nmse": float((d < 0).mean()), "verdict": verdict,
                }
            )

    pc = pd.DataFrame(rows).sort_values(["phase", "candidate"]) if rows else pd.DataFrame()
    pc.to_csv(OUT / "paired_vs_reference.csv", index=False)
    print(f"paired rows: {len(pc)}")
    if len(pc):
        print(pc.to_string(index=False))

    # ---------------- tree ----------------
    lines = []
    lines.append("# 进度树 / Progress tree")
    lines.append("")
    lines.append("> 证据源: `metacentrum_runs/`（本地镜像）。判定: paired Δmacro-NMSE vs 参照，"
                 "Δ≥0.02 且 bootstrap 95% CI 排除 0。表见 `reports/progress/`。")
    lines.append("")
    lines.append("```")
    lines.append("C0  以 Mark-CC 为参照，稳定提升 HC-CE 表现 / beat Mark-CC reference stably")
    for phase in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "ER", "CE_NEW", "EXDBN", "REPLAY"]:
        sub = cell[cell.phase == phase]
        if sub.empty:
            lines.append(f"├─ {PHASE_TITLE.get(phase, phase)}   ○ open (无 metacentrum 证据)")
            continue
        arms = sorted(sub.arm.unique())
        ref = REFERENCE.get(phase, "-")
        lines.append(f"├─ {PHASE_TITLE.get(phase, phase)}")
        lines.append(f"│    ref={ref}  arms={len(arms)}  cells={len(sub)}")
        psub = pc[(pc.phase == phase)] if len(pc) else pd.DataFrame()
        for arm in arms:
            mark = "·"
            extra = ""
            if len(psub):
                r = psub[psub.candidate == arm]
                if len(r):
                    r = r.iloc[0]
                    sym = {"improve": "√ 已证(改善)", "regress": "√ 已证(回退)", "flat": "⚠ 未达阈值"}[r["verdict"]]
                    extra = f"   [{sym}  Δ={r['delta_mean']:+.4f} CI=({r['ci_lo']:+.4f},{r['ci_hi']:+.4f}) n={int(r['n_pairs'])}]"
            lines.append(f"│    {mark} {arm}{extra}")
    lines.append("```")
    (OUT / "progress_tree.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote: {OUT/'progress_tree.md'}")
    print(f"wrote: {OUT/'paired_vs_reference.csv'}")


if __name__ == "__main__":
    main()
