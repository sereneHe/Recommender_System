#!/usr/bin/env python3
"""Render the current strict evidence tree from evidence_nodes.csv.

``progress_tree.md`` is the dashboard source.  A review copy is also kept so
older links remain useful.  Both are generated from the same node registry.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "progress" / "progress_tree.md"
REVIEW_OUT = REPORTS / "progress" / "evidence_tree_review.md"
nodes = pd.read_csv(REPORTS / "evidence_nodes.csv", keep_default_na=False)

IMPL = {"implemented": "实现✓", "registered": "注册✓", "not_implemented": "实现✗",
        "hypothesis": "假设", "partial": "实现◐", "-": "—"}
SRC = {"code": "代码", "evidence": "实验", "literature": "文献", "-": "-"}
by_id = {r.node_id: r for r in nodes.itertuples()}


def children(nid):
    return [r for r in nodes.itertuples() if r.parent == nid]


def tag(r):
    cross = " ×crossed" if str(r.crossed_factor) == "True" else ""
    return f"[{IMPL.get(r.implementation, r.implementation)} · 证据{r.evidence} · {r.claim}{cross}]"


def runs_str(r):
    return f"runs u{r.unique_runs}/v{r.valid_runs} · pairs {r.paired_units}"


lines = ["# Evidence Tree（搜索空间 v3；严格配对）", ""]
lines.append("> 六个实验轴：**A1 优化器与后端 / A2 约束嵌入 / A3 条件独立统计量 / A4 图估计与先验 / A5 模型容量与预算 / A6 数据、时间与鲁棒损失**。")
lines.append("> S0 是注册表 + fallback，不宣称科学最优性。状态分为：实现、证据、scope-specific 主张。")
lines.append("> 证据来自 `metacentrum_runs/`（严格配对，仅 valid 指标）；`mark_cc_100` 缺失时锚点用 `mark_100` 兜底。")
lines.append("> 数据族和变量类型合同见 `reports/progress/evidence_tree_data_contract.md`；B0 负责 canonical 基线，A2 不重复运行 W-only/CE-only/W+CE。")
lines.append("> 脚本与节点逐项对照见 `reports/progress/evidence_tree_script_audit.md`。")
lines.append("")
lines.append("```text")

seen = set()


def emit(nid, prefix="", is_last=True, is_root=True):
    r = by_id.get(nid)
    if r is None:
        return
    seen.add(nid)
    head = f"{r.node_id}  {r.zh}  {tag(r)}  {runs_str(r)}  ({SRC.get(r.source, r.source)})"
    if is_root:
        lines.append(head)
    else:
        lines.append(f"{prefix}{'└─ ' if is_last else '├─ '}{head}")
    lit_prefix = "" if is_root else prefix + ("   " if is_last else "│  ")
    if r.literature and str(r.literature) not in ("nan", ""):
        lines.append(f"{lit_prefix}   📚 {r.literature}")
    if getattr(r, "cohort_claims", "") and str(r.cohort_claims) not in ("nan", ""):
        lines.append(f"{lit_prefix}   ◇ cohort: {r.cohort_claims}")
    if getattr(r, "metrics", "") and str(r.metrics) not in ("nan", ""):
        lines.append(f"{lit_prefix}   ▤ {r.metrics}")
    kids = children(nid)
    child_prefix = "" if is_root else prefix + ("   " if is_last else "│  ")
    for i, c in enumerate(kids):
        emit(c.node_id, child_prefix, i == len(kids) - 1, False)


roots = [r for r in nodes.itertuples() if not r.parent or str(r.parent) in ("nan", "")]
for i, r in enumerate(roots):
    emit(r.node_id, "", i == len(roots) - 1, True)
for r in nodes.itertuples():
    if r.node_id not in seen:
        emit(r.node_id, "", True, True)

lines.append("```")
lines.append("")
lines.append("## 状态总表")
lines.append("")
lines.append("| node | axis | source | crossed | 实现 | 证据 | 主张 | unique | valid | pairs |")
lines.append("|---|---|---|---|---|---|---|---|---:|---:|---:|")
for r in nodes.itertuples():
    lines.append(f"| {r.node_id} | {r.axis} | {SRC.get(r.source,r.source)} | {r.crossed_factor} | {IMPL.get(r.implementation,r.implementation)} | {r.evidence} | {r.claim} | {r.unique_runs} | {r.valid_runs} | {r.paired_units} |")

text = "\n".join(lines)
OUT.write_text(text, encoding="utf-8")
REVIEW_OUT.write_text(text, encoding="utf-8")
print(f"wrote {OUT}")
print(f"wrote {REVIEW_OUT}")
