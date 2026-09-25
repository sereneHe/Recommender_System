"""Read-only optimization diagnostics for ExDBN and Clique/Lazy-All.

The collector never adds constraints or changes solver decisions.  It records
root-relaxation summaries, MIPSOL structure classes, periodic MIP progress and
a final JSON summary so that H1--H4 can be tested from paired runs.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import time
from collections import Counter

import networkx as nx
import numpy as np
from gurobipy import GRB


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OptimizationDiagnostics:
    def __init__(self, output_dir, method, d, max_clique_size, big_m,
                 progress_interval=10.0, metadata=None):
        self.output_dir = os.path.abspath(output_dir)
        self.method = method
        self.d = int(d)
        self.max_clique_size = int(max_clique_size)
        self.big_m = float(big_m)
        self.progress_interval = float(progress_interval)
        self.metadata = dict(metadata or {})
        self.root_rows = []
        self.mipsol_rows = []
        self.progress_rows = []
        self.class_counts = Counter()
        self.accepted_mipsol_count = 0
        self.actual_mipsol_rejected_count = 0
        self.actual_mipsol_accepted_count = 0
        self.accepted_incumbent_improvement_count = 0
        self.best_accepted_objective = float("inf")
        self.callback_costs = Counter()
        self.cut_attempt_counts = Counter()
        self.separation_attempt_counts = Counter()
        self.callbacks_with_cuts = Counter()
        self.cut_unique_keys = {"cycle": set(), "clique": set()}
        self._last_progress_time = -float("inf")
        self._written = False

    def record_root(self, model, edge_values, weight_values):
        keys = list(model._edges_vars.keys())
        x = np.asarray([float(edge_values[key]) for key in keys])
        w = np.asarray([abs(float(weight_values[key])) for key in keys])
        denominator = self.big_m * np.maximum(x, 1e-12)
        ratios = w / denominator

        # For k=5,d=20 this is 38,760 inexpensive 6-set checks, performed only
        # at root.  It directly tests whether clique inequalities are active.
        y = np.zeros((self.d, self.d))
        for i, j in keys:
            y[i, j] = float(edge_values[i, j])
        y = y + y.T
        target = self.max_clique_size + 1
        rhs = target * (target - 1) / 2 - 1
        violations = []
        if target <= self.d:
            for nodes in itertools.combinations(range(self.d), target):
                score = sum(y[i, j] for i, j in itertools.combinations(nodes, 2))
                if score > rhs + 1e-9:
                    violations.append(score - rhs)

        self.root_rows.append({
            "root_round": len(self.root_rows) + 1,
            "runtime": float(model.cbGet(GRB.Callback.RUNTIME)),
            "root_bound": float(model.cbGet(GRB.Callback.MIPNODE_OBJBND)),
            "fractional_edge_count": int(np.sum((x > 1e-6) & (x < 1 - 1e-6))),
            "x_le_001_count": int(np.sum(x <= 0.01)),
            "x_001_01_count": int(np.sum((x > 0.01) & (x <= 0.1))),
            "x_01_05_count": int(np.sum((x > 0.1) & (x <= 0.5))),
            "x_05_099_count": int(np.sum((x > 0.5) & (x < 0.99))),
            "x_ge_099_count": int(np.sum(x >= 0.99)),
            "mean_x": float(np.mean(x)),
            "mean_abs_weight": float(np.mean(w)),
            "max_abs_weight": float(np.max(w)),
            "small_x_large_w_001_01": int(np.sum((x < 0.01) & (w > 0.1))),
            "small_x_large_w_005_05": int(np.sum((x < 0.05) & (w > 0.5))),
            "small_x_large_w_01_10": int(np.sum((x < 0.1) & (w > 1.0))),
            "mean_big_m_ratio": float(np.mean(ratios)),
            "max_big_m_ratio": float(np.max(ratios)),
            "violated_clique_count": len(violations),
            "max_fractional_clique_violation": max(violations, default=0.0),
        })

    def record_mipsol(self, model, edge_values):
        diagnostics_started = time.perf_counter()
        selected = [(i, j) for i, j in model._edges_vars.keys()
                    if float(edge_values[i, j]) > 0.5]
        directed = nx.DiGraph()
        directed.add_nodes_from(range(self.d))
        directed.add_edges_from(selected)
        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(self.d))
        skeleton.add_edges_from((min(i, j), max(i, j)) for i, j in selected)

        is_dag = nx.is_directed_acyclic_graph(directed)
        cliques = list(nx.find_cliques(skeleton))
        maximum_clique_size = max((len(c) for c in cliques), default=1)
        violated_cliques = sum(len(c) > self.max_clique_size for c in cliques)
        cycle_violation = not is_dag
        clique_violation = maximum_clique_size > self.max_clique_size
        if cycle_violation and clique_violation:
            category = "both"
        elif cycle_violation:
            category = "cycle_only"
        elif clique_violation:
            category = "clique_only"
        else:
            category = "neither"
        self.class_counts[category] += 1

        objective = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        bound = float(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
        gap = abs(objective - bound) / max(abs(objective), 1e-12)
        accepted_by_method = is_dag if self.method == "exdbn" else (
            is_dag and not clique_violation)
        improves_accepted_incumbent = False
        if accepted_by_method:
            self.accepted_mipsol_count += 1
            if objective < self.best_accepted_objective - 1e-9:
                self.best_accepted_objective = objective
                self.accepted_incumbent_improvement_count += 1
                improves_accepted_incumbent = True
        self.mipsol_rows.append({
            "mipsol_id": len(self.mipsol_rows) + 1,
            "runtime": float(model.cbGet(GRB.Callback.RUNTIME)),
            "objective": objective,
            "best_bound": bound,
            "gap": gap,
            "selected_edge_count": len(selected),
            "is_dag": is_dag,
            "maximum_clique_size": maximum_clique_size,
            "maximal_clique_count": len(cliques),
            "violated_clique_count": violated_cliques,
            "cycle_violation": cycle_violation,
            "clique_violation": clique_violation,
            "violation_class": category,
            "combined_structure_valid": not cycle_violation and not clique_violation,
            "accepted_by_method": accepted_by_method,
            "improves_accepted_incumbent": improves_accepted_incumbent,
        })
        self.callback_costs["diagnostics_classification_time"] += (
            time.perf_counter() - diagnostics_started)

    def record_separation(self, kind, detection_time, submission_time, cut_keys):
        """Record separation cost without changing which cuts are submitted."""
        keys = list(cut_keys)
        self.callback_costs[f"{kind}_detection_time"] += float(detection_time)
        self.callback_costs[f"{kind}_submission_time"] += float(submission_time)
        self.separation_attempt_counts[kind] += 1
        if keys:
            self.callbacks_with_cuts[kind] += 1
        self.cut_attempt_counts[kind] += len(keys)
        self.cut_unique_keys[kind].update(keys)

    def record_mipsol_outcome(self, rejected):
        """Record the solver's actual lazy-constraint decision for a candidate."""
        if rejected:
            self.actual_mipsol_rejected_count += 1
        else:
            self.actual_mipsol_accepted_count += 1

    def record_progress(self, model):
        runtime = float(model.cbGet(GRB.Callback.RUNTIME))
        if runtime - self._last_progress_time < self.progress_interval:
            return
        self._last_progress_time = runtime
        incumbent = float(model.cbGet(GRB.Callback.MIP_OBJBST))
        bound = float(model.cbGet(GRB.Callback.MIP_OBJBND))
        gap = None
        if np.isfinite(incumbent) and np.isfinite(bound):
            gap = abs(incumbent - bound) / max(abs(incumbent), 1e-12)
        iterations = float(model.cbGet(GRB.Callback.MIP_ITRCNT))
        explored_nodes = float(model.cbGet(GRB.Callback.MIP_NODCNT))
        self.progress_rows.append({
            "runtime": runtime,
            "best_incumbent": incumbent,
            "best_bound": bound,
            "gap": gap,
            "explored_nodes": explored_nodes,
            "open_nodes": float(model.cbGet(GRB.Callback.MIP_NODLFT)),
            "solution_count": int(model.cbGet(GRB.Callback.MIP_SOLCNT)),
            "nodes_per_second": (
                explored_nodes / max(runtime, 1e-12)
            ),
            "lp_iterations": iterations,
            "lp_iterations_per_second": iterations / max(runtime, 1e-12),
            "lp_iterations_per_node": iterations / max(explored_nodes, 1.0),
        })

    def write(self, final_summary=None):
        if self._written:
            return
        self._written = True
        os.makedirs(self.output_dir, exist_ok=True)
        prefix = os.path.join(self.output_dir, self.method)
        _write_csv(prefix + "_root_diagnostics.csv", self.root_rows)
        _write_csv(prefix + "_mipsol_diagnostics.csv", self.mipsol_rows)
        _write_csv(prefix + "_progress_diagnostics.csv", self.progress_rows)

        dense_dags = [r for r in self.mipsol_rows
                      if r["is_dag"] and r["clique_violation"]]
        mipsol_count = len(self.mipsol_rows)
        rejected_count = self.actual_mipsol_rejected_count
        violated_count = sum(self.class_counts[name] for name in
                             ("cycle_only", "clique_only", "both"))
        separation_attempts = sum(self.separation_attempt_counts.values())
        callbacks_with_cuts = sum(self.callbacks_with_cuts.values())
        callback_time = float(sum(self.callback_costs.values()))
        unique_cut_count = sum(len(keys) for keys in self.cut_unique_keys.values())
        final = dict(final_summary or {})
        runtime = final.get("runtime")
        nodes = final.get("node_count")
        iterations = final.get("simplex_iteration_count")
        final_bound = final.get("best_bound")
        initial_bound = None
        if self.root_rows:
            initial_bound = self.root_rows[0].get("root_bound")
        elif self.progress_rows:
            initial_bound = self.progress_rows[0].get("best_bound")
        bound_gain = None
        if initial_bound is not None and final_bound is not None:
            bound_gain = float(final_bound) - float(initial_bound)
        derived_metrics = {
            "mipsol_rejection_rate": (
                rejected_count / mipsol_count if mipsol_count else None),
            "violation_overlap_rate": (
                self.class_counts["both"] / violated_count if violated_count else None),
            "cut_hit_rate": (
                callbacks_with_cuts / separation_attempts
                if separation_attempts else None),
            "initial_bound": initial_bound,
            "bound_gain": bound_gain,
            "dual_gain_per_cut": (
                bound_gain / unique_cut_count
                if bound_gain is not None and unique_cut_count else None),
            # Aggregate whole-run efficiency. Runtime includes separation,
            # reoptimization and search, matching the experiment definition.
            "dual_gain_per_second": (
                bound_gain / float(runtime)
                if bound_gain is not None and runtime else None),
            "bound_gain_per_1000_nodes": (
                bound_gain / (float(nodes) / 1000.0)
                if bound_gain is not None and nodes else None),
            "lp_iterations_per_node": (
                float(iterations) / float(nodes) if iterations is not None and nodes else None),
            "lp_iterations_per_second": (
                float(iterations) / float(runtime)
                if iterations is not None and runtime else None),
            "callback_time": callback_time,
            "callback_time_over_total_runtime": (
                callback_time / float(runtime) if runtime else None),
        }
        summary = {
            "method": self.method,
            "experiment_metadata": self.metadata,
            "d": self.d,
            "max_clique_size": self.max_clique_size,
            "big_m": self.big_m,
            "mipsol_count": len(self.mipsol_rows),
            "violation_counts": dict(self.class_counts),
            "dense_dag_count": len(dense_dags),
            "best_dense_dag_objective": min(
                (r["objective"] for r in dense_dags), default=None),
            "first_dense_dag_time": min(
                (r["runtime"] for r in dense_dags), default=None),
            "accepted_mipsol_count": self.accepted_mipsol_count,
            "actual_mipsol_rejected_count": self.actual_mipsol_rejected_count,
            "actual_mipsol_accepted_count": self.actual_mipsol_accepted_count,
            "accepted_incumbent_improvement_count": (
                self.accepted_incumbent_improvement_count),
            "best_accepted_objective": (
                self.best_accepted_objective
                if np.isfinite(self.best_accepted_objective) else None),
            "callback_costs": dict(self.callback_costs),
            "separation_attempt_counts": dict(self.separation_attempt_counts),
            "callbacks_with_cuts": dict(self.callbacks_with_cuts),
            "cut_attempt_counts": dict(self.cut_attempt_counts),
            "cut_unique_counts": {
                kind: len(keys) for kind, keys in self.cut_unique_keys.items()
            },
            "cut_duplicate_counts": {
                kind: self.cut_attempt_counts[kind] - len(keys)
                for kind, keys in self.cut_unique_keys.items()
            },
            "root_records": self.root_rows,
            "derived_metrics": derived_metrics,
        }
        if final:
            summary["final"] = final
        with open(prefix + "_summary_diagnostics.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        print("[EXPERIMENT-METADATA] " + json.dumps(
            self.metadata, separators=(",", ":"), sort_keys=True), flush=True)
        print("[EXPERIMENT-METRICS] " + json.dumps(
            derived_metrics, separators=(",", ":"), sort_keys=True), flush=True)


def diagnostics_enabled(cfg):
    """Return the single YAML-controlled diagnostics switch.

    ``optimization_diagnostics`` remains a compatibility alias for existing
    command lines, but new experiments should use ``experiment_diagnostics``.
    Neither switch changes the mathematical model or submits solver cuts.
    """
    return bool(getattr(
        cfg,
        "experiment_diagnostics",
        getattr(cfg, "optimization_diagnostics", False),
    ))


def create_diagnostics(cfg, method, d, max_clique_size, big_m):
    if not diagnostics_enabled(cfg):
        return None
    return OptimizationDiagnostics(
        output_dir=str(getattr(cfg, "diagnostics_output_dir", ".")),
        method=method,
        d=d,
        max_clique_size=max_clique_size,
        big_m=big_m,
        progress_interval=float(getattr(cfg, "diagnostics_progress_interval", 10.0)),
        metadata=dict(getattr(cfg, "experiment_metadata", {}) or {}),
    )
