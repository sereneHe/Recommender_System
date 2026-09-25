import itertools
import json
import os
import time
from collections import defaultdict
import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB
from gurobipy import Model
from notears import utils
from omegaconf import DictConfig, OmegaConf

from dagsolvers.dagsolver_utils import find_minimal_dag_threshold
from dagsolvers.metrics_utils import (
    apply_threshold,
    count_accuracy as count_project_accuracy,
    find_optimal_threshold_for_shd,
)
from dagsolvers.optimization_diagnostics import create_diagnostics, diagnostics_enabled
from typing import Tuple, Dict


class CliqueMethod:
    """Strategy interface used by the shared MILP model.

    All BASE and A--G behavior is selected directly through solver options.
    """

    family_name = "clique_method"

    def __init__(self, cfg):
        self.cfg = cfg
        self.variant = str(getattr(cfg, "experiment_variant", os.environ.get("DAGSOLVER_VARIANT", "BASE"))).upper()

    @property
    def method_name(self):
        return f"{self.family_name}_{self.variant.lower()}"

    def prepare_big_m(self, X, Y, global_bound):
        bounds = {(i, j): float(global_bound)
                  for i in range(X.shape[1]) for j in range(X.shape[1])
                  if i != j}
        return float(global_bound), "scalar", bounds, {
            "mode": "scalar", "fallback_count": 0,
        }

    def requires_precrush(self):
        return False

    def initialize_model(self, model):
        pass

    def integer_clique_candidates(self, model, selected_edges, edge_values):
        return None

    def on_mip(self, model):
        pass

    def on_mipnode(self, model):
        pass

    def on_mipsol(self, model, selected_edges, edge_values, weight_values,
                  lag_edge_values, lag_weight_values, objective):
        if model._separation_order == "clique_first":
            clique_added = (_add_clique_cuts(model, selected_edges, edge_values)
                            if model._enable_clique_constraints else False)
            run_cycle = not model._delayed_second_separation or not clique_added
            cycle_added = (_add_cycle_cuts(model, selected_edges)
                           if run_cycle and model._enable_cycle_constraints
                           else False)
        else:
            cycle_added = (_add_cycle_cuts(model, selected_edges)
                           if model._enable_cycle_constraints else False)
            run_clique = not model._delayed_second_separation or not cycle_added
            clique_added = (_add_clique_cuts(model, selected_edges, edge_values)
                            if run_clique and model._enable_clique_constraints
                            else False)
        return cycle_added, clique_added

    def on_rejected(self, model, selected_edges, edge_values, weight_values,
                    lag_edge_values, lag_weight_values, objective):
        pass

    def on_accepted(self, model, selected_edges, objective):
        pass

    def finalize(self, model):
        pass

    def summary_fields(self, model):
        return {}


def _callback_log(model, event, **fields):
    """Emit one machine-readable line for Clique callback timing analysis."""
    if not getattr(model, "_print_callback_events", False):
        return
    try:
        runtime = float(model.cbGet(GRB.Callback.RUNTIME))
    except gp.GurobiError:
        runtime = float("nan")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[CALLBACK] event={event} runtime={runtime:.6f}s{suffix}", flush=True)


def _print_retained_solutions(model):
    """Print every solution retained in Gurobi's solution pool as JSON."""
    if not getattr(model, "_print_retained_solutions", False):
        return
    solution_count = int(model.SolCount)
    for solution_id in range(solution_count):
        model.Params.SolutionNumber = solution_id
        edges = []
        for i, j in model._edges_vars.keys():
            if float(model._edges_vars[i, j].Xn) > 0.5:
                edges.append({
                    "from": int(i),
                    "to": int(j),
                    "weight": float(model._edges_weights[i, j].Xn),
                })
        record = {
            "event": "retained_feasible_solution",
            "solution_id": solution_id,
            "objective": float(model.PoolObjVal),
            "directed_edge_count": len(edges),
            "edges": edges,
        }
        print(f"[SOLUTION] {json.dumps(record, separators=(',', ':'))}", flush=True)



def _maximal_cliques(edges, d, max_clique_size, time_budget=0.0):
    graph = nx.Graph()
    graph.add_nodes_from(range(d))
    graph.add_edges_from((min(i, j), max(i, j)) for i, j in edges if i != j)
    started = time.perf_counter()
    violated = []
    for clique in nx.find_cliques(graph):
        if len(clique) > max_clique_size:
            violated.append(tuple(sorted(clique)))
        if time_budget > 0 and time.perf_counter() - started >= time_budget:
            break
    return violated


def find_violated_cliques(edges, d, max_clique_size, mode):
    """Compatibility helper retained for the copied solver interface."""
    violated = _maximal_cliques(edges, d, max_clique_size)
    if not violated:
        return []
    if mode == "all_cliques":
        return violated
    if mode == "largest_clique":
        return [max(violated, key=len)]
    if mode == "first_clique":
        return [violated[0]]
    if mode == "top_k_cliques":
        return sorted(violated, key=len, reverse=True)
    assert False, f"Invalid mode{mode}"


def _score_clique_candidate(clique, edge_values):
    """Build the common candidate record for an integer clique inequality."""
    nodes = tuple(sorted(int(node) for node in clique))
    pairs = tuple(itertools.combinations(nodes, 2))
    lhs = sum(float(edge_values.get((i, j), 0.0))
              + float(edge_values.get((j, i), 0.0))
              for i, j in pairs)
    rhs = len(pairs) - 1
    return {
        "nodes": nodes,
        "pairs": pairs,
        "lhs": lhs,
        "rhs": rhs,
        "violation": lhs - rhs,
    }


def _add_clique_cuts(model, selected_edges, edge_values):
    constr_added = False
    clique_mode = getattr(model, "_clique_callback_mode", "first_clique")
    max_clique_size = getattr(model, "_max_clique_size", 3)
    print_clique_cuts = getattr(model, "_print_clique_cuts", False)
    _callback_log(model, "clique_detection_start",
                  mipsol_id=model._mipsol_count,
                  selected_edges=len(selected_edges),
                  max_clique_size=max_clique_size)
    detection_started = time.perf_counter()
    candidates = model._method.integer_clique_candidates(
        model, selected_edges, edge_values)
    if candidates is None:
        cliques = find_violated_cliques(selected_edges, model._d, max_clique_size, clique_mode)
        if clique_mode == "top_k_cliques":
            cliques = cliques[:model._clique_top_k]
        candidates = [_score_clique_candidate(clique, edge_values) for clique in cliques]
    if model._max_clique_cuts_per_callback > 0:
        candidates = candidates[:model._max_clique_cuts_per_callback]
    detection_time = time.perf_counter() - detection_started
    model._clique_detection_seconds += detection_time
    _callback_log(model, "clique_constraints_filtered",
                  mipsol_id=model._mipsol_count,
                  violated=len(candidates),
                  detection_seconds=f"{detection_time:.9f}")
    if candidates and print_clique_cuts:
        rt = model.cbGet(GRB.Callback.RUNTIME)
        print(
            f"[CLIQUE] runtime={rt:.2f}s mode={clique_mode} "
            f"max_k={max_clique_size} violated={len(candidates)}"
        )
    cut_keys = []
    submission_started = time.perf_counter()
    for candidate in candidates:
        clique = candidate["nodes"]
        key = tuple(clique)
        if model._deduplicate_clique_cuts and key in model._added_clique_cut_keys:
            continue
        cut_keys.append(key)
        clique_pairs = candidate["pairs"]
        model._lazy_count += 1
        model._clique_lazy_count += 1
        model.cbLazy(
            gp.quicksum(model._edges_vars[i, j] + model._edges_vars[j, i] for i, j in clique_pairs)
            <= candidate["rhs"]
        )
        model._added_clique_cut_keys.add(key)
        if print_clique_cuts:
            print(
                f"[CLIQUE-CUT] nodes={clique} pairs={len(clique_pairs)} "
                f"rhs={candidate['rhs']} violation={candidate['violation']:.6g}"
            )
        constr_added = True
    submission_time = time.perf_counter() - submission_started
    model._clique_submission_seconds += submission_time
    if candidates:
        _callback_log(model, "clique_constraints_added",
                      mipsol_id=model._mipsol_count,
                      added=len(cut_keys),
                      submission_seconds=f"{submission_time:.9f}",
                      clique_lazy_total=model._clique_lazy_count)
    if model._optimization_diagnostics is not None:
        model._optimization_diagnostics.record_separation(
            "clique", detection_time, submission_time, cut_keys)
    return constr_added


def find_cycles(edges, mode):
    vertices = set(e[0] for e in edges)
    vertices.update(e[1] for e in edges)

    visited = set()
    on_stack = set()
    parent = {}
    stack = []
    shortest_cycle = None
    found_cycles = []
    number_of_cycles = 0

    for root in vertices:
        if root in visited:
            continue
        stack.append(root)
        while stack:
            v = stack[-1]
            if v not in visited:
                visited.add(v)
                on_stack.add(v)
            else:
                if v in on_stack:
                    on_stack.remove(v)
                # else:
                #     print('DEBUG')
                stack.pop()

            neighbors = [e[1] for e in edges if e[0] == v]
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
                    parent[neighbor] = v
                elif neighbor in on_stack:
                    number_of_cycles += 1
                    # Found a cycle
                    cycle = [neighbor, v] # Back edge
                    p = parent[v]
                    while p != neighbor:
                        cycle.append(p)
                        p=parent[p]
                    #print(cycle)
                    if mode == 'first_cycle':
                        return [cycle] # return the first found cycle
                    found_cycles.append(cycle)
                    if shortest_cycle is None or len(shortest_cycle) > len(cycle):
                        shortest_cycle = cycle

    #print(f'number of cycles: {number_of_cycles}')
    if mode == 'shortest_cycle':
        if shortest_cycle is not None:
            return [shortest_cycle]
        else:
            return []
    elif mode in {'all_cycles', 'top_k_cycles'}:
        return found_cycles
    elif mode == 'first_cycle':
        return []
    else:
        assert False, f'Invalid mode{mode}'


def extract_adj_matrix(edges_vals, weights_vals, d):
    W = np.zeros((d,d))
    for v1 in range(d):
        for v2 in range(d):
            if v1 != v2:
                if edges_vals[v1, v2] > 0.5:
                    W[v1, v2] = weights_vals[(v1, v2)]
    return W

def extract_adj_matrix_no_vars(weights_vals, d):
    A = np.zeros((d,d))
    for v1 in range(d):
        for v2 in range(d):
            A[v1, v2] = weights_vals[(v1, v2)]
    return A

def _cycle_edges_and_key(cycle):
    edges_of_cycle = [(cycle[i + 1], cycle[i])
                      for i in range(len(cycle) - 1)]
    edges_of_cycle.append((cycle[0], cycle[-1]))
    # Sorting directed edges canonicalizes cyclic rotations while preserving
    # direction, so a reversed directed cycle remains a different key.
    return edges_of_cycle, tuple(sorted(edges_of_cycle))


def _add_cycle_cuts(model, selected_edges, cycles_override=None):
    constr_added = False
    _callback_log(model, "cycle_detection_start",
                  mipsol_id=model._mipsol_count,
                  selected_edges=len(selected_edges))
    detection_started = time.perf_counter()
    cycles = (find_cycles(selected_edges, model._callback_mode)
              if cycles_override is None else list(cycles_override))
    if cycles_override is None and model._callback_mode == "top_k_cycles":
        cycles = sorted(cycles, key=len)[:model._cycle_top_k]
    if model._max_cycle_cuts_per_callback > 0:
        cycles = cycles[:model._max_cycle_cuts_per_callback]
    detection_time = time.perf_counter() - detection_started
    model._cycle_detection_seconds += detection_time
    _callback_log(model, "cycle_constraints_filtered",
                  mipsol_id=model._mipsol_count,
                  violated=len(cycles),
                  detection_seconds=f"{detection_time:.9f}")
    cut_keys = []
    submission_started = time.perf_counter()
    for cycle in cycles:
        edges_of_cycle, key = _cycle_edges_and_key(cycle)
        if model._deduplicate_cycle_cuts and key in model._added_cycle_cut_keys:
            continue
        cut_keys.append(key)
        model._lazy_count += 1
        model._cycle_lazy_count += 1
        model.cbLazy(gp.quicksum(model._edges_vars[i, j] for i, j in edges_of_cycle)
                     <= len(edges_of_cycle)-1)
        model._added_cycle_cut_keys.add(key)
        constr_added = True
    submission_time = time.perf_counter() - submission_started
    model._cycle_submission_seconds += submission_time
    if cycles:
        _callback_log(model, "cycle_constraints_added",
                      mipsol_id=model._mipsol_count,
                      added=len(cycles),
                      submission_seconds=f"{submission_time:.9f}",
                      cycle_lazy_total=model._cycle_lazy_count)
    if model._optimization_diagnostics is not None:
        model._optimization_diagnostics.record_separation(
            "cycle", detection_time, submission_time, cut_keys)
    return constr_added


def method_callback(model, where):
    """Common callback lifecycle; A--G decisions are delegated to a method."""
    if where == GRB.Callback.MESSAGE:
        return
    if where == GRB.Callback.MIP:
        if model._optimization_diagnostics is not None:
            model._optimization_diagnostics.record_progress(model)
        model._method.on_mip(model)
        return
    if where == GRB.Callback.MIPNODE:
        if (model._optimization_diagnostics is not None
                and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
                and model.cbGet(GRB.Callback.MIPNODE_NODCNT) < 0.5):
            edge_vals = model.cbGetNodeRel(model._edges_vars)
            weight_vals = model.cbGetNodeRel(model._edges_weights)
            model._optimization_diagnostics.record_root(
                model, edge_vals, weight_vals)
        model._method.on_mipnode(model)
        return
    if where != GRB.Callback.MIPSOL:
        return

    model._mipsol_count += 1
    edge_values = model.cbGetSolution(model._edges_vars)
    weight_values = model.cbGetSolution(model._edges_weights)
    lag_edge_values = [model.cbGetSolution(values)
                       for values in model._lag_edges_vars]
    lag_weight_values = [model.cbGetSolution(values)
                         for values in model._lag_edges_weights]
    objective = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
    _callback_log(model, "mipsol_candidate", mipsol_id=model._mipsol_count,
                  objective=f"{objective:.12g}")
    if model._optimization_diagnostics is not None:
        model._optimization_diagnostics.record_mipsol(model, edge_values)
    selected_edges = gp.tuplelist(
        (i, j) for i, j in model._edges_vars.keys()
        if edge_values[i, j] > 0.5)

    cycle_added, clique_added = model._method.on_mipsol(
        model, selected_edges, edge_values, weight_values,
        lag_edge_values, lag_weight_values, objective)
    rejected = bool(cycle_added or clique_added)
    if model._optimization_diagnostics is not None:
        model._optimization_diagnostics.record_mipsol_outcome(rejected)

    if rejected:
        model._method.on_rejected(
            model, selected_edges, edge_values, weight_values,
            lag_edge_values, lag_weight_values, objective)
        model._mipsol_rejected_count += 1
        runtime = float(model.cbGet(GRB.Callback.RUNTIME))
        if model._first_constraint_runtime is None:
            model._first_constraint_runtime = runtime
        model._last_constraint_runtime = runtime
        _callback_log(
            model, "mipsol_rejected", mipsol_id=model._mipsol_count,
            objective=f"{objective:.12g}", cycle_added=int(cycle_added),
            clique_added=int(clique_added),
            rejected_total=model._mipsol_rejected_count)
        _callback_log(model, "solver_resume_after_constraints",
                      mipsol_id=model._mipsol_count,
                      lazy_total=model._lazy_count)
        return

    model._mipsol_accepted_count += 1
    accepted_runtime = float(model.cbGet(GRB.Callback.RUNTIME))
    if model._first_valid_incumbent_time is None:
        model._first_valid_incumbent_time = accepted_runtime
    model._method.on_accepted(model, selected_edges, objective)
    if objective < model._best_accepted_objective - 1e-9:
        model._best_accepted_objective = objective
        W_candidate = extract_adj_matrix(edge_values, weight_values, model._d)
        W_eval = W_candidate.copy()
        W_eval[np.abs(W_eval) < model._w_threshold] = 0
        accuracy = (count_project_accuracy(
            np.asarray(model._B_ref).astype(int),
            (W_eval != 0).astype(int), [], [], test_dag=True)
            if model._B_ref is not None else {})
        model._best_accepted_snapshot = {
            "objective": objective,
            "edges": list(selected_edges),
            "weights": {str(key): float(weight_values[key])
                        for key in model._edges_weights.keys()},
            "nnz": int(np.count_nonzero(W_eval)),
            "shd": accuracy.get("shd"),
            "f1": accuracy.get("f1score"),
        }
        model._accepted_incumbent_trajectory.append({
            "runtime": accepted_runtime, "objective": objective,
            "nnz": int(np.count_nonzero(W_eval)),
            "shd": accuracy.get("shd"), "f1": accuracy.get("f1score"),
        })
    _callback_log(model, "mipsol_accepted", mipsol_id=model._mipsol_count,
                  objective=f"{objective:.12g}",
                  accepted_total=model._mipsol_accepted_count)

    rt = model.cbGet(GRB.Callback.RUNTIME)
    if model._B_ref is not None and rt - model._last_time_stats > 60:
        model._last_time_stats = rt
        W_sol = extract_adj_matrix(edge_values, weight_values, model._d)
        dag_t, W_sol = find_minimal_dag_threshold(W_sol)
        default_threshold = 0.3
        W_t = apply_threshold(W_sol, default_threshold)
        shd = utils.count_accuracy(model._B_ref, W_t != 0)['shd']
        best_t, best_shd = find_optimal_threshold_for_shd(
            model._B_ref, W_sol, [], [], np.zeros_like(model._B_ref),
            np.zeros_like(model._B_ref))
        print(f't{default_threshold}_SHD: {shd} BEST_SHD: {best_shd} '
              f'BEST_t: {best_t} OBJ: {objective} DAG_t: {dag_t}')
        model._stats.append(
            (round(rt), shd, best_shd, best_t, objective, dag_t))


def _write_method_summary(model, cfg):
    if getattr(model, "_summary_written", False):
        return
    model._summary_written = True
    output_dir = str(getattr(cfg, "diagnostics_output_dir", "."))
    os.makedirs(output_dir, exist_ok=True)
    runtime = float(model.Runtime)
    nodes = float(model.NodeCount)
    has_solution = int(model.SolCount) > 0
    callback_time = (model._cycle_detection_seconds
                     + model._cycle_submission_seconds
                     + model._clique_detection_seconds
                     + model._clique_submission_seconds
                     + float(getattr(model, "_fractional_separation_seconds", 0.0))
                     + float(getattr(model, "_delayed_classification_seconds", 0.0))
                     + float(getattr(model, "_repair_metrics", {}).get("time", 0.0)))
    summary = {
        "method": model._method_name,
        "method_variant": model._method_variant,
        "effective_variant_config": OmegaConf.to_container(cfg, resolve=True),
        "status": int(model.Status), "runtime": runtime,
        "objective": float(model.ObjVal) if has_solution else None,
        "best_bound": float(model.ObjBound),
        "gap": float(model.MIPGap) if has_solution else None,
        "node_count": nodes,
        "mipsol_total": model._mipsol_count,
        "mipsol_rejected": model._mipsol_rejected_count,
        "mipsol_accepted": model._mipsol_accepted_count,
        "cycle_cuts": model._cycle_lazy_count,
        "clique_cuts": model._clique_lazy_count,
        "clique_user_cuts": model._clique_user_count,
        "unique_cycle_cuts": len(model._added_cycle_cut_keys),
        "unique_clique_lazy_cuts": len(model._added_clique_cut_keys),
        "callback_time": callback_time,
        "callback_time_over_total_runtime": callback_time / max(runtime, 1e-12),
        "nodes_per_second": nodes / max(runtime, 1e-12),
        "lp_iterations": float(model.IterCount),
        "lp_iterations_per_node": float(model.IterCount) / max(nodes, 1.0),
        "first_valid_incumbent_time": model._first_valid_incumbent_time,
        "accepted_incumbent_trajectory": model._accepted_incumbent_trajectory,
        "best_accepted_snapshot": model._best_accepted_snapshot,
        "big_m_mode": model._big_m_mode,
        "effective_weights_bound": model._global_big_m,
        "edge_bound_summary": model._edge_bound_summary,
        "method_details": model._method.summary_fields(model),
    }
    path = os.path.join(output_dir, f"{model._method_name}_summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[METHOD-SUMMARY] " + json.dumps({
        "path": path, "method": model._method_name,
        "runtime": runtime, "gap": summary["gap"],
    }, separators=(",", ":")), flush=True)


def construct_matrix_vars(m: Model, d: int, matrix_name: str, constraints_mode, weights_bound, tabu_edges, diagonal=False) -> Tuple[Dict, Dict]:
    edges_vars = {}
    edges_weights = {}
    for v1 in range(d):
        for v2 in range(d):
            if diagonal or v1 != v2:
                if constraints_mode != 'no-vars':
                    edges_vars[v1,v2] = m.addVar(vtype=GRB.BINARY, name=f'{matrix_name}_{v1}->{v2}')
                edges_weights[v1,v2] = m.addVar(lb = float('-inf'),vtype=GRB.CONTINUOUS, name=f'{matrix_name}_weight{v1}->{v2}')

                if constraints_mode == 'no-weights':
                    m.addConstr(edges_weights[v1,v2] == edges_vars[v1,v2])
                elif constraints_mode == 'no-vars':
                    bound = (weights_bound.get((v1, v2), max(weights_bound.values()))
                             if isinstance(weights_bound, dict) else weights_bound)
                    m.addConstr(edges_weights[v1,v2] <= bound)
                    m.addConstr(-edges_weights[v1,v2] <= bound)
                else:
                    bound = (weights_bound.get((v1, v2), max(weights_bound.values()))
                             if isinstance(weights_bound, dict) else weights_bound)
                    m.addConstr(edges_weights[v1,v2] <= bound * edges_vars[v1,v2])
                    m.addConstr(-edges_weights[v1,v2] <= bound * edges_vars[v1,v2])
    if tabu_edges is not None:
        for (v1,v2) in tabu_edges:
            m.addConstr(edges_vars[v1,v2] == 0)
            m.addConstr(edges_weights[v1,v2] == 0)
    return edges_vars, edges_weights



class DelayedSeparationMethod(CliqueMethod):
    family_name = "delayed_separation"

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.variant != "A":
            raise ValueError(f"delayed_separation expects A, got {self.variant!r}")

    def initialize_model(self, model):
        model._separation_order = "cycle_first"
        model._delayed_second_separation = True
        model._callback_mode = "first_cycle"
        model._clique_callback_mode = "first_clique"
        model._max_cycle_cuts_per_callback = 1
        model._max_clique_cuts_per_callback = 1
        model._deduplicate_cycle_cuts = True
        model._deduplicate_clique_cuts = True
        model._skeleton_hits = defaultdict(int)
        model._dense_region_hits = defaultdict(int)
        model._delayed_skeleton_cycle_keys = defaultdict(set)
        model._delayed_violation_counts = defaultdict(int)
        model._delayed_classification_seconds = 0.0
        model._delayed_first_cycle_cuts = 0
        model._delayed_recurrence_escalations = 0
        model._delayed_clique_only_or_dag_dense_cuts = 0
        model._max_skeleton_hits = 0
        model._dense_escalation_hits = max(
            1, int(getattr(self.cfg, "dense_escalation_hits", 2)))
        model._clique_separation_time_budget = max(
            0.0, float(getattr(self.cfg, "clique_separation_time_budget", 0.1)))
        model._delayed_escalation_reasons = defaultdict(int)

    @staticmethod
    def _record_dense_history(model, selected_edges):
        signature = tuple(sorted({
            (min(i, j), max(i, j)) for i, j in selected_edges if i != j
        }))
        model._skeleton_hits[signature] += 1
        hits = model._skeleton_hits[signature]
        model._max_skeleton_hits = max(model._max_skeleton_hits, hits)
        graph = nx.Graph()
        graph.add_nodes_from(range(model._d))
        graph.add_edges_from(signature)
        started = time.perf_counter()
        violated = []
        for clique in nx.find_cliques(graph):
            if len(clique) > model._max_clique_size:
                violated.append(tuple(sorted(clique)))
                # Record every canonical minimal (k+1) dense region. Taking
                # only one prefix would hide other recurring regions inside a
                # larger maximal clique.
                for region in itertools.combinations(
                        sorted(clique), model._max_clique_size + 1):
                    model._dense_region_hits[tuple(region)] += 1
            if (model._clique_separation_time_budget > 0
                    and time.perf_counter() - started
                    >= model._clique_separation_time_budget):
                break
        return signature, hits, violated

    def on_mipsol(self, model, selected_edges, edge_values, weight_values,
                  lag_edge_values, lag_weight_values, objective):
        started = time.perf_counter()
        signature, skeleton_hits, violated_cliques = self._record_dense_history(
            model, selected_edges)
        cycles = find_cycles(selected_edges, "all_cycles")
        records = [(*_cycle_edges_and_key(cycle), cycle)
                   for cycle in cycles]
        seen = model._delayed_skeleton_cycle_keys[signature]
        seen_before = len(seen)
        unseen = [record for record in records if record[1] not in seen]
        cycle_violation = bool(cycles)
        clique_violation = bool(violated_cliques)
        if cycle_violation and clique_violation:
            violation_class = "both"
        elif cycle_violation:
            violation_class = "cycle_only"
        elif clique_violation:
            violation_class = "clique_only"
        else:
            violation_class = "neither"
        model._delayed_violation_counts[violation_class] += 1
        different_cycle_recurrence = bool(seen and unseen)
        dense_hits = max((model._dense_region_hits[tuple(region)]
            for clique in violated_cliques
            for region in itertools.combinations(
                sorted(clique), model._max_clique_size + 1)), default=0)
        model._delayed_classification_seconds += time.perf_counter() - started

        cycle_added = clique_added = False
        action = "accept"
        if cycle_violation:
            recurrence_ready = (
                skeleton_hits >= model._dense_escalation_hits
                or dense_hits >= model._dense_escalation_hits)
            if clique_violation and recurrence_ready:
                clique_added = _add_clique_cuts(
                    model, selected_edges, edge_values)
                if clique_added:
                    model._delayed_recurrence_escalations += 1
                    reason = ("different_cycle" if different_cycle_recurrence
                              else "dense_region_hits")
                    model._delayed_escalation_reasons[reason] += 1
                    action = "repeat_skeleton_add_clique"
            else:
                chosen = unseen[0] if unseen else records[0]
                cycle_added = _add_cycle_cuts(
                    model, selected_edges, cycles_override=[chosen[2]])
                if cycle_added:
                    seen.add(chosen[1])
                    model._delayed_first_cycle_cuts += 1
                    action = "first_add_cycle_store_skeleton"
        elif clique_violation:
            clique_added = _add_clique_cuts(
                model, selected_edges, edge_values)
            if clique_added:
                model._delayed_clique_only_or_dag_dense_cuts += 1
                action = "clique_only_or_dag_dense_add_clique"

        _callback_log(
            model, "method_a_delayed_separation",
            mipsol_id=model._mipsol_count, violation_class=violation_class,
            skeleton_hits=skeleton_hits, previously_seen_cycles=seen_before,
            dense_region_hits=dense_hits,
            different_cycle_recurrence=int(different_cycle_recurrence),
            action=action)
        return cycle_added, clique_added

    def summary_fields(self, model):
        return {
            "violation_counts": dict(model._delayed_violation_counts),
            "classification_seconds": model._delayed_classification_seconds,
            "first_cycle_cuts": model._delayed_first_cycle_cuts,
            "recurrent_skeleton_clique_escalations": (
                model._delayed_recurrence_escalations),
            "clique_only_or_dag_dense_cuts": (
                model._delayed_clique_only_or_dag_dense_cuts),
            "unique_skeletons": len(model._skeleton_hits),
            "max_skeleton_hits": model._max_skeleton_hits,
            "dense_escalation_threshold": model._dense_escalation_hits,
            "unique_dense_regions": len(model._dense_region_hits),
            "dense_region_hit_histogram": dict(sorted(
                __import__("collections").Counter(
                    model._dense_region_hits.values()).items())),
            "escalation_reasons": dict(model._delayed_escalation_reasons),
        }

def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _integer_clique_candidate(nodes, edge_values):
    nodes = tuple(sorted(nodes))
    pairs = list(itertools.combinations(nodes, 2))
    lhs = sum(float(edge_values[i, j]) + float(edge_values[j, i])
              for i, j in pairs)
    rhs = len(pairs) - 1
    violation = float(lhs - rhs)
    return {
        "nodes": nodes, "pairs": pairs, "rhs": rhs,
        "violation": violation,
        "efficiency": violation / max(len(pairs), 1),
    }


class CliqueSeparationOptionsMethod(CliqueMethod):
    family_name = "clique_separation_options"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.formulation = str(getattr(cfg, "clique_cut_formulation", os.environ.get(
            "CLIQUE_SEPERATION_FORMULATION", "k_plus_1"))).lower()
        self.selection = str(getattr(cfg, "clique_cut_selection", os.environ.get(
            "CLIQUE_SEPERATION_SELECTION", "first"))).lower()
        valid_formulations = {"maximal", "k_plus_1"}
        valid_selections = {
            "first", "all", "most_violated", "history",
            "top_k", "top_k_diverse",
        }
        if self.formulation not in valid_formulations:
            raise ValueError(
                "CLIQUE_SEPERATION_FORMULATION must be maximal or k_plus_1; got "
                f"{self.formulation!r}")
        if self.selection not in valid_selections:
            raise ValueError(
                "invalid CLIQUE_SEPERATION_SELECTION: " f"{self.selection!r}")
        self.variant = f"{self.formulation}_{self.selection}".upper()

    def initialize_model(self, model):
        model._clique_option_candidate_hits = defaultdict(int)
        model._clique_option_selected_hits = defaultdict(int)
        model._clique_option_similarity_rejections = 0
        model._controlled_max_jaccard = float(getattr(
            self.cfg, "max_cut_jaccard_similarity", os.environ.get(
                "CLIQUE_SEPERATION_MAX_JACCARD", 0.8)))
        model._clique_top_k = int(getattr(
            self.cfg, "clique_cut_top_k", os.environ.get(
                "CLIQUE_SEPERATION_TOP_K", model._clique_top_k)))
        model._deduplicate_clique_cuts = bool(getattr(
            self.cfg, "deduplicate_clique_cuts", _env_bool(
                "CLIQUE_SEPERATION_DEDUP", model._deduplicate_clique_cuts)))

    @staticmethod
    def _jaccard(left, right):
        left, right = set(left), set(right)
        return len(left & right) / max(len(left | right), 1)

    def integer_clique_candidates(self, model, selected_edges, edge_values):
        maximals = _maximal_cliques(
            selected_edges, model._d, model._max_clique_size)
        if self.formulation == "maximal":
            node_sets = sorted(maximals, key=lambda nodes: (-len(nodes), nodes))
        else:
            k = model._max_clique_size
            node_sets = sorted({
                tuple(nodes) for maximal in maximals
                for nodes in itertools.combinations(sorted(maximal), k + 1)
            })

        candidates = [_integer_clique_candidate(nodes, edge_values) for nodes in node_sets]
        for item in candidates:
            key = item["nodes"]
            model._clique_option_candidate_hits[key] += 1
            item["history_hits"] = model._clique_option_candidate_hits[key]
            item["score"] = item["efficiency"] + 1e-3 * item["history_hits"]

        ranked = sorted(candidates, key=lambda item: (
            -item["violation"], -item["efficiency"], item["nodes"]))
        if self.selection == "first":
            chosen = candidates[:1]
        elif self.selection == "all":
            chosen = candidates
        elif self.selection == "most_violated":
            chosen = ranked[:1]
        elif self.selection == "history":
            chosen = sorted(candidates, key=lambda item: (
                -item["score"], -item["violation"], item["nodes"]))[:1]
        elif self.selection == "top_k":
            chosen = ranked[:model._clique_top_k]
        else:
            history_ranked = sorted(candidates, key=lambda item: (
                -item["score"], -item["violation"], item["nodes"]))
            chosen = []
            for item in history_ranked:
                if any(self._jaccard(item["nodes"], other["nodes"])
                       > model._controlled_max_jaccard for other in chosen):
                    model._clique_option_similarity_rejections += 1
                    continue
                chosen.append(item)
                if len(chosen) >= model._clique_top_k:
                    break

        for item in chosen:
            model._clique_option_selected_hits[item["nodes"]] += 1
        return chosen

    def summary_fields(self, model):
        return {
            "clique_separation_formulation": self.formulation,
            "clique_separation_selection": self.selection,
            "clique_separation_top_k": model._clique_top_k,
            "clique_separation_deduplicate": model._deduplicate_clique_cuts,
            "max_jaccard_similarity": model._controlled_max_jaccard,
            "unique_candidates_seen": len(model._clique_option_candidate_hits),
            "unique_candidates_selected": len(model._clique_option_selected_hits),
            "similarity_rejections": model._clique_option_similarity_rejections,
        }

class RepairIncumbentMethod(CliqueMethod):
    family_name = "repair_incumbent"
    _modes = {"C1": "none", "C2": "greedy_delete",
              "C3": "local_ls", "C4": "local_ridge",
              "C5": "local_ridge", "C6": "local_ridge"}

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.variant not in self._modes:
            raise ValueError(f"repair_incumbent expects C1--C6, got {self.variant!r}")

    def initialize_model(self, model):
        model._incumbent_repair_mode = self._modes[self.variant]
        model._repair_ridge_alpha = float(
            getattr(self.cfg, "repair_ridge_alpha", 1e-3))
        model._repair_refit_solver = str(
            getattr(self.cfg, "repair_refit_solver", "auto")).lower()
        model._repair_deletion_score = str(
            getattr(self.cfg, "repair_deletion_score", "violation_loss")).lower()
        model._repair_violation_weight = float(
            getattr(self.cfg, "repair_violation_weight", 1.0))
        model._repair_loss_weight = float(
            getattr(self.cfg, "repair_loss_weight", 1.0))
        model._repair_max_retries = int(
            getattr(self.cfg, "repair_max_retries", 2))
        model._repair_metrics = {
            "attempts": 0, "submitted": 0, "matched_accepted": 0,
            "failed": 0, "edges_removed": 0,
            "columns_refitted": 0, "time": 0.0,
            "validation_failures": 0, "retry_attempts": 0,
            "cbuse_accepted": 0, "cbuse_rejected": 0,
            "cbuse_deferred": 0,
            "matched_improved_incumbent": 0,
        }
        model._repair_failure_reasons = defaultdict(int)
        model._repair_submitted_supports = set()
        model._repair_objective_degradations = []
        model._repair_violation_edge_hits = defaultdict(int)
        model._repair_best_accepted_support = set()
        model._repair_removed_records = []
        model._repair_cache_loaded_cycle = 0
        model._repair_cache_loaded_clique = 0
        model._repair_cache_saved_cycle = 0
        model._repair_cache_saved_clique = 0
        model._repair_cut_cache_mode = str(
            getattr(self.cfg, "repair_cut_cache_mode", "off")).lower()
        model._repair_cut_cache_path = self._cache_path(model)
        if self.variant == "C6" and model._repair_cut_cache_mode in {
                "read", "read_write"}:
            self._load_valid_cut_cache(model)

    def _cache_path(self, model):
        directory = str(getattr(self.cfg, "repair_cut_cache_dir", "repair_cut_cache"))
        metadata = dict(getattr(self.cfg, "experiment_metadata", {}) or {})
        seed = metadata.get("seed", "unknown")
        return os.path.join(
            directory,
            f"valid_cuts_d{model._d}_k{model._max_clique_size}_seed{seed}.json")

    @staticmethod
    def _load_valid_cut_cache(model):
        path = model._repair_cut_cache_path
        if not os.path.exists(path):
            return
        with open(path) as handle:
            payload = json.load(handle)
        if (int(payload.get("d", -1)) != model._d
                or int(payload.get("max_clique_size", -1))
                != model._max_clique_size):
            return
        for raw_cycle in payload.get("cycle_cuts", []):
            edges = tuple(sorted((int(i), int(j)) for i, j in raw_cycle))
            if not edges or edges in model._added_cycle_cut_keys:
                continue
            model.addConstr(
                sum(model._edges_vars[i, j] for i, j in edges)
                <= len(edges) - 1)
            model._added_cycle_cut_keys.add(edges)
            model._repair_cache_loaded_cycle += 1
        for raw_clique in payload.get("clique_cuts", []):
            clique = tuple(sorted(int(node) for node in raw_clique))
            if (len(clique) <= model._max_clique_size
                    or clique in model._added_clique_cut_keys):
                continue
            pairs = list(itertools.combinations(clique, 2))
            model.addConstr(sum(
                model._edges_vars[i, j] + model._edges_vars[j, i]
                for i, j in pairs) <= len(pairs) - 1)
            model._added_clique_cut_keys.add(clique)
            model._repair_cache_loaded_clique += 1

    @staticmethod
    def _save_valid_cut_cache(model):
        path = model._repair_cut_cache_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "d": model._d,
            "max_clique_size": model._max_clique_size,
            "cycle_cuts": [[list(edge) for edge in cycle]
                           for cycle in sorted(model._added_cycle_cut_keys)],
            "clique_cuts": [list(clique)
                            for clique in sorted(model._added_clique_cut_keys)],
        }
        temporary = path + f".tmp.{os.getpid()}"
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
        model._repair_cache_saved_cycle = len(payload["cycle_cuts"])
        model._repair_cache_saved_clique = len(payload["clique_cuts"])

    @staticmethod
    def _deletion_loss(model, edge, weights, support, lag_weight_values):
        """Exact data-fit SSE change if one contemporaneous edge is removed."""
        source, target = edge
        prediction = sum(
            model._X[:, parent] * float(weights[parent, target])
            for parent, child in support if child == target)
        for lag_index, lag_data in enumerate(model._Y):
            prediction = prediction + lag_data @ np.asarray([
                float(lag_weight_values[lag_index][i, target])
                for i in range(model._d)])
        residual = model._X[:, target] - prediction
        removed_contribution = model._X[:, source] * float(weights[edge])
        new_residual = residual + removed_contribution
        return float((new_residual @ new_residual - residual @ residual)
                     / len(model._X))

    def _choose_edge(self, model, choices, weights, support,
                     lag_weight_values):
        losses = {
            edge: self._deletion_loss(
                model, edge, weights, support, lag_weight_values)
            for edge in choices}
        if model._repair_deletion_score == "weight_only":
            edge = min(choices, key=lambda item: abs(float(weights[item])))
            return edge, {"loss_delta": losses[edge], "score": None}
        best_support = model._repair_best_accepted_support
        scale = max(1e-12, max(abs(value) for value in losses.values()))
        def priority(edge):
            history = model._repair_violation_edge_hits[edge]
            protected = 1 if edge in best_support else 0
            score = (model._repair_violation_weight * (1.0 + history)
                     - model._repair_loss_weight * losses[edge] / scale
                     - 2.0 * protected)
            return score, -abs(float(weights[edge])), edge
        edge = max(choices, key=priority)
        return edge, {
            "loss_delta": losses[edge],
            "violation_hits": model._repair_violation_edge_hits[edge],
            "score": priority(edge)[0],
        }

    def _repair_support(self, model, selected_edges, weights,
                        lag_weight_values):
        support, removed, decisions = set(selected_edges), [], []
        while True:
            directed = nx.DiGraph()
            directed.add_nodes_from(range(model._d))
            directed.add_edges_from(support)
            try:
                cycle = nx.find_cycle(directed, orientation="original")
            except nx.NetworkXNoCycle:
                cycle = []
            if cycle:
                choices = [(u, v) for u, v, *_ in cycle]
                edge, details = self._choose_edge(
                    model, choices, weights, support, lag_weight_values)
                support.remove(edge)
                removed.append(edge)
                decisions.append({"reason": "cycle", "edge": list(edge),
                                  **details})
                continue
            skeleton = nx.Graph()
            skeleton.add_nodes_from(range(model._d))
            skeleton.add_edges_from((min(i, j), max(i, j)) for i, j in support)
            violated = [tuple(clique) for clique in nx.find_cliques(skeleton)
                        if len(clique) > model._max_clique_size]
            if not violated:
                return support, removed, decisions
            clique = max(violated, key=len)
            choices = [(i, j) for i, j in support
                       if i in clique and j in clique]
            if not choices:
                return None, removed, decisions
            edge, details = self._choose_edge(
                model, choices, weights, support, lag_weight_values)
            support.remove(edge)
            removed.append(edge)
            decisions.append({"reason": "clique", "edge": list(edge),
                              "clique": list(clique), **details})

    @staticmethod
    def _refit(model, support, original_weights, mode, affected,
               lag_weight_values):
        repaired = {key: (float(original_weights[key]) if key in support else 0.0)
                    for key in model._edges_weights.keys()}
        if mode not in {"local_ls", "local_ridge"}:
            return repaired
        alpha = model._repair_ridge_alpha if mode == "local_ridge" else 0.0
        for target in affected:
            parents = sorted(i for i, j in support if j == target)
            for source in range(model._d):
                if source != target:
                    repaired[source, target] = 0.0
            if not parents:
                continue
            design = model._X[:, parents]
            response = model._X[:, target].copy()
            for lag_index, lag_data in enumerate(model._Y):
                response -= lag_data @ np.asarray([
                    float(lag_weight_values[lag_index][i, target])
                    for i in range(model._d)])
            gram = design.T @ design + alpha * np.eye(len(parents))
            solver = model._repair_refit_solver
            try:
                if solver == "qr" or (solver == "auto" and alpha == 0.0):
                    if alpha > 0.0:
                        augmented_x = np.vstack(
                            [design, np.sqrt(alpha) * np.eye(len(parents))])
                        augmented_y = np.concatenate(
                            [response, np.zeros(len(parents))])
                    else:
                        augmented_x, augmented_y = design, response
                    q, r = np.linalg.qr(augmented_x, mode="reduced")
                    beta = np.linalg.solve(r, q.T @ augmented_y)
                else:
                    beta = np.linalg.solve(gram, design.T @ response)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(design, response, rcond=None)[0]
            for parent, coefficient in zip(parents, beta):
                bound = model._edge_specific_bounds.get(
                    (parent, target), model._global_big_m)
                repaired[parent, target] = float(
                    np.clip(coefficient, -bound, bound))
        return repaired

    @staticmethod
    def _validate_repair(model, support, repaired, lag_edge_values,
                         lag_weight_values, tolerance=1e-7):
        """Validate every structural/linking constraint affected by repair."""
        errors = []
        graph = nx.DiGraph()
        graph.add_nodes_from(range(model._d))
        graph.add_edges_from(support)
        if not nx.is_directed_acyclic_graph(graph):
            errors.append("cycle")
        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(model._d))
        skeleton.add_edges_from((min(i, j), max(i, j)) for i, j in support)
        max_size = max((len(c) for c in nx.find_cliques(skeleton)), default=0)
        if max_size > model._max_clique_size:
            errors.append("clique")
        if any((j, i) in support for i, j in support):
            errors.append("antiparallel")
        for edge, value in repaired.items():
            bound = model._edge_specific_bounds.get(edge, model._global_big_m)
            selected = edge in support
            if abs(float(value)) > bound + tolerance:
                errors.append(f"weight_bound:{edge}")
            if not selected and abs(float(value)) > tolerance:
                errors.append(f"weight_without_edge:{edge}")
        for edge_values, weight_values in zip(
                lag_edge_values, lag_weight_values):
            for edge, value in weight_values.items():
                bound = model._global_big_m
                if abs(float(value)) > bound * float(edge_values[edge]) + tolerance:
                    errors.append(f"lag_link:{edge}")
        return {"valid": not errors, "errors": errors,
                "max_clique_size": max_size}

    def on_rejected(self, model, selected_edges, edge_values, weight_values,
                    lag_edge_values, lag_weight_values, objective):
        mode = model._incumbent_repair_mode
        if mode == "none":
            return
        started = time.perf_counter()
        model._repair_metrics["attempts"] += 1
        support, removed, decisions = self._repair_support(
            model, selected_edges, weight_values, lag_weight_values)
        if support is None or not removed:
            model._repair_metrics["failed"] += 1
            model._repair_failure_reasons["no_repairable_support"] += 1
            model._repair_metrics["time"] += time.perf_counter() - started
            return
        original_support = set(selected_edges)
        record = {
            "mipsol_id": model._mipsol_count,
            "deletion_decisions": decisions,
            "attempts": [],
        }
        cbuse_result = None
        submission_status = "not_submitted"
        repaired = None
        for retry in range(model._repair_max_retries + 1):
            affected = {j for _, j in removed}
            original_parents = {
                str(target): sorted(i for i, j in original_support
                                    if j == target)
                for target in affected}
            repaired_parents = {
                str(target): sorted(i for i, j in support if j == target)
                for target in affected}
            repaired = self._refit(
                model, support, weight_values, mode, affected,
                lag_weight_values)
            validation = self._validate_repair(
                model, support, repaired, lag_edge_values, lag_weight_values)
            attempt = {
                "retry": retry,
                "removed_edges": [list(edge) for edge in removed],
                "affected_targets": sorted(affected),
                "original_parents": original_parents,
                "repaired_parents": repaired_parents,
                "validation": validation,
            }
            record["attempts"].append(attempt)
            if not validation["valid"]:
                model._repair_metrics["validation_failures"] += 1
                for error in validation["errors"]:
                    model._repair_failure_reasons[error.split(":", 1)[0]] += 1
                cbuse_result = None
            else:
                model.cbSetSolution(
                    list(model._edges_vars.values()),
                    [1.0 if key in support else 0.0
                     for key in model._edges_vars.keys()])
                model.cbSetSolution(
                    list(model._edges_weights.values()),
                    [repaired[key] for key in model._edges_weights.keys()])
                for variables, values in zip(
                        model._lag_edges_vars, lag_edge_values):
                    model.cbSetSolution(
                        list(variables.values()),
                        [float(values[key]) for key in variables.keys()])
                for variables, values in zip(
                        model._lag_edges_weights, lag_weight_values):
                    model.cbSetSolution(
                        list(variables.values()),
                        [float(values[key]) for key in variables.keys()])
                cbuse_result = float(model.cbUseSolution())
                attempt["cbuse_result"] = cbuse_result
                # on_rejected is called from MIPSOL.  Gurobi documents that
                # cbUseSolution stores the proposed solution for later in
                # this callback context and therefore returns GRB.INFINITY;
                # that value is not a rejection.  Actual later acceptance is
                # matched by support in on_accepted().
                attempt["cbuse_immediate"] = False
                attempt["cbuse_status"] = "deferred"
                model._repair_metrics["cbuse_deferred"] += 1
                submission_status = "deferred"
                break
            if retry >= model._repair_max_retries or not support:
                break
            # Explicit fallback: remove one additional loss-aware edge, refit
            # its target column, validate again, and resubmit.
            fallback, details = self._choose_edge(
                model, list(support), weight_values, support,
                lag_weight_values)
            support.remove(fallback)
            removed.append(fallback)
            decision = {"reason": "repair_retry", "edge": list(fallback),
                        **details}
            decisions.append(decision)
            model._repair_metrics["retry_attempts"] += 1
        record["final_cbuse_result"] = cbuse_result
        record["submission_status"] = submission_status
        record["success"] = submission_status in {"accepted", "deferred"}
        model._repair_removed_records.append(record)
        if not record["success"]:
            model._repair_metrics["failed"] += 1
            model._repair_failure_reasons["submission_not_reached"] += 1
            model._repair_metrics["time"] += time.perf_counter() - started
            return
        support_key = tuple(sorted(support))
        model._repair_submitted_supports.add(support_key)
        model._repair_metrics["submitted"] += 1
        model._repair_metrics["edges_removed"] += len(removed)
        if mode != "greedy_delete":
            model._repair_metrics["columns_refitted"] += len(affected)
        residual = model._X - model._X @ np.asarray([
            [repaired.get((i, j), 0.0) for j in range(model._d)]
            for i in range(model._d)])
        for lag_index, lag_data in enumerate(model._Y):
            lag_matrix = np.asarray([
                [float(lag_weight_values[lag_index][i, j])
                 for j in range(model._d)] for i in range(model._d)])
            residual -= lag_data @ lag_matrix
        approximate_objective = float(np.sum(residual ** 2) / len(model._X))
        model._repair_objective_degradations.append(
            approximate_objective - float(objective))
        model._repair_metrics["time"] += time.perf_counter() - started

    def on_accepted(self, model, selected_edges, objective):
        model._repair_best_accepted_support = set(selected_edges)
        support_key = tuple(sorted(selected_edges))
        if support_key in model._repair_submitted_supports:
            model._repair_metrics["matched_accepted"] += 1
            if objective < model._best_accepted_objective - 1e-9:
                model._repair_metrics["matched_improved_incumbent"] += 1
            model._repair_submitted_supports.discard(support_key)

    def on_mipsol(self, model, selected_edges, edge_values, weight_values,
                  lag_edge_values, lag_weight_values, objective):
        if self.variant == "C5":
            for cycle in find_cycles(selected_edges, "all_cycles"):
                edges, _ = _cycle_edges_and_key(cycle)
                for edge in edges:
                    model._repair_violation_edge_hits[edge] += 1
            skeleton = nx.Graph()
            skeleton.add_nodes_from(range(model._d))
            skeleton.add_edges_from((min(i, j), max(i, j))
                                    for i, j in selected_edges)
            selected = set(selected_edges)
            for clique in nx.find_cliques(skeleton):
                if len(clique) <= model._max_clique_size:
                    continue
                for i, j in selected:
                    if i in clique and j in clique:
                        model._repair_violation_edge_hits[i, j] += 1
        return super().on_mipsol(
            model, selected_edges, edge_values, weight_values,
            lag_edge_values, lag_weight_values, objective)

    def finalize(self, model):
        if self.variant == "C6" and model._repair_cut_cache_mode in {
                "write", "read_write"}:
            self._save_valid_cut_cache(model)

    def summary_fields(self, model):
        submitted = model._repair_metrics["submitted"]
        signed_changes = model._repair_objective_degradations
        positive_degradations = [value for value in signed_changes if value > 0]
        return {
            "repair_config": {
                "mode": model._incumbent_repair_mode,
                "deletion_score": model._repair_deletion_score,
                "violation_weight": model._repair_violation_weight,
                "loss_weight": model._repair_loss_weight,
                "refit_solver": model._repair_refit_solver,
                "ridge_alpha": model._repair_ridge_alpha,
                "max_retries": model._repair_max_retries,
            },
            "repair_metrics": dict(model._repair_metrics),
            "repair_failure_reasons": dict(model._repair_failure_reasons),
            "repair_success_rate": (
                model._repair_metrics["matched_accepted"] / submitted
                if submitted else None),
            "repair_incumbent_improvement_rate": (
                model._repair_metrics["matched_improved_incumbent"] / submitted
                if submitted else None),
            "mean_objective_degradation": (
                float(np.mean(positive_degradations))
                if positive_degradations else 0.0),
            "mean_signed_objective_change": (
                float(np.mean(signed_changes)) if signed_changes else None),
            "repair_objective_improvement_fraction": (
                sum(value < 0 for value in signed_changes) / len(signed_changes)
                if signed_changes else None),
            "repair_records": model._repair_removed_records,
            "trajectory_violation_edge_count": len(
                model._repair_violation_edge_hits),
            "valid_cut_cache": {
                "mode": model._repair_cut_cache_mode,
                "path": model._repair_cut_cache_path,
                "loaded_cycle": model._repair_cache_loaded_cycle,
                "loaded_clique": model._repair_cache_loaded_clique,
                "saved_cycle": model._repair_cache_saved_cycle,
                "saved_clique": model._repair_cache_saved_clique,
            },
        }

def _fractional_clique_candidate(y, nodes):
    nodes = tuple(sorted(nodes))
    pairs = list(itertools.combinations(nodes, 2))
    lhs = sum(y[i, j] for i, j in pairs)
    rhs = len(pairs) - 1
    violation = float(lhs - rhs)
    return {"nodes": nodes, "pairs": pairs, "rhs": rhs,
            "violation": violation,
            "efficiency": violation / max(len(pairs), 1)}


class FractionalSeparationMethod(CliqueMethod):
    family_name = "fractional_separation"

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.variant not in {"D1", "D2", "D3", "D4", "D5"}:
            raise ValueError(f"fractional_separation expects D1--D5, got {self.variant!r}")

    def requires_precrush(self):
        return self.variant != "D1"

    def initialize_model(self, model):
        model._enable_fractional_clique_cuts = self.variant != "D1"
        model._fractional_root_only = self.variant in {"D2", "D3"}
        model._root_max_cuts_per_round = int(
            getattr(self.cfg, "root_max_cuts_per_round", 20))
        model._nonroot_max_cuts_per_callback = int(
            getattr(self.cfg, "nonroot_max_cuts_per_callback", 5))
        model._nonroot_separation_frequency = max(
            1, int(getattr(self.cfg, "nonroot_separation_frequency", 10)))
        model._fractional_violation_tolerance = float(
            getattr(self.cfg, "fractional_violation_tolerance", 1e-6))
        model._fractional_greedy_starts = int(
            getattr(self.cfg, "fractional_greedy_starts", 10))
        model._dense_history_min_hits = int(
            getattr(self.cfg, "dense_history_min_hits", 2))
        model._fractional_clique_cut_keys = set()
        model._fractional_candidates_detected = 0
        model._fractional_separation_seconds = 0.0
        model._dense_history_hits = defaultdict(int)
        model._fractional_callbacks_checked = 0
        model._fractional_callbacks_skipped = 0
        model._fractional_zero_candidate_rounds = 0
        model._fractional_disabled_after_zero_rounds = False
        model._fractional_max_zero_rounds = max(
            0, int(getattr(self.cfg, "fractional_max_zero_rounds", 3)))
        model._fractional_time_budget = max(
            0.0, float(getattr(
                self.cfg, "fractional_separation_time_budget", 0.02)))

    @staticmethod
    def _mass(edge_values, d):
        y = np.zeros((d, d), dtype=float)
        for i, j in edge_values.keys():
            y[i, j] = float(edge_values[i, j])
        return y + y.T

    @staticmethod
    def _exact(model, y, region=None, deadline=None):
        universe = tuple(range(model._d) if region is None else sorted(region))
        size = model._max_clique_size + 1
        found = []
        if len(universe) >= size:
            for nodes in itertools.combinations(universe, size):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                item = _fractional_clique_candidate(y, nodes)
                if item["violation"] > model._fractional_violation_tolerance:
                    found.append(item)
        return found

    @staticmethod
    def _greedy(model, y, region=None, deadline=None):
        universe = tuple(range(model._d) if region is None else sorted(region))
        target = model._max_clique_size + 1
        if len(universe) < target:
            return []
        degrees = {i: sum(y[i, j] for j in universe if j != i)
                   for i in universe}
        starts = sorted(universe, key=lambda i: (-degrees[i], i))[
            :model._fractional_greedy_starts]
        found = {}
        for start in starts:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            selected, remaining = [start], set(universe) - {start}
            while len(selected) < target and remaining:
                nxt = max(remaining,
                          key=lambda v: (sum(y[v, u] for u in selected), -v))
                selected.append(nxt)
                remaining.remove(nxt)
            improved = True
            while improved:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                improved = False
                current = _candidate(y, selected)
                for old in list(selected):
                    for new in sorted(set(universe) - set(selected)):
                        if (deadline is not None
                                and time.perf_counter() >= deadline):
                            break
                        trial = [v for v in selected if v != old] + [new]
                        item = _candidate(y, trial)
                        if item["violation"] > current["violation"] + 1e-12:
                            selected, improved = trial, True
                            break
                    if improved:
                        break
            item = _candidate(y, selected)
            if item["violation"] > model._fractional_violation_tolerance:
                found[item["nodes"]] = item
        return list(found.values())

    def _select(self, model, y, node_count, deadline=None):
        if self.variant == "D3":
            return self._greedy(model, y, deadline=deadline)
        if self.variant == "D4":
            return (self._exact(model, y, deadline=deadline) if node_count == 0
                    else self._greedy(model, y, deadline=deadline))
        if self.variant == "D5" and node_count != 0:
            found = {}
            for nodes, hits in model._dense_history_hits.items():
                if hits < model._dense_history_min_hits:
                    continue
                for item in self._greedy(
                        model, y, region=nodes, deadline=deadline):
                    found[item["nodes"]] = item
            return list(found.values())
        return self._exact(model, y, deadline=deadline)

    def on_mipnode(self, model):
        if self.variant == "D1":
            return
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
            return
        node_count = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
        if model._fractional_root_only and node_count != 0:
            model._fractional_callbacks_skipped += 1
            return
        if node_count and node_count % model._nonroot_separation_frequency:
            model._fractional_callbacks_skipped += 1
            return
        if model._fractional_disabled_after_zero_rounds:
            model._fractional_callbacks_skipped += 1
            return
        model._fractional_callbacks_checked += 1
        values = model.cbGetNodeRel(model._edges_vars)
        y = self._mass(values, model._d)
        started = time.perf_counter()
        deadline = (started + model._fractional_time_budget
                    if model._fractional_time_budget > 0 else None)
        candidates = sorted(self._select(model, y, node_count, deadline),
                            key=lambda item: item["violation"], reverse=True)
        model._fractional_candidates_detected += len(candidates)
        separation_elapsed = time.perf_counter() - started
        model._fractional_separation_seconds += separation_elapsed
        if candidates:
            model._fractional_zero_candidate_rounds = 0
        else:
            model._fractional_zero_candidate_rounds += 1
            if ((model._fractional_root_only or node_count > 0)
                    and model._fractional_max_zero_rounds > 0
                    and model._fractional_zero_candidate_rounds
                    >= model._fractional_max_zero_rounds):
                model._fractional_disabled_after_zero_rounds = True
        limit = (model._root_max_cuts_per_round if node_count == 0
                 else model._nonroot_max_cuts_per_callback)
        added = 0
        for item in candidates:
            if (model._fractional_time_budget > 0
                    and time.perf_counter() - started
                    >= model._fractional_time_budget):
                break
            key = tuple(item["nodes"])
            if key in model._fractional_clique_cut_keys:
                continue
            model.cbCut(gp.quicksum(
                model._edges_vars[i, j] + model._edges_vars[j, i]
                for i, j in item["pairs"]) <= item["rhs"])
            model._fractional_clique_cut_keys.add(key)
            model._clique_user_count += 1
            added += 1
            if limit > 0 and added >= limit:
                break

    def on_mipsol(self, model, selected_edges, edge_values, weight_values,
                  lag_edge_values, lag_weight_values, objective):
        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(model._d))
        skeleton.add_edges_from((min(i, j), max(i, j))
                                for i, j in selected_edges)
        for clique in nx.find_cliques(skeleton):
            if len(clique) > model._max_clique_size:
                model._dense_history_hits[tuple(sorted(clique))] += 1
        return super().on_mipsol(
            model, selected_edges, edge_values, weight_values,
            lag_edge_values, lag_weight_values, objective)

    def summary_fields(self, model):
        return {
            "fractional_candidates_detected": model._fractional_candidates_detected,
            "fractional_user_cuts": model._clique_user_count,
            "fractional_separation_seconds": model._fractional_separation_seconds,
            "unique_fractional_cuts": len(model._fractional_clique_cut_keys),
            "fractional_callbacks_checked": model._fractional_callbacks_checked,
            "fractional_callbacks_skipped": model._fractional_callbacks_skipped,
            "zero_candidate_rounds": model._fractional_zero_candidate_rounds,
            "disabled_after_zero_rounds": (
                model._fractional_disabled_after_zero_rounds),
            "separation_time_budget": model._fractional_time_budget,
        }

class BigMRelaxationMethod(CliqueMethod):
    family_name = "bigm_relaxation"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._big_m_mode = str(getattr(cfg, "big_m_mode", "scalar")).lower()
        if self._big_m_mode not in {"scalar", "edge_specific"}:
            raise ValueError(
                "big_m_mode must be 'scalar' or 'edge_specific'; got "
                f"{self._big_m_mode!r}")
        self._edge_specific = self._big_m_mode == "edge_specific"

    def prepare_big_m(self, X, Y, global_bound):
        bound = float(global_bound)
        if bound <= 0:
            raise ValueError(f"weights_bound must be positive, got {bound}")
        d = X.shape[1]
        fallback = {(i, j): bound for i in range(d) for j in range(d) if i != j}
        if not self._edge_specific:
            return bound, "scalar", fallback, {
                "mode": "scalar", "fallback_count": 0,
            }
        if str(getattr(self.cfg, "loss_type", "l2")) != "l2" or bool(
                getattr(self.cfg, "robust", False)):
            return bound, "edge_specific", fallback, {
                "mode": "edge_specific", "fallback_count": len(fallback),
                "fallback_reason": "requires non-robust l2",
            }
        safety = float(getattr(self.cfg, "edge_specific_safety_factor", 1.01))
        condition_limit = float(
            getattr(self.cfg, "edge_specific_condition_limit", 1e10))
        zero_sse = float(np.sum(X ** 2))
        bounds = dict(fallback)
        fallback_count, condition_numbers = 0, []
        for target in range(d):
            contemporaneous = [i for i in range(d) if i != target]
            blocks = [X[:, contemporaneous]] + list(Y)
            design = np.concatenate(blocks, axis=1)
            gram = design.T @ design
            condition = float(np.linalg.cond(gram)) if gram.size else float("inf")
            condition_numbers.append(condition)
            if not np.isfinite(condition) or condition > condition_limit:
                fallback_count += len(contemporaneous)
                continue
            try:
                inverse = np.linalg.inv(gram)
                beta = inverse @ design.T @ X[:, target]
            except np.linalg.LinAlgError:
                fallback_count += len(contemporaneous)
                continue
            residual = X[:, target] - design @ beta
            radius_squared = max(zero_sse - float(residual @ residual), 0.0)
            for position, source in enumerate(contemporaneous):
                candidate = abs(float(beta[position])) + np.sqrt(max(
                    radius_squared * float(inverse[position, position]), 0.0))
                bounds[source, target] = min(
                    bound, max(1e-6, safety * candidate))
        values = np.asarray(list(bounds.values()), dtype=float)
        return bound, "edge_specific", bounds, {
            "mode": "edge_specific", "fallback_count": fallback_count,
            "min": float(np.min(values)),
            "p25": float(np.quantile(values, .25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, .75)),
            "max": float(np.max(values)),
            "max_condition_number": max(condition_numbers, default=None),
        }

    def initialize_model(self, model):
        model._big_m_candidate_count = 0
        model._big_m_near_bound_count = 0
        model._big_m_max_observed_ratio = 0.0
        model._big_m_small_x_large_w_count = 0
        model._big_m_saturation_tolerance = float(
            getattr(self.cfg, "big_m_saturation_tolerance", 0.98))

    def on_mipsol(self, model, selected_edges, edge_values, weight_values,
                  lag_edge_values, lag_weight_values, objective):
        model._big_m_candidate_count += 1
        for edge, value in weight_values.items():
            bound = model._edge_specific_bounds.get(edge, model._global_big_m)
            ratio = abs(float(value)) / max(float(bound), 1e-12)
            model._big_m_max_observed_ratio = max(
                model._big_m_max_observed_ratio, ratio)
            if edge_values[edge] > 0.5 and ratio >= model._big_m_saturation_tolerance:
                model._big_m_near_bound_count += 1
            if edge_values[edge] <= 0.1 and abs(float(value)) >= 0.5:
                model._big_m_small_x_large_w_count += 1
        return super().on_mipsol(
            model, selected_edges, edge_values, weight_values,
            lag_edge_values, lag_weight_values, objective)

    def summary_fields(self, model):
        return {"big_m_mode": model._big_m_mode,
                "weights_bound": model._global_big_m,
                "edge_bound_summary": model._edge_bound_summary,
                "candidate_count": model._big_m_candidate_count,
                "near_bound_count": model._big_m_near_bound_count,
                "max_observed_bound_ratio": model._big_m_max_observed_ratio,
                "small_x_large_w_count": model._big_m_small_x_large_w_count,
                "bound_safety_warning": bool(
                    model._big_m_near_bound_count > 0),
                "saturation_tolerance": model._big_m_saturation_tolerance}

class TimeLimitOptimizationMethod(CliqueMethod):
    family_name = "time_limit_optimization"
    _defaults = [5, 60, 120, 300, 600, 1200, 1800, 2700,
                 3600, 5400, 7200]

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.variant != "G":
            raise ValueError(f"time_limit_optimization expects G, got {self.variant!r}")

    def initialize_model(self, model):
        values = list(getattr(self.cfg, "checkpoint_times", [])) or self._defaults
        model._checkpoint_times = sorted(float(value) for value in values)
        model._checkpoint_rows = []
        model._next_checkpoint_index = 0

    @staticmethod
    def _callback_time(model):
        return (model._cycle_detection_seconds + model._cycle_submission_seconds
                + model._clique_detection_seconds + model._clique_submission_seconds)

    def on_mip(self, model):
        runtime = float(model.cbGet(GRB.Callback.RUNTIME))
        while (model._next_checkpoint_index < len(model._checkpoint_times)
               and runtime >= model._checkpoint_times[model._next_checkpoint_index]):
            target = model._checkpoint_times[model._next_checkpoint_index]
            incumbent = float(model.cbGet(GRB.Callback.MIP_OBJBST))
            bound = float(model.cbGet(GRB.Callback.MIP_OBJBND))
            gap = (abs(incumbent - bound) / max(abs(incumbent), 1e-12)
                   if np.isfinite(incumbent) and np.isfinite(bound) else None)
            snapshot = model._best_accepted_snapshot
            model._checkpoint_rows.append({
                "checkpoint_target_seconds": target,
                "observed_runtime_seconds": runtime,
                "best_incumbent": incumbent, "best_bound": bound, "gap": gap,
                "explored_nodes": float(model.cbGet(GRB.Callback.MIP_NODCNT)),
                "open_nodes": float(model.cbGet(GRB.Callback.MIP_NODLFT)),
                "lp_iterations": float(model.cbGet(GRB.Callback.MIP_ITRCNT)),
                "solution_count": int(model.cbGet(GRB.Callback.MIP_SOLCNT)),
                "cycle_lazy_cuts": model._cycle_lazy_count,
                "clique_lazy_cuts": model._clique_lazy_count,
                "clique_user_cuts": model._clique_user_count,
                "callback_time": self._callback_time(model),
                "best_accepted_objective": (
                    snapshot["objective"] if snapshot else None),
                "best_accepted_edges": snapshot["edges"] if snapshot else None,
                "best_accepted_weights": snapshot["weights"] if snapshot else None,
                "best_accepted_nnz": snapshot["nnz"] if snapshot else None,
                "best_accepted_shd": snapshot["shd"] if snapshot else None,
                "best_accepted_f1": snapshot["f1"] if snapshot else None,
            })
            model._next_checkpoint_index += 1

    def finalize(self, model):
        runtime = float(model.Runtime)
        while (model._next_checkpoint_index < len(model._checkpoint_times)
               and runtime >= model._checkpoint_times[model._next_checkpoint_index]):
            target = model._checkpoint_times[model._next_checkpoint_index]
            snapshot = model._best_accepted_snapshot
            has_solution = int(model.SolCount) > 0
            model._checkpoint_rows.append({
                "checkpoint_target_seconds": target,
                "observed_runtime_seconds": runtime,
                "best_incumbent": float(model.ObjVal) if has_solution else None,
                "best_bound": float(model.ObjBound),
                "gap": float(model.MIPGap) if has_solution else None,
                "explored_nodes": float(model.NodeCount), "open_nodes": None,
                "lp_iterations": float(model.IterCount),
                "solution_count": int(model.SolCount),
                "cycle_lazy_cuts": model._cycle_lazy_count,
                "clique_lazy_cuts": model._clique_lazy_count,
                "clique_user_cuts": model._clique_user_count,
                "callback_time": self._callback_time(model),
                "best_accepted_objective": (
                    snapshot["objective"] if snapshot else None),
                "best_accepted_edges": snapshot["edges"] if snapshot else None,
                "best_accepted_weights": snapshot["weights"] if snapshot else None,
                "best_accepted_nnz": snapshot["nnz"] if snapshot else None,
                "best_accepted_shd": snapshot["shd"] if snapshot else None,
                "best_accepted_f1": snapshot["f1"] if snapshot else None,
            })
            model._next_checkpoint_index += 1

    def summary_fields(self, model):
        return {"checkpoints": model._checkpoint_rows}

def _selected_method(cfg):
    """Select one behavior from direct YAML switches; no solver-family factory."""
    if bool(getattr(cfg, "enable_incumbent_repair", False)):
        cfg.experiment_variant = str(getattr(cfg, "repair_variant", "C5"))
        return RepairIncumbentMethod(cfg)
    if bool(getattr(cfg, "enable_fractional_separation", False)):
        cfg.experiment_variant = str(getattr(cfg, "fractional_variant", "D2"))
        return FractionalSeparationMethod(cfg)
    if str(getattr(cfg, "big_m_mode", "scalar")).lower() == "edge_specific":
        return BigMRelaxationMethod(cfg)
    if bool(getattr(cfg, "enable_checkpoints", False)):
        cfg.experiment_variant = "G"
        return TimeLimitOptimizationMethod(cfg)
    if bool(getattr(cfg, "enable_delayed_separation", False)):
        cfg.experiment_variant = "A"
        return DelayedSeparationMethod(cfg)
    formulation = str(getattr(cfg, "clique_cut_formulation", "original")).lower()
    selection = str(getattr(cfg, "clique_cut_selection", "original")).lower()
    if formulation != "original" or selection != "original":
        return CliqueSeparationOptionsMethod(cfg)
    return CliqueMethod(cfg)

def solve(X, cfg: DictConfig, w_threshold, Y=None, B_ref=None, tabu_edges=None,
          method=None):

    time_limit = cfg.time_limit
    lambda1 = cfg.lambda1
    lambda2 = cfg.lambda2
    loss_type = cfg.loss_type
    constraints_mode = cfg.constraints_mode
    mode = cfg.callback_mode
    robust = cfg.robust
    if method is None:
        method = _selected_method(cfg)
    method_variant = method.variant
    reg_type = cfg.reg_type
    a_reg_type = cfg.a_reg_type
    target_mip_gap = cfg.target_mip_gap

    n, d = X.shape
    if Y is None:
        Y = []
    p = len(Y) # The number of historical data slices
    weights_bound, big_m_mode, edge_specific_bounds, edge_bound_summary = \
        method.prepare_big_m(X, Y, float(cfg.weights_bound))
    W_link_bounds = (edge_specific_bounds
                     if big_m_mode == "edge_specific"
                     else weights_bound)
     # 'no-weights'
    # if loss_type == 'l2':
    #     X = X - np.mean(X, axis=0, keepdims=True)


    m = gp.Model()
    W_edges_vars, W_edges_weights = construct_matrix_vars(
        m, d, 'W', constraints_mode, W_link_bounds, tabu_edges)
    for v1 in range(d):
        for v2 in range(v1):
            m.addConstr(W_edges_vars[v2,v1] + W_edges_vars[v1,v2] <= 1)

    A_edges_vars = []
    A_edges_weights = []

    a_constraints_mode = '' if a_reg_type == 'l1' else 'no-vars'

    for t in range(p):
        A_t_edges_vars, A_t_edges_weights = construct_matrix_vars(m, d, f'A_{t}', a_constraints_mode, weights_bound, diagonal=True, tabu_edges=None)
        A_edges_vars.append(A_t_edges_vars)
        A_edges_weights.append(A_t_edges_weights)

    robust_vars = {}
    quad_diff = {}
    if robust:
        for i in range(n):
            robust_vars[i] = m.addVar(vtype=GRB.BINARY, name=f's{i}')
            for j in range(d):
                quad_diff[i, j] = m.addVar(lb = float('-inf'),vtype=GRB.CONTINUOUS, name=f'q{i}-{j}')
        r = round(0.9 * n)
        m.addConstr(gp.quicksum(robust_vars[i] for i in range(n)) >= r)
        for i in range(n):
            for j in range(d):
                m.addConstr((X[i,j] - gp.quicksum(X[i, k] * W_edges_weights[k, j] for k in range(d) if k != j) - gp.quicksum(Y[t][i, k] * A_edges_weights[t][k, j] for k in range(d) for t in range(p)))**2 == quad_diff[i,j])
        robust_objective = gp.quicksum(quad_diff[i,j] * robust_vars[i] for i in range(n) for j in range(d))
    #callback_constraints = {}
    #callback_constraints[1] = 0


    # regulazition
    if reg_type == 'l2':
        reg = gp.quicksum(w**2 for w in W_edges_weights.values())
        # reg2 = 0
        # for A_t_edges_weights in A_edges_weights:
        #     reg2 = reg2 + gp.quicksum(a**2 for a in A_t_edges_weights.values())
    elif reg_type == 'l1':
        reg = gp.quicksum(w for w in W_edges_vars.values())
        # reg2 = 0
        # l2 reg for As becouse we dont have decision variables for them.
        # for A_t_edges_weights in A_edges_weights:
        #     reg2 = reg2 + gp.quicksum(a**2 for a in A_t_edges_weights.values())
        # for A_t_edges_vars in A_edges_vars:
        #     reg = reg + gp.quicksum(a for a in A_t_edges_vars.values())
    else:
        assert False

    if a_reg_type == 'l2':
        reg2 = 0
        for A_t_edges_weights in A_edges_weights:
            reg2 = reg2 + gp.quicksum(a**2 for a in A_t_edges_weights.values())
    elif a_reg_type == 'l1':
        reg2 = 0
        for A_t_edges_vars in A_edges_vars:
            reg = reg + gp.quicksum(a for a in A_t_edges_vars.values())
    else:
        assert False


    # Cost function
    if loss_type == 'l2':
        if robust:
            m.setObjective(robust_objective + lambda1 * reg / d + lambda2 * reg2 / d, GRB.MINIMIZE)
        else:
            m.setObjective(gp.quicksum((X[i,j] - gp.quicksum(X[i, k] * W_edges_weights[k, j] for k in range(d) if k != j)
                                        - gp.quicksum(Y[t][i, k] * A_edges_weights[t][k, j] for k in range(d) for t in range(p))
                                        )**2 for i in range(n) for j in range(d))/n + lambda1 * reg / d + lambda2 * reg2 / d, GRB.MINIMIZE)
            print(m.getObjective().getValue())
    elif loss_type == 'l1':

        abs_vars = {}
        for i in range(n):
            for j in range(d):
                abs_vars[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name=f'abs{i}-{j})')
                m.addConstr((X[i,j] - gp.quicksum(X[i, k] * W_edges_weights[k, j] for k in range(d) if k != j) - gp.quicksum(Y[t][i, k] * A_edges_weights[t][k, j] for k in range(d) for t in range(p))) <= abs_vars[i,j])
                m.addConstr(-(X[i,j] - gp.quicksum(X[i, k] * W_edges_weights[k, j] for k in range(d) if k != j) - gp.quicksum(Y[t][i, k] * A_edges_weights[t][k, j] for k in range(d) for t in range(p))) <= abs_vars[i,j])

        # abs_edges_weights={}
        # for v1 in range(d):
        #     for v2 in range(d):
        #         if v1 != v2:
        #             abs_edges_weights[v1,v2] = m.addVar(vtype=GRB.CONTINUOUS, name=f'abs_weight{v1}->{v2}')
        #             m.addConstr(W_edges_weights[v1,v2] <= abs_edges_weights[v1,v2])
        #             m.addConstr(-W_edges_weights[v1,v2] <= abs_edges_weights[v1,v2])


        m.setObjective(gp.quicksum(abs_vars[i,j] for i in range(n) for j in range(d))/n + lambda1 * reg / d + lambda2 * reg2 / d, GRB.MINIMIZE)

    m.Params.lazyConstraints = 1
    if method.requires_precrush():
        m.Params.PreCrush = 1
    m.Params.MIPGap = target_mip_gap
    m.params.TimeLimit = time_limit
    if hasattr(cfg, "gurobi_seed"):
        m.Params.Seed = int(cfg.gurobi_seed)
    if hasattr(cfg, "gurobi_threads"):
        m.Params.Threads = int(cfg.gurobi_threads)
    m._edges_vars = W_edges_vars
    m._edges_weights = W_edges_weights
    m._lag_edges_vars = A_edges_vars
    m._lag_edges_weights = A_edges_weights
    m._lazy_count = 0
    m._cycle_lazy_count = 0
    m._clique_lazy_count = 0
    m._clique_user_count = 0
    m._added_cycle_cut_keys = set()
    m._added_clique_cut_keys = set()
    m._mipsol_count = 0
    m._mipsol_rejected_count = 0
    m._mipsol_accepted_count = 0
    m._first_constraint_runtime = None
    m._last_constraint_runtime = None
    m._cycle_detection_seconds = 0.0
    m._cycle_submission_seconds = 0.0
    m._clique_detection_seconds = 0.0
    m._clique_submission_seconds = 0.0
    m._last_time_stats = 0
    m._B_ref = B_ref
    m._stats = []
    m._d = d
    m._X = np.asarray(X)
    m._Y = Y
    m._w_threshold = float(w_threshold)
    m._method = method
    m._method_variant = method_variant
    m._method_name = method.method_name
    m._global_big_m = float(weights_bound)
    m._big_m_mode = big_m_mode
    m._edge_specific_bounds = edge_specific_bounds
    m._edge_bound_summary = edge_bound_summary
    m._callback_mode = mode
    m._clique_callback_mode = getattr(cfg, "clique_callback_mode", "first_clique")
    m._max_clique_size = getattr(cfg, "max_clique_size", 3)
    m._enable_cycle_constraints = bool(getattr(cfg, "enable_cycle_constraints", True))
    m._enable_clique_constraints = bool(getattr(cfg, "enable_clique_constraints", True))
    m._separation_order = str(getattr(cfg, "separation_order", "cycle_first"))
    m._delayed_second_separation = bool(
        getattr(cfg, "delayed_second_separation", False))
    m._cycle_top_k = int(getattr(cfg, "cycle_top_k", 5))
    m._clique_top_k = int(getattr(cfg, "clique_top_k", 5))
    m._max_cycle_cuts_per_callback = int(
        getattr(cfg, "max_cycle_cuts_per_callback", 0))
    m._max_clique_cuts_per_callback = int(
        getattr(cfg, "max_clique_cuts_per_callback", 0))
    m._deduplicate_cycle_cuts = bool(getattr(cfg, "deduplicate_cycle_cuts", False))
    m._deduplicate_clique_cuts = bool(getattr(cfg, "deduplicate_clique_cuts", False))
    m._best_accepted_objective = float("inf")
    m._best_accepted_snapshot = None
    m._first_valid_incumbent_time = None
    m._accepted_incumbent_trajectory = []
    method.initialize_model(m)
    diagnostics_on = diagnostics_enabled(cfg)
    # Preserve the original solver output when diagnostics are disabled:
    # print_clique_cuts predates the diagnostics collector and therefore
    # remains independently controlled by its original YAML field.
    m._print_clique_cuts = bool(getattr(cfg, "print_clique_cuts", False))
    # These two outputs were introduced by the diagnostics instrumentation,
    # so the master switch must be on before their optional sub-flags apply.
    m._print_callback_events = diagnostics_on and bool(
        getattr(cfg, "print_callback_events", True))
    m._print_retained_solutions = diagnostics_on and bool(
        getattr(cfg, "print_retained_solutions", True))
    m._optimization_diagnostics = create_diagnostics(
        cfg, m._method_name, d,
        int(m._max_clique_size), weights_bound)
    try:
        solve_wall_started = time.perf_counter()
        if m._print_callback_events:
            print("[SOLVER] event=gurobi_solve_start runtime=0.000000s", flush=True)
        m.optimize(method_callback)
        method.finalize(m)
        solve_wall_seconds = time.perf_counter() - solve_wall_started
        post_constraint_seconds = (
            float(m.Runtime) - m._last_constraint_runtime
            if m._last_constraint_runtime is not None else None
        )
        if m._print_callback_events:
            print(
                f"[SOLVER] event=gurobi_solve_end runtime={float(m.Runtime):.6f}s "
                f"wall_seconds={solve_wall_seconds:.6f} status={int(m.Status)}",
                flush=True,
            )
            print(
                "[CALLBACK-SUMMARY] "
                f"mipsol_total={m._mipsol_count} "
                f"mipsol_rejected={m._mipsol_rejected_count} "
                f"mipsol_accepted={m._mipsol_accepted_count} "
                f"cycle_constraints_added={m._cycle_lazy_count} "
                f"clique_constraints_added={m._clique_lazy_count} "
                f"cycle_filter_seconds={m._cycle_detection_seconds:.9f} "
                f"cycle_add_seconds={m._cycle_submission_seconds:.9f} "
                f"clique_filter_seconds={m._clique_detection_seconds:.9f} "
                f"clique_add_seconds={m._clique_submission_seconds:.9f} "
                f"first_constraint_runtime_seconds="
                f"{m._first_constraint_runtime if m._first_constraint_runtime is not None else 'NA'} "
                f"last_constraint_runtime_seconds="
                f"{m._last_constraint_runtime if m._last_constraint_runtime is not None else 'NA'} "
                f"solve_after_last_constraint_seconds="
                f"{post_constraint_seconds if post_constraint_seconds is not None else 'NA'}",
                flush=True,
            )
        _print_retained_solutions(m)

        sol_count = m.getAttr(GRB.attr.SolCount)
        if sol_count == 0:
            if m._optimization_diagnostics is not None:
                m._optimization_diagnostics.write({"solution_count": 0})
            return None, None, None, None, None, {
                "cycle_lazy_count": m._cycle_lazy_count,
                "clique_lazy_count": m._clique_lazy_count,
                "clique_user_count": m._clique_user_count,
            }
        gap = m.MIPGap
        lazy_count = m._lazy_count
        lazy_breakdown = {
            "cycle_lazy_count": m._cycle_lazy_count,
            "clique_lazy_count": m._clique_lazy_count,
            "clique_user_count": m._clique_user_count,
        }
        stats = m._stats

        #print(f'add constraints: {callback_constraints[1]}')

        W_edges_vals = m.getAttr('x', W_edges_vars)
        W_weights_vals = m.getAttr('x', W_edges_weights)

        W = extract_adj_matrix(W_edges_vals, W_weights_vals, d)

        A = []
        for t in range(p):
            #A_t_edges_vals = m.getAttr('x', A_edges_vars[t])
            A_t_weights_vals = m.getAttr('x', A_edges_weights[t])
            A_t = extract_adj_matrix_no_vars(A_t_weights_vals, d)
            A.append(A_t)

        if m._enable_cycle_constraints:
            assert utils.is_dag(W)

        if m._optimization_diagnostics is not None:
            W_eval = W.copy()
            W_eval[np.abs(W_eval) < w_threshold] = 0
            final_accuracy = count_project_accuracy(
                np.asarray(B_ref).astype(int), (W_eval != 0).astype(int), [], [],
                test_dag=m._enable_cycle_constraints) \
                if B_ref is not None else {}
            m._optimization_diagnostics.write({
                "solution_count": int(sol_count),
                "objective": float(m.ObjVal),
                "best_bound": float(m.ObjBound),
                "gap": float(gap),
                "runtime": float(m.Runtime),
                "node_count": float(m.NodeCount),
                "simplex_iteration_count": float(m.IterCount),
                "work": float(m.Work),
                "nodes_per_second": float(m.NodeCount / max(m.Runtime, 1e-12)),
                "final_shd_at_solver_threshold": final_accuracy.get("shd"),
                "final_f1_at_solver_threshold": final_accuracy.get("f1score"),
                "lazy_count": int(lazy_count),
                **lazy_breakdown,
            })

        # threshold_func = np.vectorize(lambda x: (x if abs(x) > threshold else 0.0))
        # W_t = threshold_func(W)

        W[np.abs(W) < w_threshold] = 0

        return W, A, gap, lazy_count, stats, lazy_breakdown
    finally:
        if m._optimization_diagnostics is not None:
            m._optimization_diagnostics.write()
        _write_method_summary(m, cfg)
        m.dispose()
        gp.disposeDefaultEnv()


if __name__ == '__main__':
    from notears import utils
    utils.set_random_seed(1)


    n, d, s0, graph_type, sem_type = 2000, 10, 30, 'ER', 'gauss'
    #n, d, s0, graph_type, sem_type = 100, 3, 20, 'PATHPERM', 'gauss'  # 7 funguje, 25 GOBNILP COUNTER EXAMPL
    #n, d, s0, graph_type, sem_type = 10, 2, 20, 'PATH', 'gauss' # 7 funguje, 25
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')
    #W_true = np.loadtxt('W_true.csv', delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=1)
    Y = [] # TODO: add historical data
    xcol = X[:,1] / X[:,0]
    x1avg = X[:,0].sum()/n
    x2avg = X[:,1].sum()/n
    print('debug')
    print(x2avg/x1avg)

    xrat = (X[:,1]/X[:,0]).sum()/n
    print(xrat)

    np.savetxt('X.csv', X, delimiter=',')
    #X = np.loadtxt('X.csv', delimiter=',')
    cfg = OmegaConf.create()
    cfg.time_limit = 18000
    cfg.constraints_mode = 'weights'
    cfg.callback_mode = 'all_cycles'
    cfg.lambda1=1
    cfg.lambda2=1
    cfg.loss_type='l2'
    cfg.reg_type='l1'
    cfg.a_reg_type='l1'
    cfg.robust = False
    cfg.weights_bound = 100
    cfg.target_mip_gap = 0.001
    cfg.tabu_edges = False

    W_est, A_est, gap, lazy_count, stats = solve(X, cfg, 0, Y=Y, B_ref=B_true) # lambda1=0.0009
    assert utils.is_dag(W_est)
    np.savetxt('W_est_milp.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(f'gap: {gap}')
    print(f'lazy_count: {lazy_count}')
    print(stats)
    print(acc)
