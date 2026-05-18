from typing import Tuple, Dict

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from gurobipy import Model
from omegaconf import DictConfig, OmegaConf

# from dagsolvers.dagsolver_utils import find_minimal_dag_threshold
# from dagsolvers.metrics_utils import apply_threshold, find_optimal_threshold_for_shd


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
    elif mode == 'all_cycles':
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

def check_for_cycles(model, where):
    if where == GRB.Callback.MESSAGE:
        pass
        # edges_vals = model.cbGetSolution(model._edges_vars)
        # weights_vals = model.cbGetSolution(model._edges_weights)
        # W = extract_adj_matrix(edges_vals, weights_vals)
        # print(W)

    if where == GRB.Callback.MIPSOL:
        #print('CALLBACK')
        # make a list of edges selected in the solution
        constr_added = False
        vals = model.cbGetSolution(model._edges_vars)
        weights_vals = model.cbGetSolution(model._edges_weights)
        selected_edges = gp.tuplelist((i, j) for i, j in model._edges_vars.keys()
                                      if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        cycles = find_cycles(selected_edges, model._callback_mode)
        for cycle in cycles:
            edges_of_cycle = []
            for i in range(len(cycle)-1):
                edges_of_cycle.append((cycle[i+1], cycle[i]))
            edges_of_cycle.append((cycle[0], cycle[-1]))
            #callback_constraints[1] = callback_constraints[1] + 1
            #print('NEW CONSTRAINT')
            model._lazy_count += 1
            model.cbLazy(gp.quicksum(model._edges_vars[i, j] for i, j in edges_of_cycle)
                         <= len(edges_of_cycle)-1)
            constr_added = True

        # Compute solving statistics
        rt = model.cbGet(GRB.Callback.RUNTIME)
        # if not constr_added and model._B_ref is not None and (rt - model._last_time_stats > 60): # Compute statistics every 60 seconds
        #     B_true = model._B_ref
        #     model._last_time_stats = rt
        #     W_sol = extract_adj_matrix(vals, weights_vals, model._d)
        #     dag_t, W_sol = find_minimal_dag_threshold(W_sol)
        #     #W_sol = apply_threshold(W_sol, 0.3)
        #     default_threshold = 0.3
        #     W_t = apply_threshold(W_sol, default_threshold)
        #     shd = utils.count_accuracy(B_true, W_t != 0)['shd']
        #     objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        #
        #     best_t, best_shd = find_optimal_threshold_for_shd(B_true, W_sol, [], [], np.zeros_like(B_true), np.zeros_like(B_true))
        #
        #     print(f't{default_threshold}_SHD: {shd} BEST_SHD: {best_shd} BEST_t: {best_t} OBJ: {objval} DAG_t: {dag_t}')
        #     model._stats.append((round(rt), shd, best_shd, best_t, objval, dag_t))






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
                    m.addConstr(edges_weights[v1,v2] <= weights_bound)
                    m.addConstr(-edges_weights[v1,v2] <= weights_bound)
                else:
                    m.addConstr(edges_weights[v1,v2] <= weights_bound * edges_vars[v1,v2])
                    m.addConstr(-edges_weights[v1,v2] <= weights_bound * edges_vars[v1,v2])
    if tabu_edges is not None:
        for (v1,v2) in tabu_edges:
            m.addConstr(edges_vars[v1,v2] == 0)
            m.addConstr(edges_weights[v1,v2] == 0)
    return edges_vars, edges_weights


def solve(X, cfg: DictConfig, w_threshold, Y=None, B_ref=None, tabu_edges=None):

    time_limit = cfg.time_limit
    lambda1 = cfg.lambda1
    lambda2 = cfg.lambda2
    loss_type = cfg.loss_type
    constraints_mode = cfg.constraints_mode
    mode = cfg.callback_mode
    robust = cfg.robust
    weights_bound = cfg.weights_bound
    reg_type = cfg.reg_type
    a_reg_type = cfg.a_reg_type
    target_mip_gap = cfg.target_mip_gap

    n, d = X.shape
    if Y is None:
        Y = []
    p = len(Y) # The number of historical data slices
     # 'no-weights'
    # if loss_type == 'l2':
    #     X = X - np.mean(X, axis=0, keepdims=True)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    #m.setParam("OutputFlag", 0)
    W_edges_vars, W_edges_weights = construct_matrix_vars(m, d, 'W', constraints_mode, weights_bound, tabu_edges)
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
            #print(m.getObjective().getValue())
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
    m.Params.MIPGap = target_mip_gap
    m.Params.NonConvex = 2
    m.params.TimeLimit = time_limit
    m._edges_vars = W_edges_vars
    m._edges_weights = W_edges_weights
    m._lazy_count = 0
    m._last_time_stats = 0
    m._B_ref = B_ref
    m._stats = []
    m._d = d
    m._callback_mode = mode
    m.optimize(check_for_cycles)

    sol_count = m.getAttr(GRB.attr.SolCount)
    if sol_count == 0:
        return None, None, None, None, None
    gap = m.MIPGap
    lazy_count = m._lazy_count
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

    #assert utils.is_dag(W)
    m.dispose()
    gp.disposeDefaultEnv()

    # threshold_func = np.vectorize(lambda x: (x if abs(x) > threshold else 0.0))
    # W_t = threshold_func(W)

    W[np.abs(W) < w_threshold] = 0

    return W, A, gap, lazy_count, stats


# if __name__ == '__main__':
#     from notears import utils
#     utils.set_random_seed(1)
#
#
#     n, d, s0, graph_type, sem_type = 2000, 10, 30, 'ER', 'gauss'
#     #n, d, s0, graph_type, sem_type = 100, 3, 20, 'PATHPERM', 'gauss'  # 7 funguje, 25 GOBNILP COUNTER EXAMPL
#     #n, d, s0, graph_type, sem_type = 10, 2, 20, 'PATH', 'gauss' # 7 funguje, 25
#     B_true = utils.simulate_dag(d, s0, graph_type)
#     W_true = utils.simulate_parameter(B_true)
#     np.savetxt('W_true.csv', W_true, delimiter=',')
#     #W_true = np.loadtxt('W_true.csv', delimiter=',')
#
#     X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=1)
#     Y = [] # TODO: add historical data
#     xcol = X[:,1] / X[:,0]
#     x1avg = X[:,0].sum()/n
#     x2avg = X[:,1].sum()/n
#     print('debug')
#     print(x2avg/x1avg)
#
#     xrat = (X[:,1]/X[:,0]).sum()/n
#     print(xrat)
#
#     np.savetxt('X.csv', X, delimiter=',')
#     #X = np.loadtxt('X.csv', delimiter=',')
#     cfg = OmegaConf.create()
#     cfg.time_limit = 1800
#     cfg.constraints_mode = 'weights'
#     cfg.callback_mode = 'all_cycles'
#     cfg.lambda1=1
#     cfg.lambda2=1
#     cfg.loss_type='l2'
#     cfg.reg_type='l1'
#     cfg.a_reg_type='l1'
#     cfg.robust = False
#     cfg.weights_bound = 100
#     cfg.target_mip_gap = 0.001
#     cfg.tabu_edges = False
#
#     W_est, A_est, _, _, stats = solve(X, cfg, 0, Y=Y, B_ref=B_true) # lambda1=0.0009
#     assert utils.is_dag(W_est)
#     np.savetxt('W_est_milp.csv', W_est, delimiter=',')
#     acc = utils.count_accuracy(B_true, W_est != 0)
#     print(stats)
#     print(acc)

