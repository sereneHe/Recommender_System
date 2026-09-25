import networkx as nx
import numpy as np
from networkx import from_numpy_array, DiGraph

import notears.utils as notears_utils
from dagsolvers.metrics_utils import count_accuracy, apply_threshold
from dagsolvers.shd_utils import calculate_shd


class ExDagDataException(Exception):
    pass


def find_minimal_dag_threshold(W):
    if notears_utils.is_dag(W):
        return 0, W
    possible_thresholds = sorted((abs(t) for t in W.flatten() if abs(t) > 0))
    for t_candidate in possible_thresholds:
        W[np.abs(W) < t_candidate] = 0
        if notears_utils.is_dag(W):
            return t_candidate, W
    assert False  # Should always find a dag


def find_optimal_multiple_thresholds(W_true, W_est, A_true, A_est, W_bi_true, Wbi):
    assert False, 'this method needs to be fixed'
    best_w_t, best_w_shd, best_W, _ = find_optimal_threshold_single_matrix(W_true, W_est)
    best_a_t = []
    best_a_shd = []
    best_a = []
    for a_true_i, a_est_i in zip(A_true, A_est):
        best_a_i_t, best_a_i_shd, best_a_i, _ = find_optimal_threshold_single_matrix(a_true_i, a_est_i)
        best_a_t.append(best_a_i_t)
        best_a_shd.append(best_a_i_shd)
        best_a.append(best_a_i)

    best_acc = count_accuracy(W_true, best_W != 0, A_true, best_a)

    return best_w_t, best_a_t, best_w_shd + sum(best_a_shd), best_W, best_a, best_acc


def find_optimal_threshold_single_matrix(W_true, W_est):
    assert False, 'this method needs to be fixed'
    possible_thresholds = sorted((abs(t) for t in W_est.flatten() if abs(t) > 0))
    best_t = max(possible_thresholds) if possible_thresholds else 0
    best_shd, _, _ = calculate_shd(W_true, W_est != 0, [], [], test_dag=False) # W_true.shape[0]**2
    best_acc = count_accuracy(W_true, W_est != 0, [], [], test_dag=False)
    best_W = W_est
    for t_candidate in possible_thresholds:
        W_est_t = apply_threshold(W_est, t_candidate)
        shd, _, _ = calculate_shd(W_true, W_est_t != 0, [], [], test_dag=False)
        #shd, acc = compute_shd(W_true, W_est_t)
        if shd < best_shd:
            best_t = t_candidate
            best_shd = shd
            best_acc = count_accuracy(W_true, W_est_t != 0, [], [], test_dag=False)
            best_W = W_est_t
    return best_t, best_shd, best_W, best_acc


def plot(W, nodelist, filename=None, dpi=None):
    import matplotlib.pyplot as plt
    # if abbrev:
    #     ls = dict((x,x[:3]) for x in self.nodes)
    # else:
    #     ls = None
    # try:
    #     edge_colors = [self._edge_colour[compelled] for (u,v,compelled) in self.edges.data('compelled')]
    # except KeyError:
    #     edge_colors = 'k'
    graph = from_numpy_array(W, create_using=DiGraph, nodelist=nodelist)
    fig, ax = plt.subplots()
    try:
        # Prefer Graphviz when available for cleaner layered DAG layouts.
        pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
    except Exception:
        # Fall back to a pure NetworkX layout so plotting never blocks experiments.
        pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx(graph, ax=ax, pos=pos, node_color="white", arrowsize=15)
    if filename is not None:
        fig.savefig(filename,format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def plot_heatmap(W, names_x, names_y, filename=None, dpi=None):
    import matplotlib.pyplot as plt

    # Remove '_lag0' suffix from names
    names_x = [name.split("_lag")[0] for name in names_x]
    names_y = [name.split("_lag")[0] for name in names_y]

    fig, ax = plt.subplots()

    # Create the heatmap using imshow
    limit = max(abs(W.min()), abs(W.max()))

    cax = ax.imshow(W, cmap='YlGnBu', interpolation='nearest', vmin=-limit, vmax=limit) #cmaps: # YlGnBu # coolwarm

    ax.set_xticks(np.arange(len(names_x)))
    ax.set_xticklabels(names_x, rotation=90)
    ax.set_yticks(np.arange(len(names_y)))
    ax.set_yticklabels(names_y)

    # Add a colorbar to the figure
    fig.colorbar(cax, ax=ax)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename,format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def is_pag(W):
    return (W == -3).any() or (W == 2).any()


def compute_combined_shd(W_true, W_est, A_true, A_est):
    sum_est = np.copy(W_est)
    sum_true = np.copy(W_true)
    for A_true_i, A_est_i in zip(A_true, A_est):
        sum_est += A_est_i
        sum_true += A_true_i

    return find_optimal_threshold_single_matrix(sum_true, sum_est)


def tupledict_to_np_matrix(tuple_dict, d):
    matrix = np.zeros((d, d))
    for (i, j), value in tuple_dict.items():
        matrix[i, j] = value
    return matrix
