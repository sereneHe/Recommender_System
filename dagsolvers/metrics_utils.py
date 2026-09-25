import numpy as np

from dagsolvers.graphs_utils import project_pag_on_mag
from dagsolvers.shd_utils import calculate_shd, calculate_dag_shd
from notears import utils
from numpy.linalg import norm


def log_acc_metrics(acc, run, prefix):
    for key in ['shd', 'shd_w', 'shd_a', 'fdr', 'least_square_cost', 'norm_distance', 'cond', 'true_pos', 'tpr', 'false_pos', 'nnz', 'false_neg', 'precision', 'f1score', 'g_score']:
        if key in acc:
            run.log_metric(f'{prefix}_{key}', acc[key])


def calculate_metrics_pag(B_true, B_bi_true, B_est, B_bi_est, run, cfg):
    B_bi_est = np.triu(B_bi_est, k=1)
    B_bi_true = np.triu(B_bi_true, k=1)
    B_all_true = B_true + (-1 * B_bi_true)
    B_all_est = B_est + (-1 * B_bi_est)
    shd = calculate_dag_shd(B_all_true, B_all_est, test_dag=False)

    B_mag_est, B_bi_mag_est = project_pag_on_mag(B_true, B_bi_true, B_est, B_bi_est)
    B_all_mag_est = B_mag_est + (-1 * B_bi_mag_est)

    acc_all = count_accuracy(B_all_true, B_all_mag_est, [], [])

    assert shd == acc_all['shd'], f'{shd} != {acc_all["shd"]}'

    metric_infix = 'no_threshold'
    #acc_all = {'shd': shd,}
    log_acc_metrics(acc_all, run, f'{metric_infix}')

def calculate_metrics(X, Y, W_true, B_true, W_lags_true, B_lags_true, W_est, W_lags_est, W_bi_true, B_bi_true, W_bi_est, run, cfg):
    # THIS function does not work for PAGs!!!
    cost_W_true = least_square_cost(X, W_true, Y, W_lags_true)
    run.log_metric('true_least_square_cost', cost_W_true)

    if W_bi_est is None:
        W_bi_est = np.zeros_like(W_true)
        W_bi_true = np.zeros_like(W_true)
        B_bi_true = np.zeros_like(W_true)
    W_bi_est = np.triu(W_bi_est, k=1)
    W_bi_true = np.triu(W_bi_true, k=1)
    #B_bi_true = (W_bi_true != 0).astype(int)
    B_all_true = B_true - B_bi_true

    best_t, best_shd = find_optimal_threshold_for_shd(B_true, W_est, B_lags_true, W_lags_est, W_bi_true, W_bi_est)

    thresholds = [0.5, 0.3, 0.15, 0.05, best_t]
    best_W = None
    best_Wbi = None
    best_A = None
    for threshold in thresholds:
        W_est_t = apply_threshold(W_est, threshold)
        B_est_t = (W_est_t != 0).astype(int)
        W_bi_est_t = apply_threshold(W_bi_est, threshold)
        B_bi_est_t = (W_bi_est_t != 0).astype(int)
        B_all_est_t = B_est_t + (-1 * B_bi_est_t) # CPDAG - undirected edges have -1
        W_lag_est_t = [apply_threshold(m, threshold) for m in W_lags_est]
        B_lag_est_t = [(m != 0).astype(int) for m in W_lag_est_t]
        acc_all = count_accuracy(B_all_true, B_all_est_t, B_lags_true, B_lag_est_t)
        acc_all['least_square_cost'] = least_square_cost(X, W_est_t, Y, W_lag_est_t) - cost_W_true
        acc_all['norm_distance'] = compute_norm_distance(W_true, W_est_t, W_lags_true, W_lag_est_t)
        if threshold == best_t:
            best_W = W_est_t
            best_Wbi = W_bi_est_t
            best_A = W_lag_est_t
        metric_infix = 'best' if threshold == best_t else f't{threshold}'
        log_acc_metrics(acc_all, run, f'{metric_infix}')
        acc_dir = count_accuracy(B_true, B_est_t, B_lags_true, B_lag_est_t)
        log_acc_metrics(acc_dir, run, f'dir_{metric_infix}')
        acc_bi = count_accuracy(B_bi_true, B_bi_est_t, B_lags_true, B_lag_est_t)
        log_acc_metrics(acc_bi, run, f'bi_{metric_infix}')
        print(metric_infix)
        print(acc_all)

    assert best_W is not None

    run.log_metric('best_threshold', best_t)

    return best_W, best_Wbi, best_A


def _count_accuracy_stats(B_true, B_est):
    #d = B_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_est == 1)
    positive = np.flatnonzero(B_true)
    # cond_reversed = np.flatnonzero(B_true.T)
    # cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, positive, assume_unique=True)
    # treat undirected edge favorably
    #true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    #true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, positive, assume_unique=True)
    # false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    # false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    #extra = np.setdiff1d(pred, positive, assume_unique=True)
    #reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = B_true.shape[0] * B_true.shape[1] - len(positive)
    fdr = float(len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(positive), 1)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)
    precision = len(true_pos) / max((len(true_pos) + len(false_pos)), 1)
    f1 = 2 * (tpr*precision)/max((tpr+precision), 1)
    g_score = max((len(true_pos)-len(false_pos),0))/max(len(positive), 1)
    return {
        'fdr': fdr,
        'tpr': tpr, # recall, sensitivity
        'fpr': fpr,
        'nnz': pred_size,
        'true_pos': len(true_pos),
        'false_pos': len(false_pos),
        'false_neg': len(positive) - len(true_pos),
        'cond': len(positive),
        'precision': precision,
        'f1score': f1,
        'g_score': g_score
    }


def count_accuracy(B_true, B_est, B_lags_true, B_lags_est, test_dag=True):
    assert len(B_lags_true) == len(B_lags_est)
    shd, w_shd, a_shd = calculate_shd(B_true, B_est, B_lags_true, B_lags_est, test_dag=test_dag)

    B_true_dir = np.where(B_true == 1, B_true, 0)
    B_true_bidir = np.where(B_true == -1, -B_true, 0)
    B_true_bidir = np.triu(B_true_bidir, k=1)
    B_true_bidir2 = np.where(B_true == -2, (-1/2 *B_true), 0)
    B_true_bidir2 = np.triu(B_true_bidir2, k=1)

    B_est_dir = np.where(B_est == 1, B_est, 0)
    B_est_bidir = np.where(B_est == -1, -B_est, 0)
    B_est_bidir = np.triu(B_est_bidir, k=1)
    B_est_bidir2 = np.where(B_est == -2, (-1/2 * B_est), 0)
    B_est_bidir2 = np.triu(B_est_bidir2, k=1)

    m_true = np.copy(B_true_dir)
    m_est = np.copy(B_est_dir)


    m_est = np.concatenate([m_est, B_est_bidir, B_est_bidir2] + B_lags_est, axis=0)
    m_true = np.concatenate([m_true, B_true_bidir, B_true_bidir2] + B_lags_true, axis=0)

    #norm_dist = norm(m_est - m_true)
    assert np.all(np.isin(m_true, [0, 1]))
    assert np.all(np.isin(m_est, [0, 1]))


    acc = _count_accuracy_stats(m_true, m_est)
    acc['shd'] = shd
    acc['shd_w'] = w_shd
    acc['shd_a'] = a_shd
    #acc['norm_distance'] = norm_dist
    return acc


def compute_norm_distance(W_true, W_est, A_true, A_est):
    m_true = np.copy(W_true)
    m_est = np.copy(W_est)
    m_est = np.concatenate([m_est] + A_est, axis=0)
    m_true = np.concatenate([m_true] + A_true, axis=0)

    norm_dist = norm(m_est - m_true)
    return norm_dist


def least_square_cost(X, W, Y, A):
    n, d = X.shape
    p = len(A)
    assert len(Y) == len(A)
    val = sum((X[i,j] - sum(X[i, k] * W[k, j] for k in range(d) if k != j) - sum(Y[t][i, k] * A[t][k, j] for k in range(d) for t in range(p)))**2 for i in range(n) for j in range(d))
    return val


def apply_threshold(W, w_threshold):
    W_t = np.copy(W)
    W_t[np.abs(W) < w_threshold] = 0
    return W_t


def find_optimal_threshold_for_shd(B_true, W_est, B_lags_true, A_est, W_bi_true, W_bi_est):
    values = set((abs(t) for t in W_est.flatten() if abs(t) > 0))
    for A_i_est in A_est:
        values.update((abs(t) for t in A_i_est.flatten() if abs(t) > 0))

    possible_thresholds = values #sorted((abs(t) for t in W_est.flatten() if abs(t) > 0))
    if not possible_thresholds:
        possible_thresholds = [0]

    best_t = max(possible_thresholds) if possible_thresholds else 0
    best_shd = B_true.shape[0] ** 2 # calculate_shd(W_true, W_est != 0, A_true, A_est) # W_true.shape[0]**2
    B_bi_true = W_bi_true != 0
    B_all_true = B_true - B_bi_true
    for t_candidate in possible_thresholds:
        W_est_t = apply_threshold(W_est, t_candidate)
        A_est_t = [apply_threshold(A_i_est, t_candidate) for A_i_est in A_est]
        W_bi_est_t = apply_threshold(W_bi_est, t_candidate)
        B_bi_est_t = (W_bi_est_t != 0)
        B_est_t = W_est_t != 0
        B_all_est_t = B_est_t + (-1 * B_bi_est_t) # CPDAG - undirected edges have -1
        shd, _, _ = calculate_shd(B_all_true, B_all_est_t, B_lags_true, A_est_t)

        if shd < best_shd:
            best_t = t_candidate
            best_shd = shd
    return best_t, best_shd
