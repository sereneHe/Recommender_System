import numpy as np

from notears import utils as notears_utils


# def compute_shd(W_true, W_est):
#     acc = notears_utils.count_accuracy(W_true != 0, W_est != 0)
#     return acc['shd'], acc


def calculate_dag_shd(B_true, B_est, test_dag=True):
    # this needs some refactoring
    assert B_true.shape == B_est.shape
    if (B_est == -2).any() or (B_est == -3).any() or (B_est == 2).any(): # PAG - MAG markov equivalence class
        pass
    elif (B_est == -1).any():  # cpdag or mag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        # if ((B_est == -1) & (B_est.T == -1)).any():
        #     raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if test_dag and not notears_utils.is_dag(B_est):
            raise ValueError('B_est should be a DAG')

    shd = 0
    for i in range(B_true.shape[0]):
        for j in range(i):
            e_ij = (B_est[i,j], B_est[j,i])
            # Ensure that undirected edges are symmetrical - maybe transorm it into asserts?
            if min(e_ij) == -1:
                e_ij = (-1,-1) # <->

            if min(e_ij) == -2:
                e_ij = (-2,-2) # ---

            if min(e_ij) == -3:
                e_ij = (-3,-3) # o-o

            t_ij = (B_true[i,j], B_true[j,i])
            if min(t_ij) == -1:
                t_ij = (-1,-1)
            if min(t_ij) == -2:
                t_ij = (-2,-2) # ---

            if min(t_ij) == -3:
                t_ij = (-3,-3) # o-o

            # PAG modifications


            def check_if_e_is_compatible_to_t(e_ij, t_ij):
                if e_ij == t_ij:
                    return 0
                elif e_ij == (1,0) and (t_ij == (2,0) or t_ij == (-3,-3)):
                    return 0

                elif e_ij == (2,0) and (t_ij == (-3,-3) or t_ij == (-1,-1) or t_ij == (0,2)):
                    return 0

                elif e_ij == (-3,-3) and (t_ij == (-1,-1) or t_ij == (-2,-2)):
                    return 0

                elif e_ij == (1,0) and (t_ij == (0,1) or t_ij == (0,2) or t_ij == (-1,-1) or t_ij == (-2,-2)):
                    return 0.5

                elif e_ij == (2,0) and (t_ij == (0,1) or t_ij == (-2,-2)):
                    return 0.5
                elif e_ij == (-1,-1) and t_ij == (-2,-2):
                    return 0.5
                else:
                    return 1




            shd1 = check_if_e_is_compatible_to_t(e_ij, t_ij)
            shd2 = check_if_e_is_compatible_to_t(t_ij, e_ij)

            shd3 = check_if_e_is_compatible_to_t(e_ij[::-1], t_ij[::-1])
            shd4 = check_if_e_is_compatible_to_t(t_ij[::-1], e_ij[::-1])
            shd += min(shd1, shd2, shd3, shd4)

            # if e_ij != t_ij:
            #     if e_ij == t_ij[::-1]:
            #         shd += 0.5
            #     elif (e_ij == (-1,-1) and t_ij == (0,0)) or (e_ij == (0,0) and t_ij == (-1,-1)):
            #         shd += 1
            #     elif e_ij == (-1, -1) or t_ij == (-1, -1):
            #         shd += 0.5
            #     else:
            #         shd += 1

    return shd


def calculate_dag_shd_old(B_true, B_est, test_dag=True):
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if test_dag and not notears_utils.is_dag(B_est):
            raise ValueError('B_est should be a DAG')

    d = B_true.shape[0]
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    #cond_skeleton = np.concatenate([cond, cond_reversed])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)

    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return shd


def calculate_shd(B_true, B_est, B_lags_true, A_est, test_dag=True):
    shd = calculate_dag_shd(B_true, B_est, test_dag=test_dag)
    a_shd = 0
    for i in range(len(B_lags_true)):
        a_i_shd = calculate_dag_shd(B_lags_true[i] != 0, A_est[i] != 0, test_dag=False)
        a_shd += a_i_shd
    return shd + a_shd, shd, a_shd


if __name__ == '__main__':
    n = 10
    B = np.triu(np.random.randint(2, size=(n, n)), k=1)
    print(f"Number of non-zero entries in B: {np.count_nonzero(B)}")
    print(calculate_shd(B, B, [], []))
    B_est = np.zeros_like(B)
    print(calculate_shd(B, B_est, [], []))

    B = -1 * B
    print(calculate_shd(B, B_est, [], []))

    B = np.zeros((n, n))
    B[0, 1] = -1
    print(calculate_shd(B, B_est, [], []))

    B[0, 1] = -1
    B_est[1, 0] = 1
    print(calculate_shd(B, B_est, [], []))

    B[0, 1] = 1
    B_est[1, 0] = 1
    print(calculate_shd(B, B_est, [], []))

    B = np.zeros((n, n))
    B_est = np.zeros((n, n))
    B[0, 1] = -1
    B_est[1, 0] = -1
    print(calculate_shd(B, B_est, [], []))

    B = np.zeros((n, n))
    B_est = np.zeros((n, n))
    B[0, 1] = -1
    B_est[1, 0] = 1
    print(calculate_shd(B, B_est, [], []))
