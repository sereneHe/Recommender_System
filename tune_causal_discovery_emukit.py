"""
Bayesian optimization for tuning causal discovery hyperparameters with Emukit.

Example use case:
    Tune hyperparameters of a causal discovery algorithm by minimizing SHD,
    BIC score, validation loss, or another causal-discovery metric.

Install:
    pip install emukit GPy numpy pandas scikit-learn
"""

import json
import numpy as np

import GPy
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement


# ============================================================
# 1. Replace this with your own causal discovery algorithm
# ============================================================

def run_causal_discovery(X, lambda_reg, threshold, alpha):
    """
    Placeholder causal discovery function.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_variables)
        Observed data.
    lambda_reg : float
        Regularization strength.
    threshold : float
        Edge threshold.
    alpha : float
        Test level / penalty / algorithm-specific parameter.

    Returns
    -------
    adjacency : np.ndarray, shape (p, p)
        Estimated adjacency matrix.
        adjacency[i, j] = 1 means i -> j.
    """

    p = X.shape[1]

    # --------------------------------------------------------
    # Replace this part with your real algorithm.
    #
    # Examples:
    #   adjacency = notears_linear(X, lambda1=lambda_reg, w_threshold=threshold)
    #   adjacency = pc_algorithm(X, alpha=alpha)
    #   adjacency = ges_algorithm(X, penalty=lambda_reg)
    # --------------------------------------------------------

    # Dummy example: random sparse graph, only for script testing.
    rng = np.random.default_rng(123)
    W = rng.normal(size=(p, p))
    np.fill_diagonal(W, 0.0)

    adjacency = (np.abs(W) > threshold).astype(int)

    # Remove lower-triangular edges just to avoid cycles in this dummy example.
    adjacency = np.triu(adjacency, k=1)

    return adjacency


# ============================================================
# 2. Define evaluation metric
# ============================================================

def structural_hamming_distance(A_hat, A_true):
    """
    Simple SHD between two adjacency matrices.

    Parameters
    ----------
    A_hat : np.ndarray, shape (p, p)
        Estimated adjacency matrix.
    A_true : np.ndarray, shape (p, p)
        Ground-truth adjacency matrix.

    Returns
    -------
    shd : int
        Number of edge additions/deletions/orientation mismatches.
    """
    return int(np.sum(A_hat != A_true))


def evaluate_hyperparams(X, A_true, lambda_reg, threshold, alpha):
    """
    Objective function to minimize.

    You can replace SHD with:
        - BIC
        - validation negative log-likelihood
        - held-out prediction error
        - SID
        - custom graph score
    """

    A_hat = run_causal_discovery(
        X=X,
        lambda_reg=lambda_reg,
        threshold=threshold,
        alpha=alpha,
    )

    shd = structural_hamming_distance(A_hat, A_true)

    return float(shd)


# ============================================================
# 3. Emukit objective wrapper
# ============================================================

def make_emukit_objective(X, A_true, log_path="bo_results.jsonl"):
    """
    Emukit expects objective input X_bo with shape (n_points, n_params)
    and output Y_bo with shape (n_points, 1).
    """

    def objective(X_bo):
        results = []

        for row in X_bo:
            # Parameters are passed in the same order as ParameterSpace.
            log10_lambda = float(row[0])
            threshold = float(row[1])
            alpha = float(row[2])

            lambda_reg = 10.0 ** log10_lambda

            loss = evaluate_hyperparams(
                X=X,
                A_true=A_true,
                lambda_reg=lambda_reg,
                threshold=threshold,
                alpha=alpha,
            )

            record = {
                "log10_lambda": log10_lambda,
                "lambda_reg": lambda_reg,
                "threshold": threshold,
                "alpha": alpha,
                "loss": loss,
            }

            print(record)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            results.append([loss])

        return np.array(results)

    return objective


# ============================================================
# 4. Main Bayesian optimization routine
# ============================================================

def bayesopt_causal_discovery(
    X,
    A_true,
    n_initial_points=8,
    n_bo_iterations=25,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)

    # --------------------------------------------------------
    # Search space
    #
    # lambda_reg is optimized on log10 scale:
    #     log10_lambda in [-4, 1] means lambda in [1e-4, 10]
    #
    # threshold controls edge pruning.
    #
    # alpha can represent PC test level, penalty factor,
    # or any continuous hyperparameter.
    # --------------------------------------------------------

    space = ParameterSpace(
        [
            ContinuousParameter("log10_lambda", -4.0, 1.0),
            ContinuousParameter("threshold", 0.01, 1.0),
            ContinuousParameter("alpha", 0.001, 0.2),
        ]
    )

    objective = make_emukit_objective(X, A_true)

    # --------------------------------------------------------
    # Initial random design
    # --------------------------------------------------------

    X_init = space.sample_uniform(n_initial_points)
    Y_init = objective(X_init)

    # --------------------------------------------------------
    # Gaussian Process surrogate model
    # --------------------------------------------------------

    input_dim = X_init.shape[1]

    kernel = GPy.kern.Matern52(input_dim=input_dim, ARD=True)
    gp = GPy.models.GPRegression(X_init, Y_init, kernel)

    # SHD or validation loss can be noisy, so keep noise nonzero.
    gp.Gaussian_noise.variance = 1e-3
    gp.Gaussian_noise.variance.constrain_bounded(1e-6, 1.0)

    model = GPyModelWrapper(gp)

    # Expected Improvement is a standard acquisition function for BO.
    acquisition = ExpectedImprovement(model)

    loop = BayesianOptimizationLoop(
        space=space,
        model=model,
        acquisition=acquisition,
    )

    loop.run_loop(objective, n_bo_iterations)

    # --------------------------------------------------------
    # Extract best result
    # --------------------------------------------------------

    X_all = loop.loop_state.X
    Y_all = loop.loop_state.Y

    best_idx = int(np.argmin(Y_all))
    best_x = X_all[best_idx]
    best_y = float(Y_all[best_idx, 0])

    best_params = {
        "log10_lambda": float(best_x[0]),
        "lambda_reg": float(10.0 ** best_x[0]),
        "threshold": float(best_x[1]),
        "alpha": float(best_x[2]),
        "best_loss": best_y,
    }

    return best_params, X_all, Y_all


# ============================================================
# 5. Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Replace this with your real data.
    #
    # X shape:
    #     n_samples x n_variables
    #
    # For EEG causal discovery, X might be:
    #     windows x channels
    # or:
    #     samples x brain_regions
    # depending on your preprocessing.
    # --------------------------------------------------------

    rng = np.random.default_rng(0)

    n_samples = 300
    p = 8

    X = rng.normal(size=(n_samples, p))

    # Dummy ground-truth DAG for testing.
    A_true = np.zeros((p, p), dtype=int)
    A_true[0, 1] = 1
    A_true[1, 2] = 1
    A_true[0, 3] = 1
    A_true[4, 5] = 1

    best_params, X_all, Y_all = bayesopt_causal_discovery(
        X=X,
        A_true=A_true,
        n_initial_points=8,
        n_bo_iterations=25,
        random_seed=42,
    )

    print("\nBest hyperparameters:")
    print(json.dumps(best_params, indent=2))

    np.save("bo_X_all.npy", X_all)
    np.save("bo_Y_all.npy", Y_all)

    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)


#def run_causal_discovery(X, lambda_reg, threshold, alpha):
# A_hat = notears_linear(
# X,
# lambda1=lambda_reg,
# w_threshold=threshold,
# )
# A_hat = pc_algorithm(
# X,
# alpha=alpha,
# )

# loss = bic_score(X, A_hat)
# loss = validation_negative_log_likelihood(X_train, X_val, A_hat)
# bo_results.jsonl      每一次 BO 评估记录
# best_params.json      最优超参数
# bo_X_all.npy          所有尝试过的超参数
# bo_Y_all.npy          所有对应 loss