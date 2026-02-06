from loguru import logger
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import kendalltau

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency for XGB mode only
    XGBRegressor = None  # type: ignore[assignment]

@dataclass
class SelectionResult:
    model: object
    train_mse: float
    test_mse: float
    train_ratio: float
    test_ratio: float
    test_kendalltau: float


def compute_predictor_errors(
    prep_data: pd.DataFrame,
    hei_feats: list[str],
    target_col: str,
    *,
    model_name: str,
    custom_objective: str,
    random_seed: int = 42,
) -> SelectionResult:
    """Port of project_hei predictor evaluation for REG/XGB."""
    assert model_name in {"REG", "XGB"}
    assert custom_objective in {"lagrange", "mse_builtin"}

    reg_dat = prep_data[hei_feats + [target_col]].dropna()
    x = reg_dat[hei_feats].to_numpy()
    y = reg_dat[target_col].to_numpy()

    if x.shape[0] < 8:
        raise ValueError(f"Not enough samples for {target_col}: {x.shape[0]}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=random_seed, shuffle=True
    )

    def custom_mse_obj(y_true: np.ndarray, y_pred: np.ndarray):
        grad = y_pred - y_true
        hess = np.ones_like(y_pred)
        return grad, hess

    if model_name == "XGB":
        if XGBRegressor is None:
            raise ImportError("xgboost is required for model_name='XGB'. Install package 'xgboost'.")
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=20,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=random_seed,
                        base_score=float(y_train.mean()),
                        objective=custom_mse_obj if custom_objective == "lagrange" else "reg:squarederror",
                    ),
                ),
            ]
        )
    else:
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("linreg", LinearRegression()),
            ]
        )

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_mse = float(mean_squared_error(y_train, y_train_pred))
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    train_bench = float(mean_squared_error(y_train, np.ones_like(y_train) * y_train.mean()))
    test_bench = float(mean_squared_error(y_test, np.ones_like(y_test) * y_train.mean()))
    train_ratio = train_mse / max(train_bench, 1e-12)
    test_ratio = test_mse / max(test_bench, 1e-12)
    test_tau = float(kendalltau(y_test, y_test_pred, nan_policy="omit").statistic)
    if np.isnan(test_tau):
        test_tau = 0.0

    return SelectionResult(
        model=model,
        train_mse=train_mse,
        test_mse=test_mse,
        train_ratio=float(train_ratio),
        test_ratio=float(test_ratio),
        test_kendalltau=test_tau,
    )


def get_mean_average_errors(
    prep_data: pd.DataFrame,
    run_feats: list[str],
    target_col: str,
    model_name: str,
    custom_objective: str,
    n_runs: int,
    seed: int = 42,
) -> tuple[float, float]:
    run_res: list[SelectionResult] = []
    for i in range(n_runs):
        run_res.append(
            compute_predictor_errors(
                prep_data,
                run_feats,
                target_col,
                model_name=model_name,
                custom_objective=custom_objective,
                random_seed=seed + i,
            )
        )
    train_ratio_mean = float(np.mean([v.train_ratio for v in run_res]))
    test_ratio_mean = float(np.mean([v.test_ratio for v in run_res]))
    return train_ratio_mean, test_ratio_mean


def run_feature_selection(
    prep_data: pd.DataFrame,
    *,
    model_name: str,
    custom_objective: str,
    target_col: str,
    n_runs: int,
    n_features: int,
    full_features: list[str],
    seed: int = 42,
) -> tuple[list[str], list[float], list[float], SelectionResult | None]:
    """Forward feature selection based on mean test variance ratio."""
    curr_feats: list[str] = []
    curr_test_errs: list[float] = []
    curr_train_errs: list[float] = []
    curr_test_error = 1e9
    best_final: SelectionResult | None = None

    logger.info(f"====== Target: {target_col}")
    for i in range(int(n_features)):
        candidate_feats = [s for s in full_features if s not in curr_feats]
        if not candidate_feats:
            break

        res: list[tuple[str, float, float]] = []
        for feat in candidate_feats:
            train_ratio_mean, test_ratio_mean = get_mean_average_errors(
                prep_data,
                curr_feats + [feat],
                target_col,
                model_name,
                custom_objective,
                int(n_runs),
                seed=seed,
            )
            res.append((feat, train_ratio_mean, test_ratio_mean))

        res = sorted(res, key=lambda t: t[2])
        best_feat, best_train_error, best_test_error = res[0]

        if best_test_error >= curr_test_error:
            logger.info("Failed to improve further, stopping feature selection.")
            break

        curr_feats.append(best_feat)
        curr_train_errs.append(float(best_train_error))
        curr_test_errs.append(float(best_test_error))
        curr_test_error = float(best_test_error)

        best_final = compute_predictor_errors(
            prep_data,
            curr_feats,
            target_col,
            model_name=model_name,
            custom_objective=custom_objective,
            random_seed=seed,
        )
        logger.info(
            f"Round {i}: selected={best_feat}, train_var_reduction={100 * (1.0 - best_train_error):.3f}, "
            f"test_var_reduction={100 * (1.0 - best_test_error):.3f}"
        )

    return curr_feats, curr_train_errs, curr_test_errs, best_final


def model_factory(d: int):
    """
    Backward-compatible factory from mark-ori notebook.
    Returns (model_class, model_params) for Gaussian Process pipeline.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    kernel = RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
    model_class = Pipeline
    model_params = {
        "steps": [
            ("scale", StandardScaler()),
            (
                "GP",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1.0,
                    normalize_y=True,
                    n_restarts_optimizer=3,
                ),
            ),
        ]
    }
    return model_class, model_params
