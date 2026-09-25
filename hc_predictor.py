"""Base HC neural predictor with selectable ALM/SPBM training.

Runtime switches are intentionally environment variables so cluster shell scripts
can select the implementation without introducing another Hydra sweep dimension.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from humancompatible.train.dual_optim import MoreauEnvelope, PBM
from nn_lagrangian import (
    MLPRegressor,
    W_constraint,
    fit_aug_lagrangian_nn_constraint as fit_alm_nn_constraint,
    make_regression_loss,
)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}."
    )


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def get_constraint_backend() -> str:
    backend = os.getenv("HC_CONSTRAINT_BACKEND", "alm").strip().lower()
    if backend == "pbm":
        backend = "spbm"
    if backend not in {"alm", "spbm"}:
        raise ValueError(
            "HC_CONSTRAINT_BACKEND must be 'alm' or 'spbm' "
            f"(alias 'pbm' is accepted), got {backend!r}."
        )
    return backend


def weibull_gaussianization_enabled() -> bool:
    return _env_bool("HC_WEIBULL_GAUSSIANIZE", default=False)


def _dweibull_cdf(values: np.ndarray, q_alpha: float, beta: float) -> np.ndarray:
    """Numerically stable shifted discrete-Weibull CDF."""
    values = np.asarray(values, dtype=np.float64)
    cdf = np.zeros_like(values)
    valid = values >= 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exponent = np.power(values[valid] + 1.0, beta)
        cdf[valid] = -np.expm1(np.log(q_alpha) * exponent)
    return np.clip(np.nan_to_num(cdf, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _estimate_dweibull_column(values: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import minimize

    values = np.asarray(values, dtype=np.float64)

    def nll(params):
        q_alpha, beta = params
        if not (0.0 < q_alpha < 1.0 and beta > 0.0):
            return 1e100
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            term1 = np.power(q_alpha, np.power(values, beta))
            term2 = np.power(q_alpha, np.power(values + 1.0, beta))
            pmf = term1 - term2
        if not np.all(np.isfinite(pmf)):
            return 1e100
        pmf = np.clip(pmf, 1e-12, 1.0)
        return float(-np.log(pmf).sum())

    result = minimize(
        nll,
        x0=np.array([0.5, 1.0]),
        bounds=((1e-5, 1.0 - 1e-5), (1e-5, 50.0)),
        method="L-BFGS-B",
    )
    if result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
        return float(result.x[0]), float(result.x[1])
    logging.warning("Discrete-Weibull fit failed; using q_alpha=0.5, beta=1.0.")
    return 0.5, 1.0


def weibull_gaussianize_table(table, *, mode: str | None = None, seed: int | None = None,
                              require_count_support: bool = True):
    """Gaussianize non-negative integer COUNT columns through a Weibull PIT.

    The discrete-Weibull PIT is only defined on non-negative integer support.
    Continuous or signed columns (e.g. FRED growth rates) are therefore
    rejected rather than silently shifted; use a continuous distribution or an
    empirical copula for those.  Set ``HC_WEIBULL_ALLOW_NONCOUNT=1`` (or pass
    ``require_count_support=False``) to override explicitly.

    NOTE: the transform is currently fitted on the whole supplied table.  For a
    leakage-free run it must be fitted inside each training fold only; callers
    that pass a full dataset before cross-validation are responsible for making
    the fit fold-local before any conclusion is drawn from it.
    """
    import pandas as pd
    from scipy.stats import norm

    if require_count_support and _env_bool("HC_WEIBULL_ALLOW_NONCOUNT", default=False):
        require_count_support = False

    is_frame = isinstance(table, pd.DataFrame)
    columns = table.columns if is_frame else None
    index = table.index if is_frame else None
    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Weibull Gaussianization expects a 2-D table, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))[0]
        column = columns[int(bad[1])] if columns is not None else int(bad[1])
        raise ValueError(
            "Weibull Gaussianization received NaN/Inf before CDF evaluation: "
            f"row={int(bad[0])}, column={column!r}. Clean missing values first."
        )

    selected_mode = (mode or os.getenv("HC_WEIBULL_MODE", "rand")).strip().lower()
    if selected_mode not in {"mid", "rand"}:
        raise ValueError("HC_WEIBULL_MODE must be 'mid' or 'rand'.")
    random_seed = seed if seed is not None else _env_int("HC_WEIBULL_RANDOM_SEED", 42)
    rng = np.random.default_rng(random_seed)
    transformed = np.zeros_like(values)

    for column_index in range(values.shape[1]):
        column_values = values[:, column_index]
        if np.ptp(column_values) == 0.0:
            continue
        if require_count_support:
            finite = column_values[np.isfinite(column_values)]
            if np.any(finite < 0) or not np.all(np.equal(np.mod(finite, 1.0), 0.0)):
                name = columns[column_index] if columns is not None else column_index
                raise ValueError(
                    "Weibull Gaussianization is defined for non-negative integer count "
                    f"columns only; column {name!r} is continuous or signed. Use a "
                    "continuous distribution / empirical copula instead, or set "
                    "HC_WEIBULL_ALLOW_NONCOUNT=1 to override explicitly."
                )
        shifted = column_values - min(0.0, float(column_values.min()))
        q_alpha, beta = _estimate_dweibull_column(shifted)
        upper = _dweibull_cdf(shifted, q_alpha, beta)
        lower = _dweibull_cdf(shifted - 1.0, q_alpha, beta)
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        if np.any(lower > upper + 1e-12):
            name = columns[column_index] if columns is not None else column_index
            raise ValueError(f"Non-monotone Weibull CDF bounds in column {name!r}.")
        if selected_mode == "mid":
            probabilities = 0.5 * (lower + upper)
        else:
            # Clip before uniform(): NumPy rejects NaN/Inf bounds before a later clip.
            probabilities = rng.uniform(lower, np.maximum(lower, upper))
        probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
        transformed[:, column_index] = norm.ppf(probabilities)

    if is_frame:
        return pd.DataFrame(transformed, index=index, columns=columns)
    return transformed


def fit_spbm_nn_constraint(
    X, y, W, cfg, verbose=False, device="cpu", X_val=None, y_val=None,
):
    """Algorithm 1 SPBM using humancompatible.train's PBM implementation."""
    del X_val, y_val
    torch.set_default_dtype(torch.float32)

    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, device=device).reshape(-1)
    W = torch.tensor(np.asarray(W), dtype=torch.float32, device=device)
    _, d = X.shape
    if W.shape != (d + 1, d + 1):
        raise ValueError(f"W must be {(d + 1, d + 1)}, got {tuple(W.shape)}.")

    model = MLPRegressor(
        input_dim=d,
        hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth),
    ).to(device)

    batch_size = min(_env_int("HC_SPBM_BATCH_SIZE", 128), len(X))
    if batch_size <= 0:
        raise ValueError("HC_SPBM_BATCH_SIZE must be positive and training data cannot be empty.")
    generator = torch.Generator()
    generator.manual_seed(_env_int("HC_SPBM_RANDOM_SEED", 42))
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    base_optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    center_decay = _env_float("HC_SPBM_CENTER_DECAY", 0.9)
    if not 0.0 <= center_decay < 1.0:
        raise ValueError("HC_SPBM_CENTER_DECAY must be in [0, 1).")
    primal_optimizer = MoreauEnvelope(
        base_optimizer,
        mu=_env_float("HC_SPBM_PROX_MU", 2.0),
        beta=1.0 - center_decay,
    )

    dual_optimizer = PBM(
        m=d + 1,
        penalty_mult=_env_float("HC_SPBM_PENALTY_MULT", 0.9),
        gamma=_env_float("HC_SPBM_GAMMA", 0.9),
        delta=_env_float("HC_SPBM_ADAPT_DELTA", 1.0),
        penalty_update=os.getenv("HC_SPBM_PENALTY_UPDATE", "dimin_adapt"),
        pbf=os.getenv("HC_SPBM_PBF", "quadratic_logarithmic"),
        init_duals=_env_float("HC_SPBM_INIT_DUALS", float(cfg.lambda0)),
        init_penalties=_env_float("HC_SPBM_INIT_PENALTIES", 1.0),
        dual_range=(
            _env_float("HC_SPBM_DUAL_MIN", 1e-4),
            _env_float("HC_SPBM_DUAL_MAX", 100.0),
        ),
        penalty_range=(
            _env_float("HC_SPBM_PENALTY_MIN", 0.1),
            _env_float("HC_SPBM_PENALTY_MAX", 1.0),
        ),
        primal_update_process_length=1,
        device=device,
    )

    M = W - torch.eye(d + 1, dtype=W.dtype, device=device)
    tolerance = _env_float("HC_SPBM_CONSTRAINT_TOLERANCE", 1e-3)
    loss_fn = make_regression_loss(cfg)
    total_steps = int(cfg.n_outer) * int(cfg.n_inner)
    loader_iterator = iter(loader)

    logging.info(
        "HC constraint backend=spbm, steps=%d, batch_size=%d, gamma=%g, "
        "penalty_update=%s, weibull_gaussianize=%s",
        total_steps,
        batch_size,
        _env_float("HC_SPBM_GAMMA", 0.9),
        os.getenv("HC_SPBM_PENALTY_UPDATE", "dimin_adapt"),
        weibull_gaussianization_enabled(),
    )

    for step in range(total_steps):
        try:
            X_batch, y_batch = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            X_batch, y_batch = next(loader_iterator)

        base_optimizer.zero_grad(set_to_none=True)
        predictions = model(X_batch)
        mse = loss_fn(predictions, y_batch)
        if bool(cfg.constrained):
            xy_mean = torch.cat((X_batch.mean(dim=0), predictions.mean().reshape(1)))
            residual = M @ xy_mean
            constraints = residual.abs() - tolerance
            objective = dual_optimizer.forward_update(mse, constraints)
        else:
            constraints = torch.zeros(d + 1, dtype=X.dtype, device=device)
            objective = mse
        objective.backward()
        primal_optimizer.step()

        if verbose and step % max(1, total_steps // 10) == 0:
            print(
                f"step={step:04d}  MSE={mse.item():.6e}  "
                f"max(g)={constraints.max().item():+.6e}  "
                f"lambda_norm={dual_optimizer.duals.norm().item():.6e}  "
                f"p=[{dual_optimizer.penalties.min().item():.3e}, "
                f"{dual_optimizer.penalties.max().item():.3e}]"
            )

    model.spbm_duals_ = dual_optimizer.duals.detach().cpu()
    model.spbm_penalties_ = dual_optimizer.penalties.detach().cpu()
    return model, model.spbm_duals_.numpy()


def fit_aug_lagrangian_nn_constraint(
    X, y, W, cfg, verbose=False, device="cpu", X_val=None, y_val=None,
):
    """Stable public entry point used by ``recommender_estimator``."""
    backend = get_constraint_backend()
    if backend == "alm":
        logging.info(
            "HC constraint backend=alm, weibull_gaussianize=%s",
            weibull_gaussianization_enabled(),
        )
        model, lam = fit_alm_nn_constraint(
            X,
            y,
            W,
            cfg,
            verbose=verbose,
            device=device,
            X_val=X_val,
            y_val=y_val,
        )
        d = np.asarray(X).shape[1]
        model.constraint_counts_ = {
            "backend": backend,
            "independent_constraints": 0,
            "dependent_constraints": 0,
            "alm_constraints": int(d + 1 if bool(cfg.constrained) else 0),
            "pbm_constraints": 0,
            "w_constraints": int(d + 1 if bool(cfg.constrained) else 0),
            "total_constraints": int(d + 1 if bool(cfg.constrained) else 0),
        }
        return model, lam

    model, lam = fit_spbm_nn_constraint(
        X,
        y,
        W,
        cfg,
        verbose=verbose,
        device=device,
        X_val=X_val,
        y_val=y_val,
    )
    d = np.asarray(X).shape[1]
    model.constraint_counts_ = {
        "backend": backend,
        "independent_constraints": 0,
        "dependent_constraints": 0,
        "alm_constraints": 0,
        "pbm_constraints": int(d + 1 if bool(cfg.constrained) else 0),
        "w_constraints": int(d + 1 if bool(cfg.constrained) else 0),
        "total_constraints": int(d + 1 if bool(cfg.constrained) else 0),
    }
    return model, lam


__all__ = [
    "MLPRegressor",
    "W_constraint",
    "fit_aug_lagrangian_nn_constraint",
    "fit_spbm_nn_constraint",
    "get_constraint_backend",
    "weibull_gaussianization_enabled",
    "weibull_gaussianize_table",
]
