"""Bayesian tuning of the real HC predictor with fold-local CV.

The former version of this file optimised SHD for a fixed random dummy DAG.
This version tunes parameters that are actually consumed by the project's
plain HC implementation and minimises the same normalized outer-CV MSE used
by ``run_experiments.py``.

The repository environment does not currently contain GPy/Emukit, so the
Bayesian loop uses scikit-learn's Gaussian-process regressor and expected
improvement.  The objective and fold handling are independent of that choice.

Example (run on the cluster with the current SVN problem):
    python tune_causal_discovery_emukit.py --target SVN \
        --n-initial-points 2 --n-bo-iterations 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error

from industry_utils import load_data
from recommender_utils import (
    _cross_validate_hc_with_fold_local_dag,
    _make_site_gender_cv_splits,
    create_model,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent
SOLVER_CONFIG = REPO_ROOT / "experiments_conf" / "solver" / "hc_predictor.yaml"
PROBLEM_CONFIG_TEMPLATE = (
    REPO_ROOT
    / "experiments_conf"
    / "problem"
    / "FRED_16country_monthly"
    / "industry_eu_svn.yaml"
)

# Parameter order in the BO vector. All four are used by solve_milp or the HC
# optimizer; unlike the old alpha placeholder, none is a no-op.
PARAMETERS = (
    "learning_rate",
    "weight_decay",
    "lambda1",
    "nonzero_threshold",
)

# Log10 bounds. The known SVN run is lr=.25, wd=.25, lambda1=.1,
# nonzero_threshold=.001 and is included explicitly as the first observation.
BO_BOUNDS = np.asarray(
    [
        [math.log10(0.01), math.log10(0.25)],
        [math.log10(0.01), math.log10(1.0)],
        [math.log10(0.01), math.log10(1.0)],
        [math.log10(0.0001), math.log10(0.05)],
    ],
    dtype=float,
)
KNOWN_BASELINE = np.log10([0.25, 0.25, 0.1, 0.001])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="SVN")
    parser.add_argument(
        "--problem-config",
        type=Path,
        default=PROBLEM_CONFIG_TEMPLATE,
        help="Hydra problem YAML used to load the data.",
    )
    parser.add_argument("--solver-config", type=Path, default=SOLVER_CONFIG)
    parser.add_argument("--n-initial-points", type=int, default=2)
    parser.add_argument("--n-bo-iterations", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--target-mip-gap", type=float, default=0.13)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results/bo_hc_cv_results.jsonl"),
    )
    parser.add_argument(
        "--best-path",
        type=Path,
        default=Path("results/best_hc_cv_params.json"),
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_context(args: argparse.Namespace):
    solver_cfg = OmegaConf.load(_absolute_path(args.solver_config))
    OmegaConf.resolve(solver_cfg)
    solver_cfg.recalculate_dag = True
    solver_cfg.time_limit = float(args.time_limit)
    solver_cfg.target_mip_gap = float(args.target_mip_gap)
    solver_cfg.n_runs = int(args.n_runs)
    solver_cfg.random_state = int(args.random_seed)
    solver_cfg.cv_random_state = int(args.random_seed)

    problem_cfg = OmegaConf.load(_absolute_path(args.problem_config))
    data_path = _absolute_path(Path(str(problem_cfg.data_path)))
    target = str(args.target)
    features = [str(feature) for feature in problem_cfg.features if str(feature) != target]
    prep_data = load_data(
        data_path=data_path,
        frequency=problem_cfg.get("frequency"),
        target=target,
        features=features,
        start_date=problem_cfg.get("start_date"),
        end_date=problem_cfg.get("end_date"),
        impute=problem_cfg.get("impute", "none"),
        dropna_selected=problem_cfg.get("dropna_selected", True),
    )
    prep_data = prep_data.dropna(subset=[target]).dropna(axis=1)
    features = [feature for feature in features if feature in prep_data.columns]
    X = prep_data[features]
    y = prep_data[target]
    cv_splits, cv_kind = _make_site_gender_cv_splits(prep_data, target, args.n_runs, solver_cfg)
    LOGGER.info(
        "Tuning HC target=%s, rows=%d, features=%d, CV=%s, parameters=%s",
        target,
        len(prep_data),
        len(features),
        cv_kind,
        PARAMETERS,
    )
    return solver_cfg, prep_data, X, y, cv_splits, target


def _decode(vector: np.ndarray) -> dict[str, float]:
    values = 10.0 ** np.asarray(vector, dtype=float)
    return {name: float(value) for name, value in zip(PARAMETERS, values)}


def _candidate_config(base_cfg, params: dict[str, float]):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    for name, value in params.items():
        setattr(cfg, name, value)
    return cfg


def evaluate_hc_cv(
    params: dict[str, float],
    base_cfg,
    prep_data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    target: str,
) -> dict:
    """Evaluate one real HC configuration using fold-local DAG learning."""
    cfg = _candidate_config(base_cfg, params)
    zero_w = np.zeros((len(X.columns) + 1, len(X.columns) + 1), dtype=float)
    model = create_model(
        "HC",
        zero_w,
        target,
        list(X.columns) + [target],
        "lagrange",
        prep_data,
        cfg,
    )
    results = _cross_validate_hc_with_fold_local_dag(
        model,
        X,
        y,
        cv_splits,
        target,
    )
    baseline_mse = mean_squared_error(y, np.full(len(y), y.mean()))
    fold_test = np.asarray(results["test_score"], dtype=float) / baseline_mse
    fold_train = np.asarray(results["train_score"], dtype=float) / baseline_mse
    return {
        "params": params,
        "cv_test_mean": float(fold_test.mean()),
        "cv_test_std": float(fold_test.std(ddof=1)) if len(fold_test) > 1 else 0.0,
        "cv_train_mean": float(fold_train.mean()),
        "cv_test_errs": fold_test.tolist(),
        "cv_train_errs": fold_train.tolist(),
        "baseline_mse": float(baseline_mse),
    }


def _write_record(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _suggest_with_gp(X_observed: np.ndarray, y_observed: np.ndarray, rng) -> np.ndarray:
    """Suggest a point by expected improvement over a GP surrogate."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(len(PARAMETERS)), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=0,
    )
    gp.fit(X_observed, y_observed)
    pool = rng.uniform(BO_BOUNDS[:, 0], BO_BOUNDS[:, 1], size=(4096, len(PARAMETERS)))
    mean, std = gp.predict(pool, return_std=True)
    std = np.maximum(std, 1e-9)
    best = float(np.min(y_observed))
    improvement = best - mean
    z = improvement / std
    expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)
    return pool[int(np.argmax(expected_improvement))]


def bayesopt_hc_cv(
    base_cfg,
    prep_data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    target: str,
    n_initial_points: int,
    n_bo_iterations: int,
    random_seed: int,
    log_path: Path,
) -> tuple[dict, list[dict]]:
    if n_initial_points < 1 or n_bo_iterations < 0:
        raise ValueError("n_initial_points must be >=1 and n_bo_iterations must be >=0")

    rng = np.random.default_rng(random_seed)
    n_random = max(0, n_initial_points - 1)
    X_observed = np.vstack(
        [
            KNOWN_BASELINE,
            rng.uniform(
                BO_BOUNDS[:, 0],
                BO_BOUNDS[:, 1],
                size=(n_random, len(PARAMETERS)),
            ),
        ]
    )
    records = []
    y_observed = []

    for iteration, vector in enumerate(X_observed, start=1):
        params = _decode(vector)
        try:
            result = evaluate_hc_cv(params, base_cfg, prep_data, X, y, cv_splits, target)
            objective = result["cv_test_mean"]
            status = "ok"
        except Exception as exc:  # Keep BO alive, but make failures visible.
            LOGGER.exception("HC candidate failed: %s", params)
            result = {"params": params, "error": repr(exc)}
            objective = 1e6
            status = "failed"
        record = {
            "iteration": iteration,
            "phase": "initial",
            "status": status,
            "objective": float(objective),
            **result,
        }
        print(json.dumps(record, sort_keys=True))
        _write_record(log_path, record)
        records.append(record)
        y_observed.append(objective)

    for iteration in range(n_initial_points + 1, n_initial_points + n_bo_iterations + 1):
        vector = _suggest_with_gp(X_observed, np.asarray(y_observed), rng)
        params = _decode(vector)
        try:
            result = evaluate_hc_cv(params, base_cfg, prep_data, X, y, cv_splits, target)
            objective = result["cv_test_mean"]
            status = "ok"
        except Exception as exc:
            LOGGER.exception("HC BO candidate failed: %s", params)
            result = {"params": params, "error": repr(exc)}
            objective = 1e6
            status = "failed"
        record = {
            "iteration": iteration,
            "phase": "bayesian",
            "status": status,
            "objective": float(objective),
            **result,
        }
        print(json.dumps(record, sort_keys=True))
        _write_record(log_path, record)
        records.append(record)
        y_observed.append(objective)
        X_observed = np.vstack([X_observed, vector])

    successful = [record for record in records if record["status"] == "ok"]
    if not successful:
        raise RuntimeError("All HC/CV tuning candidates failed")
    best = min(successful, key=lambda record: record["objective"])
    return best, records


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    base_cfg, prep_data, X, y, cv_splits, target = _load_context(args)
    best, records = bayesopt_hc_cv(
        base_cfg=base_cfg,
        prep_data=prep_data,
        X=X,
        y=y,
        cv_splits=cv_splits,
        target=target,
        n_initial_points=args.n_initial_points,
        n_bo_iterations=args.n_bo_iterations,
        random_seed=args.random_seed,
        log_path=_absolute_path(args.log_path),
    )
    best_output = {
        "target": target,
        "objective": best["objective"],
        "best_cv_test_mean": best.get("cv_test_mean"),
        "best_cv_test_std": best.get("cv_test_std"),
        "best_cv_train_mean": best.get("cv_train_mean"),
        "params": best["params"],
        "n_evaluations": len(records),
        "note": "CV test error is used as the BO objective; confirm the selected configuration on a fresh final holdout.",
    }
    best_path = _absolute_path(args.best_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best_output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best_output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
