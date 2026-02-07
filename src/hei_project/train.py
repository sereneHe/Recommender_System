from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import hydra
import mlflow
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from hei_project.hei.data_helper import load_all_data
from hei_project.model import run_recommender


def _load_w_est(zip_path: Path, inner_csv: str) -> np.ndarray:
    if not zip_path.exists():
        raise FileNotFoundError(f"W_est zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        if inner_csv not in z.namelist():
            raise FileNotFoundError(f"{inner_csv} not found in zip")
        with z.open(inner_csv) as f:
            return np.loadtxt(f, delimiter=",")


def _load_intra_nodes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"intra_nodes file not found: {path}")
    s = path.read_text(encoding="utf-8").strip()
    return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def train(cfg: DictConfig) -> None:
    logger.info("Hydra config:\n{}", OmegaConf.to_yaml(cfg))

    # ------------------
    # MLflow setup
    # ------------------
    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_URI")
        or cfg.mlflow.tracking_uri
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # ------------------
    # Load data
    # ------------------
    food_feats, non_food_feats, prep_data = load_all_data()

    # ------------------
    # Load assets
    # ------------------
    project_root = Path(__file__).resolve().parents[2]
    assets_dir = Path(os.environ.get("ASSETS_DIR", str(cfg.paths.assets_dir)))
    if not assets_dir.is_absolute():
        assets_dir = (project_root / assets_dir).resolve()
    if not assets_dir.exists():
        fallback_assets = project_root / "data"
        logger.warning(
            "Resolved assets_dir does not exist ({}). Falling back to {}",
            assets_dir,
            fallback_assets,
        )
        assets_dir = fallback_assets
    w_est = _load_w_est(
        assets_dir / cfg.assets.w_est_zip,
        cfg.assets.w_est_inner_csv,
    )
    row_and_col_names = _load_intra_nodes(
        assets_dir / cfg.assets.intra_nodes_txt
    )

    # ------------------
    # Run experiment
    # ------------------
    run_name = f"{cfg.solver.model_name}_{cfg.solver.custom_objective}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": cfg.solver.model_name,
                "custom_objective": cfg.solver.custom_objective,
                "n_select_features": cfg.solver.n_select_features,
                "n_runs": cfg.solver.n_runs,
                "targets": list(cfg.targets.columns),
            }
        )
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

        start = time.time()
        result = run_recommender(
            food_feats=food_feats,
            non_food_feats=non_food_feats,
            prep_data=prep_data,
            w_est=w_est,
            row_and_col_names=row_and_col_names,
            model_name=cfg.solver.model_name,
            custom_objective=cfg.solver.custom_objective,
            n_select_features=cfg.solver.n_select_features,
            n_runs=cfg.solver.n_runs,
            target_columns=list(cfg.targets.columns),
            cfg=cfg,
        )
        mlflow.log_metric("runtime_seconds", time.time() - start)

        for target, (feats, train_errs, test_errs) in result.items():
            safe = target.replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_text(
                "[" + ", ".join(feats) + "]",
                f"{safe}_selected_features.txt",
            )
            if train_errs:
                mlflow.log_metric(f"{safe}_train_err", train_errs[-1])
            if test_errs:
                mlflow.log_metric(f"{safe}_test_err", test_errs[-1])

        logger.success("Training finished for targets: {}", list(result.keys()))


if __name__ == "__main__":
    train()
