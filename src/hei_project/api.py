from __future__ import annotations

import io
import json
import os
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

# Project imports
from hei_project.hei.data_helper import load_all_data
from hei_project.model import run_recommender

# -------------------------
# Paths / Env
# -------------------------
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "src/hei_project/config/config.yaml"))
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "data")).resolve()

# W_est + intra_nodes (can also be configured via config.yaml)
DEFAULT_W_EST_ZIP = "W_est.csv.zip"
DEFAULT_W_EST_INNER = "W_est.csv"
DEFAULT_INTRA_NODES = "intra_nodes.txt"

# Prometheus metrics
API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

# -------------------------
# Helpers
# -------------------------
def _load_cfg(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"CONFIG_PATH not found: {path}")
    return OmegaConf.load(path)


def _resolve_assets(cfg: DictConfig) -> Dict[str, Path]:
    """
    Resolve assets with env override:
      ASSETS_DIR overrides cfg.paths.assets_dir if present.
    """
    assets_dir = Path(os.environ.get("ASSETS_DIR", str(getattr(cfg.paths, "assets_dir", ASSETS_DIR)))).resolve()
    w_est_zip = assets_dir / str(getattr(cfg.assets, "w_est_zip", DEFAULT_W_EST_ZIP))
    intra_nodes = assets_dir / str(getattr(cfg.assets, "intra_nodes_txt", DEFAULT_INTRA_NODES))
    inner_csv = str(getattr(cfg.assets, "w_est_inner_csv", DEFAULT_W_EST_INNER))
    return {
        "assets_dir": assets_dir,
        "w_est_zip": w_est_zip,
        "w_est_inner_csv": Path(inner_csv),  # keep name separately
        "intra_nodes": intra_nodes,
    }


def _load_w_est(zip_path: Path, inner_csv_name: str) -> np.ndarray:
    if not zip_path.exists():
        raise FileNotFoundError(f"W_est zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        if inner_csv_name not in z.namelist():
            raise FileNotFoundError(f"{inner_csv_name} not found in {zip_path}. Found: {z.namelist()}")
        with z.open(inner_csv_name) as f:
            return np.loadtxt(f, delimiter=",")


def _load_intra_nodes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"intra_nodes file not found: {path}")
    s = path.read_text(encoding="utf-8").strip()
    return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


def _safe_target_name(target: str) -> str:
    return (
        target.replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# -------------------------
# Lifespan: load cfg/data/assets once
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        cfg = _load_cfg(CONFIG_PATH)
        logger.info(f"Loaded config from {CONFIG_PATH}")

        # MLflow setup (optional)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or str(getattr(cfg.mlflow, "tracking_uri", "") or "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(getattr(cfg.mlflow, "experiment_name", "codiet_recommender")))

        # Load data from HEI pipeline
        food_feats, non_food_feats, prep_data = load_all_data()
        data_dict: Dict[str, Any] = {
            "food_feats": list(food_feats),
            "non_food_feats": list(non_food_feats),
            "prep_data": prep_data,
        }

        # Resolve and load assets
        assets = _resolve_assets(cfg)
        w_est_inner_name = str(getattr(cfg.assets, "w_est_inner_csv", DEFAULT_W_EST_INNER))
        w_est = _load_w_est(assets["w_est_zip"], w_est_inner_name)
        row_and_col_names = _load_intra_nodes(assets["intra_nodes"])

        # Store in app state
        app.state.cfg = cfg
        app.state.data_dict = data_dict
        app.state.prep_data = prep_data
        app.state.food_feats = list(data_dict["food_feats"])
        app.state.non_food_feats = list(data_dict["non_food_feats"])
        app.state.w_est = w_est
        app.state.row_and_col_names = row_and_col_names
        app.state.assets = {
            "assets_dir": str(assets["assets_dir"]),
            "w_est_zip": str(assets["w_est_zip"]),
            "intra_nodes": str(assets["intra_nodes"]),
        }

        logger.success(
            f"Startup complete | prep_data={prep_data.shape} | food_feats={len(app.state.food_feats)} "
            f"| non_food_feats={len(app.state.non_food_feats)}"
        )

        yield

    except Exception as e:
        logger.exception(f"API startup failed: {e}")
        raise


app = FastAPI(title="HEI Recommender Inference API", lifespan=lifespan)


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health_check():
    loaded = hasattr(app.state, "prep_data") and hasattr(app.state, "cfg")
    return {
        "status": "healthy" if loaded else "starting",
        "loaded": loaded,
        "assets": getattr(app.state, "assets", {}),
    }


@app.post("/recommend")
async def recommend_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-based recommend endpoint.
    Payload supports optional overrides:
      {
        "targets": ["GLU (mg/dL)"],
        "solver": {
          "model_name": "XGB",
          "custom_objective": "lagrange",
          "n_select_features": 5,
          "n_runs": 40
        }
      }
    """
    if not hasattr(app.state, "cfg"):
        raise HTTPException(status_code=503, detail="Service not ready.")

    cfg: DictConfig = app.state.cfg
    cfg2 = OmegaConf.copy(cfg)

    # Apply overrides
    targets = payload.get("targets", None)
    if targets is not None:
        if not isinstance(targets, list) or len(targets) == 0:
            raise HTTPException(status_code=400, detail="'targets' must be a non-empty list.")
        cfg2.targets.columns = targets

    solver_over = payload.get("solver", {})
    if isinstance(solver_over, dict):
        if "model_name" in solver_over:
            cfg2.solver.model_name = str(solver_over["model_name"])
        if "custom_objective" in solver_over:
            cfg2.solver.custom_objective = str(solver_over["custom_objective"])
        if "n_select_features" in solver_over:
            cfg2.solver.n_select_features = int(solver_over["n_select_features"])
        if "n_runs" in solver_over:
            cfg2.solver.n_runs = int(solver_over["n_runs"])

    # Run recommender
    try:
        result = run_recommender(
            cfg=cfg2,
            food_feats=app.state.food_feats,
            non_food_feats=app.state.non_food_feats,
            prep_data=app.state.prep_data,
            w_est=app.state.w_est,
            row_and_col_names=app.state.row_and_col_names,
        )
    except Exception as e:
        logger.exception(f"run_recommender failed: {e}")
        raise HTTPException(status_code=500, detail=f"run_recommender failed: {e}")

    # JSON-friendly output
    out: Dict[str, Any] = {"targets": list(cfg2.targets.columns), "results": {}}
    for t, (feats, train_errs, test_errs) in result.items():
        out["results"][t] = {
            "selected_features": list(feats),
            "train_errors": [float(x) for x in train_errs],
            "test_errors": [float(x) for x in test_errs],
        }
    return out


@app.post("/recommend-csv")
async def recommend_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a CSV file with columns including:
      - ID
      - VISIT
    Optionally can include:
      - targets column(s) in header? (not required)
    The endpoint will:
      1) parse CSV for ID/VISIT pairs
      2) subset prep_data to those samples (if present)
      3) run recommender on full dataset (selection), then return selected features + errors
         and include which IDs were found.
    """
    if not hasattr(app.state, "prep_data"):
        raise HTTPException(status_code=503, detail="Service not ready.")

    filename = file.filename or "<uploaded>"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if "ID" not in df.columns or "VISIT" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain columns: ID, VISIT")

    # Find which samples exist in prep_data (for reporting)
    prep = app.state.prep_data
    merged = df[["ID", "VISIT"]].merge(prep[["ID", "VISIT"]], on=["ID", "VISIT"], how="left", indicator=True)
    found = merged["_merge"].eq("both").sum()
    total = len(merged)

    # We still run recommender on full prep_data (same as your pipeline),
    # but we report which requested IDs exist.
    try:
        cfg: DictConfig = app.state.cfg
        result = run_recommender(
            cfg=cfg,
            food_feats=app.state.food_feats,
            non_food_feats=app.state.non_food_feats,
            prep_data=prep,
            w_est=app.state.w_est,
            row_and_col_names=app.state.row_and_col_names,
        )
    except Exception as e:
        logger.exception(f"run_recommender failed: {e}")
        raise HTTPException(status_code=500, detail=f"run_recommender failed: {e}")

    out: Dict[str, Any] = {
        "filename": filename,
        "requested_samples": int(total),
        "found_in_prep_data": int(found),
        "targets": list(app.state.cfg.targets.columns),
        "results": {},
    }
    for t, (feats, train_errs, test_errs) in result.items():
        out["results"][t] = {
            "selected_features": list(feats),
            "train_errors": [float(x) for x in train_errs],
            "test_errors": [float(x) for x in test_errs],
        }

    return out


# -------------------------
# Prometheus middleware + endpoint
# -------------------------
@app.middleware("http")
async def prometheus_request_counter(request: Request, call_next):
    response = await call_next(request)
    API_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -------------------------
# uv script entrypoint
# -------------------------
def main() -> None:
    """
    For pyproject.toml:
      [project.scripts]
      api = "hei_project.api:main"
    """
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("hei_project.api:app", host=host, port=port, reload=False)
