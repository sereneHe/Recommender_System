from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from loguru import logger
import numpy as np
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
import torch
import joblib  # type: ignore[import-untyped]

from hei_project.guardrails import DataGuard  # type: ignore

MODEL_PATH = Path("models/recommender/GLU_mg_dL_model.joblib")
PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "feature_columns.json"
FOOD_FEATURES_PATH = PROCESSED_DIR / "food_feats.json"
NON_FOOD_FEATURES_PATH = PROCESSED_DIR / "non_food_feats.json"

# Prometheus metrics
API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)


def _load_feature_columns() -> list[str]:
    if FOOD_FEATURES_PATH.exists() and NON_FOOD_FEATURES_PATH.exists():
        food = json.loads(FOOD_FEATURES_PATH.read_text(encoding="utf-8"))
        non_food = json.loads(NON_FOOD_FEATURES_PATH.read_text(encoding="utf-8"))
        if isinstance(food, list) and isinstance(non_food, list):
            cols = [str(c) for c in food + non_food]
            # dedup keep order
            seen: set[str] = set()
            out: list[str] = []
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
    if FEATURES_PATH.exists():
        cols = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        if isinstance(cols, list):
            return [str(c) for c in cols]
    raise RuntimeError(
        f"Missing feature metadata. Expected one of: {FOOD_FEATURES_PATH} + {NON_FOOD_FEATURES_PATH}, or {FEATURES_PATH}"
    )


def _predict_with_model(model: Any, x_np: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        preds = np.asarray(model.predict(x_np), dtype=float).reshape(-1)
        return preds

    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        raw = model(x_t)  # type: ignore[operator]
    if isinstance(raw, torch.Tensor):
        return raw.detach().cpu().numpy().reshape(-1).astype(float)
    return np.asarray(raw, dtype=float).reshape(-1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model at {MODEL_PATH}. Run CoDiet training first.")

    app.state.feature_columns = _load_feature_columns()
    app.state.model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded CoDiet model from {MODEL_PATH}")
    logger.info(f"Expected feature count: {len(app.state.feature_columns)}")
    yield


app = FastAPI(title="CoDiet Nutrition Inference API", lifespan=lifespan)


@app.post("/evaluate-csv")
async def evaluate_csv(file: UploadFile = File(...)) -> dict:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded.")

    filename = file.filename
    if filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Optional label columns for supervised evaluation.
    label_col = None
    for candidate in ["target", "label", "y_true", "GLU (mg/dL)"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None:
        y_true = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
        x_df = df.drop(columns=[label_col])
    else:
        y_true = None
        x_df = df

    feature_columns = app.state.feature_columns
    missing = set(feature_columns) - set(x_df.columns)
    extra = set(x_df.columns) - set(feature_columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing)}")
    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected extra columns: {sorted(extra)}")

    x_df = x_df[feature_columns]
    x_np = x_df.to_numpy(dtype=float)

    bouncer = DataGuard()
    if not bouncer.validate(torch.tensor(x_np, dtype=torch.float32)):
        raise HTTPException(status_code=400, detail="Input rejected by Guardrails (Drift or Anomaly detected)")

    preds = _predict_with_model(app.state.model, x_np)

    response: dict[str, Any] = {
        "filename": filename,
        "n_samples": int(x_np.shape[0]),
        "n_features": int(x_np.shape[1]),
        "prediction_mean": float(np.mean(preds)) if preds.size else float("nan"),
        "prediction_std": float(np.std(preds)) if preds.size else float("nan"),
    }

    if y_true is not None and len(y_true) == len(preds):
        mask = np.isfinite(y_true)
        if mask.any():
            err = preds[mask] - y_true[mask]
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(np.square(err))))
            response.update(
                {
                    "has_labels": True,
                    "label_column": label_col,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
        else:
            response.update({"has_labels": True, "label_column": label_col, "message": "Labels are all NaN after parsing."})
    else:
        response.update({"has_labels": False, "message": "No numeric target column provided; returning predictions summary only."})

    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


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
