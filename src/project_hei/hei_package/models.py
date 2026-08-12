# hei_package/models.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBRegressor


def build_model(model_name: str, n_features: int):
    """
    Factory function to build ML models.

    Parameters
    ----------
    model_name : str
        One of {"GP", "REG", "XGB"}
    n_features : int

    Returns
    -------
    model_class, model_params
    """
    if model_name == "GP":
        kernel = RBF(length_scale=[1.0] * n_features)
        return Pipeline, {
            "steps": [
                ("scale", StandardScaler()),
                ("gp", GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1.0,
                    normalize_y=True,
                    n_restarts_optimizer=3,
                ))
            ]
        }

    if model_name == "REG":
        return Pipeline, {
            "steps": [
                ("scale", StandardScaler()),
                ("linreg", LinearRegression())
            ]
        }

    if model_name == "XGB":
        return Pipeline, {
            "steps": [
                ("scale", StandardScaler()),
                ("xgb", XGBRegressor(
                    n_estimators=10,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                ))
            ]
        }

    raise ValueError(f"Unknown model: {model_name}")

import pandas as pd
from pathlib import Path

def export_pca_tables(
    pca_model,
    feature_names,
    prefix: str,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    loadings = pd.DataFrame(
        pca_model.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
    )
    loadings.to_csv(out_dir / f"{prefix}_pca_loadings.csv")

    variance = pd.DataFrame({
        "explained_variance_ratio": pca_model.explained_variance_ratio_
    })
    variance.to_csv(out_dir / f"{prefix}_pca_variance.csv", index=False)
