from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from hei_project import api


@pytest.fixture
def api_app(monkeypatch):
    api.app.state.cfg = OmegaConf.create(
        {
            "targets": {"columns": ["GLU (mg/dL)"]},
            "solver": {
                "model_name": "XGB",
                "custom_objective": "lagrange",
                "n_select_features": 2,
                "n_runs": 1,
            },
        }
    )
    api.app.state.prep_data = object()
    api.app.state.food_feats = ["food_a"]
    api.app.state.non_food_feats = ["bio_a"]
    api.app.state.w_est = [[1.0]]
    api.app.state.row_and_col_names = ["food_a", "bio_a"]
    api.app.state.assets = {"assets_dir": "data"}

    def fake_run_recommender(**kwargs):
        return {
            "GLU (mg/dL)": (
                ["food_a", "bio_a"],
                [0.1, 0.2],
                [0.3, 0.4],
            )
        }

    monkeypatch.setattr(api, "run_recommender", fake_run_recommender)
    return api.app
