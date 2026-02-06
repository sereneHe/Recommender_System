from __future__ import annotations

import json
import pickle
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline

from hei_project import api
from hei_project.data import (
    average_visits,
    clean,
    corr,
    load_all_data,
    perm_test_pval,
    split_visits_from_column,
    split_visits_from_visit_labels,
)
from hei_project.model import SelectionResult, model_factory, run_feature_selection
from hei_project.train import train_recommender
from hei_project.visualize import (
    get_pca,
    get_results,
    load_pattern,
    plot_grouped_var_reduction,
    plot_nolip_lip_var_reduction,
    print_res_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw"


def _required_codiet_pickles() -> dict[str, Path]:
    return {
        "hei.pkl": RAW_ROOT / "HEI" / "hei.pkl",
        "blood_data.pkl": RAW_ROOT / "UpdatedDataFromSara" / "blood_data.pkl",
        "body_comp.pkl": RAW_ROOT / "body_composition" / "body_comp.pkl",
        "average_expenditure.pkl": RAW_ROOT / "energy_expenditure" / "average_expenditure.pkl",
    }


class DummyScaler:
    def transform(self, x):
        return np.asarray(x, dtype=np.float32)


class DummyGuard:
    def validate(self, x: torch.Tensor) -> bool:
        return True


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


@pytest.fixture
def api_app(monkeypatch):
    monkeypatch.setattr(api, "DataGuard", lambda: DummyGuard())

    api.app.state.feature_columns = ["f1", "f2", "f3"]
    api.app.state.scaler = DummyScaler()
    api.app.state.model = DummyModel()

    return api.app


@pytest.fixture
async def api_client(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client


def to_csv(rows: list[dict]) -> bytes:
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row[c]) for c in cols))
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_evaluate_csv_no_labels(api_client):
    csv_bytes = to_csv(
        [
            {"f1": 1, "f2": 2, "f3": 3},
            {"f1": -1, "f2": -2, "f3": -3},
        ]
    )
    files = {"file": ("data.csv", csv_bytes, "text/csv")}
    r = await api_client.post("/evaluate-csv", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["has_labels"] is False
    assert body["n_samples"] == 2
    assert body["n_features"] == 3


@pytest.mark.asyncio
async def test_missing_columns(api_client):
    csv_bytes = to_csv([{"f1": 1, "f2": 2}])
    files = {"file": ("data.csv", csv_bytes, "text/csv")}
    r = await api_client.post("/evaluate-csv", files=files)
    assert r.status_code == 400
    assert "Missing columns" in r.json()["detail"]


@pytest.mark.asyncio
async def test_health(api_client):
    r = await api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_codiet_raw_pickles_exist() -> None:
    required = _required_codiet_pickles()
    missing = [str(path) for path in required.values() if not path.exists()]
    assert not missing, f"Missing CoDiet nutrition pickle files: {missing}"


def test_codiet_raw_pickles_have_id_and_rows() -> None:
    for name, path in _required_codiet_pickles().items():
        df = pd.read_pickle(path)
        assert "ID" in df.columns, f"{name} is missing ID column"
        assert len(df) > 0, f"{name} is empty"


def test_codiet_load_all_data_merges_without_duplicate_columns(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    for name, src in _required_codiet_pickles().items():
        df = pd.read_pickle(src).head(120)
        df.to_pickle(cache_dir / name)

    out_prep = tmp_path / "processed" / "prep_data.pkl"
    bundle = load_all_data(data_dir=RAW_ROOT, cache_dir=cache_dir, prep_out_path=out_prep)

    assert out_prep.exists()
    assert not bundle.prep_data.empty
    assert "ID" in bundle.prep_data.columns
    assert bundle.prep_data.columns.duplicated().sum() == 0
    assert len(bundle.food_feats) > 0
    assert len(bundle.non_food_feats) > 0


def test_mark_ori_clean_drops_nan_pairs() -> None:
    v = np.array([1.0, np.nan, 3.0, 4.0])
    u = np.array([1.0, 2.0, np.nan, 4.0])
    v_out, u_out = clean(v, u)
    assert np.allclose(v_out, np.array([1.0, 4.0]))
    assert np.allclose(u_out, np.array([1.0, 4.0]))


def test_mark_ori_corr_and_permutation_pval() -> None:
    u = np.array([1, 2, 3, 4, 5], dtype=float)
    v = np.array([2, 4, 6, 8, 10], dtype=float)
    cval = corr(u, v, corr_type="pearson")
    assert np.isclose(cval, 1.0)

    pval = perm_test_pval(u, v, corr_type="pearson", n_permutes=30, seed=0)
    assert 0.0 <= pval <= 1.0


def test_mark_ori_visit_split_helpers() -> None:
    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "VISIT": [1, 3, 1, 3],
            "x": [10.0, 20.0, 30.0, 40.0],
            "feat_visit_3": [0.5, 0.6, 0.7, 0.8],
        }
    )

    visit_3_df, other_df = split_visits_from_visit_labels(df)
    assert "ID" in visit_3_df.columns
    assert "feat_visit_3" in visit_3_df.columns
    assert "x" in other_df.columns

    visit1_df, visit3_df = split_visits_from_column(df, [1], [3])
    assert "VISIT" not in visit1_df.columns
    assert "VISIT" not in visit3_df.columns
    assert list(visit1_df["ID"]) == [1, 2]
    assert list(visit3_df["ID"]) == [1, 2]

    avg_visits_df, _ = split_visits_from_column(df, [1, 3], [3])
    assert np.isclose(avg_visits_df.loc[avg_visits_df["ID"] == 1, "x"].iloc[0], 15.0)


def test_average_visits_groups_by_id() -> None:
    df = pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "VISIT": [1, 2, 1, 2],
            "food_a": [2.0, 4.0, 6.0, 8.0],
        }
    )
    out = average_visits(df)
    assert list(out["ID"]) == [1, 2]
    assert np.isclose(out.loc[out["ID"] == 1, "food_a"].iloc[0], 3.0)
    assert np.isclose(out.loc[out["ID"] == 2, "food_a"].iloc[0], 7.0)


def test_load_all_data_merges_pickles_and_exports_prep(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    hei = pd.DataFrame({"ID": [1, 2], "food_a": [0.1, 0.4], "VISIT": [1, 1], "GLU (mg/dL)": [90.0, 100.0]})
    blood = pd.DataFrame({"ID": [1, 2], "dbs_rbc_lip_x": [1.0, 2.0]})
    body = pd.DataFrame({"ID": [1, 2], "weight": [60.0, 70.0]})
    expend = pd.DataFrame({"ID": [1, 2], "age": [30, 40]})

    hei.to_pickle(raw / "hei.pkl")
    blood.to_pickle(raw / "blood_data.pkl")
    body.to_pickle(raw / "body_comp.pkl")
    expend.to_pickle(raw / "average_expenditure.pkl")

    out_prep = tmp_path / "processed" / "prep_data.pkl"
    bundle = load_all_data(data_dir=raw, cache_dir=tmp_path / "missing_cache", prep_out_path=out_prep)
    assert "food_a" in bundle.food_feats
    assert "weight" in bundle.non_food_feats
    assert out_prep.exists()
    assert set(["ID", "food_a", "dbs_rbc_lip_x", "weight", "age"]).issubset(bundle.prep_data.columns)


def test_run_feature_selection_reg_selects_features() -> None:
    prep = pd.DataFrame(
        {
            "food_a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "GLU (mg/dL)": [80, 81, 83, 84, 86, 87, 89, 90, 92, 93],
        }
    )
    feats, train_errs, test_errs, final = run_feature_selection(
        prep,
        model_name="REG",
        custom_objective="mse_builtin",
        target_col="GLU (mg/dL)",
        n_runs=2,
        n_features=2,
        full_features=["food_a", "age"],
        seed=1,
    )
    assert len(feats) >= 1
    assert len(train_errs) == len(test_errs) == len(feats)
    assert isinstance(final, SelectionResult)
    assert np.isfinite(final.test_ratio)


def test_train_recommender_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"

    prep = pd.DataFrame(
        {
            "food_a": [0.1, 0.2, 0.3],
            "age": [30, 40, 50],
            "GLU (mg/dL)": [90.0, 95.0, 100.0],
        }
    )
    prep.to_pickle(processed / "prep_data.pkl")
    (processed / "food_feats.json").write_text(json.dumps(["food_a"]), encoding="utf-8")
    (processed / "non_food_feats.json").write_text(json.dumps(["age"]), encoding="utf-8")

    def fake_selection(*args, **kwargs):
        return (
            ["food_a"],
            [0.8],
            [0.7],
            SelectionResult(
                model={"kind": "fake-model"},
                train_mse=1.0,
                test_mse=2.0,
                train_ratio=0.8,
                test_ratio=0.7,
                test_kendalltau=0.5,
            ),
        )

    monkeypatch.setattr("hei_project.train.run_feature_selection", fake_selection)

    train_recommender(
        processed_dir=processed,
        model_dir=model_dir,
        report_dir=report_dir,
        model_name="REG",
        custom_objective="mse_builtin",
        n_select_features=1,
        n_runs=1,
        targets="GLU (mg/dL)",
        seed=42,
    )

    summary = report_dir / "recommender_training_results.json"
    assert summary.exists()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["targets"] == ["GLU (mg/dL)"]
    assert data["results"]["GLU (mg/dL)"]["selected_features"] == ["food_a"]
    assert Path(data["results"]["GLU (mg/dL)"]["model_path"]).exists()


def test_visualize_get_pca_and_pattern_batch(tmp_path: Path) -> None:
    prep = pd.DataFrame({"ID": [1, 2, 3], "GLU (mg/dL)": [90.0, 100.0, 95.0]})
    objective = pd.DataFrame({"ID": [1, 2, 3], "a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    pca_df = get_pca(prep, objective, "obj", n_dims=2, label_col="GLU (mg/dL)", do_plot=False)
    assert list(pca_df.columns)[0] == "ID"
    assert pca_df.shape[0] == 3

    pkl_dir = tmp_path / "pkl_outs"
    pkl_dir.mkdir(parents=True)
    payload = {"T1": (None, None, [0.2]), "T2": (None, None, [0.4])}
    for i in range(2):
        with (pkl_dir / f"run_seed{i}.pkl").open("wb") as f:
            pickle.dump(payload, f)

    res = get_results(pkl_dir, [r"^run_seed\d+\.pkl$"])
    assert "^run_seed\\d+\\.pkl$" in res
    assert res["^run_seed\\d+\\.pkl$"].shape[1] == 2


def test_notebook_compat_get_results_and_load_pattern(tmp_path: Path) -> None:
    pkl_dir = tmp_path / "pkl_outs"
    pkl_dir.mkdir(parents=True)
    payload = {"T1": (["a"], [0.9], [0.2]), "T2": (["b"], [0.8], [0.4]), "whtr(waist-height_ratio)": ([], [], [1.0])}
    for i in range(3):
        with (pkl_dir / f"run_seed{i}.pkl").open("wb") as f:
            pickle.dump(payload, f)

    names, pred_vals, raw = get_results(pkl_dir / "run_seed0.pkl")
    assert list(names) == ["T1", "T2"]
    assert pred_vals.shape == (2,)
    assert isinstance(raw, dict)

    combined = load_pattern(r"^run_seed\d+\.pkl$", pkl_dir)
    assert combined is not None
    assert combined.shape == (2, 3)


def test_notebook_compat_print_res_dict(capsys) -> None:
    raw = {"T1": (["f1", "f2"], [0.9, 0.8], [0.4, 0.2])}
    print_res_dict(raw, header="TEST")
    out = capsys.readouterr().out
    assert "TEST" in out
    assert "T1" in out


def test_model_factory_exists_and_returns_pipeline_definition() -> None:
    model_class, model_params = model_factory(3)
    assert model_class is Pipeline
    assert "steps" in model_params
    assert model_params["steps"][1][0] == "GP"


def test_model_factory_returns_pipeline_for_recommender() -> None:
    model_class, model_params = model_factory(4)
    assert model_class is Pipeline
    assert "steps" in model_params
    assert model_params["steps"][1][0] == "GP"


def test_run_feature_selection_with_nutrition_features() -> None:
    prep = pd.DataFrame(
        {
            "food_A": [0.1, 0.4, 0.2, 0.8, 0.5, 0.7, 0.9, 0.3, 0.6, 1.0],
            "food_B": [1.0, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.9, 0.6, 0.5],
            "age": [20, 25, 22, 35, 28, 31, 45, 33, 40, 38],
            "TEE": [2100, 2200, 2150, 2300, 2250, 2350, 2500, 2400, 2450, 2550],
            "GLU (mg/dL)": [88, 92, 90, 101, 95, 99, 108, 103, 106, 110],
        }
    )

    feats, train_errs, test_errs, final = run_feature_selection(
        prep_data=prep,
        model_name="REG",
        custom_objective="mse_builtin",
        target_col="GLU (mg/dL)",
        n_runs=2,
        n_features=2,
        full_features=["food_A", "food_B", "age", "TEE"],
        seed=42,
    )
    assert len(feats) >= 1
    assert len(train_errs) == len(test_errs) == len(feats)
    assert isinstance(final, SelectionResult)
    assert np.isfinite(final.test_ratio)


def test_train_recommender_writes_codiet_style_outputs(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"

    prep = pd.DataFrame(
        {
            "food_A": [0.1, 0.2, 0.3],
            "food_B": [0.3, 0.2, 0.1],
            "age": [30, 40, 50],
            "TEE": [2200, 2300, 2400],
            "GLU (mg/dL)": [90.0, 95.0, 100.0],
        }
    )
    prep.to_pickle(processed / "prep_data.pkl")
    (processed / "food_feats.json").write_text(json.dumps(["food_A", "food_B"]), encoding="utf-8")
    (processed / "non_food_feats.json").write_text(json.dumps(["age", "TEE"]), encoding="utf-8")

    def fake_selection(*args, **kwargs):
        return (
            ["food_A"],
            [0.8],
            [0.7],
            SelectionResult(
                model={"kind": "fake-model"},
                train_mse=1.0,
                test_mse=2.0,
                train_ratio=0.8,
                test_ratio=0.7,
                test_kendalltau=0.5,
            ),
        )

    monkeypatch.setattr("hei_project.train.run_feature_selection", fake_selection)

    train_recommender(
        processed_dir=processed,
        model_dir=model_dir,
        report_dir=report_dir,
        model_name="REG",
        custom_objective="mse_builtin",
        n_select_features=1,
        n_runs=1,
        targets="GLU (mg/dL)",
        seed=42,
    )

    summary = report_dir / "recommender_training_results.json"
    assert summary.exists()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["targets"] == ["GLU (mg/dL)"]
    assert data["results"]["GLU (mg/dL)"]["selected_features"] == ["food_A"]
    assert Path(data["results"]["GLU (mg/dL)"]["model_path"]).exists()


def test_codiet_grouped_var_reduction_plot_is_saved(tmp_path: Path) -> None:
    combined_list = [
        np.array(
            [
                [12.0, 13.5, 11.2],
                [8.0, 8.8, 9.1],
                [15.0, 14.7, 15.3],
            ]
        ),
        np.array(
            [
                [14.0, 14.8, 13.9],
                [10.2, 10.0, 9.8],
                [17.1, 16.8, 17.5],
            ]
        ),
    ]
    trgt_names = np.array(["GLU (mg/dL)", "TRIG (mg/dL)", "CHOL (mg/dL)"])
    names_lst = ["no lipids", "with lipids"]

    out = plot_grouped_var_reduction(
        combined_list=combined_list,
        trgt_names=trgt_names,
        names_lst=names_lst,
        save_dir=tmp_path,
        figure_name="codiet_grouped.png",
        show_plot=False,
    )

    assert out == tmp_path / "codiet_grouped.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_codiet_nolip_lip_plot_is_saved(tmp_path: Path) -> None:
    food_pred = np.array([9.2, 12.1, 7.0, 6.8], dtype=float)
    food_pred_lipids = np.array([10.0, 13.4, 8.3, 7.5], dtype=float)
    trgt_names = np.array(["GLU", "CHOL", "TRIG", "HDL"])

    out = plot_nolip_lip_var_reduction(
        food_pred=food_pred,
        food_pred_lipids=food_pred_lipids,
        trgt_names=trgt_names,
        save_dir=tmp_path,
        figure_name="codiet_nolip_lip.png",
        show_plot=False,
    )

    assert out == tmp_path / "codiet_nolip_lip.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_codiet_grouped_var_reduction_shape_validation() -> None:
    with pytest.raises(ValueError, match="same target dimension"):
        plot_grouped_var_reduction(
            combined_list=[np.ones((3, 2)), np.ones((4, 2))],
            trgt_names=np.array(["A", "B", "C"]),
            names_lst=["x", "y"],
            show_plot=False,
        )


def test_codiet_nolip_lip_length_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_nolip_lip_var_reduction(
            food_pred=np.array([1.0, 2.0, 3.0]),
            food_pred_lipids=np.array([1.1, 2.2]),
            trgt_names=np.array(["A", "B", "C"]),
            show_plot=False,
        )
