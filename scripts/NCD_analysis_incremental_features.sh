#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/reports/results/pkl_outs}"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${ROOT_DIR}/src" \
ROOT_DIR="${ROOT_DIR}" \
OUT_DIR="${OUT_DIR}" \
MODEL_NAME="${MODEL_NAME:-DT}" \
N_RUNS="${N_RUNS:-150}" \
N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}" \
N_EXPERIMENTS="${N_EXPERIMENTS:-20}" \
MAX_WORKERS="${MAX_WORKERS:-1}" \
DO_PERMUTE="${DO_PERMUTE:-False}" \
USE_LIPIDOMICS="${USE_LIPIDOMICS:-True}" \
TARGET_COLUMNS="${TARGET_COLUMNS:-GLU (mg/dL),HDL (mg/dL),LDL (mg/dL),TRIG (mg/dL),HbA1c (%),Systolic Blood Pressure (mm Hg),Diastolic Blood Pressure (mm Hg),CRP (mg/dL),whtr(waist-height_ratio)}" \
BASE_FEATS="${BASE_FEATS:-age,gender_numeric,stress_index,fatigue_index,mean_hrt,site_continental,weight,height,GMWI,microbiome_Shannon}" \
"${PY_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


ROOT_DIR = Path(os.environ["ROOT_DIR"])
OUT_DIR = Path(os.environ["OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ["MODEL_NAME"].strip().upper()  # REG | DT | LINEAR
N_RUNS = int(os.environ["N_RUNS"])
N_SELECT_FEATURES = int(os.environ["N_SELECT_FEATURES"])
N_EXPERIMENTS = int(os.environ["N_EXPERIMENTS"])
MAX_WORKERS = int(os.environ["MAX_WORKERS"])
DO_PERMUTE = as_bool(os.environ["DO_PERMUTE"])
USE_LIPIDOMICS = as_bool(os.environ["USE_LIPIDOMICS"])

target_columns = [x.strip() for x in os.environ["TARGET_COLUMNS"].split(",") if x.strip()]
base_feats = [x.strip() for x in os.environ["BASE_FEATS"].split(",") if x.strip()]

processed_dir = ROOT_DIR / "data" / "processed"
prep_path = processed_dir / "prep_data.pkl"
food_path = processed_dir / "food_feats.json"
non_food_path = processed_dir / "non_food_feats.json"

if not (prep_path.exists() and food_path.exists() and non_food_path.exists()):
    from hei_project.hei.data_helper import load_all_data

    food_feats, non_food_feats, prep_data = load_all_data()
    processed_dir.mkdir(parents=True, exist_ok=True)
    prep_data.to_pickle(prep_path)
    food_path.write_text(json.dumps(list(food_feats)), encoding="utf-8")
    non_food_path.write_text(json.dumps(list(non_food_feats)), encoding="utf-8")
else:
    prep_data = pd.read_pickle(prep_path)
    food_feats = json.loads(food_path.read_text(encoding="utf-8"))
    non_food_feats = json.loads(non_food_path.read_text(encoding="utf-8"))

if MODEL_NAME == "REG":
    model_class = Pipeline
    model_params = {"steps": [("scale", StandardScaler()), ("linreg", LinearRegression())]}
elif MODEL_NAME == "DT":
    model_class = Pipeline
    model_params = {"steps": [("DT", DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42))]}
elif MODEL_NAME == "LINEAR":
    model_class = LinearRegression
    model_params = {}
else:
    raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME}. Use REG|DT|LINEAR")

full_feats = [f for f in (list(food_feats) + list(non_food_feats) + list(base_feats)) if f in prep_data.columns]
if not USE_LIPIDOMICS:
    full_feats = [f for f in full_feats if "dbs_rbc_lip" not in f]
full_feats = list(dict.fromkeys(full_feats))

valid_targets = [t for t in target_columns if t in prep_data.columns]
missing_targets = [t for t in target_columns if t not in prep_data.columns]
if missing_targets:
    print(f"[WARN] Missing targets skipped: {missing_targets}")
if not valid_targets:
    raise ValueError("No valid target columns found in prep_data")

prep_data_orig = prep_data.copy(deep=True)


def compute_predictor_errors_local(df: pd.DataFrame, selected_feats: list[str], target_col: str, seed: int):
    reg = df[selected_feats + [target_col]].dropna()
    X = reg[selected_feats].to_numpy()
    y = reg[target_col].to_numpy()
    if len(y) < 10:
        return 1.0, 1.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_bench = mean_squared_error(y_train, np.ones_like(y_train) * y_train.mean())
    test_bench = mean_squared_error(y_test, np.ones_like(y_test) * y_train.mean())
    return train_mse / train_bench, test_mse / test_bench


def run_feature_selection_local(df: pd.DataFrame, target_col: str, n_runs: int, n_features: int, feature_pool: list[str]):
    curr_feats: list[str] = []
    curr_train_errs: list[float] = []
    curr_test_errs: list[float] = []
    curr_test_error = 1e9

    print(f"====== {target_col}")
    for i in range(n_features):
        candidate_feats = [f for f in feature_pool if f not in curr_feats]
        if not candidate_feats:
            break

        res = []
        for feat in candidate_feats:
            train_rs = []
            test_rs = []
            for run_i in range(n_runs):
                tr, te = compute_predictor_errors_local(df, curr_feats + [feat], target_col, seed=run_i + 1234)
                train_rs.append(tr)
                test_rs.append(te)
            res.append((feat, float(np.mean(train_rs)), float(np.mean(test_rs))))

        res.sort(key=lambda t: t[2])
        best_feat, best_train, best_test = res[0]

        print(f"Round {i}: best={best_feat} test_var_reduction={100*(1-best_test):.3f}")
        if best_test >= curr_test_error:
            print("Failed to improve further")
            break

        curr_feats.append(best_feat)
        curr_train_errs.append(best_train)
        curr_test_errs.append(best_test)
        curr_test_error = best_test

    return curr_feats, curr_train_errs, curr_test_errs


def run_and_save_rnd_experiment(seed: int):
    print(f"Running with seed: {seed}")
    local_df = prep_data_orig.copy(deep=True)
    rng = np.random.default_rng(seed)
    res_dict = {}

    for target_col in valid_targets:
        if DO_PERMUTE:
            local_df[target_col] = rng.permutation(local_df[target_col].values)

        curr_feats, curr_train_errs, curr_test_errs = run_feature_selection_local(
            local_df,
            target_col=target_col,
            n_runs=N_RUNS,
            n_features=N_SELECT_FEATURES,
            feature_pool=full_feats,
        )
        res_dict[target_col] = (curr_feats, curr_train_errs, curr_test_errs)

    out_name = (
        f"NCD_analysis_incremental_feat_{MODEL_NAME}_"
        f"lipids_{USE_LIPIDOMICS}_maxfeat_{N_SELECT_FEATURES}_permute_{DO_PERMUTE}_{seed}.pkl"
    )
    with (OUT_DIR / out_name).open("wb") as f:
        pkl.dump(res_dict, f)
    print(f"[OK] saved {OUT_DIR / out_name}")
    return out_name


seed_rng = np.random.default_rng(99837643)
seeds = seed_rng.integers(low=0, high=100000, size=N_EXPERIMENTS)
print(f"seeds={list(map(int, seeds))}")

if MAX_WORKERS <= 1:
    for s in seeds:
        run_and_save_rnd_experiment(int(s))
else:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        _ = list(executor.map(run_and_save_rnd_experiment, map(int, seeds)))

print("[DONE] NCD incremental feature experiments completed")
PY
