#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi

PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/data/processed}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/reports/results}"
MODEL_NAME="${MODEL_NAME:-XGB}"
CUSTOM_OBJECTIVE="${CUSTOM_OBJECTIVE:-mse_builtin}"
N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}"
N_RUNS="${N_RUNS:-1}"
TARGETS="${TARGETS:-GLU (mg/dL)}"
SEEDS="${SEEDS:-13042,95863}"

mkdir -p "${RESULTS_DIR}"

PYTHONPATH="${ROOT_DIR}/src" \
PROCESSED_DIR="${PROCESSED_DIR}" \
RESULTS_DIR="${RESULTS_DIR}" \
MODEL_NAME="${MODEL_NAME}" \
CUSTOM_OBJECTIVE="${CUSTOM_OBJECTIVE}" \
N_SELECT_FEATURES="${N_SELECT_FEATURES}" \
N_RUNS="${N_RUNS}" \
TARGETS="${TARGETS}" \
SEEDS="${SEEDS}" \
"${PY_BIN}" - <<'PY'
from __future__ import annotations

import ast
import json
import os
import pickle as pkl
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from hei_project.model import build_full_feats
from hei_project.hei.recommender_utils import run_feature_selection

processed_dir = Path(os.environ["PROCESSED_DIR"])
results_dir = Path(os.environ["RESULTS_DIR"])
model_name = os.environ["MODEL_NAME"]
custom_objective = os.environ["CUSTOM_OBJECTIVE"]
n_select_features = int(os.environ["N_SELECT_FEATURES"])
n_runs = int(os.environ["N_RUNS"])
targets = [x.strip() for x in os.environ["TARGETS"].split(",") if x.strip()]
seeds = [int(x.strip()) for x in os.environ["SEEDS"].split(",") if x.strip()]

prep_data = pd.read_pickle(processed_dir / "prep_data.pkl")
food_feats = json.loads((processed_dir / "food_feats.json").read_text(encoding="utf-8"))
non_food_feats = json.loads((processed_dir / "non_food_feats.json").read_text(encoding="utf-8"))

root_dir = processed_dir.parents[1]
zip_path = root_dir / "data" / "W_est.csv.zip"
intra_path = root_dir / "data" / "intra_nodes.txt"

with zipfile.ZipFile(zip_path) as z:
    with z.open("W_est.csv") as f:
        w_est = np.loadtxt(f, delimiter=",")
row_and_col_names = ast.literal_eval(intra_path.read_text(encoding="utf-8").strip())

for t in targets:
    if t not in prep_data.columns:
        raise KeyError(f"Target '{t}' not found in prep_data")

for seed in seeds:
    for use_lipids in (False, True):
        full_feats = build_full_feats(prep_data, list(food_feats), list(non_food_feats), cfg=None)
        if not use_lipids:
            full_feats = [f for f in full_feats if "dbs_rbc_lip" not in f]

        for do_permute in (False, True):
            work = prep_data.copy()
            if do_permute:
                rng = np.random.default_rng(seed)
                for t in targets:
                    work[t] = rng.permutation(work[t].to_numpy())

            res_dict = {}
            for target_col in targets:
                curr_feats, curr_train_errs, curr_test_errs = run_feature_selection(
                    prep_data=work,
                    model_name=model_name,
                    custom_objective=custom_objective,
                    target_col=target_col,
                    w_est=w_est,
                    row_and_col_names=row_and_col_names,
                    n_runs=n_runs,
                    n_features=n_select_features,
                    full_feats=full_feats,
                    model_factory=None,
                )
                res_dict[target_col] = (curr_feats, curr_train_errs, curr_test_errs)

            out_name = f"NCD_analysis_incremental_feat_lipids_{use_lipids}_maxfeat_{n_select_features}_permute_{do_permute}_{seed}.pkl"
            out_path = results_dir / out_name
            with out_path.open("wb") as f:
                pkl.dump(res_dict, f)
            print(f"[OK] {out_path}")
PY
