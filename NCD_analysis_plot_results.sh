#!/usr/bin/env bash
set -euo pipefail

# Generate: reports/figures/ncd_risk_incremental_nolip_lip.png
# Intermediate data path is fixed to: reports/results/
# Keep notebook intermediate filenames unchanged:
#   intake_incremental_feat_results.json
#   intake_incremental_feat_with_lipids_results.json
#
# Optional overrides:
#   NO_LIPID_FILE=/path/to/intake_incremental_feat_results.json \
#   WITH_LIPID_FILE=/path/to/intake_incremental_feat_with_lipids_results.json \
#   OUT_FIG=/path/to/ncd_risk_incremental_nolip_lip.png \
#   bash NCD_analysis_plot_results.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/reports/results}"
NO_LIPID_FILE="${NO_LIPID_FILE:-${RESULTS_DIR}/intake_incremental_feat_results.json}"
WITH_LIPID_FILE="${WITH_LIPID_FILE:-${RESULTS_DIR}/intake_incremental_feat_with_lipids_results.json}"
OUT_FIG="${OUT_FIG:-${ROOT_DIR}/reports/figures/ncd_risk_incremental_nolip_lip.png}"
OUT_FIG_5FEAT="${OUT_FIG_5FEAT:-${ROOT_DIR}/reports/figures/ncd_risk_incremental_5feat.png}"

PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/datasets/processed}"
MODEL_NAME="${MODEL_NAME:-REG}"
CUSTOM_OBJECTIVE="${CUSTOM_OBJECTIVE:-mse_builtin}"
N_SELECT_FEATURES="${N_SELECT_FEATURES:-5}"
N_RUNS="${N_RUNS:-20}"
SEED="${SEED:-42}"
TARGETS="${TARGETS:-}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
PATTERN_SEEDS="${PATTERN_SEEDS:-${SEED}}"
GENERATE_PATTERN_PKLS="${GENERATE_PATTERN_PKLS:-1}"

PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi

mkdir -p "${RESULTS_DIR}" "$(dirname "${OUT_FIG}")"

PYTHONPATH="${ROOT_DIR}/src" \
NO_LIPID_FILE="${NO_LIPID_FILE}" \
WITH_LIPID_FILE="${WITH_LIPID_FILE}" \
OUT_FIG="${OUT_FIG}" \
OUT_FIG_5FEAT="${OUT_FIG_5FEAT}" \
PROCESSED_DIR="${PROCESSED_DIR}" \
MODEL_NAME="${MODEL_NAME}" \
CUSTOM_OBJECTIVE="${CUSTOM_OBJECTIVE}" \
N_SELECT_FEATURES="${N_SELECT_FEATURES}" \
N_RUNS="${N_RUNS}" \
SEED="${SEED}" \
TARGETS="${TARGETS}" \
FORCE_REBUILD="${FORCE_REBUILD}" \
PATTERN_SEEDS="${PATTERN_SEEDS}" \
GENERATE_PATTERN_PKLS="${GENERATE_PATTERN_PKLS}" \
"${PY_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle as pkl

import pandas as pd
import numpy as np

from hei_project.model import run_feature_selection
from hei_project.visualize import (
    get_results,
    load_pattern,
    plot_grouped_var_reduction,
    plot_nolip_lip_var_reduction,
)

no_lipid_file = Path(os.environ["NO_LIPID_FILE"])
with_lipid_file = Path(os.environ["WITH_LIPID_FILE"])
out_fig = Path(os.environ["OUT_FIG"])
out_fig_5feat = Path(os.environ["OUT_FIG_5FEAT"])
processed_dir = Path(os.environ["PROCESSED_DIR"])
model_name = os.environ["MODEL_NAME"]
custom_objective = os.environ["CUSTOM_OBJECTIVE"]
n_select_features = int(os.environ["N_SELECT_FEATURES"])
n_runs = int(os.environ["N_RUNS"])
seed = int(os.environ["SEED"])
targets_env = os.environ.get("TARGETS", "").strip()
force_rebuild = os.environ.get("FORCE_REBUILD", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
pattern_seeds_env = os.environ.get("PATTERN_SEEDS", str(seed))
generate_pattern_pkls = os.environ.get("GENERATE_PATTERN_PKLS", "1").strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_features(prep_data: pd.DataFrame, food_feats: list[str], non_food_feats: list[str], include_lipids: bool) -> list[str]:
    feats: list[str] = []
    feats += list(food_feats)
    feats += [
        "age",
        "gender_numeric",
        "stress_index",
        "fatigue_index",
        "mean_hrt",
        "site_continental",
        "weight",
        "height",
        "GMWI",
        "microbiome_Shannon",
    ]
    if include_lipids:
        feats += [c for c in non_food_feats if "dbs_rbc_lip" in c]
    feats += [c for c in prep_data.columns if "microb_clean15_" in c]
    feats = _dedup_keep_order(feats)
    feats = [c for c in feats if c in prep_data.columns]
    return feats


def _resolve_targets(prep_data: pd.DataFrame) -> list[str]:
    if targets_env:
        t = [x.strip() for x in targets_env.split(",") if x.strip()]
        if not t:
            raise ValueError("TARGETS is set but empty after parsing")
        for col in t:
            if col not in prep_data.columns:
                raise KeyError(f"Target '{col}' not found in prep_data")
        return t

    default_report = Path("reports/recommender_training_results.json")
    if default_report.exists():
        payload = json.loads(default_report.read_text(encoding="utf-8"))
        maybe_targets = payload.get("targets")
        if isinstance(maybe_targets, list) and maybe_targets:
            t = [str(x) for x in maybe_targets if str(x) in prep_data.columns]
            if t:
                return t
    if "GLU (mg/dL)" in prep_data.columns:
        return ["GLU (mg/dL)"]
    raise ValueError("No targets resolved. Set TARGETS='col1,col2,...'")


def _generate_incremental_result(out_path: Path, include_lipids: bool) -> None:
    prep_path = processed_dir / "prep_data.pkl"
    food_path = processed_dir / "food_feats.json"
    non_food_path = processed_dir / "non_food_feats.json"
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing {prep_path}")
    if not food_path.exists() or not non_food_path.exists():
        raise FileNotFoundError(f"Missing feature list files in {processed_dir}")

    prep_data = pd.read_pickle(prep_path)
    food_feats = json.loads(food_path.read_text(encoding="utf-8"))
    non_food_feats = json.loads(non_food_path.read_text(encoding="utf-8"))
    full_feats = _build_features(prep_data, food_feats, non_food_feats, include_lipids=include_lipids)
    targets = _resolve_targets(prep_data)

    result_map: dict[str, dict[str, object]] = {}
    for target_col in targets:
        curr_feats, curr_train_errs, curr_test_errs, _ = run_feature_selection(
            prep_data,
            model_name=model_name,
            custom_objective=custom_objective,
            target_col=target_col,
            n_runs=n_runs,
            n_features=n_select_features,
            full_features=full_feats,
            seed=seed,
        )
        result_map[target_col] = {
            "selected_features": curr_feats,
            "train_var_ratio_history": curr_train_errs,
            "test_var_ratio_history": curr_test_errs,
        }

    summary = {
        "model_name": model_name,
        "custom_objective": custom_objective,
        "n_select_features": n_select_features,
        "n_runs": n_runs,
        "targets": targets,
        "full_feats": full_feats,
        "results": result_map,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Generated incremental results: {out_path}")


def _parse_seed_list(text: str) -> list[int]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    out: list[int] = []
    for x in items:
        out.append(int(x))
    if not out:
        out = [seed]
    return out


def _run_notebook_style_result(
    prep_data: pd.DataFrame,
    full_feats: list[str],
    targets: list[str],
    *,
    do_permute: bool,
    run_seed: int,
) -> dict[str, tuple[list[str], list[float], list[float]]]:
    work_data = prep_data.copy()
    if do_permute:
        rng = np.random.default_rng(run_seed)
        for target_col in targets:
            work_data[target_col] = rng.permutation(work_data[target_col].to_numpy())

    res_dict: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for target_col in targets:
        curr_feats, curr_train_errs, curr_test_errs, _ = run_feature_selection(
            work_data,
            model_name=model_name,
            custom_objective=custom_objective,
            target_col=target_col,
            n_runs=n_runs,
            n_features=n_select_features,
            full_features=full_feats,
            seed=run_seed,
        )
        res_dict[target_col] = (curr_feats, curr_train_errs, curr_test_errs)
    return res_dict


def _write_notebook_pickle(path: Path, payload: dict[str, tuple[list[str], list[float], list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pkl.dump(payload, f)
    print(f"[OK] Generated notebook-style result: {path}")


def _generate_pattern_pickles() -> None:
    prep_path = processed_dir / "prep_data.pkl"
    food_path = processed_dir / "food_feats.json"
    non_food_path = processed_dir / "non_food_feats.json"
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing {prep_path}")
    if not food_path.exists() or not non_food_path.exists():
        raise FileNotFoundError(f"Missing feature list files in {processed_dir}")

    prep_data = pd.read_pickle(prep_path)
    food_feats = json.loads(food_path.read_text(encoding="utf-8"))
    non_food_feats = json.loads(non_food_path.read_text(encoding="utf-8"))
    targets = _resolve_targets(prep_data)

    full_feats_no_lip = _build_features(prep_data, food_feats, non_food_feats, include_lipids=False)
    full_feats_lip = _build_features(prep_data, food_feats, non_food_feats, include_lipids=True)
    seeds = _parse_seed_list(pattern_seeds_env)

    for run_seed in seeds:
        filename_map = [
            (
                _run_notebook_style_result(
                    prep_data, full_feats_no_lip, targets, do_permute=True, run_seed=run_seed
                ),
                [
                    f"NCD_analysis_incremental_feat_lipids_False_maxfeat_{n_select_features}_permute_True_{run_seed}.pkl",
                    f"NCD_analysis_incremental_feat_no_lipids_results_permute_{run_seed}.pkl",
                ],
            ),
            (
                _run_notebook_style_result(
                    prep_data, full_feats_no_lip, targets, do_permute=False, run_seed=run_seed
                ),
                [
                    f"NCD_analysis_incremental_feat_lipids_False_maxfeat_{n_select_features}_permute_False_{run_seed}.pkl",
                    f"NCD_analysis_incremental_feat_no_lipids_results_{run_seed}.pkl",
                ],
            ),
            (
                _run_notebook_style_result(
                    prep_data, full_feats_lip, targets, do_permute=True, run_seed=run_seed
                ),
                [
                    f"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_True_{run_seed}.pkl",
                    f"NCD_analysis_incremental_feat_with_lipids_results_permute_{run_seed}.pkl",
                ],
            ),
            (
                _run_notebook_style_result(
                    prep_data, full_feats_lip, targets, do_permute=False, run_seed=run_seed
                ),
                [
                    f"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_False_{run_seed}.pkl",
                ],
            ),
        ]

        for payload, names in filename_map:
            for name in names:
                out_path = no_lipid_file.parent / name
                if force_rebuild or not out_path.exists():
                    _write_notebook_pickle(out_path, payload)
                else:
                    print(f"[SKIP] Existing notebook-style result: {out_path}")

if force_rebuild or not no_lipid_file.exists():
    _generate_incremental_result(no_lipid_file, include_lipids=False)
else:
    print(f"[SKIP] Existing no-lipid result file: {no_lipid_file}")

if force_rebuild or not with_lipid_file.exists():
    _generate_incremental_result(with_lipid_file, include_lipids=True)
else:
    print(f"[SKIP] Existing with-lipid result file: {with_lipid_file}")

if generate_pattern_pkls:
    _generate_pattern_pickles()

trgt_names, food_pred, _ = get_results(no_lipid_file)
_, food_pred_lipids, _ = get_results(with_lipid_file, trgt_names=trgt_names)

pattern_base = no_lipid_file.parent
no_lip_perm = load_pattern(
    rf"NCD_analysis_incremental_feat_lipids_False_maxfeat_{n_select_features}_permute_True_\d+\.pkl",
    pattern_base,
)
no_lip_no_perm = load_pattern(
    rf"NCD_analysis_incremental_feat_lipids_False_maxfeat_{n_select_features}_permute_False_\d+\.pkl",
    pattern_base,
)
lip_perm = load_pattern(
    rf"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_True_\d+\.pkl",
    pattern_base,
)
lip_no_perm = load_pattern(
    rf"NCD_analysis_incremental_feat_lipids_True_maxfeat_{n_select_features}_permute_False_\d+\.pkl",
    pattern_base,
)

if any(v is None for v in [no_lip_perm, no_lip_no_perm, lip_perm, lip_no_perm]):
    missing = [
        name
        for name, arr in [
            ("no_lip_perm", no_lip_perm),
            ("no_lip_no_perm", no_lip_no_perm),
            ("lip_perm", lip_perm),
            ("lip_no_perm", lip_no_perm),
        ]
        if arr is None
    ]
    raise FileNotFoundError(
        f"Missing grouped-plot pattern results in {pattern_base}. Missing: {', '.join(missing)}"
    )

plot_grouped_var_reduction(
    combined_list=[no_lip_perm, no_lip_no_perm, lip_perm, lip_no_perm],
    trgt_names=trgt_names,
    names_lst=["permute no lipids", "no lipids", "permute with lipids", "with lipids"],
    title="Risk Prediction From Biomarkers (5 features max)",
    figure_name=out_fig_5feat.name,
    save_dir=out_fig_5feat.parent,
    show_plot=False,
)

plot_nolip_lip_var_reduction(
    food_pred=food_pred,
    food_pred_lipids=food_pred_lipids,
    trgt_names=trgt_names,
    figure_name=out_fig.name,
    save_dir=out_fig.parent,
    show_plot=False,
)

print(f"[OK] Figure saved: {out_fig}")
print(f"[OK] Figure saved: {out_fig_5feat}")
print(f"[OK] Intermediate inputs used: {no_lipid_file} | {with_lipid_file}")
PY
