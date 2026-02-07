from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig


from hei_project.hei.recommender_utils import run_feature_selection


def build_full_feats(
    prep_data: pd.DataFrame,
    food_feats: List[str],
    non_food_feats: List[str],
    cfg: Optional[DictConfig] = None,
) -> List[str]:
    """
    Build full feature list for selection.
    If cfg.features exists -> cfg-driven; else fallback to your hard-coded rules.
    """
    # ---- cfg-driven (recommended) ----
    if cfg is not None and "features" in cfg:
        full: List[str] = []
        full += list(food_feats)

        full += list(getattr(cfg.features, "extra_base", []))

        if bool(getattr(cfg.features, "include_anthropometrics", True)):
            full += list(getattr(cfg.features, "anthropometrics", ["weight", "height"]))

        full += list(getattr(cfg.features, "extra_non_food", []))

        if bool(getattr(cfg.features, "include_dbs_rbc_lip", True)):
            kw = str(getattr(cfg.features, "dbs_rbc_lip_keyword", "dbs_rbc_lip"))
            full += [c for c in non_food_feats if kw in c]

        for prefix in list(getattr(cfg.features, "include_prefixes", ["microb_clean15_"])):
            full += [c for c in prep_data.columns if str(c).startswith(prefix)]

        # Keep only existing cols, de-dup preserve order
        seen = set()
        out: List[str] = []
        for f in full:
            if f in prep_data.columns and f not in seen:
                seen.add(f)
                out.append(f)
        return out

    # ---- fallback: your original hard-coded logic ----
    full_feats: List[str] = []
    full_feats += list(food_feats)
    full_feats += [
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
    full_feats += [c for c in non_food_feats if "dbs_rbc_lip" in c]
    full_feats += [n for n in prep_data.columns if "microb_clean15_" in n]

    full_feats = [f for f in full_feats if f in prep_data.columns]
    seen = set()
    out = []
    for f in full_feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def run_recommender(
    *,
    food_feats: List[str],
    non_food_feats: List[str],
    prep_data: pd.DataFrame,
    w_est: np.ndarray,
    row_and_col_names: List[str],
    model_name: str,
    custom_objective: str,
    n_select_features: int,
    n_runs: int,
    target_columns: List[str],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, Tuple[List[str], List[float], List[float]]]:
    """
    Core recommender routine:
      - builds full feature list
      - runs run_feature_selection for each target
      - returns {target: (selected_feats, train_errs, test_errs)}
    """
    assert custom_objective in ["lagrange", "mse_custom", "mse_builtin"], \
        "custom_objective must be one of: lagrange, mse_custom, mse_builtin"

    full_feats = build_full_feats(prep_data, food_feats, non_food_feats, cfg)

    # Log full feat list once per run
    mlflow.log_text("[" + ", ".join(full_feats) + "]", "full_feats_list.txt")

    res_dict: Dict[str, Tuple[List[str], List[float], List[float]]] = {}

    for target_col in target_columns:
        curr_feats, curr_train_errs, curr_test_errs = run_feature_selection(
            prep_data=prep_data,
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
        res_dict[target_col] = (list(curr_feats), list(curr_train_errs), list(curr_test_errs))

    return res_dict
