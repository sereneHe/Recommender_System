from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# 复用你已有的 Hydra 化 loader（建议）
# 如果你的文件路径不同，请改这里
from data_helper import load_all_data


def resolve_assets_from_cfg(cfg: DictConfig) -> Dict[str, Path]:
    """
    Resolve assets paths with env override.
    Required in cfg:
      paths.assets_dir
      assets.w_est_zip
      assets.w_est_inner_csv
      assets.intra_nodes_txt
    """
    project_root = Path(__file__).resolve().parents[2]
    assets_dir = Path(os.environ.get("ASSETS_DIR", str(cfg.paths.assets_dir))).expanduser()
    if not assets_dir.is_absolute():
        assets_dir = (project_root / assets_dir).resolve()
    else:
        assets_dir = assets_dir.resolve()

    w_est_zip = assets_dir / str(cfg.assets.w_est_zip)
    intra_nodes = assets_dir / str(cfg.assets.intra_nodes_txt)

    return {"assets_dir": assets_dir, "w_est_zip": w_est_zip, "intra_nodes": intra_nodes}


def load_w_est(zip_path: Path, inner_csv: str = "W_est.csv") -> np.ndarray:
    if not zip_path.exists():
        raise FileNotFoundError(f"W_est zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        if inner_csv not in z.namelist():
            raise FileNotFoundError(f"'{inner_csv}' not found in zip. Found: {z.namelist()}")
        with z.open(inner_csv) as f:
            w_est = np.loadtxt(f, delimiter=",")
    return w_est


def load_intra_nodes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"intra_nodes file not found: {path}")
    s = path.read_text(encoding="utf-8").strip()
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = [x.strip().strip("'").strip('"') for x in s.strip("[]").split(",") if x.strip()]
    return [str(x).strip() for x in parsed if str(x).strip()]


def load_cfg(config_path: str | Path = "src/project_hei/configs/config.yaml") -> DictConfig:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return OmegaConf.load(p)


def get_dataset_and_assets(
    cfg: Optional[DictConfig] = None,
    config_path: str | Path = "src/project_hei/configs/config.yaml",
) -> Tuple[List[str], List[str], pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
    """
    One-shot helper:
      - loads cfg (if not provided)
      - load_all_data(cfg) -> expects {"food_feats","non_food_feats","prep_data", ...}
      - loads assets: W_est + intra_nodes
    Returns:
      food_feats, non_food_feats, prep_data, w_est, row_and_col_names, meta
    """
    if cfg is None:
        cfg = load_cfg(config_path)

    logger.info("Loading data via load_all_data(cfg)...")
    data_dict = load_all_data(cfg)

    # Required outputs (per your earlier change request)
    if "food_feats" not in data_dict or "non_food_feats" not in data_dict or "prep_data" not in data_dict:
        raise RuntimeError(
            "load_all_data(cfg) must return keys: 'food_feats', 'non_food_feats', 'prep_data'."
        )

    food_feats: List[str] = list(data_dict["food_feats"])
    non_food_feats: List[str] = list(data_dict["non_food_feats"])
    prep_data: pd.DataFrame = data_dict["prep_data"]

    logger.info(f"prep_data shape: {prep_data.shape} | food_feats={len(food_feats)} non_food_feats={len(non_food_feats)}")

    assets = resolve_assets_from_cfg(cfg)
    w_est = load_w_est(assets["w_est_zip"], inner_csv=str(cfg.assets.w_est_inner_csv))
    row_and_col_names = load_intra_nodes(assets["intra_nodes"])

    meta = {
        "data_dir": data_dict.get("data_dir"),
        "cache_dir": data_dict.get("cache_dir"),
        "assets_dir": str(assets["assets_dir"]),
        "w_est_zip": str(assets["w_est_zip"]),
        "intra_nodes": str(assets["intra_nodes"]),
    }
    return food_feats, non_food_feats, prep_data, w_est, row_and_col_names, meta
