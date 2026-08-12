#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NCD Analysis - HEI project
Multi-target GPR with ARD, consumes load_all_data() output
"""

import argparse
import os
import pickle as pkl
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer

from hei_package.data_load import load_all_data
from hei_package.build_prep_data import build_prep_data


# =========================================================
# Feature construction
# =========================================================

def build_features(prep_data: pd.DataFrame, max_features: int):
    """
    Build feature list for modeling.
    """
    food_feats = [c for c in prep_data.columns if 'food_' in c]
    non_food_feats = [c for c in prep_data.columns if 'dbs_rbc_lip' in c]

    full_feats = []
    full_feats += food_feats
    base_feats = ['age', 'gender_numeric', 'stress_index', 'fatigue_index', 'mean_hrt']
    full_feats += base_feats

    # 自动加入所有 site_continental one-hot 列
    site_cols = [c for c in prep_data.columns if c.startswith('site_continental')]
    full_feats += site_cols

    full_feats += ['weight', 'height', 'BMI']
    full_feats += ['GMWI', 'microbiome_Shannon']
    full_feats += non_food_feats
    full_feats += [c for c in prep_data.columns if 'microb_clean15_' in c]

    return full_feats[:max_features]

# =========================================================
# Model
# =========================================================

def build_model(n_features: int):
    """
    Build GP pipeline with imputer + scaler + GP
    """
    kernel = RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # 均值填充 NaN
        ("scale", StandardScaler()),
        ("gp", GaussianProcessRegressor(kernel=kernel, alpha=1.0, normalize_y=True, random_state=42))
    ])
    return model


def train_model(prep_data, features, target):
    X = prep_data[features]
    y = prep_data[target]

    # 删除 X 或 y 中含 NaN 的行
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    print(f"[train_model] {target}: {X_clean.shape[0]} samples after dropping NaN")

    model = build_model(len(features))
    model.fit(X_clean, y_clean)

    gp = model.named_steps['gp']
    length_scale = gp.kernel_.length_scale_ if hasattr(gp.kernel_, 'length_scale_') else None

    return model, length_scale


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HEI NCD Analysis (Multi-target GPR)")
    parser.add_argument("--data-dir", type=str, default="./data/raw", help="Folder containing raw datasets")
    parser.add_argument("--targets", nargs="+", default=["GLU (mg/dL)"], help="List of target columns")
    parser.add_argument("--max-features", type=int, default=20, help="Maximum number of features")
    parser.add_argument("--out-dir", type=str, default="models", help="Directory to save models and length-scales")
    return parser.parse_args()


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    # 自动创建输出目录，防止保存模型时报错
    # 绝对路径保存模型，防止路径混乱
    abs_out_dir = os.path.abspath(args.out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    print(f"Process ID: {os.getpid()}")
    print(f"Targets: {args.targets}")
    print(f"Max features: {args.max_features}")

    # Load all HEI project data
    data_dict = load_all_data(args.data_dir)
    # 支持直接读取 prep_data_path
    if 'prep_data_path' in data_dict:
        print(f"[ncd_analysis] 直接从 {data_dict['prep_data_path']} 读取 prep_data")
        prep_data = pd.read_csv(data_dict['prep_data_path'])
    else:
        prep_data = build_prep_data(data_dict)

    features = build_features(prep_data, args.max_features)
    print(f"Using {len(features)} features: {features}")
    print(f"prep_data shape: {prep_data.shape}")
    print(f"prep_data columns: {prep_data.columns.tolist()}")
    print(f"prep_data sample (head):\n{prep_data.head()}\n")

    length_scale_records = []

    for target in args.targets:
        print(f"\nTraining model for target: {target}")
        if target not in prep_data.columns:
            print(f"  WARNING: Target {target} not in data. Skipping.")
            continue

        model, length_scale = train_model(prep_data, features, target)

        # Save model
        # 再次确保输出目录存在
        os.makedirs(abs_out_dir, exist_ok=True)
        # 替换特殊字符，保证文件名安全
        safe_target = (
            target.replace(' ', '_')
                  .replace('(', '')
                  .replace(')', '')
                  .replace('/', '_')
                  .replace('%', 'pct')
                  .replace('[', '_')
                  .replace(']', '_')
                  .replace('^', '_')
                  .replace('.', '_')
        )
        model_path = os.path.join(abs_out_dir, f"{safe_target}_model.pkl")
        with open(model_path, "wb") as f:
            pkl.dump(model, f)
        print(f"  Model saved to {model_path}")

        # Record length-scale
        if length_scale is not None:
            length_scale_records.append({
                "target": target,
                **{feat: ls for feat, ls in zip(features, length_scale)}
            })

    # Save all length-scales
    if length_scale_records:
        df_ls = pd.DataFrame(length_scale_records)
        ls_path = os.path.join(args.out_dir, "length_scales.csv")
        df_ls.to_csv(ls_path, index=False)
        print(f"\nAll ARD length-scales saved to {ls_path}")


if __name__ == "__main__":
    main()
