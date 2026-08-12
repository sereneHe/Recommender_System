import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import openpyxl
import os
import re

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import BaggingRegressor

from mk_utils import (
    MedianKNNRegressor,
    corr,
    percentile_mask,
)

# -----------------------
# utils
# -----------------------
def l1_err(u, v):
    return np.abs(u - v).mean()


def build_model(cfg):
    if cfg.model.name == "bagged_knn":
        return Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                (
                    "regressor",
                    BaggingRegressor(
                        estimator=MedianKNNRegressor(
                            n_neighbors=cfg.model.knn.n_neighbors
                        ),
                        n_estimators=cfg.model.bagging.n_estimators,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")


# -----------------------
# main
# -----------------------
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    print("==== Hydra config ====")
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)

    # -----------------------
    # load prepared data (你原来的逻辑)
    # -----------------------
    hei_data = pd.read_pickle(cfg.data.paths.hei)
    
    try:
        blood_data = pd.read_pickle("/Users/xiaoyuhe/Recommender_System/data/process/blood_data.pkl")
        # blood_discription = pd.read_pickle("data/blood_biochemistry/blood_discription.pkl")
        site_data = pd.read_pickle("/Users/xiaoyuhe/Recommender_System/data/process/site_data.pkl")
    except FileNotFoundError:
        file = "/Users/xiaoyuhe/Recommender_System/data/raw/blood_biochemistry/bloodbiochemistry.xlsx"
        data = pd.read_excel(file)
        # 你可以在这里添加数据预处理和保存逻辑

    try:
        energy_expenditure = pd.read_pickle("/Users/xiaoyuhe/Recommender_System/data/process/expenditure.pkl")
        average_expenditure = pd.read_pickle("/Users/xiaoyuhe/Recommender_System/data/raw/energy_expenditure/average_expenditure.pkl")
    except FileNotFoundError:
        energy_expenditure = pd.DataFrame()
        for file in os.listdir("/Users/xiaoyuhe/Recommender_System/data/raw/energy_expenditure"):
            if file.endswith(".csv"):
                tee = pd.read_csv(os.path.join("/Users/xiaoyuhe/Recommender_System/data/raw/energy_expenditure", file))
                tee = tee.dropna(subset=["timepoint"])
                tee = tee[ tee["sample_id"].apply(lambda x:   len(re.findall(r"\d+", str(x))) == 2  ) ]
                tee.insert(1, "ID", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", str(x))[0])))
                tee.insert(2, "VISIT", tee["sample_id"].apply(lambda x: int(re.findall(r"\d+", str(x))[1])))
                tee.drop(columns=["sample_id"], inplace=True)
                tee.reset_index(drop=True, inplace=True)
                if energy_expenditure.empty:
                    energy_expenditure = tee
                else:
                    energy_expenditure = pd.concat([energy_expenditure, tee], ignore_index=True)
        energy_expenditure.rename(columns={"TEE2": "TEE", "TEE":"TEE_orig"}, inplace=True)
        energy_expenditure["timepoint"].astype(int)
        average_expenditure = energy_expenditure.groupby(["ID", "VISIT"]).agg({"TEE": "mean"}).reset_index()
        average_expenditure.to_pickle("/Users/xiaoyuhe/Recommender_System/data/process/average_expenditure.pkl")
        energy_expenditure.to_pickle("/Users/xiaoyuhe/Recommender_System/data/process/expenditure.pkl")

    base_data = hei_data.select_dtypes(include=["number"])
    X = base_data[["heitotpro"]].to_numpy()

    # target
    y_col = f"{cfg.target.name}_delta_normed"
    print("[DEBUG] Columns in base_data:", list(base_data.columns))
    print("[DEBUG] Attempting to use target column:", y_col)
    if y_col not in base_data.columns:
        print(f"[ERROR] Target column '{y_col}' not found in base_data! Please check your config and data preparation.")
        raise KeyError(y_col)
    Y = base_data[y_col].to_numpy() * 100

    # clean NaN
    mask = ~np.isnan(Y)
    X, Y = X[mask], Y[mask]

    mask = ~np.isnan(X).any(axis=1)
    X, Y = X[mask], Y[mask]

    # percentile clip
    p = cfg.experiment.percentile_clip
    keep = percentile_mask(Y, p, 100 - p)
    X, Y = X[keep], Y[keep]

    # -----------------------
    # CV
    # -----------------------
    rs = ShuffleSplit(
        n_splits=cfg.experiment.n_splits,
        test_size=int(cfg.experiment.test_ratio * len(X)),
        random_state=np.random.randint(1_000_000),
    )

    train_err, test_err = [], []
    train_corr, test_corr = [], []

    for i, (tr, te) in enumerate(rs.split(X, Y)):
        if i % 10 == 0:
            print(f"Split {i}")

        model = build_model(cfg)
        model.fit(X[tr], Y[tr])

        y_tr_pred = model.predict(X[tr])
        y_te_pred = model.predict(X[te])

        train_err.append(l1_err(y_tr_pred, Y[tr]))
        test_err.append(l1_err(y_te_pred, Y[te]))

        train_corr.append(corr(y_tr_pred, Y[tr], cfg.experiment.corr_type))
        test_corr.append(corr(y_te_pred, Y[te], cfg.experiment.corr_type))

    print("==== Results ====")
    print(f"Train err: {np.mean(train_err):.4f}")
    print(f"Test  err: {np.mean(test_err):.4f}")
    print(f"Train corr: {np.mean(train_corr):.4f}")
    print(f"Test  corr: {np.mean(test_corr):.4f}")


if __name__ == "__main__":
    main()
