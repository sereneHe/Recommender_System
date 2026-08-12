import json
import pandas as pd

def load_json_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def hei_analysis_from_config(config_path):
    cfg = load_json_config(config_path)
    prep_data = pd.read_csv(cfg["data_path"])
    return ncd_food_conditioning_analysis(
        prep_data,
        cfg["food_feats"],
        cfg["non_food_feats"],
        cfg["target_columns"],
        model_type=cfg.get("model_type", "gp"),
        n_runs=cfg.get("n_runs", 50),
        n_select_features=cfg.get("n_select_features", 5)
    )

# 用法
results = hei_analysis_from_config("/Users/xiaoyuhe/Recommender_System/hei_analysis.json")