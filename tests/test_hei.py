# ncd_analysis_food_and_conditioning_model.py

from hei_package.data_preparation import prep_data, food_feats, non_food_feats
from hei_package.experiment import load_or_run_feature_selection

TARGET = "GLU (mg/dL)"
MODEL = "XGB"
N_RUNS = 40
N_SELECT_FEATURES = 5

full_feats = (
    food_feats
    + ["age", "gender_numeric", "stress_index", "fatigue_index"]
    + ["weight", "height"]
    + ["GMWI", "microbiome_Shannon"]
    + [c for c in non_food_feats if "dbs_rbc_lip" in c]
)

selected_feats, train_errs, test_errs = load_or_run_feature_selection(
    prep_data,
    full_feats,
    TARGET,
    MODEL,
    N_SELECT_FEATURES,
    N_RUNS,
)
