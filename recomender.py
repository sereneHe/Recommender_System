from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import pickle as pkl

from xgboost import XGBRegressor

from data_helper import load_all_data
from recommender_utils import run_feature_selection

food_feats, non_food_feats, prep_data = load_all_data()

model_class = Pipeline
model_params = {
    'steps': [
        ("scale", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
         )
    ]
}
model_name = 'XGB'

model_factory = None

N_SELECT_FEATURES = 5

# evaluate a feature sequence
n_runs = 40
# n_runs = 2

food_feats_orig = ["TKCAL", "WHOLEFRT", "MONOPOLY", "ALLMEAT", "SEAPLANT", "ADDSUGC", "SOLFATC", "TALCO",
                   "T_F_TOTAL", "T_G_WHOLE", "T_D_TOTAL", "TSFAT", "TSODI", "T_G_REFINED", "EMPTYCAL10",
                   "T_V_TOTAL", "T_V_DRKGR", "T_V_LEGUMES"
                   ]

full_feats = [] + food_feats

# full_feats += base_feats
full_feats += ['age', 'gender_numeric', 'stress_index',
               'fatigue_index', 'mean_hrt', 'site_continental'
               ]

# exclude if doing whtr target
full_feats += ['weight', 'height']

full_feats += ['GMWI', 'microbiome_Shannon']
full_feats += [c for c in non_food_feats if 'dbs_rbc_lip' in c]

# full_feats += [n for n in prep_data.columns if 'microb_4cl_' in n]
# full_feats += [n for n in prep_data.columns if 'microb_phyl4cl_' in n]
# full_feats += [n for n in prep_data.columns if 'microb_large_' in n]
full_feats += [n for n in prep_data.columns if 'microb_clean15_' in n]

# print(full_feats)


"""
target_columns = ['GLU (mg/dL)', 'HDL (mg/dL)', 'LDL (mg/dL)', 'TRIG (mg/dL)', 'HbA1c (%)',
                  'Systolic Blood Pressure (mm Hg)', 'Diastolic Blood Pressure (mm Hg)',
                  'CRP (mg/dL)', 'whtr(waist-height_ratio)']
"""
target_columns = ['GLU (mg/dL)']
# target_columns = ['whtr(waist-height_ratio)']


prep_data_orig = prep_data.copy(deep=True)

# seed_rng = np.random.default_rng(99837643)
# seed = seed_rng.integers(low=0, high=100000, size=1)[0]
# print(f'Running with seed: {seed}')
# rng = np.random.default_rng(seed)

res_dict = {}

for target_col in target_columns:
    curr_feats, curr_train_errs, curr_test_errs = run_feature_selection(
        prep_data,
        model_class, model_params,
        target_col,
        n_runs, N_SELECT_FEATURES,
        full_feats,
        model_factory=model_factory
    )

    res_dict[target_col] = (curr_feats, curr_train_errs, curr_test_errs)


