import mlflow
import numpy as np
import zipfile
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import pickle as pkl

from xgboost import XGBRegressor

from data_helper import load_all_data
from recommender_utils import run_feature_selection



def run_recommender(food_feats, non_food_feats, prep_data, w_est, row_and_col_names, model_name, custom_objective, N_SELECT_FEATURES, n_runs):

    def custom_mse_obj(y_true, y_pred):
        # L = 0.5 * (y_pred - y_true)^2
        grad = y_pred - y_true # grad = dL/dy_pred = (y_pred - y_true)
        hess = np.ones_like(y_pred) # hess = d^2L/dy_pred^2 = 1
        return grad, hess

    if model_name == 'XGB':
        model_class = Pipeline
        model_params = {
            'steps': [
                ("scale", StandardScaler()),
                ("xgb", XGBRegressor(
                    n_estimators=10,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    objective=custom_mse_obj if custom_objective == 'lagrange' else "reg:squarederror" #custom_se #
                )
                 )
            ]
        }
    #model_name = 'XGB'
    elif model_name == 'REG':

        model_class = Pipeline
        model_params = {
            'steps': [
                ("scale", StandardScaler()),
                ("linreg", LinearRegression())
            ]
        }
        #model_name = 'REG'

    model_factory = None

    #N_SELECT_FEATURES = 5

    # evaluate a feature sequence
    #n_runs = 40
    ## n_runs = 2

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

    mlflow.log_text("\n".join(full_feats) + "\n", "full_feats_list.txt")

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

    return res_dict


if __name__ == "__main__":
    food_feats, non_food_feats, prep_data = load_all_data()
    with zipfile.ZipFile("./data/W_est.csv.zip") as z:
        with z.open("W_est.csv") as f:
            w_est = np.loadtxt(f, delimiter=",")
        # w_est = np.loadtxt("./data/W_est.csv", delimiter=",")

    s = Path('./data/intra_nodes.txt').read_text(encoding="utf-8").strip()
    row_and_col_names = [x.strip() for x in s.strip("[]").split(",")]

    print(row_and_col_names)
    result = run_recommender(food_feats, non_food_feats, prep_data, w_est, row_and_col_names)

    print(result)


"""
model_class = Pipeline
model_params = {
    'steps': [
        ("scale", StandardScaler()),
        ("linreg", LinearRegression())
    ]
}
model_name = 'REG'
"""

"""
model_class = Pipeline
model_params = {
    'steps': [
        ("DT",DecisionTreeRegressor(max_depth=3,min_samples_leaf=10))        
    ]
}
model_name = 'DT'
"""


"""
model_class = Pipeline
model_params = {
    'steps': [
        ("RF",RandomForestRegressor(n_estimators=3, 
            max_depth=3,min_samples_leaf=10,
            n_jobs = 10)
        )        
    ]
}
model_name = 'RF'
"""

"""
(n_estimators=100, 
criterion='squared_error', 
max_depth=None, 
min_samples_split=2, min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, 
max_features=1.0, 
max_leaf_nodes=None, 
min_impurity_decrease=0.0, 
bootstrap=True, oob_score=False, 
n_jobs=None, random_state=None, verbose=0, 
"""

"""
from sklearn.kernel_ridge import KernelRidge
model_class = Pipeline
model_params = {
    'steps': [
        ("scale", StandardScaler()),
        ("kerreg", KernelRidge(alpha=.5, kernel='rbf', gamma=1e-2))
    ]
}
model_name = 'KerREG'
"""

""""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import warnings
from sklearn.exceptions import ConvergenceWarning

# Option 1: ignore all ConvergenceWarning (global, simplest)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def model_factory(d):
    #d = len(full_feats)
    kernel = RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))

    model_class = Pipeline
    model_params = {
        'steps': [
            ("scale", StandardScaler()),
            ("GP",GaussianProcessRegressor(kernel=kernel, 
                                           alpha=1., 
                                           normalize_y=True, 
                                           n_restarts_optimizer=3,
                                          )
            )        
        ]
    }
    model_name = 'GP'
    return model_class, model_params

model_class = None
model_params = None

"""
