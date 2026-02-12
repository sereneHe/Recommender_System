import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


from compute_tools import percentile_mask
from xgboost_lagrangian import fit_aug_lagrangian_W_constraint
from xgboost import XGBRegressor


def compute_predictor_errors_scikit(estimator, X, y):
    test_mse = mean_squared_error(y, estimator.predict(X))
    test_bench = mean_squared_error(y, np.ones_like(y) * estimator._y_train_mean_)

    return test_mse / test_bench


class RecommenderBaseEstimator(BaseEstimator):
    def __init__(self, w_est, target_col, row_and_col_names, custom_objective, prep_data):
        if custom_objective is None or custom_objective not in ['lagrange', 'mse_builtin', 'mse_custom']:
            raise ValueError("Custom objective can be only lagrange, mse_builtin, mse_custom")

        self.w_est = w_est  # the exDBN matrix
        self.target_col = target_col
        self.row_and_col_names = row_and_col_names
        self.custom_objective = custom_objective
        self.prep_data = prep_data
        self._rf_model_ = None
        self.feature_names_in_ = None

    def predict(self, X):
        return self._rf_model_.predict(X)

    def preprocess_data(self, X, y):
        mask = percentile_mask(y, 5)
        X = X[mask]
        if y is not None:
            y = y[mask]
        return mask, X, y


class XGBRecommenderPredictor(RecommenderBaseEstimator):
    def get_current_column_names(self, X):
        # SFS posílá X jako numpy array s vyházenými řádky a sloupci
        current_feature_names = []

        for i in range(X.shape[1]):
            col_data = X[:, i]
            for col_name in self.prep_data.columns:
                if col_name in current_feature_names:
                    continue
                it = iter(self.prep_data[col_name])
                if all(any(a == b for a in it) for b in col_data):
                    current_feature_names.append(col_name)
                    break

        return current_feature_names

    def fit(self, X, y=None):
        _, X, y = self.preprocess_data(X, y)
        self._y_train_mean_ = y.mean()
        if self.custom_objective in ['lagrange', 'mse_custom']:
            self.scaler_ = StandardScaler()
            self.scaler_.fit_transform(X)

            row_and_col_names_indices = {name: i for i, name in enumerate(self.row_and_col_names)}
            idx_list = [row_and_col_names_indices[f] for f in self.get_current_column_names(X)]
            predict_idx = row_and_col_names_indices[self.target_col]
            w_est = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]
            self._rf_model_, lam = fit_aug_lagrangian_W_constraint(X, y, w_est, num_boost_round=10)
        else:
            model_class = Pipeline
            model_params = {
                'steps': [
                    ("scale", StandardScaler()),
                    ("xgb", XGBRegressor(
                        n_estimators=10,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42,
                        # tree_method="hist",
                        base_score=y.mean(),
                        objective='reg:squarederror'
                    )
                     )
                ]
            }
            rf_model = model_class(**model_params)
            self._rf_model_ = rf_model.fit(X, y)
            return self

    def predict(self, X):
        if self.custom_objective in ['lagrange', 'mse_custom']:
            X = self.scaler_.transform(X)
            return self._rf_model_.predict(xgb.DMatrix(X))
        else:
            return super().predict(X)


class REGRecommenderPredictor(RecommenderBaseEstimator):
    def fit(self, X, y=None):
        _, X, y = self.preprocess_data(X, y)
        self._y_train_mean_ = y.mean()

        model_class = Pipeline
        model_params = {
            'steps': [
                ("scale", StandardScaler()),
                ("linreg", LinearRegression())
            ]
        }
        rf_model = model_class(**model_params)

        # Train the model
        self._rf_model_ = rf_model.fit(X, y)
        return self


