import networkx as nx
import numpy as np
import xgboost as xgb
import logging

from os.path import join

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from hc_predictor import fit_aug_lagrangian_nn_constraint as fit_hc_lagrangian_nn_constraint
from nn_causal_constraints_lagrangian import (
    fit_aug_lagrangian_nn_constraint as fit_ci_ce_lagrangian_nn_constraint,
)
import solve_milp
from compute_tools import percentile_mask
from xgboost_lagrangian import fit_aug_lagrangian_W_constraint
from xgboost import XGBRegressor

import torch
from omegaconf import open_dict


@torch.no_grad
def compute_predictor_errors_scikit(estimator, X, y):
    test_mse = mean_squared_error(y, estimator.predict(X))
    return test_mse

@torch.no_grad
def compute_predictor_errors_and_cs_scikit(estimator, X, y, W):
    y_pred = np.array(estimator.predict(X))
    test_mse = mean_squared_error(y, y_pred)
    if estimator._y_normalized:
        y_pred_norm = (y_pred - estimator._y_mean)/ estimator._y_std
    X = estimator.scaler_.transform(X) 
    M = W - np.eye(X.shape[1] + 1)
    muX = X.mean(axis=0)
    g0 = M[:, :-1] @ muX
    v = M[:, -1]
    test_c = np.linalg.norm(g0 + y_pred_norm.mean(axis=0) * v)

    return {'score': test_mse, 'c': test_c}


def compute_recalculated_w_est(prep_data, target_col, cfg):
    """Compute a single MILP DAG once for the full problem and reuse it later."""
    prep_data = prep_data.dropna(subset=[target_col])
    prep_data = prep_data.dropna(axis=1)
    configured_features = getattr(cfg, "current_feature_names", None)
    if configured_features is not None:
        current_column_names = [str(col) for col in configured_features if str(col) in prep_data.columns]
    else:
        current_column_names = [str(col) for col in prep_data.columns if col != target_col]
    X = prep_data.drop(target_col, axis=1) if target_col in prep_data.columns else prep_data
    X = X[current_column_names]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = prep_data[target_col]
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    d = X.shape[1] + 1
    X_y = np.column_stack((X, y_scaled.to_numpy()))
    G = nx.read_graphml(join(cfg.data_path, cfg.knowledge_graph_filename))
    H = G.subgraph(current_column_names + [target_col]).copy()
    H = nx.complement(H)
    col_to_idx = {col: idx for idx, col in enumerate(current_column_names + [target_col])}
    tabu_edges = [(col_to_idx[s], col_to_idx[e]) for (s, e) in H.edges()]
    w_est, _, _, _, _ = solve_milp.solve(
        X_y,
        cfg,
        cfg.nonzero_threshold,
        Y=[],
        B_ref=np.zeros((d, d)),
        tabu_edges=tabu_edges,
    )
    return w_est

class RecommenderBaseEstimator(BaseEstimator):
    def __init__(self, w_est, target_col, row_and_col_names, custom_objective, prep_data, cfg):
        if custom_objective is None or custom_objective not in ['lagrange', 'mse_builtin', 'mse_custom', 'reg:squarederror']:
            raise ValueError("Custom objective can be only lagrange, mse_builtin, mse_custom")

        self.w_est = w_est  # the exDBN matrix
        self.target_col = target_col
        self.row_and_col_names = row_and_col_names
        self.custom_objective = custom_objective
        self.prep_data = prep_data
        self._rf_model_ = None
        self.feature_names_in_ = None
        self._w_est = None
        self.cfg = cfg

    def predict(self, X):
        return self._rf_model_.predict(X)

    def preprocess_data(self, X, y):
        mask = percentile_mask(y, 5)
        X = X[mask]
        if y is not None:
            y = y[mask]
        return mask, X, y

    def get_current_column_names(self, X):
        if hasattr(X, "columns"):
            return [str(col) for col in X.columns]

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


class XGBRecommenderPredictor(RecommenderBaseEstimator):
    def get_current_column_names(self, X):
        if hasattr(X, "columns"):
            return [str(col) for col in X.columns]

        # SFS posílá X jako numpy array s vyházenými řádky a sloupci
        current_feature_names = []

        for i in range(X.shape[1]):
            col_data = X[:, i]
            ati = set()
            for col_name in self.prep_data.columns:
                if col_name in current_feature_names:
                    continue
                it = iter(self.prep_data[col_name])
                if all(any(a == b for a in it) for b in col_data):
                    current_feature_names.append(col_name)
                    if not self.cfg.debug:
                        break
                    ati.add(col_name)
                    if(len(ati) > 1):
                        logging.warning(f"Multiple features mapping to a column: {ati}")

        return current_feature_names

    def fit(self, X, y=None):
        _, X, y = self.preprocess_data(X, y)
        current_column_names = self.get_current_column_names(X)
        #self._y_train_mean_ = y.mean()
        if self.custom_objective in ['lagrange', 'mse_custom']:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X) # y?
            self._y_mean = y.mean()
            self._y_std = y.std()
            y_scaled = (y - self._y_mean) / self._y_std # ExDBN may not work well, if we do not normalize also y

            if self.cfg.recalculate_dag:
                w_est = compute_recalculated_w_est(self.prep_data, self.target_col, self.cfg)
                self._w_est = w_est

            else:
                row_and_col_names_indices = {name: i for i, name in enumerate(self.row_and_col_names)}
                idx_list = [row_and_col_names_indices[f] for f in current_column_names]
                predict_idx = row_and_col_names_indices[self.target_col]
                w_est = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]
                self._w_est = w_est

            # call exdbn


            self._rf_model_, lam = fit_aug_lagrangian_W_constraint(X, y_scaled, w_est, self.cfg)
            self._y_normalized = True
        else:
            model_class = Pipeline
            model_params = {
                'steps': [
                    ("scale", StandardScaler()),
                    ("xgb", XGBRegressor(
                        n_estimators=self.cfg.n_estimators,
                        max_depth=self.cfg.max_depth,
                        learning_rate=self.cfg.learning_rate,
                        random_state=self.cfg.random_state,
                        # tree_method="hist",
                        # base_score=y.mean(),
                        #objective='reg:squarederror'
                    )
                     )
                ]
            }
            rf_model = model_class(**model_params)
            self._rf_model_ = rf_model.fit(X, y)
            self._y_normalized = False
        return self

    def predict(self, X):
        if self.custom_objective in ['lagrange', 'mse_custom']:
            X = self.scaler_.transform(X)
            prediction = self._rf_model_.predict(xgb.DMatrix(X))
        else:
            prediction = super().predict(X)

        if self._y_normalized:
            prediction = prediction * self._y_std + self._y_mean
        return prediction





class HCRecommenderPredictor(RecommenderBaseEstimator):
    fit_lagrangian_nn_constraint = staticmethod(fit_hc_lagrangian_nn_constraint)
    inject_ci_context = False
    use_validation_selection = False

    def fit(self, X, y=None):
        _, X, y = self.preprocess_data(X, y)
    # self._y_train_mean_ = y.mean()
        current_column_names = self.get_current_column_names(X)
        X_fit = X
        y_fit = y
        X_val = None
        y_val = None
        use_validation = self.use_validation_selection and bool(
            getattr(self.cfg, "use_validation", True)
        )
        if use_validation:
            val_fraction = float(getattr(self.cfg, "validation_fraction", 0.2))
            if 0.0 < val_fraction < 1.0 and len(y) >= 3:
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X,
                    y,
                    test_size=val_fraction,
                    random_state=int(getattr(self.cfg, "validation_random_state", 0)),
                    shuffle=True,
                )
            else:
                logging.warning(
                    "Skipping validation split: validation_fraction=%s, n_samples=%s",
                    val_fraction,
                    len(y),
                )
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X_fit)
        X_val_scaled = self.scaler_.transform(X_val) if X_val is not None else None
        if self.inject_ci_context:
            with open_dict(self.cfg):
                self.cfg.current_feature_names = list(current_column_names)
                self.cfg.current_target_name = self.target_col
        self._y_mean = y_fit.mean()
        self._y_std = y_fit.std()
        y = (y_fit - self._y_mean) / self._y_std
        y_val_scaled = (y_val - self._y_mean) / self._y_std if y_val is not None else None
        self._y_normalized = True

        if self.cfg.recalculate_dag:
            w_est = compute_recalculated_w_est(self.prep_data, self.target_col, self.cfg)
            self._w_est = w_est

        else:
            # breakpoint()
            row_and_col_names_indices = {name: i for i, name in enumerate(self.row_and_col_names)}
            idx_list = [row_and_col_names_indices[f] for f in current_column_names]
            predict_idx = row_and_col_names_indices[self.target_col]
            w_est = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]
            self._w_est = w_est

        self._rf_model_, lam = self.fit_lagrangian_nn_constraint(
            X,
            y,
            w_est,
            self.cfg,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
        )
        self.validation_history_ = getattr(self._rf_model_, "validation_history_", None)

        return self
        

    def predict(self, X):
        if self.custom_objective == 'lagrange':
            X = self.scaler_.transform(X)
            prediction = self._rf_model_.predict(X)
            if self._y_normalized:
                prediction = prediction * self._y_std + self._y_mean
            return prediction 
        else:
            return super().predict(X)


class HCCIRecommenderPredictor(HCRecommenderPredictor):
    fit_lagrangian_nn_constraint = staticmethod(fit_ci_ce_lagrangian_nn_constraint)
    inject_ci_context = True


class HCCERecommenderPredictor(HCRecommenderPredictor):
    fit_lagrangian_nn_constraint = staticmethod(fit_ci_ce_lagrangian_nn_constraint)
    inject_ci_context = True
    use_validation_selection = True


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


