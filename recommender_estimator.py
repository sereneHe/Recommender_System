import networkx as nx
import numpy as np
import xgboost as xgb
import logging

from os.path import join

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from nn_lagrangian import fit_aug_lagrangian_nn_constraint
import solve_milp
from compute_tools import percentile_mask
from xgboost_lagrangian import fit_aug_lagrangian_W_constraint
from xgboost import XGBRegressor

import torch


@torch.no_grad
def compute_predictor_errors_scikit(estimator, X, y):
    test_mse = mean_squared_error(y, estimator.predict(X))
    #test_bench = mean_squared_error(y, np.ones_like(y) * estimator._y_train_mean_)

    return test_mse # / test_bench


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


class XGBRecommenderPredictor(RecommenderBaseEstimator):
    def get_current_column_names(self, X):
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
        #self._y_train_mean_ = y.mean()
        if self.custom_objective in ['lagrange', 'mse_custom']:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X) # y?
            self._y_mean = y.mean()
            self._y_std = y.std()
            y_scaled = (y - self._y_mean) / self._y_std # ExDBN may not work well, if we do not normalize also y

            if self.cfg.recalculate_dag:
                d = X.shape[1] + 1 # adding one for y
                X_y = np.column_stack((X, y_scaled.to_numpy()))
                # if d <= 2:
                #     w_est = np.zeros((d,d))
                # else:
                #print("max X", X.max(), "min X", X.min())
                current_column_names = self.get_current_column_names(X)
                G = nx.read_graphml(join(self.cfg.data_path, self.cfg.knowledge_graph_filename))
                #g_nodes = list(G.nodes())
                #print(G.nodes())
                #print(current_column_names + [self.target_col])
                H = G.subgraph(current_column_names + [self.target_col]).copy()
                # if H.number_of_nodes() > 0:
                #     print('not emty')
                H = nx.complement(H)
                col_to_idx = {col: idx for idx, col in enumerate(current_column_names + [self.target_col])}
                tabu_edges = list((col_to_idx[s],col_to_idx[e]) for (s,e) in H.edges())
                # if tabu_edges:
                #     print(tabu_edges)
                w_est, _, _, _, _ = solve_milp.solve(X_y, self.cfg, self.cfg.nonzero_threshold,
                                                                        Y=[],
                                                                        B_ref=np.zeros((d,d)),
                                                                        tabu_edges=tabu_edges )
                self._w_est = w_est # dont read this property if feature selector is used.
                #print(w_est)

                # row_and_col_names_indices = {name: i for i, name in enumerate(self.row_and_col_names)}
                # idx_list = [row_and_col_names_indices[f] for f in self.get_current_column_names(X)]
                # predict_idx = row_and_col_names_indices[self.target_col]
                # w_est2 = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]
                #print(w_est2)

            else:
                row_and_col_names_indices = {name: i for i, name in enumerate(self.row_and_col_names)}
                idx_list = [row_and_col_names_indices[f] for f in self.get_current_column_names(X)]
                predict_idx = row_and_col_names_indices[self.target_col]
                w_est = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]

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
    def get_current_column_names(self, X):
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
    # self._y_train_mean_ = y.mean()
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        if self.cfg.recalculate_dag:
            d = X.shape[1] + 1 # adding one for y
            X_y = np.column_stack((X, y.to_numpy()))
            # if d <= 2:
            #     w_est = np.zeros((d,d))
            # else:
            # print("max X", X.max(), "min X", X.min())
            current_column_names = self.get_current_column_names(X)
            G = nx.read_graphml(join(self.cfg.data_path, self.cfg.knowledge_graph_filename))
            # print(G.nodes())
            # print(current_column_names + [self.target_col])
            H = G.subgraph(current_column_names + [self.target_col]).copy()
            if H.number_of_nodes() > 0:
                print('not emty')
            H = nx.complement(H)
            col_to_idx = {col: idx for idx, col in enumerate(current_column_names + [self.target_col])}
            tabu_edges = list((col_to_idx[s],col_to_idx[e]) for (s,e) in H.edges())
            # if tabu_edges:
            #     print(tabu_edges)
            w_est, _, _, _, _ = solve_milp.solve(X_y, self.cfg, self.cfg.nonzero_threshold,
                                                                    Y=[],
                                                                    B_ref=np.zeros((d,d)),
                                                                    tabu_edges=tabu_edges )
            self._w_est = w_est

        else:
            row_and_col_names_indices = {name: i for i, name in enumerate(self.row_and_col_names)}
            idx_list = [row_and_col_names_indices[f] for f in self.get_current_column_names(X)]
            predict_idx = row_and_col_names_indices[self.target_col]
            w_est = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]
                
        self._rf_model_, lam = fit_aug_lagrangian_nn_constraint(X, y, w_est, self.cfg)
        

    def predict(self, X):
        if self.custom_objective == 'lagrange':
            X = self.scaler_.transform(X)
            return self._rf_model_.predict(X)
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


