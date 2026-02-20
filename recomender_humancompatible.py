import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb

from os.path import join

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import solve_milp
from compute_tools import percentile_mask
from recommender_estimator import RecommenderBaseEstimator
from xgboost_lagrangian import fit_aug_lagrangian_W_constraint
from xgboost import XGBRegressor



class HCRecommenderPredictor(RecommenderBaseEstimator):
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
        # self._y_train_mean_ = y.mean()
        if self.custom_objective in ['lagrange', 'mse_custom']:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

            if self.cfg.recalculate_dag:
                d = X.shape[1] + 1 # adding one for y
                X_y = np.column_stack((X, y.to_numpy()))
                # if d <= 2:
                #     w_est = np.zeros((d,d))
                # else:
                print("max X", X.max(), "min X", X.min())
                current_column_names = self.get_current_column_names(X)
                G = nx.read_graphml(join(self.cfg.data_path, self.cfg.knowledge_graph_filename))
                print(G.nodes())
                print(current_column_names + [self.target_col])
                H = G.subgraph(current_column_names + [self.target_col]).copy()
                if H.number_of_nodes() > 0:
                    print('not emty')
                H = nx.complement(H)
                col_to_idx = {col: idx for idx, col in enumerate(current_column_names + [self.target_col])}
                tabu_edges = list((col_to_idx[s],col_to_idx[e]) for (s,e) in H.edges())
                if tabu_edges:
                    print(tabu_edges)
                w_est, _, _, _, _ = solve_milp.solve(X_y, self.cfg, self.cfg.nonzero_threshold,
                                                                        Y=[],
                                                                        B_ref=np.zeros((d,d)),
                                                                        tabu_edges=tabu_edges )
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

