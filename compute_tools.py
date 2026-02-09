from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def get_pca(prep_data, objective_df, pca_objective_name, n_dims=5,
            label_col='TRIG (mg/dL)',  # 'triglyceride',
            do_print=True
            ):
    ids = objective_df['ID'].to_numpy()

    X_orig = objective_df.drop(columns=['ID']).to_numpy()

    # Standardize the features (recommended for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)

    if label_col is not None:
        y = prep_data[label_col].to_numpy()
    else:
        y = np.ones(prep_data.shape[0])

    # Initialize PCA
    pca = PCA(n_components=n_dims, random_state=42)

    # Fit and transform the data
    X_pca = pca.fit_transform(X_scaled)

    if do_print:
        # Plot the first 2 dimensions
        plt.figure(figsize=(12, 5))

        # Subplot 1: PCA scatter plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'PCA {pca_objective_name}: First 2 Principal Components')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Explained variance ratio
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, alpha=0.7)
        plt.title('Explained Variance by Principal Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary information
        print(f"Scaled shape: {X_scaled.shape}")
        print(f"PCA shape: {X_pca.shape}")
        print(f"Number of components: {pca.n_components_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        print(f"First 2 components explain: {pca.explained_variance_ratio_[:2].sum():.1%} of variance")

        # Print individual explained variance ratios
        print("\nExplained variance ratio per component:")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i + 1}: {ratio:.3f} ({ratio:.1%})")

        # Optional: Save the PCA coordinates
        # np.savetxt('pca_coordinates.csv', X_pca, delimiter=',')

        # Access PCA components (loadings) if needed
        print(f"\nPCA components shape: {pca.components_.shape}")
        # Each row is a principal component, each column corresponds to original features

    pca_df = pd.DataFrame(X_pca, columns=[pca_objective_name + f'.pca_{i}' for i in range(n_dims)])
    pca_df.insert(0, 'ID', ids)

    return pca_df

from scipy.stats import spearmanr, kendalltau
from mk_utils import clean,corr, perm_test_pval, average_visits, split_visits_from_column


from mk_utils import MedianKNNRegressor, percentile_mask, corr, perm_test_pval
from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import LinearRegression


def compute_cov(u, v, corr_type, n_permutes=500, seed=None):
    u, v = clean(u, v)

    """
    if len(u)< size_threshold:
        return np.nan, np.nan

    #the pval will be bad if small sample
    """
    if len(u) == 0:
        return np.nan, np.nan

    corr_v = corr(u, v, corr_type=corr_type)
    pval_v = perm_test_pval(u, v, corr_type=corr_type, n_permutes=n_permutes, seed=seed)

    return corr_v, pval_v


def percentile_mask(Y, outlier_percentile):
    # clean target outliers
    Y_num = Y[~np.isnan(Y)]

    lower_threshold = np.percentile(Y_num, outlier_percentile)
    upper_threshold = np.percentile(Y_num, 100 - outlier_percentile)

    mask = (Y >= lower_threshold) & (Y <= upper_threshold)
    return mask


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


def compute_predictor_errors(prep_data, hei_feats, target_col, w_est, row_and_col_names,
                             do_print=True, stack_linear=False,
                             save_details=None,
                             model_name=None,
                             custom_objective=None,
                             compute_covs=True
                             ):
    assert not (stack_linear and (save_details is not None)), 'unsupported combination'

    # data_cols = prep_data.columns
    # print(len(data_cols))

    reg_dat = prep_data[hei_feats + [target_col]]
    reg_dat = reg_dat.dropna()

    X = reg_dat[hei_feats].to_numpy()
    Y = reg_dat[target_col].to_numpy()

    mask = percentile_mask(Y, 5)
    X = X[mask]
    Y = Y[mask]

    # split_seed = np.random.randint(1000000)
    rng = np.random.default_rng(seed=None)
    split_seed = rng.integers(1000000)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.3,
        random_state=split_seed
    )
    # """
    """
    if model_class is None:
        model_class = RandomForestRegressor
    if model_params is None:
        model_params = {
            'n_estimators':100,        # number of trees
            'max_depth':None,          # no limit on tree depth (can cause overfitting)
            'min_samples_split':15,     # minimum samples required to split a node
            'min_samples_leaf':15,      # minimum samples required at leaf node
            'max_features':'sqrt',     # number of features to consider for best split
            'random_state':42,         # for reproducibility
            'n_jobs':-1               # use all available cores
        }
    """
    # assert model_class is not None
    # assert model_params is not None
    assert model_name is not None
    assert custom_objective is not None and custom_objective in ['lagrange', 'mse_builtin', 'mse_custom']
    number_of_features, number_of_samples = X_train.shape

    row_and_col_names_indices = {name: i for i, name in enumerate(row_and_col_names)}

    idx_list = [row_and_col_names_indices[f] for f in hei_feats]


    predict_idx = row_and_col_names_indices[target_col]
    w_est_y = w_est[:, predict_idx]
    w_est_y = w_est_y[idx_list]

    #w_est_y = np.zeros((number_of_features,))


    if model_name == 'XGB':
        lambda_features = np.zeros_like(w_est_y)
        lambda_predict = 0.0
        if custom_objective in ['lagrange', 'mse_custom']:
            if custom_objective == 'lagrange':
                def custom_obj(y_true, y_pred):
                    # L = 0.5 * (y_pred - y_true)^2
                    grad = y_pred - y_true + np.dot(lambda_features,
                                                    w_est_y) + lambda_predict  # grad = dL/dy_pred = (y_pred - y_true)
                    hess = np.ones_like(y_pred)  # hess = d^2L/dy_pred^2 = 1
                    return grad, hess
            else:
                def custom_obj(y_true, y_pred):
                    # L = 0.5 * (y_pred - y_true)^2
                    grad = y_pred - y_true  # grad = dL/dy_pred = (y_pred - y_true)
                    hess = np.ones_like(y_pred)  # hess = d^2L/dy_pred^2 = 1
                    return grad, hess

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            dtrain = xgb.DMatrix(X_train_s, label=y_train)
            dtest = xgb.DMatrix(X_test_s, label=y_test)

            def obj_for_train(preds, dtrain):
                y_true = dtrain.get_label()
                grad, hess = custom_obj(y_true, preds)  # uses your existing custom_obj
                return grad, hess

            params = {
                "max_depth": 3,
                "eta": 0.1,  # learning_rate
                "seed": 42,  # random_state
                "base_score": float(np.mean(y_train)),
                # "tree_method": "hist",           # uncomment if you want
            }

            num_boost_round = 10

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                obj=obj_for_train,
                evals=[(dtrain, "train"), (dtest, "test")],  # optional
                verbose_eval=False
            )

            # --- 6) Predict ---
            y_test_pred = booster.predict(dtest)
            y_train_pred = booster.predict(dtrain)

        else:
            custom_obj = 'reg:squarederror'


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
                        base_score=y_train.mean(),
                        objective=custom_obj
                    )
                     )
                ]
            }
            rf_model = model_class(**model_params)

            # Train the model
            rf_model.fit(X_train, y_train)
            y_train_pred = rf_model.predict(X_train)
            y_test_pred = rf_model.predict(X_test)
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
        rf_model = model_class(**model_params)

        # Train the model
        rf_model.fit(X_train, y_train)
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)



    if stack_linear:
        lr_model = LinearRegression()
        lr_model.fit(y_train_pred[:, None], y_train)
        y_train_pred = lr_model.predict(y_train_pred[:, None])

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_bench = mean_squared_error(y_train, np.ones_like(y_train) * y_train.mean())

    train_ratio = train_mse / train_bench

    # y_test_pred = rf_model.predict(X_test)
    if stack_linear:
        y_test_pred = lr_model.predict(y_test_pred[:, None])

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_bench = mean_squared_error(y_test, np.ones_like(y_test) * y_train.mean())

    if compute_covs:
        test_covs_res = compute_cov(y_test, y_test_pred, "kendalltau")
    else:
        test_covs_res = None

    if do_print:
        print(f"Train Mean Squared Error: {train_mse:.4f}")
        print(f"Train Bench Error: {train_bench:.4f}")

        print('\n')

        print(f"Test Mean Squared Error: {test_mse:.4f}")
        print(f"Test Bench Error: {test_bench:.4f}")
        print(f'ratio: {test_mse / test_bench}')

        print(f'test covs: {test_covs_res}')

        # if False:
        feature_importance = rf_model.feature_importances_
        print(f"\nTop 5 Most Important Features:")
        # Assuming X is a DataFrame with column names, otherwise use indices
        feature_names = hei_feats
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(importance_pairs[:5]):
            print(f"{i + 1}. {feature}: {importance:.4f}")

    if do_print:
        print('\n ---------------------- \n')

    if save_details is not None:
        save_details['model'] = rf_model
        save_details['mask'] = mask
        save_details['X_train'] = X_train
        save_details['X_test'] = X_test
        save_details['y_train'] = y_train
        save_details['y_test'] = y_test

    return train_mse, test_mse, test_mse / test_bench, test_covs_res, train_ratio


#########   feature selectors


from sklearn.inspection import permutation_importance


def get_feature_covs(prep_data, cols, target_col,
                     outlier_percent=None, cov_type='kendalltau',
                     explanatory_outlier_percent=None,
                     seed=None,
                     do_sort=True
                     ):
    cov_res_lst = []
    v = np.squeeze(prep_data[target_col].to_numpy())

    if outlier_percent is None:
        mask1 = np.ones_like(v, dtype=bool)
    else:
        mask1 = percentile_mask(v, outlier_percent)

    # print(mask1.sum())
    for c in cols:
        u = np.squeeze(prep_data[c].to_numpy())

        if explanatory_outlier_percent is not None:
            mask2 = mask1 & percentile_mask(u, explanatory_outlier_percent)
        else:
            mask2 = mask1

        corr_v, pval_v = compute_cov(u[mask2], v[mask2], cov_type, n_permutes=1000, seed=seed)
        cov_res_lst.append((c, corr_v, pval_v))
        # print(f'{c}: {corr_v:.2f}, pval: {pval_v:.2f}')

    srt_cov_res_lst = sorted(cov_res_lst, key=lambda t: t[2]) if do_sort else cov_res_lst

    return srt_cov_res_lst


def forest_feat_selector(prep_data, cols, target_col, method='builtin'):
    reg_dat = prep_data[cols + [target_col]]
    reg_dat = reg_dat.dropna()

    X = reg_dat[cols].to_numpy()
    Y = reg_dat[target_col].to_numpy()

    mask = percentile_mask(Y, 5)
    X = X[mask]
    Y = Y[mask]

    rf_model = RandomForestRegressor(
        n_estimators=100,  # number of trees
        max_depth=None,  # no limit on tree depth (can cause overfitting)
        min_samples_split=15,  # minimum samples required to split a node
        min_samples_leaf=15,  # minimum samples required at leaf node
        max_features='sqrt',  # number of features to consider for best split
        random_state=42,  # for reproducibility
        n_jobs=-1  # use all available cores
    )
    # """

    if method == 'builtin':
        # Train the model
        rf_model.fit(X, Y)

        importances = rf_model.feature_importances_
        res = [(cols[i], importance) for i, importance in enumerate(importances)]
        res = sorted(res, key=lambda t: -t[1])

    elif method == 'permutation':

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        rf_model.fit(X_train, y_train)

        perm_importance = permutation_importance(rf_model, X_test, y_test,
                                                 scoring='r2',  # Can also use 'neg_mean_squared_error'
                                                 n_repeats=10,
                                                 random_state=42)

        res = [(cols[i], importance) for i, importance in enumerate(perm_importance.importances_mean)]
        res = sorted(res, key=lambda t: -t[1])
        # have also perm_importance.importances_std

    else:
        assert False, 'unknown selection type'

    return res