import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate

from compute_tools import compute_predictor_errors
from recomender_humancompatible import HCRecommenderPredictor
from recommender_estimator import XGBRecommenderPredictor, REGRecommenderPredictor, compute_predictor_errors_scikit


def get_mean_average_errors(prep_data, run_feats, target_col, w_est, row_and_col_names,
                            model_name, custom_objective,
                            n_runs):
    run_res = []
    for i in range(n_runs):
        run_res.append(
            compute_predictor_errors(prep_data, run_feats, target_col, w_est, row_and_col_names,
                                     model_name=model_name,
                                     custom_objective=custom_objective,
                                     do_print=False, stack_linear=False,
                                     compute_covs=False
                                     )
        )

    train_ratio_mean = np.array([v[4] for v in run_res]).mean()
    test_ratio_mean = np.array([v[2] for v in run_res]).mean()

    return train_ratio_mean, test_ratio_mean


from time import time


def run_feature_selection(prep_data, model_name,custom_objective,
                          target_col,
                          w_est, row_and_col_names,
                          n_runs, n_features,
                          full_feats,
                          model_factory=None
                          ):
    curr_feats = []

    # these are actually variance ratios, not pure errors.
    curr_test_errs = []
    curr_train_errs = []

    curr_test_error = 100000.

    print(f'====== {target_col} ')
    for i in range(n_features):

        if model_factory is not None:
            model_class, model_params = model_factory(len(curr_feats) + 1)

        start_time = time()

        candidate_feats = [s for s in full_feats if s not in curr_feats]

        res = []
        for feat in candidate_feats:
            train_ratio_mean, test_ratio_mean = get_mean_average_errors(
                prep_data, curr_feats + [feat], target_col,
                w_est, row_and_col_names,
                model_name,custom_objective,
                n_runs
            )

            res.append((feat, train_ratio_mean, test_ratio_mean))

        res = sorted(res, key=lambda t: t[2])

        best_res = res[0]

        best_feat = best_res[0]
        best_test_error = best_res[2]
        best_train_error = best_res[1]

        print('')
        print(f'Round {i}')

        if best_test_error >= curr_test_error:
            print('Failed to impove further')
            break

        curr_feats += [best_feat]
        curr_test_errs += [best_test_error]
        curr_train_errs += [best_train_error]

        curr_test_error = best_test_error

        print(f'   Features: {curr_feats}')
        print(f'   Mean Train var reduction: {100 * (1. - best_res[1])}')
        print(f'   Mean Test var reduction: {100 * (1. - best_test_error)}')

        ctime = time()
        print(f'   Round completed in {(ctime - start_time) / 60:.2f} min.')

    return curr_feats, curr_train_errs, curr_test_errs


def run_feature_selection_scikit(prep_data, model_name, custom_objective,
                          target_col,
                          w_est, row_and_col_names,
                          n_runs, n_features,
                          full_feats, solver_cfg, model_factory=None
                          ):
    model = None
    if model_name == "XGB":
        model = XGBRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data, solver_cfg)
    elif model_name == "REG":
        model = REGRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data, solver_cfg)
    elif model_name == "HC":
        model = HCRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data, solver_cfg)
    if model is None:
        raise ValueError("Model can be only XGB or REG.")

    sfs = SequentialFeatureSelector(
        model,
        direction="forward",
        scoring=compute_predictor_errors_scikit,
        cv=n_runs,
        n_features_to_select=n_features
    )

    assert all(isinstance(col, str) for col in prep_data.columns)

    y = prep_data[target_col]

    #TODO this deletes only 30 columns out of 508, so I will do this so that scikit feature selection works (no nas)
    prep_data = prep_data[full_feats]
    prep_data = prep_data.dropna(axis=1)
    X = prep_data.drop(target_col, axis=1) if target_col in prep_data.columns else prep_data

    sfs.fit(X, y)

    # here are the selected features
    best_features = X.columns[sfs.get_support()]
    selected_indices = sfs.get_support(indices=True)

    print(f"Best features {best_features}, {selected_indices}")

    X_selected = X.iloc[:, selected_indices]
    X_selected = X_selected.to_numpy() # CV did not work with pandas data frame, IDKW


    results = cross_validate(model, X_selected, y,
        cv=10,
        scoring=compute_predictor_errors_scikit,
        return_train_score=True
    )

    return best_features, [results["train_score"].mean()], [results["test_score"].mean()]