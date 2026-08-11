import numpy as np
import logging
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback.
    StratifiedGroupKFold = None
import torch


from artifact_utils import write_text_artifact, write_yaml_artifact
from compute_tools import compute_predictor_errors
from recommender_estimator import (
    XGBRecommenderPredictor,
    REGRecommenderPredictor,
    HCRecommenderPredictor,
    HCCIRecommenderPredictor,
    HCCERecommenderPredictor,
    compute_predictor_errors_and_cs_scikit,
    compute_predictor_errors_scikit,
)


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


def _make_site_gender_cv_splits(prep_data, target_col, n_splits, solver_cfg=None):
    n_splits = int(n_splits)
    random_state = int(getattr(solver_cfg, "cv_random_state", 2227070966)) if solver_cfg is not None else 2227070966

    site_col = "site_numeric" if "site_numeric" in prep_data.columns else None
    if site_col is None and "site_continental" in prep_data.columns:
        site_col = "site_continental"
    gender_col = "gender_numeric" if "gender_numeric" in prep_data.columns else None

    if site_col is None or gender_col is None:
        logging.warning(
            "Site/gender stratified CV requested but site or gender column is missing; "
            "falling back to shuffled KFold."
        )
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.arange(len(prep_data)))), "kfold"

    site_labels = prep_data[site_col].round().astype("Int64").astype(str)
    gender_labels = prep_data[gender_col].round().astype("Int64").astype(str)
    n_target_bins = int(getattr(solver_cfg, "cv_target_bins", n_splits)) if solver_cfg is not None else n_splits
    target_bins = None
    if target_col in prep_data.columns:
        try:
            target_bins = (
                pd.qcut(
                    prep_data[target_col],
                    q=min(n_target_bins, prep_data[target_col].nunique()),
                    labels=False,
                    duplicates="drop",
                )
                .astype("Int64")
                .astype(str)
            )
        except ValueError:
            target_bins = None

    target_bin_info = None
    if target_bins is not None:
        target_bin_info = {
            "target_col": target_col,
            "n_target_bins": int(target_bins.nunique()),
            "target_bin_counts": target_bins.value_counts().sort_index().to_dict(),
        }

    candidate_strata = []
    if target_bins is not None:
        candidate_strata.append(
            ("site_gender_target_bin", site_labels + "_" + gender_labels + "_" + target_bins)
        )
    candidate_strata.append(("site_gender", site_labels + "_" + gender_labels))
    if target_bins is not None:
        candidate_strata.append(("gender_target_bin", gender_labels + "_" + target_bins))
        candidate_strata.append(("site_target_bin", site_labels + "_" + target_bins))
        candidate_strata.append(("target_bin", target_bins))
    candidate_strata.append(("gender", gender_labels))
    candidate_strata.append(("site", site_labels))

    strata_name = None
    strata = None
    stratum_counts = None
    for candidate_name, candidate in candidate_strata:
        candidate_counts = candidate.value_counts()
        if candidate_counts.min() >= n_splits:
            strata_name = candidate_name
            strata = candidate
            stratum_counts = candidate_counts
            break

    if strata is None:
        strata_name, strata = candidate_strata[-1]
        stratum_counts = strata.value_counts()
        logging.warning(
            "No stratification label has at least n_splits=%d samples in every class; "
            "using %s and falling back to shuffled KFold if sklearn rejects it. Counts: %s",
            n_splits,
            strata_name,
            stratum_counts.to_dict(),
        )
    elif strata_name != "site_gender_target_bin":
        logging.warning(
            "Site/gender/target-bin CV is too sparse for n_splits=%d; using coarser "
            "stratification '%s' instead. Counts: %s",
            n_splits,
            strata_name,
            stratum_counts.to_dict(),
        )

    groups = prep_data["ID"].to_numpy() if "ID" in prep_data.columns else np.arange(len(prep_data))
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        try:
            splits = list(splitter.split(np.arange(len(prep_data)), strata.to_numpy(), groups))
        except ValueError as exc:
            logging.warning(
                "StratifiedGroupKFold failed (%s); falling back to shuffled KFold.",
                exc,
            )
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(splitter.split(np.arange(len(prep_data)))), "kfold"
        split_kind = "stratified_group_kfold"
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(np.arange(len(prep_data)), strata.to_numpy()))
        split_kind = "stratified_kfold"

    fold_summary = []
    for fold_idx, (_, test_idx) in enumerate(splits, start=1):
        fold_summary.append(
            {
                "fold": fold_idx,
                "n_test": int(len(test_idx)),
                "site_gender_counts": strata.iloc[test_idx].value_counts().sort_index().to_dict(),
            }
        )
    logging.info(
        "Using %s stratified by %s: %s",
        split_kind,
        strata_name,
        fold_summary,
    )
    write_yaml_artifact(
        "cv_site_gender_split_summary.yaml",
        {
            "cv": split_kind,
            "strata_name": strata_name,
            "site_col": site_col,
            "gender_col": gender_col,
            "target_bin_info": target_bin_info,
            "strata_counts": stratum_counts.sort_index().to_dict(),
            "folds": fold_summary,
        },
    )
    return splits, split_kind


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


def create_model(model_name, w_est, target_col, row_and_col_names, custom_objective, prep_data,
                                        solver_cfg):
    model = None
    if model_name == "XGB":
        model = XGBRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data,
                                        solver_cfg)  # prep_data[row_and_col_names]
    elif model_name == "REG":
        model = REGRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data, solver_cfg)
    elif model_name in {"HC", "HC-CI", "HC-CE"}:
        torch.manual_seed(42)
        solver_name = str(getattr(solver_cfg, "name", ""))
        if model_name == "HC-CI" or solver_name == "hc_predictor_ci":
            model_class = HCCIRecommenderPredictor
        elif model_name == "HC-CE" or solver_name == "hc_predictor_ce":
            model_class = HCCERecommenderPredictor
        else:
            model_class = HCRecommenderPredictor
        model = model_class(w_est, target_col, row_and_col_names, custom_objective,
                            prep_data, solver_cfg)
    if model is None:
        raise ValueError("Model can be only XGB, REG, HC, HC-CI, or HC-CE.")
    return model


def run_feature_selection_scikit(prep_data, model_name, custom_objective,
                          target_col,
                          w_est, row_and_col_names,
                          n_runs, n_features,
                          full_feats, solver_cfg, model_factory=None
                          ):
    logging.info(f"w_est shape {w_est.shape}")
    logging.info(f"target_col {target_col}")
    logging.info(f"row_and_col_names {len(row_and_col_names)}")
    logging.info(f"prep_data.shape {prep_data.shape}")
    logging.info(f"row_and_col_names {row_and_col_names}")



    model = create_model(model_name, w_est, target_col, row_and_col_names, custom_objective, prep_data,
                                        solver_cfg)
    assert all(isinstance(col, str) for col in prep_data.columns)
    prep_data = prep_data.dropna(subset=[target_col])
    y = prep_data[target_col]

    score_normalizer = mean_squared_error(y, np.ones_like(y) * y.mean())
    #TODO this deletes only 30 columns out of 508, so I will do this so that scikit feature selection works (no nas)
    #prep_data = prep_data[full_feats]
    prep_data = prep_data.dropna(axis=1)
    X = prep_data.drop(target_col, axis=1) if target_col in prep_data.columns else prep_data
    if full_feats is not None:
        X = X[full_feats]

    logging.info(f"Testing on columns {len(X.columns)}: {X.columns}")
    cv_splits, cv_kind = _make_site_gender_cv_splits(prep_data, target_col, n_runs, solver_cfg)
    logging.info("Using CV splitter for model evaluation: %s", cv_kind)

    if 'feature_selector' in solver_cfg and solver_cfg.feature_selector == 'SequentialFeatureSelector':
        sfs = SequentialFeatureSelector(
            model,
            direction="forward",
            scoring=compute_predictor_errors_scikit,
            cv=cv_splits,
            n_features_to_select=n_features
        )

        sfs.fit(X, y)
        # here are the selected features
        best_features = X.columns[sfs.get_support()]
        write_text_artifact("selected_features_param.txt", ", ".join(str(feature) for feature in best_features))
        write_yaml_artifact("selected_features.yaml", {'selected_best_features': list(best_features)})
        selected_indices = sfs.get_support(indices=True)

        logging.info(f"Best features {best_features}, {selected_indices}")

        X_selected = X.iloc[:, selected_indices]
        X_selected = X_selected.to_numpy() # CV did not work with pandas data frame, IDKW
        w_est = model._w_est
        
    else:
        model.fit(X, y)
        validation_history = getattr(model, "validation_history_", None)
        if validation_history:
            write_yaml_artifact(
                "validation_history.yaml",
                {
                    "fit": validation_history,
                    "best_validation_loss": getattr(model._rf_model_, "best_validation_loss_", None),
                    "best_validation_outer": getattr(model._rf_model_, "best_validation_outer_", None),
                    "restore_best_validation_model": getattr(
                        model._rf_model_, "restore_best_validation_model_", None
                    ),
                },
            )
        # Keep column names for estimators that need to align a reused W_est
        # with the current CV fold features.
        X_selected = X
        best_features = list(X.columns)
        w_est = model._w_est

    results = cross_validate(model, X_selected, y,
        cv=cv_splits,
        scoring=(lambda estimator, X, y: compute_predictor_errors_and_cs_scikit(estimator, X, y, estimator._w_est)) if isinstance(model, HCRecommenderPredictor) else compute_predictor_errors_scikit,
        return_train_score=True,
        return_estimator=True,
    )
    cv_validation_history = []
    for fold_idx, estimator in enumerate(results.get("estimator", []), start=1):
        history = getattr(estimator, "validation_history_", None)
        if history:
            cv_validation_history.append(
                {
                    "fold": fold_idx,
                    "history": history,
                    "best_validation_loss": getattr(estimator._rf_model_, "best_validation_loss_", None),
                    "best_validation_outer": getattr(estimator._rf_model_, "best_validation_outer_", None),
                    "restore_best_validation_model": getattr(
                        estimator._rf_model_, "restore_best_validation_model_", None
                    ),
                }
            )
    if cv_validation_history:
        write_yaml_artifact("cv_validation_history.yaml", cv_validation_history)

    if model_name == 'HC':
        # add fit call to get the graph for selected features
        model.fit(X_selected, y)
        w_est = model._w_est
    test_mse_fold = results["test_score"]
    test_mse = test_mse_fold.mean() / score_normalizer
    train_mse_fold = results["train_score"]
    train_mse = train_mse_fold.mean() / score_normalizer

    return (
        best_features,
        train_mse,
        test_mse,
        train_mse_fold / score_normalizer,
        test_mse_fold / score_normalizer,
        w_est,
    )
