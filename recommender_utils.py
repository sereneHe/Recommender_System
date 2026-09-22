import numpy as np
import logging
import os
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_validate
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback.
    StratifiedGroupKFold = None
import torch


from artifact_utils import write_text_artifact, write_yaml_artifact
from compute_tools import compute_predictor_errors
from hc_predictor import weibull_gaussianization_enabled, weibull_gaussianize_table
from recommender_estimator import (
    XGBRecommenderPredictor,
    REGRecommenderPredictor,
    HCRecommenderPredictor,
    HCCIRecommenderPredictor,
    HCCERecommenderPredictor,
    compute_predictor_errors_and_cs_scikit,
    compute_predictor_errors_scikit,
)
from utils import coerce_random_state


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
    random_state = (
        coerce_random_state(getattr(solver_cfg, "cv_random_state", None), 2227070966)
        if solver_cfg is not None
        else 2227070966
    )

    cv_strategy = str(
        getattr(solver_cfg, "cv_strategy", "site_gender")
        if solver_cfg is not None
        else "site_gender"
    ).strip().lower()
    if cv_strategy in {"time", "time_series", "rolling", "rolling_origin", "expanding"}:
        if n_splits < 2:
            raise ValueError("Time-series CV requires solver.n_runs (n_splits) >= 2.")
        if hasattr(prep_data, "index") and not prep_data.index.is_monotonic_increasing:
            raise ValueError(
                "Time-series CV requires rows sorted from earliest to latest. "
                "Sort the problem data before constructing splits."
            )
        raw_test_size = getattr(solver_cfg, "cv_time_test_size", None) if solver_cfg is not None else None
        test_size = None if raw_test_size in (None, "", 0, "0") else int(raw_test_size)
        gap = int(getattr(solver_cfg, "cv_time_gap", 0) if solver_cfg is not None else 0)
        if gap < 0:
            raise ValueError("cv_time_gap must be non-negative.")
        try:
            splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
            splits = list(splitter.split(np.arange(len(prep_data))))
        except ValueError as exc:
            raise ValueError(
                "Could not construct time-series CV splits. Reduce solver.n_runs or "
                "solver.cv_time_test_size for the available history."
            ) from exc
        logging.info(
            "Using expanding-window TimeSeriesSplit: n_splits=%d, test_size=%s, gap=%d.",
            n_splits,
            test_size,
            gap,
        )
        return splits, "time_series_split"

    if cv_strategy not in {"site_gender", "auto", "stratified"}:
        raise ValueError(
            "cv_strategy must be one of {'site_gender', 'time_series'}, "
            f"got {cv_strategy!r}."
        )

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


def _constraint_count_row(estimator, target_col, solver_cfg, stage, fold=None):
    counts = getattr(estimator, "constraint_counts_", None)
    if not counts:
        return None
    row = {
        "target": str(target_col),
        "solver": str(getattr(solver_cfg, "name", "")),
        "stage": stage,
        "fold": fold,
    }
    row.update(counts)
    return row


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
        # Keep the historical default while allowing paired optimizer
        # experiments to use exactly the same, explicitly selected seed.
        model_seed = coerce_random_state(getattr(solver_cfg, "random_state", None), 42)
        torch.manual_seed(model_seed)
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


def _slice_cv_rows(values, indices):
    """Take CV rows while preserving DataFrame/Series column metadata."""
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices]


def _make_outer_fold_prep_data(X_train, y_train, target_col):
    """Build the sole DAG-learning table for one outer-training fold."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "Fold-local DAG estimation requires pandas feature columns so W_est "
            "can be aligned with the active model features."
        )
    fold_prep_data = X_train.copy()
    fold_prep_data[target_col] = np.asarray(y_train)
    return fold_prep_data


def _score_to_scalar(score):
    """Extract MSE from HC's score-and-constraint diagnostic mapping."""
    if isinstance(score, dict):
        return float(score["score"])
    return float(score)


def _cross_validate_hc_with_fold_local_dag(model, X, y, cv_splits, target_col):
    """Run outer CV without fitting a DAG on full data.

    ``cross_validate`` clones an estimator but cannot pass each clone its train
    indices.  The old path consequently made every clone learn W_est from its
    full ``prep_data`` (including outer-test rows), and a standalone initial
    fit caused another unnecessary DAG solve.  Here each clone receives only
    its outer-training table before ``fit``.  Its W_est is then reused by the
    NN training and both train/test scores for that fold.
    """
    results = {
        "fit_time": [],
        "score_time": [],
        "test_score": [],
        "train_score": [],
        "estimator": [],
        "train_indices": [],
        "test_indices": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        X_train = _slice_cv_rows(X, train_idx)
        y_train = _slice_cv_rows(y, train_idx)
        X_test = _slice_cv_rows(X, test_idx)
        y_test = _slice_cv_rows(y, test_idx)

        estimator = clone(model)
        estimator.prep_data = _make_outer_fold_prep_data(X_train, y_train, target_col)
        logging.info(
            "Outer CV fold %d: estimating W_est from %d training rows only.",
            fold_idx,
            len(X_train),
        )

        fit_started = time()
        estimator.fit(X_train, y_train)
        results["fit_time"].append(time() - fit_started)

        score_started = time()
        test_score = compute_predictor_errors_and_cs_scikit(
            estimator, X_test, y_test, estimator._w_est
        )
        train_score = compute_predictor_errors_and_cs_scikit(
            estimator, X_train, y_train, estimator._w_est
        )
        results["score_time"].append(time() - score_started)
        results["test_score"].append(_score_to_scalar(test_score))
        results["train_score"].append(_score_to_scalar(train_score))
        results["estimator"].append(estimator)
        results["train_indices"].append(np.asarray(train_idx, dtype=int))
        results["test_indices"].append(np.asarray(test_idx, dtype=int))

    for key in ("fit_time", "score_time", "test_score", "train_score"):
        results[key] = np.asarray(results[key], dtype=float)
    return results


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



    original_model_prep_data = prep_data
    assert all(isinstance(col, str) for col in prep_data.columns)
    prep_data = prep_data.dropna(subset=[target_col])
    y = prep_data[target_col]

    #TODO this deletes only 30 columns out of 508, so I will do this so that scikit feature selection works (no nas)
    #prep_data = prep_data[full_feats]
    prep_data = prep_data.dropna(axis=1)
    X = prep_data.drop(target_col, axis=1) if target_col in prep_data.columns else prep_data
    if full_feats is not None:
        X = X[full_feats]

    logging.info(f"Testing on columns {len(X.columns)}: {X.columns}")
    cv_splits, cv_kind = _make_site_gender_cv_splits(prep_data, target_col, n_runs, solver_cfg)
    logging.info("Using CV splitter for model evaluation: %s", cv_kind)

    solver_name = str(getattr(solver_cfg, "name", ""))
    write_constraint_counts = solver_name == "hc_predictor_ce" or model_name == "HC-CE"
    use_weibull = (
        model_name == "HC"
        and solver_name == "hc_predictor"
        and weibull_gaussianization_enabled()
    )
    model_prep_data = original_model_prep_data
    if use_weibull:
        if not bool(getattr(solver_cfg, "recalculate_dag", False)):
            raise ValueError(
                "HC_WEIBULL_GAUSSIANIZE=1 requires solver.recalculate_dag=true "
                "so W is learned in the same Gaussianized space as X and y."
            )
        model_columns = list(X.columns) + [target_col]
        model_prep_data = weibull_gaussianize_table(prep_data[model_columns])
        X = model_prep_data[list(X.columns)]
        y = model_prep_data[target_col]
        logging.info(
            "Applied Weibull Gaussianization to %d HC columns using mode=%s.",
            len(model_columns),
            os.getenv("HC_WEIBULL_MODE", "rand"),
        )

    constraint_count_rows = []
    model = create_model(
        model_name,
        w_est,
        target_col,
        row_and_col_names,
        custom_objective,
        model_prep_data,
        solver_cfg,
    )

    if 'feature_selector' in solver_cfg and solver_cfg.feature_selector == 'SequentialFeatureSelector':
        sfs = SequentialFeatureSelector(
            model,
            direction="forward",
            scoring=compute_predictor_errors_scikit,
            cv=cv_splits,
            n_features_to_select=n_features
        )

        sfs.fit(X, y)
        if write_constraint_counts:
            row = _constraint_count_row(model, target_col, solver_cfg, "feature_selection")
            if row is not None:
                constraint_count_rows.append(row)
        # here are the selected features
        best_features = X.columns[sfs.get_support()]
        write_text_artifact("selected_features_param.txt", ", ".join(str(feature) for feature in best_features))
        write_yaml_artifact("selected_features.yaml", {'selected_best_features': list(best_features)})
        selected_indices = sfs.get_support(indices=True)

        logging.info(f"Best features {best_features}, {selected_indices}")

        # Preserve feature labels so the fold-local DAG can be aligned with the
        # selected columns during the outer CV below.
        X_selected = X.iloc[:, selected_indices]
        w_est = model._w_est
        
    else:
        X_selected = X
        best_features = list(X.columns)
        if isinstance(model, HCRecommenderPredictor):
            # Do not fit a standalone model on all rows.  Besides spending an
            # extra MILP solve, that W_est would include every outer-test row.
            # The fold-local CV helper below is the only HC fit path here.
            logging.info(
                "Skipping full-data initial HC fit; each outer fold will learn "
                "and reuse W_est from its training rows."
            )
        else:
            model.fit(X, y)
            if write_constraint_counts:
                row = _constraint_count_row(model, target_col, solver_cfg, "initial_fit")
                if row is not None:
                    constraint_count_rows.append(row)
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
            w_est = model._w_est

    if isinstance(model, HCRecommenderPredictor):
        results = _cross_validate_hc_with_fold_local_dag(
            model, X_selected, y, cv_splits, target_col
        )
        if results["estimator"]:
            # This is a representative fold-local graph for existing W_est.csv
            # and heat-map output; it is not a full-data graph.
            w_est = results["estimator"][0]._w_est
    else:
        results = cross_validate(model, X_selected, y,
            cv=cv_splits,
            scoring=compute_predictor_errors_scikit,
            return_train_score=True,
            return_estimator=True,
            error_score="raise",
        )
    cv_validation_history = []
    constraint_metadata_rows = []
    constraint_stat_audit_rows = []
    w_constraint_audit_rows = []
    ci_window_filter_audit_rows = []
    constraint_audit_enabled = bool(getattr(solver_cfg, "constraint_audit_enabled", False))
    for fold_idx, estimator in enumerate(results.get("estimator", []), start=1):
        if write_constraint_counts:
            row = _constraint_count_row(estimator, target_col, solver_cfg, "cv_fold", fold_idx)
            if row is not None:
                constraint_count_rows.append(row)
        if constraint_audit_enabled:
            train_indices = (
                results.get("train_indices", [])[fold_idx - 1]
                if results.get("train_indices")
                else cv_splits[fold_idx - 1][0]
            )
            test_indices = results.get("test_indices", [])[fold_idx - 1] if results.get("test_indices") else cv_splits[fold_idx - 1][1]
            X_train_audit = _slice_cv_rows(X_selected, train_indices)
            y_train_audit = _slice_cv_rows(y, train_indices)
            X_audit = _slice_cv_rows(X_selected, test_indices)
            y_audit = _slice_cv_rows(y, test_indices)
            if not hasattr(estimator, "constraint_audit_rows"):
                raise TypeError(
                    "constraint_audit_enabled requires an HC/HC-CE estimator with "
                    "constraint_audit_rows()."
                )
            metadata_rows, statistic_rows, w_rows = estimator.constraint_audit_rows(
                X_audit,
                y_audit,
                fold=fold_idx,
                stage="outer_test",
                # Calibrate SE-based tolerances on the training fold only, so
                # the held-out violation metric does not read held-out labels.
                tolerance_reference=(X_train_audit, y_train_audit),
            )
            _, train_statistic_rows, _ = estimator.constraint_audit_rows(
                X_train_audit,
                y_train_audit,
                fold=fold_idx,
                stage="outer_train",
            )
            for audit_row in metadata_rows:
                audit_row.update({"target": str(target_col), "solver": solver_name})
            for audit_row in statistic_rows:
                audit_row.update(
                    {
                        "target": str(target_col),
                        "solver": solver_name,
                        "evaluation_split": "outer_test",
                    }
                )
            for audit_row in train_statistic_rows:
                audit_row.update(
                    {
                        "target": str(target_col),
                        "solver": solver_name,
                        "evaluation_split": "outer_train",
                    }
                )
            for audit_row in w_rows:
                audit_row.update(
                    {
                        "target": str(target_col),
                        "solver": solver_name,
                        "evaluation_split": "outer_test",
                    }
                )
            constraint_metadata_rows.extend(metadata_rows)
            constraint_stat_audit_rows.extend(train_statistic_rows)
            constraint_stat_audit_rows.extend(statistic_rows)
            w_constraint_audit_rows.extend(w_rows)
            window_filter_rows = getattr(
                getattr(estimator, "_rf_model_", None),
                "ci_window_filter_diagnostics_",
                [],
            )
            for audit_row in window_filter_rows:
                ci_window_filter_audit_rows.append(
                    {
                        **audit_row,
                        "fold": fold_idx,
                        "target": str(target_col),
                        "solver": solver_name,
                    }
                )
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
        if isinstance(model, HCCERecommenderPredictor):
            first_fold = cv_validation_history[0]
            write_yaml_artifact(
                "validation_history.yaml",
                {
                    "source": "outer_cv_fold",
                    "fold": first_fold["fold"],
                    "fit": first_fold["history"],
                    "best_validation_loss": first_fold["best_validation_loss"],
                    "best_validation_outer": first_fold["best_validation_outer"],
                    "restore_best_validation_model": first_fold[
                        "restore_best_validation_model"
                    ],
                },
            )

    if write_constraint_counts and constraint_count_rows:
        write_text_artifact(
            "constraint_counts.csv",
            pd.DataFrame(constraint_count_rows).to_csv(index=False),
        )
    if constraint_metadata_rows:
        write_text_artifact(
            "constraint_metadata.csv",
            pd.DataFrame(constraint_metadata_rows).to_csv(index=False),
        )
    if constraint_stat_audit_rows:
        write_text_artifact(
            "constraint_stat_audit.csv",
            pd.DataFrame(constraint_stat_audit_rows).to_csv(index=False),
        )
    if w_constraint_audit_rows:
        write_text_artifact(
            "w_constraint_audit.csv",
            pd.DataFrame(w_constraint_audit_rows).to_csv(index=False),
        )
    if ci_window_filter_audit_rows:
        write_text_artifact(
            "ci_window_filter_audit.csv",
            pd.DataFrame(ci_window_filter_audit_rows).to_csv(index=False),
        )
    test_mse_fold = results["test_score"]
    train_mse_fold = results["train_score"]
    # Normalize each held-out fold with the mean-only baseline learned from
    # that fold's training rows.  The historical full-table normalizer made
    # the reported NMSE depend on outer-test targets.
    fold_normalizers = []
    y_array = np.asarray(y, dtype=float)
    for train_idx, _ in cv_splits:
        y_train_fold = y_array[train_idx]
        normalizer = mean_squared_error(
            y_train_fold,
            np.full_like(y_train_fold, y_train_fold.mean(), dtype=float),
        )
        fold_normalizers.append(max(float(normalizer), np.finfo(float).eps))
    fold_normalizers = np.asarray(fold_normalizers, dtype=float)
    if len(fold_normalizers) != len(test_mse_fold):
        raise RuntimeError("CV score count does not match the number of fold normalizers.")
    write_yaml_artifact(
        "cv_score_normalizers.yaml",
        {
            "definition": "MSE(y_train_fold, mean(y_train_fold))",
            "values": fold_normalizers.tolist(),
        },
    )
    test_mse_fold = test_mse_fold / fold_normalizers
    train_mse_fold = train_mse_fold / fold_normalizers
    test_mse = test_mse_fold.mean()
    train_mse = train_mse_fold.mean()
    write_text_artifact(
        "cv_fold_metrics.csv",
        pd.DataFrame(
            {
                "fold": np.arange(1, len(test_mse_fold) + 1),
                "train_nmse": train_mse_fold,
                "test_nmse": test_mse_fold,
                "fit_time_seconds": results.get("fit_time", np.full(len(test_mse_fold), np.nan)),
                "score_time_seconds": results.get("score_time", np.full(len(test_mse_fold), np.nan)),
            }
        ).to_csv(index=False),
    )

    return (
        best_features,
        train_mse,
        test_mse,
        train_mse_fold,
        test_mse_fold,
        w_est,
    )
