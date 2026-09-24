import networkx as nx
import numpy as np
import xgboost as xgb
import logging
import hashlib
import json
import os

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
from omegaconf import OmegaConf, open_dict
from hc_predictor_ce import (
    _is_discrete_ci_kind,
    constraint_window_statistics,
    dependent_statistic_values,
    independent_expectation_tolerances,
    signed_expectation_equalities,
)
from utils import coerce_random_state


def _matrix_sha256(matrix):
    """Return a stable fingerprint for a numeric W matrix."""
    array = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _graph_structure_metrics(true_w, estimated_w, threshold):
    """Compare two directed supports without using edge weights.

    The metrics are diagnostics only.  In particular, the true synthetic W
    is never passed into the MILP objective or used to select a model.
    """
    true_array = np.asarray(true_w, dtype=float)
    estimated_array = np.asarray(estimated_w, dtype=float)
    if true_array.shape != estimated_array.shape or true_array.ndim != 2:
        return {}
    # Ground-truth synthetic coefficients are structural zeros/non-zeros, so
    # do not apply the estimated-graph release threshold to the truth.
    true_support = np.abs(true_array) > 0.0
    estimated_support = np.abs(estimated_array) > float(threshold)
    np.fill_diagonal(true_support, False)
    np.fill_diagonal(estimated_support, False)
    tp = int(np.sum(true_support & estimated_support))
    fp = int(np.sum(~true_support & estimated_support))
    fn = int(np.sum(true_support & ~estimated_support))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    true_skeleton = true_support | true_support.T
    estimated_skeleton = estimated_support | estimated_support.T
    union = int(np.sum(true_skeleton | estimated_skeleton))
    skeleton_jaccard = (
        float(np.sum(true_skeleton & estimated_skeleton) / union)
        if union
        else 1.0
    )
    # Standard directed-pair SHD: an extra edge, missing edge, or reversal
    # contributes one operation.  Compute it locally to keep the estimator's
    # import path independent of the optional clique-solver package.
    shd = 0
    for i in range(true_support.shape[0]):
        for j in range(i):
            if (
                true_support[i, j] != estimated_support[i, j]
                or true_support[j, i] != estimated_support[j, i]
            ):
                shd += 1
    return {
        "true_w_edges": int(np.sum(true_support)),
        "estimated_w_edges": int(np.sum(estimated_support)),
        "true_edge_tp": tp,
        "false_positive_edges": fp,
        "false_negative_edges": fn,
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "shd": float(shd),
        "skeleton_jaccard": skeleton_jaccard,
    }


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


def _clique_backend_cfg(cfg):
    """Create an isolated config for the optional external clique solver.

    The HC configs predate the standalone clique solver and therefore do not
    contain all of its diagnostic/options fields.  Keep the caller's config
    untouched and fill only safe defaults required by ``solve_milp_clique``.
    """
    clique_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if getattr(clique_cfg, "max_clique_size", None) is None:
        clique_cfg.max_clique_size = 5
    # Selecting the ``milp_clique`` backend is itself the opt-in for clique
    # separation.  This keeps the ordinary HC configuration (which has the
    # legacy clique flag disabled) from silently running the external solver
    # without its defining cuts.
    clique_cfg.enable_clique_constraints = True
    if not hasattr(clique_cfg, "enable_cycle_constraints"):
        clique_cfg.enable_cycle_constraints = True
    if not hasattr(clique_cfg, "separation_order"):
        clique_cfg.separation_order = "cycle_first"
    if not hasattr(clique_cfg, "delayed_second_separation"):
        clique_cfg.delayed_second_separation = False
    if not hasattr(clique_cfg, "clique_cut_selection"):
        formulation = str(getattr(clique_cfg, "clique_cut_formulation", "original")).lower()
        clique_cfg.clique_cut_selection = "all" if formulation != "original" else "original"
    if not hasattr(clique_cfg, "clique_cut_top_k"):
        clique_cfg.clique_cut_top_k = int(getattr(clique_cfg, "clique_top_k", 5))
    if not hasattr(clique_cfg, "experiment_diagnostics"):
        clique_cfg.experiment_diagnostics = False
    if not hasattr(clique_cfg, "print_callback_events"):
        clique_cfg.print_callback_events = False
    if not hasattr(clique_cfg, "print_retained_solutions"):
        clique_cfg.print_retained_solutions = False
    if not hasattr(clique_cfg, "print_clique_cuts"):
        clique_cfg.print_clique_cuts = False
    return clique_cfg


def _solve_dag_with_backend(X, cfg, w_threshold, Y=None, B_ref=None, tabu_edges=None):
    """Dispatch DAG estimation while preserving the historical MILP default."""
    configured_backend = getattr(cfg, "dag_solver_backend", None)
    backend = str(
        configured_backend
        or os.environ.get("HC_DAG_SOLVER_BACKEND", "milp")
    ).strip().lower()
    if backend in {"milp", "solve_milp", "default"}:
        result = solve_milp.solve(
            X, cfg, w_threshold, Y=Y, B_ref=B_ref, tabu_edges=tabu_edges
        )
        if len(result) == 5:
            w_est, a_est, gap, lazy_count, stats = result
        else:
            raise RuntimeError(f"Unexpected solve_milp return length: {len(result)}")
        diagnostics = (
            dict(stats)
            if isinstance(stats, dict)
            else {"legacy_stats": stats}
        )
        diagnostics.setdefault("backend", "milp")
        return w_est, a_est, gap, lazy_count, diagnostics

    if backend in {"milp_clique", "clique", "solve_milp_clique"}:
        from dagsolvers import solve_milp_clique

        clique_cfg = _clique_backend_cfg(cfg)
        result = solve_milp_clique.solve(
            X, clique_cfg, w_threshold, Y=Y, B_ref=B_ref, tabu_edges=tabu_edges
        )
        if len(result) != 6:
            raise RuntimeError(
                "Unexpected solve_milp_clique return length: "
                f"{len(result)} (expected 6)"
            )
        w_est, a_est, gap, lazy_count, stats, clique_breakdown = result
        diagnostics = (
            dict(stats)
            if isinstance(stats, dict)
            else {"solver_stats": stats}
        )
        if isinstance(clique_breakdown, dict):
            for key, value in clique_breakdown.items():
                diagnostics.setdefault(f"clique_{key}", value)
        diagnostics.setdefault("backend", "milp_clique")
        return w_est, a_est, gap, lazy_count, diagnostics

    raise ValueError(
        "dag_solver_backend must be one of {'milp', 'milp_clique'}, "
        f"got {backend!r}"
    )


def compute_recalculated_w_est(
    prep_data,
    target_col,
    cfg,
    current_feature_names=None,
    return_diagnostics=False,
    true_w_reference=None,
):
    """Compute a MILP DAG for the active feature set.

    ``return_diagnostics`` is opt-in so existing callers still receive only
    the matrix.  The Phase-1 screen uses the diagnostics to distinguish an
    optimal DAG from a time-limited incumbent and to explain failed folds.
    """
    prep_data = prep_data.dropna(subset=[target_col])
    prep_data = prep_data.dropna(axis=1)
    configured_features = current_feature_names
    if configured_features is None:
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
    knowledge_graph_filename = getattr(cfg, "knowledge_graph_filename", None)
    tabu_edges = None
    if knowledge_graph_filename:
        G = nx.read_graphml(join(cfg.data_path, knowledge_graph_filename))
        H = G.subgraph(current_column_names + [target_col]).copy()
        H = nx.complement(H)
        col_to_idx = {col: idx for idx, col in enumerate(current_column_names + [target_col])}
        tabu_edges = [(col_to_idx[s], col_to_idx[e]) for (s, e) in H.edges()]
    else:
        logging.info("Recalculating DAG without knowledge-graph tabu edges.")
    # The optional clique solver expects a binary adjacency reference for its
    # internal diagnostics.  Keep the weighted truth separately for the
    # wrapper metrics below; neither form enters the optimization objective.
    solver_reference = (
        None
        if true_w_reference is None
        else (np.abs(np.asarray(true_w_reference, dtype=float)) > 0.0).astype(int)
    )
    w_est, _, gap, lazy_count, solver_stats = _solve_dag_with_backend(
        X_y,
        cfg,
        cfg.nonzero_threshold,
        Y=[],
        B_ref=solver_reference,
        tabu_edges=tabu_edges,
    )
    diagnostics = dict(solver_stats or {})
    diagnostics.setdefault("mip_gap", float(gap) if gap is not None else float("nan"))
    diagnostics.setdefault("lazy_constraints_added", int(lazy_count or 0))
    diagnostics.setdefault("edge_penalty_applied", float(getattr(cfg, "edge_penalty", 0.0) or 0.0))
    max_parents = getattr(cfg, "max_parents", None)
    diagnostics.setdefault(
        "max_parents_applied",
        None if max_parents in (None, "", "null", "None") else int(max_parents),
    )
    if w_est is None:
        raise RuntimeError(
            "DAG solver returned no incumbent for the active fold; "
            f"solver diagnostics={diagnostics}"
        )
    released_support = np.abs(np.asarray(w_est, dtype=float)) > float(
        getattr(cfg, "nonzero_threshold", 0.0)
    )
    np.fill_diagonal(released_support, False)
    diagnostics.setdefault(
        "w_solver_estimated_w_edges", int(np.sum(released_support))
    )
    diagnostics.setdefault("selected_w_edges", int(np.sum(released_support)))
    if true_w_reference is not None:
        diagnostics.update(
            {
                f"w_solver_{key}": value
                for key, value in _graph_structure_metrics(
                    true_w_reference,
                    w_est,
                    getattr(cfg, "nonzero_threshold", 0.0),
                ).items()
            }
        )
    if return_diagnostics:
        return w_est, diagnostics
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
        # Present only when a supplied W is a true synthetic SEM coefficient
        # matrix in raw data units. ``_w_est`` is always the matrix passed to
        # the standardized neural training path; keeping both prevents a raw
        # structural-equation audit from accidentally using scaled weights.
        self._raw_sem_w_est = None
        self._w_matrix_space_ = "model_standardized"
        self._w_est_sha256 = ""
        self._w_cache_key = ""
        self._w_cache_path = ""
        self._w_cache_hit = False
        self._w_solver_diagnostics = {}
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

    def _make_w_cache_key(self, source_prep_data, current_column_names):
        """Fingerprint one fold's DAG input and the DAG solver settings.

        The phase-1 stabilization arms use this key to reuse exactly the same
        fold-local W.  It intentionally excludes the CE backend so ALM/PBM
        arms can share the graph while changing only constraint training.
        """
        if not hasattr(source_prep_data, "columns"):
            return ""
        columns = [str(name) for name in current_column_names]
        if str(self.target_col) not in source_prep_data.columns:
            return ""
        columns = [name for name in columns if name in source_prep_data.columns]
        columns.append(str(self.target_col))
        try:
            table = source_prep_data.loc[:, columns].dropna(subset=[self.target_col])
            values = np.asarray(table.to_numpy(dtype=np.float64), dtype=np.float64)
        except (AttributeError, TypeError, ValueError):
            return ""
        digest = hashlib.sha256()
        digest.update("|".join(columns).encode("utf-8"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
        solver_signature = {
            "dag_solver_backend": str(
                getattr(self.cfg, "dag_solver_backend", "milp")
            ).lower(),
            "time_limit": str(getattr(self.cfg, "time_limit", "")),
            "target_mip_gap": str(getattr(self.cfg, "target_mip_gap", "")),
            "nonzero_threshold": str(getattr(self.cfg, "nonzero_threshold", "")),
            "lambda1": str(getattr(self.cfg, "lambda1", "")),
            "lambda2": str(getattr(self.cfg, "lambda2", "")),
            "edge_penalty": str(getattr(self.cfg, "edge_penalty", "")),
            "max_parents": str(getattr(self.cfg, "max_parents", "")),
            "enable_clique_constraints": str(
                getattr(self.cfg, "enable_clique_constraints", "")
            ),
            "max_clique_size": str(getattr(self.cfg, "max_clique_size", "")),
            "clique_callback_mode": str(
                getattr(self.cfg, "clique_callback_mode", "")
            ),
            "clique_cut_formulation": str(
                getattr(self.cfg, "clique_cut_formulation", "")
            ),
            "clique_cut_selection": str(
                getattr(self.cfg, "clique_cut_selection", "")
            ),
            "clique_top_k": str(getattr(self.cfg, "clique_top_k", "")),
            "max_clique_cuts_per_callback": str(
                getattr(self.cfg, "max_clique_cuts_per_callback", "")
            ),
            "deduplicate_clique_cuts": str(
                getattr(self.cfg, "deduplicate_clique_cuts", "")
            ),
            "weights_bound": str(getattr(self.cfg, "weights_bound", "")),
            "loss_type": str(getattr(self.cfg, "loss_type", "")),
            "constraints_mode": str(getattr(self.cfg, "constraints_mode", "")),
            "robust": str(getattr(self.cfg, "robust", "")),
            "reg_type": str(getattr(self.cfg, "reg_type", "")),
            "a_reg_type": str(getattr(self.cfg, "a_reg_type", "")),
            "callback_mode": str(getattr(self.cfg, "callback_mode", "")),
            "gurobi_seed": str(getattr(self.cfg, "gurobi_seed", "")),
            "gurobi_threads": str(getattr(self.cfg, "gurobi_threads", "")),
            "knowledge_graph_filename": str(
                getattr(self.cfg, "knowledge_graph_filename", "")
            ),
        }
        digest.update(repr(sorted(solver_signature.items())).encode("utf-8"))
        return digest.hexdigest()

    def _synthetic_true_w_for_names(self, current_column_names):
        """Return the synthetic truth in the active feature/target order.

        This path is enabled only by the explicit synthetic oracle audit flag.
        Real-data runs therefore cannot accidentally use their supplied W
        artifact as a reference metric.
        """
        oracle = str(getattr(self.cfg, "constraint_audit_oracle", "none")).strip().lower()
        if oracle not in {"synthetic_linear_sem", "synthetic_generator_oracle"}:
            return None
        try:
            reference = np.asarray(self.w_est, dtype=float)
            names = [str(name) for name in self.row_and_col_names]
            required = [str(name) for name in current_column_names] + [str(self.target_col)]
            indices = [names.index(name) for name in required]
        except (AttributeError, TypeError, ValueError, IndexError):
            return None
        if reference.ndim != 2 or reference.shape[0] != len(names) or reference.shape[1] != len(names):
            return None
        return reference[np.ix_(indices, indices)]

    def _recalculate_or_load_w_est(self, source_prep_data, current_column_names):
        """Estimate W once per fold, optionally reusing a shared cache.

        The cache is opt-in through ``HC_CE_W_CACHE_DIR``.  This keeps all
        historical experiments unchanged while allowing the Phase-1 arms to
        compare constraint handling on the identical estimated DAG.
        """
        cache_dir = os.environ.get("HC_CE_W_CACHE_DIR", "").strip()
        cache_key = self._make_w_cache_key(source_prep_data, current_column_names)
        cache_path = None
        cache_metadata_path = None
        self._w_cache_key = cache_key
        self._w_cache_path = ""
        self._w_cache_hit = False
        self._w_solver_diagnostics = {}
        true_w_reference = self._synthetic_true_w_for_names(current_column_names)

        if cache_dir and cache_key:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"w_{cache_key}.npy")
            cache_metadata_path = f"{cache_path}.json"
            self._w_cache_path = cache_path
            try:
                cached = np.load(cache_path, allow_pickle=False)
                expected_shape = (len(current_column_names) + 1,) * 2
                if cached.shape == expected_shape and np.all(np.isfinite(cached)):
                    self._w_cache_hit = True
                    logging.info("Reusing fold-local W cache: %s", cache_path)
                    self._w_est_sha256 = _matrix_sha256(cached)
                    cached_diagnostics = {
                        "status": "cached",
                        "runtime_seconds": 0.0,
                        "sol_count": 1,
                        "mip_gap": float("nan"),
                        "cache_hit": 1,
                    }
                    if cache_metadata_path and os.path.exists(cache_metadata_path):
                        try:
                            with open(cache_metadata_path, "r", encoding="utf-8") as handle:
                                cached_diagnostics.update(json.load(handle))
                        except (OSError, ValueError, TypeError) as exc:
                            logging.warning(
                                "Could not read W cache metadata %s: %s",
                                cache_metadata_path,
                                exc,
                            )
                    cached_diagnostics["cache_hit"] = 1
                    self._w_solver_diagnostics = cached_diagnostics
                    return cached
                logging.warning("Ignoring invalid W cache with shape %s: %s", cached.shape, cache_path)
            except (OSError, ValueError) as exc:
                if os.path.exists(cache_path):
                    logging.warning("Ignoring unreadable W cache %s: %s", cache_path, exc)

        w_est, solver_diagnostics = compute_recalculated_w_est(
            source_prep_data,
            self.target_col,
            self.cfg,
            current_feature_names=current_column_names,
            return_diagnostics=True,
            true_w_reference=true_w_reference,
        )
        self._w_solver_diagnostics = dict(solver_diagnostics or {})
        self._w_solver_diagnostics["cache_hit"] = 0
        w_est = np.asarray(w_est, dtype=float)
        if not np.all(np.isfinite(w_est)):
            raise ValueError("DAG solver returned a non-finite W matrix.")
        self._w_est_sha256 = _matrix_sha256(w_est)

        if cache_path is not None:
            temporary_path = f"{cache_path}.{os.getpid()}.tmp"
            try:
                with open(temporary_path, "wb") as handle:
                    np.save(handle, w_est, allow_pickle=False)
                os.replace(temporary_path, cache_path)
            except OSError as exc:
                logging.warning("Could not write W cache %s: %s", cache_path, exc)
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
            if cache_metadata_path:
                metadata_tmp = f"{cache_metadata_path}.{os.getpid()}.tmp"
                try:
                    with open(metadata_tmp, "w", encoding="utf-8") as handle:
                        json.dump(self._w_solver_diagnostics, handle, sort_keys=True)
                    os.replace(metadata_tmp, cache_metadata_path)
                except OSError as exc:
                    logging.warning("Could not write W cache metadata %s: %s", cache_metadata_path, exc)
                    try:
                        os.remove(metadata_tmp)
                    except OSError:
                        pass
        return w_est

    def _record_w_est_metadata(self, w_est, current_column_names=None):
        self._w_est_sha256 = _matrix_sha256(w_est)
        self._w_est = w_est
        released_support = np.abs(np.asarray(w_est, dtype=float)) > float(
            getattr(self.cfg, "nonzero_threshold", 0.0)
        )
        np.fill_diagonal(released_support, False)
        self._w_solver_diagnostics.setdefault(
            "w_solver_estimated_w_edges", int(np.sum(released_support))
        )
        self._w_solver_diagnostics.setdefault(
            "selected_w_edges", int(np.sum(released_support))
        )
        if current_column_names is not None:
            true_w_reference = self._synthetic_true_w_for_names(current_column_names)
            if true_w_reference is not None:
                self._w_solver_diagnostics.update(
                    {
                        f"w_solver_{key}": value
                        for key, value in _graph_structure_metrics(
                            true_w_reference,
                            w_est,
                            getattr(self.cfg, "nonzero_threshold", 0.0),
                        ).items()
                    }
                )
        self._w_solver_diagnostics.setdefault(
            "edge_penalty_applied",
            float(getattr(self.cfg, "edge_penalty", 0.0) or 0.0),
        )
        max_parents = getattr(self.cfg, "max_parents", None)
        self._w_solver_diagnostics.setdefault(
            "max_parents_applied",
            None if max_parents in (None, "", "null", "None") else int(max_parents),
        )
        return w_est

    def _get_w_est_for_fit(self, current_column_names, dag_prep_data=None):
        """Resolve W for the active features, with a safe DAG fallback.

        When ``recalculate_dag`` is false, older W artifacts may not contain
        every feature listed by a CoDiet problem.  In that case calculate a
        feature-specific W once and keep the requested false training mode.
        """
        source_prep_data = self.prep_data if dag_prep_data is None else dag_prep_data
        self._raw_sem_w_est = None
        self._w_matrix_space_ = "model_standardized"
        if self.cfg.recalculate_dag:
            return self._record_w_est_metadata(
                self._recalculate_or_load_w_est(source_prep_data, current_column_names),
                current_column_names,
            )

        row_and_col_names_indices = {
            str(name): i for i, name in enumerate(self.row_and_col_names)
        }
        required_names = [str(name) for name in current_column_names] + [str(self.target_col)]
        missing_names = [name for name in required_names if name not in row_and_col_names_indices]
        if missing_names:
            logging.warning(
                "recalculate_dag=false requested, but W artifact is missing %d active "
                "column(s), e.g. %s. Falling back to one DAG recalculation, then "
                "continuing with recalculate_dag=false training.",
                len(missing_names),
                ", ".join(missing_names[:5]),
            )
            return self._record_w_est_metadata(
                self._recalculate_or_load_w_est(source_prep_data, current_column_names),
                current_column_names,
            )

        idx_list = [row_and_col_names_indices[str(name)] for name in current_column_names]
        predict_idx = row_and_col_names_indices[str(self.target_col)]
        w_est = self.w_est[np.ix_(idx_list + [predict_idx], idx_list + [predict_idx])]
        matrix_space = str(
            getattr(self.cfg, "w_matrix_space", "model_standardized")
        ).strip().lower()
        if matrix_space not in {"model_standardized", "raw_sem"}:
            raise ValueError(
                "w_matrix_space must be 'model_standardized' or 'raw_sem', "
                f"got {matrix_space!r}."
            )
        if matrix_space == "raw_sem":
            # Synthetic SEMs supply X_i -> X_j coefficients in raw units, but
            # HC-CE trains on (X_i-mean_i)/scale_i and (Y-mean_Y)/scale_Y.
            # Convert b_ij to b'_ij=b_ij*scale_i/scale_j before it reaches the
            # W mean-moment term. The retained raw matrix is used only by the
            # raw structural-equation oracle audit below.
            if not hasattr(self, "scaler_") or not hasattr(self, "_y_std"):
                raise RuntimeError(
                    "raw_sem W requires fitted feature and target scaling before W resolution."
                )
            raw_sem_w_est = np.asarray(w_est, dtype=float).copy()
            scales = np.concatenate(
                [np.asarray(self.scaler_.scale_, dtype=float), [float(self._y_std)]]
            )
            if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
                raise ValueError("Cannot transform raw_sem W with non-positive/non-finite scales.")
            self._raw_sem_w_est = raw_sem_w_est
            # CE graph construction only needs the support, but S1 also uses
            # the numerical W term. Preserve raw W for CI-only arms so its
            # supplied graph is exactly the synthetic graph.
            if bool(getattr(self.cfg, "use_w_constraints", True)):
                w_est = raw_sem_w_est * (scales[:, None] / scales[None, :])
                self._w_matrix_space_ = "raw_sem_coefficients_transformed_to_model_standardized"
            else:
                self._w_matrix_space_ = "raw_sem_graph_only"
        return self._record_w_est_metadata(w_est, current_column_names)


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

            w_est = self._get_w_est_for_fit(current_column_names)

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
                        random_state=coerce_random_state(
                            getattr(self.cfg, "random_state", None), 42
                        ),
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
                validation_strategy = str(
                    getattr(self.cfg, "validation_split_strategy", "random")
                ).strip().lower()
                if validation_strategy in {"time", "time_series", "last_block"}:
                    n_val = max(1, int(np.ceil(len(y) * val_fraction)))
                    if len(y) - n_val < 2:
                        raise ValueError(
                            "Time validation split leaves fewer than two training rows; "
                            "reduce validation_fraction."
                        )
                    if hasattr(X, "iloc"):
                        X_fit, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
                    else:
                        X_fit, X_val = X[:-n_val], X[-n_val:]
                    if hasattr(y, "iloc"):
                        y_fit, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
                    else:
                        y_fit, y_val = y[:-n_val], y[-n_val:]
                elif validation_strategy in {"random", "shuffle"}:
                    X_fit, X_val, y_fit, y_val = train_test_split(
                        X,
                        y,
                        test_size=val_fraction,
                        random_state=coerce_random_state(
                            getattr(self.cfg, "validation_random_state", None), 0
                        ),
                        shuffle=True,
                    )
                else:
                    raise ValueError(
                        "validation_split_strategy must be one of {'random', 'time'}, "
                        f"got {validation_strategy!r}."
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

        dag_prep_data = None
        dag_fit_scope = str(getattr(self.cfg, "dag_fit_scope", "outer_train")).strip().lower()
        if use_validation and dag_fit_scope == "inner_train":
            if not hasattr(X_fit, "copy") or not hasattr(X_fit, "columns"):
                raise TypeError(
                    "dag_fit_scope=inner_train requires labeled pandas feature columns."
                )
            dag_prep_data = X_fit.copy()
            dag_prep_data[self.target_col] = np.asarray(y_fit)
        elif dag_fit_scope not in {"outer_train", "inner_train"}:
            raise ValueError(
                "dag_fit_scope must be one of {'outer_train', 'inner_train'}, "
                f"got {dag_fit_scope!r}."
            )

        w_est = self._get_w_est_for_fit(current_column_names, dag_prep_data=dag_prep_data)

        self._rf_model_, lam = self.fit_lagrangian_nn_constraint(
            X,
            y,
            w_est,
            self.cfg,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
        )
        self.validation_history_ = getattr(self._rf_model_, "validation_history_", None)
        self.constraint_counts_ = getattr(self._rf_model_, "constraint_counts_", None)
        if isinstance(self.constraint_counts_, dict):
            self.constraint_counts_ = dict(self.constraint_counts_)
            self.constraint_counts_.update(
                {
                    "w_est_sha256": self._w_est_sha256,
                    "w_cache_hit": int(bool(self._w_cache_hit)),
                    "w_solver_status": str(
                        self._w_solver_diagnostics.get("status", "unknown")
                    ),
                    "w_solver_backend": str(
                        self._w_solver_diagnostics.get(
                            "backend",
                            getattr(self.cfg, "dag_solver_backend", "milp"),
                        )
                    ),
                    "w_solver_runtime_seconds": self._w_solver_diagnostics.get(
                        "runtime_seconds", float("nan")
                    ),
                    "w_solver_mip_gap": self._w_solver_diagnostics.get(
                        "mip_gap", float("nan")
                    ),
                    "w_solver_sol_count": self._w_solver_diagnostics.get(
                        "sol_count", float("nan")
                    ),
                    "w_solver_cycle_lazy_cuts": self._w_solver_diagnostics.get(
                        "cycle_lazy_constraints_added", float("nan")
                    ),
                    "w_solver_clique_lazy_cuts": self._w_solver_diagnostics.get(
                        "clique_lazy_constraints_added", float("nan")
                    ),
                    "w_solver_selected_edges": self._w_solver_diagnostics.get(
                        "selected_w_edges", float("nan")
                    ),
                    "w_solver_max_skeleton_clique": self._w_solver_diagnostics.get(
                        "selected_w_max_skeleton_clique", float("nan")
                    ),
                    "w_solver_edge_penalty_applied": self._w_solver_diagnostics.get(
                        "edge_penalty_applied", float("nan")
                    ),
                    "w_solver_max_parents_applied": self._w_solver_diagnostics.get(
                        "max_parents_applied", float("nan")
                    ),
                    "w_solver_true_w_edges": self._w_solver_diagnostics.get(
                        "w_solver_true_w_edges", float("nan")
                    ),
                    "w_solver_estimated_w_edges": self._w_solver_diagnostics.get(
                        "w_solver_estimated_w_edges", float("nan")
                    ),
                    "w_solver_true_edge_tp": self._w_solver_diagnostics.get(
                        "w_solver_true_edge_tp", float("nan")
                    ),
                    "w_solver_false_positive_edges": self._w_solver_diagnostics.get(
                        "w_solver_false_positive_edges", float("nan")
                    ),
                    "w_solver_false_negative_edges": self._w_solver_diagnostics.get(
                        "w_solver_false_negative_edges", float("nan")
                    ),
                    "w_solver_edge_precision": self._w_solver_diagnostics.get(
                        "w_solver_edge_precision", float("nan")
                    ),
                    "w_solver_edge_recall": self._w_solver_diagnostics.get(
                        "w_solver_edge_recall", float("nan")
                    ),
                    "w_solver_edge_f1": self._w_solver_diagnostics.get(
                        "w_solver_edge_f1", float("nan")
                    ),
                    "w_solver_shd": self._w_solver_diagnostics.get(
                        "w_solver_shd", float("nan")
                    ),
                    "w_solver_skeleton_jaccard": self._w_solver_diagnostics.get(
                        "w_solver_skeleton_jaccard", float("nan")
                    ),
                }
            )
            self._rf_model_.constraint_counts_ = self.constraint_counts_

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

    def _synthetic_oracle_prediction_normalized(self, X):
        """Return E[Y|X] for the generated linear synthetic SEM, if requested.

        The oracle deliberately uses the original synthetic structural matrix
        supplied to the estimator (``self.w_est``), not the fold-specific
        ``self._w_est``.  This keeps it valid for S4, where ``self._w_est`` is
        intentionally re-estimated and is therefore not an oracle.
        """
        mode = str(getattr(self.cfg, "constraint_audit_oracle", "none")).strip().lower()
        if mode in {"", "none", "off"}:
            return None
        weights = np.asarray(self.w_est, dtype=float)
        raw_x = np.asarray(X, dtype=float)

        if mode == "synthetic_generator_oracle":
            mechanism = str(
                getattr(self.cfg, "constraint_audit_oracle_mechanism", "linear") or "linear"
            ).strip().lower()
            if mechanism == "temporal_smooth":
                # A fold's static feature matrix has no lag column, so the
                # temporal conditional mean cannot be reconstructed here by
                # reusing the static oracle.  Refuse rather than silently
                # returning a wrong (static) oracle.
                raise ValueError(
                    "synthetic_generator_oracle does not support mechanism "
                    "'temporal_smooth' from a static feature matrix; supply a "
                    "lag-augmented feature set / rolling-split oracle instead."
                )
            reference = self._synthetic_true_w_for_names(list(self.get_current_column_names(X)))
            if reference is None:
                return None
            # ``_synthetic_true_w_for_names`` returns W in [features..., target]
            # order, so the target is the last index.
            target_index = reference.shape[0] - 1
            from synthetic_utils import structural_conditional_mean

            oracle_raw = structural_conditional_mean(
                reference, raw_x, mechanism, target_index)
            return (oracle_raw - float(self._y_mean)) / float(self._y_std)

        if mode != "synthetic_linear_sem":
            raise ValueError(
                "constraint_audit_oracle must be 'none', 'synthetic_linear_sem' "
                f"or 'synthetic_generator_oracle', got {mode!r}."
            )
        if weights.shape != (raw_x.shape[1] + 1, raw_x.shape[1] + 1):
            raise ValueError(
                "Synthetic oracle W shape does not match the active predictor features: "
                f"W={weights.shape}, X={raw_x.shape}."
            )
        oracle_raw = raw_x @ weights[:-1, -1]
        return (oracle_raw - float(self._y_mean)) / float(self._y_std)

    def constraint_audit_rows(self, X, y, *, fold, stage="outer_test", tolerance_reference=None):
        """Evaluate fitted CI/W constraints on held-out rows without retraining.

        The returned rows intentionally retain both the raw signed statistic
        and the statistic used by the dependent penalty.  This makes a
        positive-direction dependence margin auditable rather than implicit.

        ``tolerance_reference`` is an optional ``(X_ref, y_ref)`` pair used to
        build SE-based independence tolerances.  Callers auditing a held-out
        fold must pass the training fold here; otherwise the tolerance (and the
        violation derived from it) would be calibrated on the held-out labels.
        """
        if self._rf_model_ is None:
            return [], [], []

        specs = [dict(spec) for spec in getattr(self._rf_model_, "ci_constraints_", [])]
        names = list(getattr(self._rf_model_, "ci_variable_names_", []))
        if not names:
            names = list(self.get_current_column_names(X)) + [str(self.target_col)]
        raw_x = np.asarray(X, dtype=float)
        x_scaled = self.scaler_.transform(X)
        y_observed = (np.asarray(y, dtype=float).reshape(-1) - float(self._y_mean)) / float(self._y_std)
        with torch.no_grad():
            raw_prediction = self.predict(X)
        if isinstance(raw_prediction, torch.Tensor):
            raw_prediction = raw_prediction.detach().cpu().numpy()
        y_prediction = (
            np.asarray(raw_prediction, dtype=float).reshape(-1) - float(self._y_mean)
        ) / float(self._y_std)
        y_oracle = self._synthetic_oracle_prediction_normalized(X)
        raw_y = np.asarray(y, dtype=float).reshape(-1)
        oracle_raw = (
            None
            if y_oracle is None
            else y_oracle * float(self._y_std) + float(self._y_mean)
        )

        # Provenance for the audit: where the constraint structure came from and
        # which oracle (if any) judged it.  ``true_dag`` means the constraints
        # were built on the supplied (synthetic-truth) W, i.e. recalculate_dag
        # was off; otherwise the DAG was estimated.
        _oracle_mode = str(getattr(self.cfg, "constraint_audit_oracle", "none")).strip().lower()
        _recalc_dag = bool(getattr(self.cfg, "recalculate_dag", True))
        constraint_source = (
            "true_dag" if (_oracle_mode.startswith("synthetic") and not _recalc_dag)
            else "estimated_dag"
        )
        if _oracle_mode == "synthetic_generator_oracle":
            oracle_type = str(
                getattr(self.cfg, "constraint_audit_oracle_mechanism", "linear") or "linear"
            ).strip().lower()
        elif _oracle_mode == "synthetic_linear_sem":
            oracle_type = "linear"
        else:
            oracle_type = "none"

        metadata_rows = []
        statistic_rows = []
        if specs:
            x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)

            def evaluate(response):
                response_tensor = torch.as_tensor(response, dtype=torch.float32)
                with torch.no_grad():
                    values = signed_expectation_equalities(
                        x_tensor, response_tensor, specs, cfg=self.cfg
                    )
                return values.detach().cpu().numpy().astype(float)

            observed_values = evaluate(y_observed)
            prediction_values = evaluate(y_prediction)
            oracle_values = evaluate(y_oracle) if y_oracle is not None else None
            tolerance = float(getattr(self.cfg, "ce_independence_tolerance", 0.0))
            independent_indices = [
                index for index, spec in enumerate(specs)
                if str(spec.get("type", "independent")) == "independent"
            ]
            tolerance_by_index = {}
            if independent_indices:
                independent_specs = [specs[index] for index in independent_indices]
                if tolerance_reference is not None:
                    ref_X, ref_y = tolerance_reference
                    reference_x_tensor = torch.as_tensor(
                        self.scaler_.transform(ref_X), dtype=torch.float32
                    )
                    reference_observed = torch.as_tensor(
                        (np.asarray(ref_y, dtype=float).reshape(-1) - float(self._y_mean))
                        / float(self._y_std),
                        dtype=torch.float32,
                    )
                else:
                    reference_x_tensor = x_tensor
                    reference_observed = torch.as_tensor(y_observed, dtype=torch.float32)
                with torch.no_grad():
                    effective_tolerances = independent_expectation_tolerances(
                        reference_x_tensor,
                        reference_observed,
                        independent_specs,
                        tolerance=tolerance,
                        cfg=self.cfg,
                    ).detach().cpu().numpy().astype(float)
                tolerance_by_index = {
                    constraint_index: float(effective_tolerances[position])
                    for position, constraint_index in enumerate(independent_indices)
                }
            dependent_statistic = str(
                getattr(self.cfg, "ci_dependent_statistic", "signed")
            ).strip().lower()

            for constraint_index, spec in enumerate(specs, start=1):
                relation = str(spec.get("type", "independent"))
                x_index = int(spec["x_index"])
                y_index = int(spec["y_index"])
                z_indices = [int(index) for index in spec.get("z_indices", []) or []]
                margin = (
                    tolerance_by_index.get(constraint_index - 1, tolerance)
                    if relation == "independent"
                    else float(spec.get("margin", 0.0))
                )
                x_name = names[x_index] if x_index < len(names) else f"x{x_index}"
                y_name = names[y_index] if y_index < len(names) else f"x{y_index}"
                z_names = [names[index] if index < len(names) else f"x{index}" for index in z_indices]
                collider_index = spec.get("collider_index")
                collider_name = (
                    names[int(collider_index)]
                    if collider_index is not None and int(collider_index) < len(names)
                    else None
                )
                metadata_rows.append(
                    {
                        "fold": int(fold),
                        "stage": stage,
                        "constraint_id": int(constraint_index),
                        "relation": relation,
                        "source": spec.get("source", "manual"),
                        "mode": spec.get("mode", "manual"),
                        "x_index": x_index,
                        "x_name": x_name,
                        "y_index": y_index,
                        "y_name": y_name,
                        "z_indices": ";".join(str(index) for index in z_indices),
                        "z_names": ";".join(z_names),
                        "collider_index": collider_index,
                        "collider_name": collider_name,
                        "margin_or_tolerance": margin,
                        "posterior_support": spec.get("posterior_support"),
                        "opposing_support": spec.get("opposing_support"),
                        "constraint_source": constraint_source,
                        "oracle_type": oracle_type,
                    }
                )

                def violation(value, independent_tolerance=0.0):
                    if relation == "independent":
                        return max(abs(value) - independent_tolerance, 0.0), abs(value)
                    value_tensor = torch.as_tensor(value, dtype=torch.float32)
                    enforced = float(
                        dependent_statistic_values(value_tensor, cfg=self.cfg).item()
                    )
                    return max(margin - enforced, 0.0), enforced

                row_tolerance = tolerance_by_index.get(constraint_index - 1, tolerance)
                tolerance_mode = str(getattr(self.cfg, "ce_tolerance_mode", "fixed"))
                tolerance_multiplier = float(getattr(self.cfg, "ce_tolerance_sd_multiplier", 1.96))
                estimated_se = (
                    max((row_tolerance - tolerance) / tolerance_multiplier, 0.0)
                    if relation == "independent"
                    and tolerance_mode in {"se", "standard_error"}
                    and tolerance_multiplier > 0
                    and not _is_discrete_ci_kind(str(getattr(self.cfg, "ci_penalty_kind", "")))
                    else None
                )
                observed_violation, observed_enforced = violation(
                    observed_values[constraint_index - 1], row_tolerance
                )
                prediction_violation, prediction_enforced = violation(
                    prediction_values[constraint_index - 1], row_tolerance
                )
                oracle_value = None if oracle_values is None else oracle_values[constraint_index - 1]
                oracle_violation, oracle_enforced = (
                    (None, None) if oracle_value is None else violation(oracle_value, row_tolerance)
                )
                # statistic_validity is per-constraint and only known after the
                # oracle violation is computed, so set it on the row we just
                # appended.
                metadata_rows[-1]["statistic_validity"] = (
                    "not_audited" if oracle_value is None
                    else ("pass" if float(oracle_violation or 0.0) <= 1e-9 else "fail")
                )
                audit_kind = str(
                    getattr(self._rf_model_, "ci_penalty_kind_", "conditional_expectation")
                )
                if _is_discrete_ci_kind(audit_kind):
                    window_summary = {
                        "window_count": 0,
                        "window_statistics": [],
                        "window_mean": None,
                        "window_std": None,
                        "window_sign_flip_rate": None,
                    }
                else:
                    window_summary = constraint_window_statistics(
                        x_tensor,
                        torch.as_tensor(y_observed, dtype=torch.float32),
                        spec,
                        cfg=self.cfg,
                    )
                window_std = window_summary["window_std"]
                window_snr = (
                    abs(window_summary["window_mean"]) / window_std
                    if window_std is not None and window_std > 1e-12
                    else None
                )
                statistic_rows.append(
                    {
                        "fold": int(fold),
                        "stage": stage,
                        "constraint_id": int(constraint_index),
                        "relation": relation,
                        "statistic_kind": getattr(
                            self._rf_model_, "ci_penalty_kind_", "conditional_expectation"
                        ),
                        "ce_statistic_kind": (
                            "not_applicable"
                            if _is_discrete_ci_kind(audit_kind)
                            else getattr(self.cfg, "ce_statistic_kind", "partial_correlation")
                        ),
                        "ce_statistic_shrinkage": (
                            None
                            if _is_discrete_ci_kind(audit_kind)
                            else float(getattr(self.cfg, "ce_statistic_shrinkage", 0.0))
                        ),
                        "ce_residualize_method": getattr(self.cfg, "ce_residualize_method", "linear"),
                        "ce_se_method": (
                            "not_applicable"
                            if _is_discrete_ci_kind(audit_kind)
                            else getattr(self.cfg, "ce_se_method", "auto")
                        ),
                        "ce_tolerance_mode": tolerance_mode,
                        "ce_tolerance_sd_multiplier": tolerance_multiplier,
                        "estimated_se": estimated_se,
                        "effective_tolerance": row_tolerance if relation == "independent" else None,
                        "window_count": window_summary["window_count"],
                        "window_statistics": ";".join(
                            f"{value:.10g}" for value in window_summary["window_statistics"]
                        ),
                        "window_mean": window_summary["window_mean"],
                        "window_std": window_std,
                        "window_abs_mean_over_spread": window_snr,
                        "window_sign_flip_rate": window_summary["window_sign_flip_rate"],
                        "window_sign_flip_filter_applicable": relation == "dependent",
                        "dependent_statistic": dependent_statistic,
                        "observed_y_statistic": float(observed_values[constraint_index - 1]),
                        "prediction_statistic": float(prediction_values[constraint_index - 1]),
                        "oracle_statistic": None if oracle_value is None else float(oracle_value),
                        "observed_enforced_statistic": float(observed_enforced),
                        "prediction_enforced_statistic": float(prediction_enforced),
                        "oracle_enforced_statistic": oracle_enforced,
                        "observed_violation": float(observed_violation),
                        "prediction_violation": float(prediction_violation),
                        "oracle_violation": oracle_violation,
                    }
                )

        w_rows = []
        if bool(getattr(self._rf_model_, "w_constraint_enabled_", False)):
            weights = np.asarray(self._w_est, dtype=float)
            if weights.shape != (x_scaled.shape[1] + 1, x_scaled.shape[1] + 1):
                raise ValueError(
                    "W constraint audit matrix does not match active predictor features: "
                    f"W={weights.shape}, X={x_scaled.shape}."
                )
            residual_matrix = weights - np.eye(weights.shape[0])

            def w_residual(response):
                return residual_matrix @ np.concatenate(
                    [x_scaled.mean(axis=0), [float(np.mean(response))]]
                )

            observed_residual = w_residual(y_observed)
            prediction_residual = w_residual(y_prediction)
            oracle_residual = w_residual(y_oracle) if y_oracle is not None else None
            # The legacy W penalty is a first-moment condition evaluated in
            # model-standardized coordinates.  For a synthetic true W we also
            # expose the raw target structural equation, which should be zero
            # for the analytic E[Y|X] oracle.  These are diagnostics, not two
            # interchangeable constraint definitions.
            raw_sem_weights = getattr(self, "_raw_sem_w_est", None)
            raw_target_weights = (
                None
                if raw_sem_weights is None
                else np.asarray(raw_sem_weights, dtype=float)[:-1, -1]
            )
            observed_target_equation = (
                None if raw_target_weights is None else raw_y - raw_x @ raw_target_weights
            )
            prediction_target_equation = (
                None
                if raw_target_weights is None
                else np.asarray(raw_prediction, dtype=float).reshape(-1) - raw_x @ raw_target_weights
            )
            oracle_target_equation = (
                None
                if oracle_raw is None or raw_target_weights is None
                else oracle_raw - raw_x @ raw_target_weights
            )
            for coordinate in range(weights.shape[0]):
                variable_name = names[coordinate] if coordinate < len(names) else f"x{coordinate}"
                w_rows.append(
                    {
                        "fold": int(fold),
                        "stage": stage,
                        "coordinate": coordinate,
                        "variable": variable_name,
                        "mean_moment_space": f"model_standardized_xy_with_{self._w_matrix_space_}",
                        "structural_target_space": (
                            "raw_xy_with_true_raw_sem_W"
                            if raw_target_weights is not None
                            else "not_available_without_true_raw_sem_W"
                        ),
                        "observed_y_residual": float(observed_residual[coordinate]),
                        "prediction_residual": float(prediction_residual[coordinate]),
                        "oracle_residual": None if oracle_residual is None else float(oracle_residual[coordinate]),
                        "observed_l2": float(np.linalg.norm(observed_residual)),
                        "prediction_l2": float(np.linalg.norm(prediction_residual)),
                        "oracle_l2": None if oracle_residual is None else float(np.linalg.norm(oracle_residual)),
                        "observed_target_equation_mean": (
                            None
                            if observed_target_equation is None
                            else float(np.mean(observed_target_equation))
                        ),
                        "prediction_target_equation_mean": (
                            None
                            if prediction_target_equation is None
                            else float(np.mean(prediction_target_equation))
                        ),
                        "oracle_target_equation_mean": (
                            None
                            if oracle_target_equation is None
                            else float(np.mean(oracle_target_equation))
                        ),
                        "observed_target_equation_rmse": (
                            None
                            if observed_target_equation is None
                            else float(np.sqrt(np.mean(np.square(observed_target_equation))))
                        ),
                        "prediction_target_equation_rmse": (
                            None
                            if prediction_target_equation is None
                            else float(np.sqrt(np.mean(np.square(prediction_target_equation))))
                        ),
                        "oracle_target_equation_rmse": (
                            None
                            if oracle_target_equation is None
                            else float(np.sqrt(np.mean(np.square(oracle_target_equation))))
                        ),
                    }
                )

        return metadata_rows, statistic_rows, w_rows


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


