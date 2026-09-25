#!/usr/bin/env python3
"""A8 lag-channel preflight runner -- SKELETON.

Fixes the *contract* of the preflight described in ``docs/a8_lag_preflight.md``
before any W alignment exists:

* three feature tiers, ``t0 = X_t``, ``t1 = X_t + X_{t-1}``,
  ``t2 = Pa_t + Pa_{t-1}`` (the target's true parents);
* one shared ``valid_rows`` mask and one shared expanding-window split for all
  three tiers, so a model difference cannot be confused with a split or a row
  difference;
* ``xgb100`` and ``nn0`` on the identical folds;
* the lag-aware oracle as a reference only;
* no CE/W arms, no strict evidence -- the output carries a diagnostic batch id
  and ``strict_inference_allowed: false``.

``nn0`` is deliberately blocked: the repo's NN needs a W matrix and its lag
columns may only enter W through the audited adapter
(``scripts/evidence_tree/w_adapter.py``).  Until that is implemented and
reviewed, the run records ``nn0: blocked`` instead of fitting a mis-aligned
model.

NMSE uses the repository definition exactly: per fold
``MSE(test) / MSE(y_train_fold, mean(y_train_fold))``, then averaged over folds
(``recommender_utils.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evidence_tree"))

from synthetic_utils import (  # noqa: E402
    add_lag_features,
    simulate_synthetic_problem,
    structural_conditional_mean,
    temporal_parent_blocks,
)
import w_adapter  # noqa: E402

TIERS = ("t0", "t1", "t2")
DEFAULT_BATCH_ID = "et_a8_lag_preflight_v1"
# The 100-round XGB reference.  Kept in one place so the estimator manifest and
# the fitted model can never drift apart.
XGB100_PARAMS = {"n_estimators": 100, "tree_method": "hist", "n_jobs": 1}
DEFAULT_VALIDATION_FRACTION = 0.2


def _hash_indices(indices) -> str:
    payload = ",".join(str(int(i)) for i in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_time_ordered(index) -> None:
    """Refuse to build splits on rows that are not time-ordered.

    Mirrors the repository contract in ``recommender_utils.py`` (the
    ``cv_strategy=time_series`` branch): a positional expanding-window split is
    only meaningful when rows run earliest -> latest, but ``TimeSeriesSplit``
    itself never checks this.  Guarding here keeps the skeleton's stated
    contract ("rows must be time-sorted") true rather than aspirational.
    """
    if not pd.Index(index).is_monotonic_increasing:
        raise ValueError(
            "Time-series CV requires rows sorted from earliest to latest. "
            "Sort the problem data before constructing splits."
        )


def last_block_validation_split(n_samples: int, val_fraction: float = 0.2
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Inner-validation split mirroring the estimator's ``strategy=time``.

    ``recommender_estimator.py`` holds out the final ``ceil(n * fraction)`` rows
    as validation and requires at least two rows to remain for fitting.  The
    NN's early-stopping split must follow the same rule; a random inner split
    would let future information select the model.  Fixing it here, before the
    NN arm is unblocked, keeps the inner and outer contracts consistent.
    """
    n = int(n_samples)
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    n_val = max(1, int(np.ceil(n * float(val_fraction))))
    if n - n_val < 2:
        raise ValueError(
            "Time validation split leaves fewer than two training rows; "
            "reduce validation_fraction."
        )
    return np.arange(0, n - n_val), np.arange(n - n_val, n)


def build_frame(graph_seed, noise_seed, n_samples, n_nodes, expected_edges, rho):
    frame, weights, _oracle, meta = simulate_synthetic_problem(
        graph_type="ER",
        n_samples=n_samples,
        n_nodes=n_nodes,
        expected_edges=expected_edges,
        mechanism="temporal_smooth",
        graph_seed=int(graph_seed),
        noise_seed=int(noise_seed),
        target_node=n_nodes - 1,
        temporal_rho=rho,
    )
    return frame, weights, meta


def build_tiers(frame: pd.DataFrame, meta: dict, target: str
                ) -> tuple[dict[str, dict], pd.DataFrame]:
    """Return the three feature tiers over one shared lag-augmented frame."""
    nodes = [c for c in frame.columns if c != target]
    parents = [str(p) for p in meta["true_parents_of_target"]]
    lagged = add_lag_features(frame, 1, columns=nodes)
    return {
        "t0": {"source": "X_t", "columns": nodes},
        "t1": {"source": "X_t + X_{t-1}",
               "columns": nodes + [f"{c}_lag1" for c in nodes]},
        "t2": {"source": "Pa_t + Pa_{t-1}",
               "columns": parents + [f"{c}_lag1" for c in parents]},
    }, lagged


def shared_valid_rows(lagged: pd.DataFrame, tiers: dict, target: str) -> np.ndarray:
    """Rows complete in *every* tier, so all tiers see the same samples."""
    mask = pd.Series(True, index=lagged.index)
    for spec in tiers.values():
        mask &= lagged[spec["columns"]].notna().all(axis=1)
    mask &= lagged[target].notna()
    return np.flatnonzero(mask.to_numpy())


def expanding_window_splits(positions: np.ndarray, n_splits: int,
                            test_size, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit over shared positions; identical for every tier.

    The shared row mask drops the rows without a predecessor, so the split
    budget must be checked against the *masked* count, not ``n_samples``.
    Failing here with a precise message beats an opaque sklearn error.
    """
    n = int(len(positions))
    if int(n_splits) < 2:
        raise ValueError("expanding-window CV needs n_splits >= 2.")
    if test_size is not None:
        required = int(n_splits) * int(test_size) + max(int(gap), 0)
        if n <= required:
            raise ValueError(
                f"shared valid rows = {n} cannot host n_splits={n_splits} x "
                f"test_size={test_size} (+gap={gap}). The lag tiers drop every "
                "row without a predecessor, so reduce --test-size or --n-splits."
            )
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    return [(positions[train], positions[test])
            for train, test in splitter.split(positions)]


def nmse_folds(y_true: np.ndarray, y_pred: np.ndarray,
               splits) -> np.ndarray:
    """Repository NMSE: per-fold MSE normalised by that fold's mean baseline."""
    out = []
    y = np.asarray(y_true, dtype=float)
    for train_idx, test_idx in splits:
        y_train = y[train_idx]
        normalizer = max(float(np.mean((y_train - y_train.mean()) ** 2)),
                         np.finfo(float).eps)
        out.append(float(np.mean((y[test_idx] - np.asarray(y_pred[test_idx])) ** 2))
                   / normalizer)
    return np.asarray(out, dtype=float)


def fit_predict_xgb100(x_train, y_train, x_test, seed: int) -> np.ndarray:
    from xgboost import XGBRegressor

    model = XGBRegressor(random_state=int(seed), **XGB100_PARAMS)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_predict_nn0(x_train, y_train, x_test, *, base_names, feature_names,
                    base_w, target):
    """Blocked: the NN path needs the audited W alignment first.

    The repo NN consumes a W matrix; on ``t1``/``t2`` the lag columns are not
    DAG nodes, so W must go through :func:`w_adapter.build_augmented_w`.  Until
    that is implemented and reviewed this hook refuses instead of fitting a
    mis-aligned model (which could also trip the missing-column MILP fallback).
    """
    try:
        w_adapter.build_augmented_w(
            base_w=base_w,
            base_names=base_names,
            feature_names=feature_names,
            target=target,
        )
    except NotImplementedError as exc:
        raise NotImplementedError(f"nn0 blocked by the W adapter: {exc}") from exc
    raise AssertionError("build_augmented_w unexpectedly returned")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--expected-edges", type=int, default=30)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=1500)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    parser.add_argument("--models", nargs="+", default=["xgb100", "nn0"])
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and print the contract without fitting models")
    args = parser.parse_args(argv)

    reports = []
    for graph_seed in args.graph_seeds:
        for noise_seed in args.noise_seeds:
            frame, weights, meta = build_frame(
                graph_seed, noise_seed, args.n_samples, args.n_nodes,
                args.expected_edges, args.rho)
            target = str(meta["target_node"])
            tiers, lagged = build_tiers(frame, meta, target)
            assert_time_ordered(lagged.index)
            rows = shared_valid_rows(lagged, tiers, target)
            splits = expanding_window_splits(rows, args.n_splits, args.test_size, args.gap)

            y_all = lagged[target].to_numpy(dtype=float)
            node_block = frame[list(frame.columns)].to_numpy(dtype=float)
            _cur, lag_block = temporal_parent_blocks(weights, node_block, int(target[1:]))
            oracle = structural_conditional_mean(
                weights, node_block, meta["mechanism"], int(target[1:]),
                temporal_parents=lag_block, temporal_rho=float(meta["temporal_rho"]),
                require_lag=True)

            split_manifest = {
                "cv_strategy": "time_series",
                "validation_split_strategy": "time",
                "n_splits": int(args.n_splits),
                "test_size": int(args.test_size),
                "gap": int(args.gap),
                "valid_rows": int(len(rows)),
                "valid_rows_hash": _hash_indices(rows),
                "folds": [
                    {"fold": i + 1,
                     "n_train": int(len(tr)), "n_test": int(len(te)),
                     "train_index_hash": _hash_indices(tr),
                     "test_index_hash": _hash_indices(te)}
                    for i, (tr, te) in enumerate(splits)
                ],
            }
            feature_manifest = {
                tier: {
                    "source": spec["source"],
                    "columns": list(spec["columns"]),
                    "n_features": len(spec["columns"]),
                    "column_class": {
                        "node_columns": [c for c in spec["columns"] if "_lag" not in c],
                        "lag_columns": [c for c in spec["columns"] if "_lag" in c],
                    },
                }
                for tier, spec in tiers.items()
            }
            oracle_nmse = nmse_folds(y_all, oracle, splits)

            model_rows = []
            for tier in args.tiers:
                columns = tiers[tier]["columns"]
                x = lagged[columns].to_numpy(dtype=float)
                if "xgb100" in args.models:
                    if args.dry_run:
                        # --dry-run must not fit anything; emit a plan-only row so
                        # the report schema stays complete without a model number
                        # that could be mistaken for a result.
                        model_rows.append({"graph_seed": graph_seed, "noise_seed": noise_seed,
                                           "tier": tier, "model": "xgb100",
                                           "fold": 0, "nmse": np.nan, "status": "dry-run"})
                    else:
                        pred = np.full(len(y_all), np.nan)
                        for tr, te in splits:
                            pred[te] = fit_predict_xgb100(
                                x[tr], y_all[tr], x[te], seed=int(graph_seed * 100000 + noise_seed))
                        for i, value in enumerate(nmse_folds(y_all, pred, splits), start=1):
                            model_rows.append({"graph_seed": graph_seed, "noise_seed": noise_seed,
                                               "tier": tier, "model": "xgb100",
                                               "fold": i, "nmse": value})
                if "nn0" in args.models:
                    model_rows.append({"graph_seed": graph_seed, "noise_seed": noise_seed,
                                       "tier": tier, "model": "nn0",
                                       "fold": 0, "nmse": np.nan, "status": "blocked"})
            for i, value in enumerate(oracle_nmse, start=1):
                model_rows.append({"graph_seed": graph_seed, "noise_seed": noise_seed,
                                   "tier": "oracle", "model": "lag_oracle",
                                   "fold": i, "nmse": value})

            report = {
                "batch_id": args.batch_id,
                "strict_inference_allowed": False,
                "scope": "synthetic/Temporal",
                "mechanism": meta["mechanism"],
                "rho": float(meta["temporal_rho"]),
                "graph_seed": int(graph_seed),
                "noise_seed": int(noise_seed),
                "target": target,
                "oracle_metrics": {
                    "definition": "MSE(test)/MSE(y_train,mean(y_train)), mean over folds",
                    # An oracle number may only be reported next to the tier
                    # that produced it, so the evaluated block is explicit.
                    "evaluated_on": "t2 features (P_t + P_{t-1})",
                    "require_lag": True,
                    "mean_nmse": float(np.nanmean(oracle_nmse)),
                    "folds": [float(v) for v in oracle_nmse],
                },
                "split_manifest": split_manifest,
                "feature_manifest": feature_manifest,
                "estimator_manifest": {
                    "xgb100": {
                        "params": dict(XGB100_PARAMS),
                        "random_state": "graph_seed*100000 + noise_seed",
                        "reference_parity": (
                            "UNVERIFIED: A8's registered xgb100 runs the repo "
                            "'mark' solver (scripts/evidence_tree/"
                            "A8_nn_favorable_synthetic.sh). This preflight fits a "
                            "bare XGBRegressor on the same 100-round budget; parity "
                            "with the registered reference is NOT claimed."
                        ),
                    },
                    "nn0": {"status": "blocked: needs w_adapter.build_augmented_w"},
                    "inner_validation": {
                        "strategy": "time",
                        "rule": "last ceil(n*validation_fraction) rows, >=2 train rows",
                        "validation_fraction": DEFAULT_VALIDATION_FRACTION,
                    },
                    "ce_w": "out of scope (no CE/W in the preflight)",
                },
                "model_status": {
                    "xgb100": ("fitted" if "xgb100" in args.models and not args.dry_run
                               else "dry-run/skipped"),
                    "nn0": "blocked: needs scripts/evidence_tree/w_adapter.py",
                    "ce_w": "out of scope (no CE/W in the preflight)",
                },
                "rows": model_rows,
            }
            reports.append(report)

            print(f"graph={graph_seed} noise={noise_seed} target={target} "
                  f"valid_rows={len(rows)} "
                  f"oracle_nmse={report['oracle_metrics']['mean_nmse']:.4f}")
            for tier in args.tiers:
                fm = feature_manifest[tier]
                if args.dry_run:
                    metrics = "dry-run"
                else:
                    values = [r["nmse"] for r in model_rows
                              if r["tier"] == tier and r["model"] == "xgb100"]
                    metrics = f"xgb100={np.nanmean(values):.4f}"
                print(f"  {tier}: n_feat={fm['n_features']:>3} "
                      f"({len(fm['column_class']['node_columns'])} node + "
                      f"{len(fm['column_class']['lag_columns'])} lag) {metrics}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"batch_id": args.batch_id, "strict_inference_allowed": False,
             "reports": reports}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
