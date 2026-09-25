#!/usr/bin/env python3
"""Non-training mixing diagnostic for the temporal synthetic SEM.

The temporal generator starts from an all-zero state, so its first samples are
transient, not draws from the stationary law.  Before any *strict* temporal
confirm we must freeze a burn-in length determined from data, not guessed.

This script never trains a model.  It generates one long series per
``(graph_seed, noise_seed)`` unit, then compares the running mean / variance /
lag-1 autocorrelation of every node against the steady-state estimates taken
from the tail of the same series.  The reported ``burn_in`` is the smallest
prefix length after which *all* nodes agree within tolerance.

Usage::

    ./codiet311/bin/python scripts/evidence_tree/burnin_mixing_diagnostic.py \
        --graph-seeds 42 43 --noise-seeds 101 --n-samples 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthetic_utils import simulate_synthetic_problem  # noqa: E402

# Relative tolerances the burn-in must satisfy before it can be frozen.
DEFAULT_TOL_MEAN = 0.05
DEFAULT_TOL_VAR = 0.05
DEFAULT_TOL_ACF = 0.05


def _acf1(series: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation per column (columns are nodes)."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    denom = np.sum(x * x, axis=0)
    numer = np.sum(x[:-1] * x[1:], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, numer / denom, 0.0)
    return out


def _stats(series: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(series, dtype=float)
    return {"mean": x.mean(axis=0), "var": x.var(axis=0), "acf1": _acf1(x)}


def _converged(candidate: dict, reference: dict, *, scale: np.ndarray,
               tol_mean: float, tol_var: float, tol_acf: float) -> bool:
    """Scale-aware stationarity check.

    A purely relative test is wrong here: a lag-1 autocorrelation whose true
    value is ~0 makes any tiny difference look like a 100% error.  So the
    tolerances are expressed in each quantity's natural scale:

    * mean  : absolute, scaled by the steady-state per-node std;
    * var   : relative to the steady-state variance;
    * acf1  : absolute (the quantity lives in [-1, 1]).
    """
    mean_ok = np.all(np.abs(candidate["mean"] - reference["mean"]) <= tol_mean * scale)
    var_ok = np.all(
        np.abs(candidate["var"] - reference["var"])
        <= tol_var * np.maximum(reference["var"], 1e-12)
    )
    acf_ok = np.all(np.abs(candidate["acf1"] - reference["acf1"]) <= tol_acf)
    return bool(mean_ok and var_ok and acf_ok)


def mixing_time(
    series: np.ndarray,
    *,
    tail_fraction: float = 0.5,
    tol_mean: float = DEFAULT_TOL_MEAN,
    tol_var: float = DEFAULT_TOL_VAR,
    tol_acf: float = DEFAULT_TOL_ACF,
    stride: int = 10,
) -> dict:
    """Return the smallest prefix to drop so the remainder is stationary."""
    x = np.asarray(series, dtype=float)
    n = x.shape[0]
    tail_start = max(1, int(n * (1.0 - tail_fraction)))
    ref = _stats(x[tail_start:])
    scale = np.sqrt(np.maximum(ref["var"], 1e-12))

    candidates = list(range(0, tail_start, max(1, int(stride))))
    for burn in candidates:
        cand = _stats(x[burn:])
        if _converged(cand, ref, scale=scale,
                      tol_mean=tol_mean, tol_var=tol_var, tol_acf=tol_acf):
            return {
                "burn_in": int(burn),
                "tail_start": int(tail_start),
                "n_samples": int(n),
                "converged": True,
                "tol": {"mean": tol_mean, "var": tol_var, "acf1": tol_acf},
            }
    return {
        "burn_in": int(tail_start),
        "tail_start": int(tail_start),
        "n_samples": int(n),
        "converged": False,
        "tol": {"mean": tol_mean, "var": tol_var, "acf1": tol_acf},
    }


def decay_profile(series: np.ndarray, *, tail_fraction: float = 0.5,
                  prefixes=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)) -> dict:
    """Sensitive, cheap transient report for the first ``K`` samples.

    ``mixing_time`` can only say "the transient is below tolerance"; with
    20 000 samples a single transient row cannot move the mean.  This profile
    reports the *normalised* deviation of the first ``K`` samples from the
    steady state, maxed over nodes, so the decay is visible even when it never
    crosses the tolerance.
    """
    x = np.asarray(series, dtype=float)
    n = x.shape[0]
    tail_start = max(1, int(n * (1.0 - tail_fraction)))
    ref_mean = x[tail_start:].mean(axis=0)
    scale = np.sqrt(np.maximum(x[tail_start:].var(axis=0), 1e-12))
    profile = {}
    for k in prefixes:
        if k >= tail_start:
            continue
        dev = np.abs(x[:k].mean(axis=0) - ref_mean) / scale
        profile[str(int(k))] = float(np.max(dev))
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--expected-edges", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    reports = []
    for graph_seed in args.graph_seeds:
        for noise_seed in args.noise_seeds:
            frame, _weights, _oracle, meta = simulate_synthetic_problem(
                graph_type="ER",
                n_samples=args.n_samples,
                n_nodes=args.n_nodes,
                expected_edges=args.expected_edges,
                mechanism="temporal_smooth",
                graph_seed=int(graph_seed),
                noise_seed=int(noise_seed),
                target_node=args.n_nodes - 1,
                temporal_rho=args.rho,
            )
            series = frame.to_numpy(dtype=float)
            result = mixing_time(series, stride=args.stride)
            result.update({
                "graph_seed": int(graph_seed),
                "noise_seed": int(noise_seed),
                "mechanism": meta.get("mechanism"),
                "rho": float(args.rho),
                "decay_profile": decay_profile(series),
                # The sequential generator never assigns row 0 (the loop starts
                # at t=1), so it stays all-zero: an impossible observation that
                # is currently fed to the models.  BURN_IN>=1 removes it; the
                # proper fix is to initialise row 0 from the stationary law.
                "first_row_all_zero": bool(np.all(series[0] == 0.0)),
                "n_all_zero_rows": int(np.sum(np.all(series == 0.0, axis=1))),
            })
            reports.append(result)
            profile = result["decay_profile"]
            head = ", ".join(f"K={k}:{v:.3f}" for k, v in list(profile.items())[:5])
            print(
                f"graph={graph_seed} noise={noise_seed} "
                f"burn_in={result['burn_in']} "
                f"converged={result['converged']} "
                f"zero_rows={result['n_all_zero_rows']} "
                f"decay[{head}]"
            )

    worst = max((r["burn_in"] for r in reports), default=0)
    summary = {
        "mechanism": "temporal_smooth",
        "rho": float(args.rho),
        "units": len(reports),
        "burn_in_required": int(worst),
        "all_converged": all(r["converged"] for r in reports),
        "reports": reports,
    }
    print(f"burn_in_required={worst} all_converged={summary['all_converged']}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
