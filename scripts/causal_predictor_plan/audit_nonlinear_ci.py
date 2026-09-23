#!/usr/bin/env python3
"""Audit CI statistics against a known synthetic graph before training.

For each triple (x, target | z) induced by the generator graph, label it with
the true d-separation status and score three conditional-independence
statistics on the same data:

  - partial_correlation : the statistic HC-CE currently uses (linear);
  - hsic                 : HSIC between the X and Y residuals on Z;
  - soft_cmi             : quantile-binned soft conditional mutual information.

A statistic that cannot separate true independence from dependence on nonlinear
data must not be pushed into the training loop.  For each statistic we report
the ROC AUC and the false-negative rate at the threshold that holds the
false-positive rate at --target-fpr (default 5%).  No permutation test is used,
so the audit is fast and threshold-consistent across statistics.

Usage:
  python audit_nonlinear_ci.py --graph-type ER --sem-type nonlinear \
    --graph-seeds 42 43 44 --noise-seeds 101 --output results/ci_audit_ER_nl.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from synthetic_utils import load_data  # noqa: E402
import hc_predictor_ce as ce  # noqa: E402


def _graph_from_weights(weights):
    graph = nx.DiGraph()
    n = weights.shape[0]
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and weights[i, j] != 0:
                graph.add_edge(i, j)
    return graph


def _hsic(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    if n < 4:
        return 0.0

    def kernel(a):
        sq = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=2)
        med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
        return np.exp(-sq / max(med, 1e-12))

    K = kernel(x)
    L = kernel(y)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / ((n - 1) ** 2))


def _residual_hsic(x, y, z):
    if z is None or len(z) == 0:
        return _hsic(x, y)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    z_t = torch.as_tensor(np.asarray(z, dtype=np.float32))
    with torch.no_grad():
        x_res = ce._residualize(x_t, z_t).cpu().numpy()
        y_res = ce._residualize(y_t, z_t).cpu().numpy()
    return _hsic(x_res, y_res)


def _partial_corr(x, y, z):
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    z_t = None if z is None or len(z) == 0 else torch.as_tensor(np.asarray(z, dtype=np.float32))
    with torch.no_grad():
        return float(abs(ce.conditional_partial_correlation(x_t, y_t, z_t)))


def _soft_cmi(x, y, z):
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    z_t = None if z is None or len(z) == 0 else torch.as_tensor(np.asarray(z, dtype=np.float32))
    with torch.no_grad():
        return float(ce.soft_discrete_conditional_mutual_information(x_t, y_t, z_t, n_bins=4))


def _candidate_triples(graph, target, max_cond):
    nodes = [n for n in graph.nodes if n != target]
    triples = []
    for x in nodes:
        others = [n for n in nodes if n != x]
        for size in range(0, max_cond + 1):
            for z in itertools.combinations(others, size):
                independent = nx.is_d_separator(graph, {x}, {target}, set(z))
                triples.append((x, list(z), bool(independent)))
    return triples


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    # pos = dependent (higher statistic expected), neg = independent.
    auc = (ranks[~labels].sum() - len(neg) * (len(neg) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def audit(args):
    rows = []
    rng = np.random.default_rng(args.seed)
    for graph_seed in args.graph_seeds:
        for noise_seed in args.noise_seeds:
            frame, weights = load_data(
                n_samples=args.n_samples,
                n_nodes=args.n_nodes,
                expected_edges=args.expected_edges,
                graph_type=args.graph_type,
                sem_type=args.sem_type,
                noise_scale=args.noise_scale,
                seed=graph_seed,
                graph_seed=graph_seed,
                noise_seed=noise_seed,
            )
            data = frame.values.astype(float)
            graph = _graph_from_weights(weights)
            target = data.shape[1] - 1
            triples = _candidate_triples(graph, target, args.max_cond)
            if args.max_triples and len(triples) > args.max_triples:
                idx = rng.choice(len(triples), size=args.max_triples, replace=False)
                triples = [triples[i] for i in idx]
            for x, z, independent in triples:
                xv = data[:, x]
                yv = data[:, target]
                zv = data[:, z] if z else None
                rows.append(
                    {
                        "graph_seed": graph_seed,
                        "noise_seed": noise_seed,
                        "x": x,
                        "z": ";".join(map(str, z)),
                        "true_independent": independent,
                        "partial_correlation": _partial_corr(xv, yv, zv),
                        "hsic": _hsic(xv, yv),
                        "residual_hsic": _residual_hsic(xv, yv, zv),
                        "soft_cmi": _soft_cmi(xv, yv, zv),
                    }
                )
    return rows


def summarize(rows, target_fpr):
    import pandas as pd

    frame = pd.DataFrame(rows)
    labels = frame["true_independent"].to_numpy(dtype=bool)
    summary = []
    for name in ("partial_correlation", "hsic", "residual_hsic", "soft_cmi"):
        scores = frame[name].to_numpy(dtype=float)
        auc = _auc(scores, labels)
        neg = scores[labels]
        threshold = float(np.quantile(neg, 1.0 - target_fpr)) if len(neg) else float("nan")
        called_dependent = scores > threshold
        fp = int((labels & called_dependent).sum())
        fn = int((~labels & ~called_dependent).sum())
        summary.append(
            {
                "statistic": name,
                "auc": auc,
                "target_fpr": target_fpr,
                "threshold": threshold,
                "n_independent": int(labels.sum()),
                "n_dependent": int((~labels).sum()),
                "false_positive_rate": fp / max(int(labels.sum()), 1),
                "false_negative_rate": fn / max(int((~labels).sum()), 1),
            }
        )
    return frame, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-type", default="ER", choices=("ER", "SF"))
    parser.add_argument("--sem-type", default="nonlinear")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--expected-edges", type=int, default=15)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--max-cond", type=int, default=2)
    parser.add_argument("--max-triples", type=int, default=250)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = audit(args)
    frame, summary = summarize(rows, args.target_fpr)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".per_triple.csv"), index=False)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Per-triple table: {output.with_suffix('.per_triple.csv')}")
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
