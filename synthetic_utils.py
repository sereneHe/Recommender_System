"""Deterministic ER/SF DAG and SEM data generation for experiments.

Two families live here:

* the historical **linear** SEM (``sem_type`` in gauss/laplace/student_t, plus
  the legacy per-edge ``nonlinear`` stress test), kept byte-for-byte compatible;
* explicit **NN-favourable mechanisms** selected by ``synthetic_mechanism``
  (smooth additive, deep compositional, high-dimensional smooth interaction,
  periodic/multiscale, temporal smooth). These are NOT overloads of
  ``sem_type``: they change the structural equations and expose the true
  conditional mean ``oracle_fn`` so a constraint audit uses the real generator
  rather than a linear ``X @ W`` surrogate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_graph_request(n_nodes: int, expected_edges: int) -> tuple[int, int]:
    n_nodes = int(n_nodes)
    expected_edges = int(expected_edges)
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")
    max_edges = n_nodes * (n_nodes - 1) // 2
    if not 1 <= expected_edges <= max_edges:
        raise ValueError(
            f"expected_edges must be in [1, {max_edges}] for {n_nodes} nodes."
        )
    return n_nodes, expected_edges


def _simulate_er_dag(
    n_nodes: int,
    expected_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an acyclic Erdos-Renyi graph with exactly ``expected_edges``."""
    candidates = np.array(
        [(source, target) for source in range(n_nodes) for target in range(source + 1, n_nodes)],
        dtype=int,
    )
    selected = rng.choice(len(candidates), size=expected_edges, replace=False)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[tuple(candidates[selected].T)] = 1.0

    # The last node is the prediction target in the supplied Hydra configs.
    # Ensure the generated benchmark actually contains a causal predictor for it.
    if not adjacency[:, -1].any():
        removable = candidates[selected]
        removable = removable[removable[:, 1] != n_nodes - 1]
        source = int(rng.integers(0, n_nodes - 1))
        old_source, old_target = removable[int(rng.integers(len(removable)))]
        adjacency[old_source, old_target] = 0.0
        adjacency[source, -1] = 1.0
    return adjacency


def _simulate_sf_dag(
    n_nodes: int,
    expected_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an acyclic preferential-attachment (scale-free) graph."""
    parents_per_node = max(1, min(n_nodes - 1, int(round(expected_edges / n_nodes))))
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    degree = np.zeros(n_nodes, dtype=float)

    for target in range(1, n_nodes):
        n_parents = min(parents_per_node, target)
        probabilities = degree[:target] + 1.0
        probabilities /= probabilities.sum()
        parents = rng.choice(target, size=n_parents, replace=False, p=probabilities)
        adjacency[parents, target] = 1.0
        degree[parents] += 1.0
        degree[target] += float(n_parents)
    return adjacency


def simulate_dag(
    n_nodes: int,
    expected_edges: int,
    graph_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a topologically ordered ER or SF adjacency matrix."""
    n_nodes, expected_edges = _validate_graph_request(n_nodes, expected_edges)
    graph_type = str(graph_type).strip().upper()
    if graph_type == "ER":
        return _simulate_er_dag(n_nodes, expected_edges, rng)
    if graph_type == "SF":
        return _simulate_sf_dag(n_nodes, expected_edges, rng)
    raise ValueError(f"graph_type must be 'ER' or 'SF', got {graph_type!r}.")


def simulate_parameters(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign NOTEARS-style signed coefficients to the edges of a DAG."""
    weights = np.zeros_like(adjacency, dtype=float)
    edge_mask = adjacency != 0
    magnitudes = rng.uniform(0.5, 2.0, size=int(edge_mask.sum()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=int(edge_mask.sum()))
    weights[edge_mask] = magnitudes * signs
    return weights


def simulate_linear_sem(
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    sem_type: str = "gauss",
    noise_scale: float = 1.0,
    noise_df: float = 5.0,
) -> np.ndarray:
    """Sample a SEM in the matrix's topological column order.

    ``sem_type`` in ``{gauss, laplace, student_t}`` keeps the structural
    equations linear with different innovation distributions.  ``nonlinear``
    keeps the same topology but applies a random bounded nonlinear link to each
    edge, so the conditional mean is no longer linear (see below).
    """
    n_samples = int(n_samples)
    noise_scale = float(noise_scale)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive.")
    noise_kind = str(sem_type).strip().lower()
    if noise_kind in {"gaussian"}:
        noise_kind = "gauss"
    if noise_kind not in {"gauss", "laplace", "student_t", "nonlinear"}:
        raise ValueError(
            "sem_type must be one of {'gauss', 'laplace', 'student_t', "
            "'nonlinear'}. Count and zero-inflated SEMs require a separate "
            "generator and are intentionally not approximated here."
        )
    noise_df = float(noise_df)
    if noise_kind == "student_t" and noise_df <= 2.0:
        raise ValueError("student_t noise_df must be greater than 2 so variance is finite.")

    n_nodes = weights.shape[0]
    samples = np.zeros((n_samples, n_nodes), dtype=float)
    if noise_kind == "nonlinear":
        # Nonlinear stress test: additive Gaussian noise, but each edge applies
        # a random bounded link, so d-separation still holds in the graph while
        # a linear partial-correlation statistic is misspecified.  The returned
        # ``weights`` describe the graph coefficients only; they do not give
        # E[Y|X], so ``constraint_audit_oracle=synthetic_linear_sem`` and the
        # raw-SEM W moment are invalid for these runs.
        for target in range(n_nodes):
            parents = np.flatnonzero(weights[:, target])
            contribution = np.zeros(n_samples, dtype=float)
            for parent in parents:
                parent_values = samples[:, parent]
                slope = float(rng.uniform(0.5, 1.5))
                link = int(rng.integers(0, 3))
                if link == 0:
                    linked = np.tanh(slope * parent_values)
                elif link == 1:
                    linked = np.sin(slope * parent_values)
                else:
                    linked = parent_values * np.tanh(slope * parent_values)
                contribution += weights[parent, target] * linked
            noise = rng.normal(loc=0.0, scale=noise_scale, size=n_samples)
            samples[:, target] = contribution + noise
        return samples

    for target in range(n_nodes):
        parents = np.flatnonzero(weights[:, target])
        if noise_kind == "gauss":
            noise = rng.normal(loc=0.0, scale=noise_scale, size=n_samples)
        elif noise_kind == "laplace":
            # Var(Laplace(0, b))=2b²; match Gaussian noise_scale variance.
            noise = rng.laplace(loc=0.0, scale=noise_scale / np.sqrt(2.0), size=n_samples)
        else:
            # Scale a Student-t variate to variance noise_scale².
            scale = noise_scale * np.sqrt((noise_df - 2.0) / noise_df)
            noise = rng.standard_t(df=noise_df, size=n_samples) * scale
        samples[:, target] = samples[:, parents] @ weights[parents, target] + noise
    return samples


def load_data(
    *,
    n_samples: int,
    n_nodes: int,
    expected_edges: int,
    graph_type: str,
    sem_type: str = "gauss",
    noise_scale: float = 1.0,
    noise_df: float = 5.0,
    seed: int = 1,
    graph_seed: int | None = None,
    noise_seed: int | None = None,
    mechanism: str = "linear",
    target_node: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return synthetic observations and the weighted ground-truth DAG.

    ``graph_seed`` controls both topology and structural coefficients, while
    ``noise_seed`` controls only SEM innovations.  Leaving both unset keeps
    the historical single-generator ``seed`` behavior exactly.

    ``mechanism`` selects the structural equation family.  ``"linear"`` with
    both seeds unset reproduces the historical generator byte-for-byte; any
    NN-favourable mechanism routes through :func:`simulate_synthetic_problem`.
    """
    mechanism = str(mechanism or "linear").strip().lower()
    if mechanism == "linear" and graph_seed is None and noise_seed is None:
        rng = np.random.default_rng(int(seed))
        adjacency = simulate_dag(n_nodes, expected_edges, graph_type, rng)
        weights = simulate_parameters(adjacency, rng)
        samples = simulate_linear_sem(
            weights, n_samples, rng,
            sem_type=sem_type, noise_scale=noise_scale, noise_df=noise_df,
        )
        columns = [f"X{index}" for index in range(int(n_nodes))]
        return pd.DataFrame(samples, columns=columns), weights

    samples_df, weights, _oracle, _meta = simulate_synthetic_problem(
        graph_type=graph_type,
        n_samples=n_samples,
        n_nodes=n_nodes,
        expected_edges=expected_edges,
        mechanism=mechanism,
        graph_seed=int(seed if graph_seed is None else graph_seed),
        noise_seed=int(seed if noise_seed is None else noise_seed),
        sem_type=sem_type,
        noise_scale=noise_scale,
        noise_df=noise_df,
        target_node=target_node,
    )
    return samples_df, weights


# ===========================================================================
# NN-favourable synthetic mechanisms (A7)
# ===========================================================================
# These change the STRUCTURAL EQUATIONS, not the noise.  They are selected by
# ``problem.synthetic_mechanism`` and deliberately kept separate from
# ``sem_type`` so a run can never be mislabelled as ordinary ER.
MECHANISMS = (
    "linear",
    "smooth_additive",
    "compositional",
    "highdim_smooth",
    "periodic",
    "temporal_smooth",
)

# Scope label each mechanism must be recorded under (see evidence _common.sh).
MECHANISM_SCOPE = {
    "linear": "synthetic/ER",
    "smooth_additive": "synthetic/SmoothER",
    "compositional": "synthetic/CompositionalER",
    "highdim_smooth": "synthetic/HighDim",
    "periodic": "synthetic/Periodic",
    "temporal_smooth": "synthetic/Temporal",
}

_SMOOTH_LINKS = ("tanh", "sin", "softplus")


def _link(name: str, z: np.ndarray) -> np.ndarray:
    name = str(name).lower()
    if name == "tanh":
        return np.tanh(z)
    if name == "sin":
        return np.sin(z)
    if name == "softplus":
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    if name == "identity":
        return z
    raise ValueError(f"unknown smooth link {name!r}")


def _simulate_layered_dag(n_nodes: int, expected_edges: int, n_layers: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Acyclic layered graph: edges only go from earlier to later layers.

    Depth (not width) is then the thing a network can exploit, so a shallow
    tree ensemble has to approximate the composition with many splits.
    """
    n_nodes = int(n_nodes)
    n_layers = max(2, min(int(n_layers), n_nodes))
    layer = np.minimum((np.arange(n_nodes) * n_layers) // n_nodes, n_layers - 1)
    candidates = np.array(
        [(s, t) for s in range(n_nodes) for t in range(n_nodes)
         if layer[s] < layer[t]],
        dtype=int,
    )
    if len(candidates) == 0:
        return _simulate_er_dag(n_nodes, expected_edges, rng)
    n_edges = min(int(expected_edges), len(candidates))
    selected = rng.choice(len(candidates), size=n_edges, replace=False)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[tuple(candidates[selected].T)] = 1.0
    if not adjacency[:, -1].any():
        later = [i for i in range(n_nodes - 1) if layer[i] < layer[-1]]
        if later:
            adjacency[int(rng.choice(later)), -1] = 1.0
    return adjacency


def _structural_mean(samples: np.ndarray, weights: np.ndarray, target: int,
                     mechanism: str, links: tuple, rng: np.random.Generator,
                     interactions: dict, temporal_parents: np.ndarray | None,
                     temporal_rho: float) -> np.ndarray:
    """True conditional mean E[X_target | Pa(target)] for the mechanism."""
    parents = np.flatnonzero(weights[:, target])
    if len(parents) == 0:
        return np.zeros(samples.shape[0], dtype=float)
    pa = samples[:, parents]
    z = pa @ weights[parents, target]
    if mechanism == "linear":
        return z
    if mechanism in ("smooth_additive", "compositional"):
        return _link(links[target % len(links)], z)
    if mechanism == "highdim_smooth":
        out = np.tanh(z)
        inter = interactions.get(target)
        if inter:
            a, b = inter[0]
            out = out + 0.2 * np.sin(samples[:, a] * samples[:, b])
            if len(inter) > 1:
                c, d = inter[1]
                out = out + 0.1 * samples[:, c] * samples[:, d]
        return out
    if mechanism == "periodic":
        return np.sin(z) + 0.5 * np.cos(2.0 * z)
    if mechanism == "temporal_smooth":
        base = np.tanh(z)
        if temporal_parents is not None and temporal_parents.shape[1] == len(parents):
            z_lag = temporal_parents @ weights[parents, target]
            base = np.tanh(z + temporal_rho * z_lag)
        return base
    raise ValueError(f"unknown mechanism {mechanism!r}")


def simulate_synthetic_problem(
    *,
    graph_type: str = "ER",
    n_samples: int = 10000,
    n_nodes: int = 20,
    expected_edges: int = 30,
    mechanism: str = "linear",
    graph_seed: int = 1,
    noise_seed: int = 101,
    sem_type: str = "gauss",
    noise_scale: float = 1.0,
    noise_df: float = 5.0,
    target_node: int | None = None,
    smooth_links: tuple = _SMOOTH_LINKS,
    n_layers: int = 3,
    temporal_lag: int = 1,
    temporal_rho: float = 0.5,
) -> tuple[pd.DataFrame, np.ndarray, "callable", dict]:
    """Generate a synthetic problem and expose its TRUE conditional mean.

    Returns ``(samples_df, w_true, oracle_fn, metadata)`` where ``oracle_fn``
    maps an ``(n, p)`` feature matrix to ``E[target | parents]`` under the real
    generating mechanism (never a linear surrogate).
    """
    mechanism = str(mechanism or "linear").strip().lower()
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}")
    n_nodes = int(n_nodes)
    target = n_nodes - 1 if target_node is None else int(target_node)

    graph_rng = np.random.default_rng(int(graph_seed))
    if mechanism == "compositional":
        adjacency = _simulate_layered_dag(n_nodes, expected_edges, n_layers, graph_rng)
    else:
        adjacency = simulate_dag(n_nodes, expected_edges, graph_type, graph_rng)
    weights = simulate_parameters(adjacency, graph_rng)

    # Deterministic per-node interaction pairs for the interaction mechanisms.
    interactions: dict[int, list] = {}
    for j in range(n_nodes):
        parents = list(np.flatnonzero(weights[:, j]))
        if len(parents) >= 2 and mechanism in ("highdim_smooth", "periodic"):
            interactions[j] = [(parents[0], parents[1])]
            if len(parents) >= 4:
                interactions[j].append((parents[2], parents[3]))

    sample_rng = np.random.default_rng(int(noise_seed))
    samples = np.zeros((int(n_samples), n_nodes), dtype=float)
    for t in range(n_nodes):
        mean = _structural_mean(samples, weights, t, mechanism, smooth_links,
                                sample_rng, interactions, None, temporal_rho)
        samples[:, t] = mean + sample_rng.normal(0.0, noise_scale, size=n_samples)

    # Temporal variant: rerun sequentially so node j at time t also sees its
    # parents at t-1 (a stationary, smooth dynamic SEM).
    temporal_parents_by_target: dict[int, np.ndarray] = {}
    if mechanism == "temporal_smooth":
        sample_rng = np.random.default_rng(int(noise_seed))
        series = np.zeros((int(n_samples), n_nodes), dtype=float)
        for t in range(1, int(n_samples)):
            for j in range(n_nodes):
                parents = np.flatnonzero(weights[:, j])
                cur = float(series[t, parents] @ weights[parents, j]) if len(parents) else 0.0
                lag = float(series[t - 1, parents] @ weights[parents, j]) if len(parents) else 0.0
                series[t, j] = np.tanh(cur + temporal_rho * lag) + sample_rng.normal(0.0, noise_scale)
        samples = series
        for j in range(n_nodes):
            parents = np.flatnonzero(weights[:, j])
            if len(parents):
                lag_block = np.zeros((int(n_samples), len(parents)), dtype=float)
                lag_block[1:] = samples[:-1, parents]
                temporal_parents_by_target[j] = lag_block

    columns = [f"X{index}" for index in range(n_nodes)]
    samples_df = pd.DataFrame(samples, columns=columns)

    def oracle_fn(features) -> np.ndarray:
        data = np.asarray(features, dtype=float)
        return _structural_mean(data, weights, target, mechanism, smooth_links,
                                np.random.default_rng(0), interactions,
                                temporal_parents_by_target.get(target),
                                temporal_rho)

    metadata = {
        "schema_version": 1,
        "mechanism": mechanism,
        "scope": MECHANISM_SCOPE[mechanism],
        "graph_type": str(graph_type).upper(),
        "n_samples": int(n_samples),
        "n_nodes": n_nodes,
        "expected_edges": int(expected_edges),
        "graph_seed": int(graph_seed),
        "noise_seed": int(noise_seed),
        "target_node": f"X{target}",
        "noise_scale": float(noise_scale),
        "smooth_links": list(smooth_links),
        "n_layers": int(n_layers) if mechanism == "compositional" else None,
        "temporal_lag": int(temporal_lag) if mechanism == "temporal_smooth" else None,
        "temporal_rho": float(temporal_rho) if mechanism == "temporal_smooth" else None,
        "oracle_type": "structural_conditional_mean",
        "true_parents_of_target": [f"X{i}" for i in np.flatnonzero(weights[:, target])],
    }
    return samples_df, weights, oracle_fn, metadata


def temporal_parent_blocks(weights, samples, target: int):
    """Return the ``(P_t, P_{t-1})`` blocks of the target's true parents.

    ``weights`` is the true structural matrix and ``samples`` the generated
    series (rows = time, columns = nodes).  Row 0 has no predecessor, so its
    lag block is zero-filled exactly as in the generator.  Both blocks are
    column-aligned with ``np.flatnonzero(weights[:, target])``.
    """
    weights = np.asarray(weights, dtype=float)
    samples = np.asarray(samples, dtype=float)
    parents = np.flatnonzero(weights[:, target])
    if len(parents) == 0:
        return (np.zeros((samples.shape[0], 0), dtype=float),
                np.zeros((samples.shape[0], 0), dtype=float))
    current = samples[:, parents]
    lagged = np.zeros_like(current)
    lagged[1:] = samples[:-1, parents]
    return current, lagged


def structural_conditional_mean(weights, features, mechanism: str, target: int,
                                temporal_parents=None, temporal_rho: float = 0.5,
                                smooth_links: tuple = _SMOOTH_LINKS,
                                require_lag: bool = False) -> np.ndarray:
    """Public, deterministic E[target | parents] for the given mechanism.

    Unlike :func:`simulate_synthetic_problem`'s closure this takes the weights
    directly, so any consumer that holds the true ``W`` (e.g. an oracle audit in
    the estimator) can compute the real conditional mean without re-running the
    generator or knowing the seed.  Interaction pairs and per-node links are
    derived deterministically from ``weights`` exactly as in generation.

    ``require_lag`` guards the dynamic mechanism: with
    ``mechanism="temporal_smooth"`` a missing ``temporal_parents`` block means
    the caller only holds ``P_t``, so the dynamic conditional mean is not
    identifiable.  Refusing is the point -- silently returning ``tanh(z)``
    would report a static oracle for a dynamic DGP.
    """
    weights = np.asarray(weights, dtype=float)
    data = np.asarray(features, dtype=float)
    if (require_lag and str(mechanism) == "temporal_smooth"
            and temporal_parents is None):
        raise ValueError(
            "refusing to evaluate a temporal_smooth oracle without the lag "
            "block P_{t-1}: a static feature matrix does not identify the "
            "dynamic conditional mean."
        )
    n_nodes = weights.shape[0]
    interactions: dict[int, list] = {}
    for j in range(n_nodes):
        parents = list(np.flatnonzero(weights[:, j]))
        if len(parents) >= 2 and mechanism in ("highdim_smooth", "periodic"):
            interactions[j] = [(parents[0], parents[1])]
            if len(parents) >= 4:
                interactions[j].append((parents[2], parents[3]))
    return _structural_mean(data, weights, int(target), str(mechanism),
                            tuple(smooth_links), np.random.default_rng(0),
                            interactions, temporal_parents, float(temporal_rho))


def add_lag_features(samples_df, lag: int, columns=None, suffix: str = "_lag"):
    """Append ``k`` lagged copies of the predictor columns to a synthetic frame.

    ``add_lag_features(df, 1)`` turns ``X_t`` into the ``[X_t, X_{t-1}]`` input
    tier.  The first ``lag`` rows have no predecessor, so their lag columns are
    left **NaN** (``Series.shift``), not zero: an undefined predictor is marked
    missing and the caller must drop those rows through a shared row mask.  (The
    generator's internal ``temporal_parent_blocks`` zero-fills instead, because
    it must reproduce its own recurrence; do not conflate the two.)  ``columns``
    defaults to every column except the last, matching the synthetic
    ``[features..., target]`` layout.
    """
    lag = int(lag)
    if lag < 0:
        raise ValueError("lag must be non-negative.")
    frame = samples_df.copy()
    if lag == 0:
        return frame
    if columns is None:
        columns = list(frame.columns[:-1])
    lagged = {}
    for column in columns:
        for k in range(1, lag + 1):
            lagged[f"{column}{suffix}{k}"] = frame[column].shift(k)
    for name, series in lagged.items():
        frame[name] = series
    return frame


def save_synthetic_artifacts(output_dir, samples_df, w_true, oracle_fn, metadata) -> dict:
    """Persist the ground truth a downstream audit must consume.

    Writes ``W_true.csv``, ``generator_metadata.yaml``, ``oracle_prediction.csv``
    (the true conditional mean for every row) and returns the paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    w_path = out / "W_true.csv"
    np.savetxt(w_path, np.asarray(w_true), delimiter=",")
    meta_path = out / "generator_metadata.yaml"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    oracle_path = out / "oracle_prediction.csv"
    oracle_vals = np.asarray(oracle_fn(samples_df.to_numpy()), dtype=float)
    pd.DataFrame({"oracle_mean": oracle_vals}).to_csv(oracle_path, index=False)
    return {"W_true": str(w_path), "generator_metadata": str(meta_path),
            "oracle_prediction": str(oracle_path)}

