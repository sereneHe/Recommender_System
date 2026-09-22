"""Deterministic ER/SF DAG and linear-SEM data generation for experiments."""

from __future__ import annotations

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
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return synthetic observations and the weighted ground-truth DAG.

    ``graph_seed`` controls both topology and structural coefficients, while
    ``noise_seed`` controls only SEM innovations.  Leaving both unset keeps
    the historical single-generator ``seed`` behavior exactly.
    """
    if graph_seed is None and noise_seed is None:
        rng = np.random.default_rng(int(seed))
        adjacency = simulate_dag(n_nodes, expected_edges, graph_type, rng)
        weights = simulate_parameters(adjacency, rng)
        sample_rng = rng
    else:
        graph_rng = np.random.default_rng(int(seed if graph_seed is None else graph_seed))
        adjacency = simulate_dag(n_nodes, expected_edges, graph_type, graph_rng)
        weights = simulate_parameters(adjacency, graph_rng)
        sample_rng = np.random.default_rng(int(seed if noise_seed is None else noise_seed))
    samples = simulate_linear_sem(
        weights,
        n_samples,
        sample_rng,
        sem_type=sem_type,
        noise_scale=noise_scale,
        noise_df=noise_df,
    )
    columns = [f"X{index}" for index in range(int(n_nodes))]
    return pd.DataFrame(samples, columns=columns), weights
