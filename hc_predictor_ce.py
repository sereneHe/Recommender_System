import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off", ""}


def _environment_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES)}, "
        f"got {value!r}"
    )


def _environment_int(name, default, minimum=None):
    value = int(os.environ.get(name, default))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _environment_float(name, default, minimum=None, maximum=None):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def birth_death_constraint_sampling_enabled():
    """Return whether posterior-stable CI constraints are enabled.

    The feature is deliberately opt-in so historical HC-CE runs keep using the
    single thresholded MILP DAG.  Set ``HC_CE_BD_MCMC=1`` to enable it.
    """

    return _environment_flag("HC_CE_BD_MCMC", False)


def _as_numpy_matrix(value, label):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _is_binary_dag(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.int8)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return False
    if np.any(np.diag(adjacency)):
        return False

    indegree = adjacency.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return visited == adjacency.shape[0]


def _local_gaussian_bic_score(data, node, parents, edge_penalty, cache):
    key = (int(node), tuple(int(parent) for parent in parents))
    if key in cache:
        return cache[key]

    target = data[:, node]
    if parents:
        design = np.column_stack(
            [np.ones(data.shape[0], dtype=float), data[:, list(parents)]]
        )
    else:
        design = np.ones((data.shape[0], 1), dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    variance = max(float(np.mean(residual * residual)), 1.0e-12)
    parent_count = len(parents)
    score = (
        -0.5 * data.shape[0] * math.log(variance)
        - 0.5 * parent_count * math.log(max(data.shape[0], 2))
        - edge_penalty * parent_count
    )
    cache[key] = float(score)
    return cache[key]


def _dag_log_score(data, adjacency, edge_penalty, cache):
    return sum(
        _local_gaussian_bic_score(
            data,
            node,
            tuple(np.flatnonzero(adjacency[:, node])),
            edge_penalty,
            cache,
        )
        for node in range(adjacency.shape[0])
    )


def _dag_neighbor_moves(adjacency, allow_reverse):
    dimension = adjacency.shape[0]
    for source in range(dimension):
        for target in range(dimension):
            if source == target:
                continue
            if adjacency[source, target]:
                proposal = adjacency.copy()
                proposal[source, target] = 0
                yield ("death", source, target), proposal

                if allow_reverse and not adjacency[target, source]:
                    proposal = adjacency.copy()
                    proposal[source, target] = 0
                    proposal[target, source] = 1
                    if _is_binary_dag(proposal):
                        yield ("reverse", source, target), proposal
            elif not adjacency[target, source]:
                proposal = adjacency.copy()
                proposal[source, target] = 1
                if _is_binary_dag(proposal):
                    yield ("birth", source, target), proposal


def birth_death_dag_samples(
    W,
    graph_data,
    *,
    initial_edge_threshold=0.1,
    steps=1000,
    burn_in=200,
    edge_penalty=1.0,
    temperature=1.0,
    allow_reverse=True,
    seed=20260916,
):
    """Sample DAGs using continuous-time add/delete/reverse graph moves.

    The target is an approximate linear-Gaussian graph posterior based on a
    decomposable BIC score plus an explicit per-edge sparsity penalty.  Move
    rates are ``min(1, exp(delta_log_score / temperature))``; paired reverse
    rates therefore satisfy detailed balance for this approximate target.

    Returned state weights are continuous-time holding times, not visit counts.
    The sampler intentionally changes only the graph used to derive CI/CE
    constraints; the original weighted ``W`` remains available to W_constraint.
    """

    weights = _as_numpy_matrix(W, "W")
    data = _as_numpy_matrix(graph_data, "graph_data")
    if weights.shape[0] != weights.shape[1]:
        raise ValueError("W must be square")
    if data.shape[1] != weights.shape[0]:
        raise ValueError(
            "graph_data columns must match W: "
            f"data={data.shape[1]}, W={weights.shape[0]}"
        )
    if data.shape[0] < 2:
        raise ValueError("birth-death DAG sampling requires at least two rows")
    if steps < 1:
        raise ValueError("steps must be positive")
    if burn_in < 0 or burn_in >= steps:
        raise ValueError("burn_in must be non-negative and smaller than steps")
    if initial_edge_threshold < 0 or edge_penalty < 0 or temperature <= 0:
        raise ValueError(
            "initial_edge_threshold and edge_penalty must be non-negative, "
            "and temperature must be positive"
        )

    adjacency = (np.abs(weights) > float(initial_edge_threshold)).astype(np.int8)
    np.fill_diagonal(adjacency, 0)
    if not _is_binary_dag(adjacency):
        raise ValueError(
            "the thresholded initial W is not a DAG; birth-death sampling "
            "requires an acyclic initial state"
        )

    rng = np.random.default_rng(int(seed))
    local_score_cache = {}
    current_score = _dag_log_score(data, adjacency, edge_penalty, local_score_cache)
    state_times = {}
    edge_time = np.zeros_like(adjacency, dtype=float)
    total_holding_time = 0.0
    move_counts = {"birth": 0, "death": 0, "reverse": 0}

    for iteration in range(int(steps)):
        moves = []
        proposals = []
        proposal_scores = []
        rates = []
        for move, proposal in _dag_neighbor_moves(adjacency, bool(allow_reverse)):
            proposal_score = _dag_log_score(
                data, proposal, edge_penalty, local_score_cache
            )
            log_rate = (proposal_score - current_score) / temperature
            rate = 1.0 if log_rate >= 0.0 else math.exp(max(log_rate, -700.0))
            moves.append(move)
            proposals.append(proposal)
            proposal_scores.append(proposal_score)
            rates.append(rate)

        total_rate = float(sum(rates))
        if not math.isfinite(total_rate) or total_rate <= 0.0:
            raise RuntimeError("birth-death sampler has no finite positive move rate")
        holding_time = float(rng.exponential(1.0 / total_rate))

        if iteration >= burn_in:
            state_key = adjacency.tobytes()
            if state_key not in state_times:
                state_times[state_key] = [adjacency.copy(), 0.0, 0]
            state_times[state_key][1] += holding_time
            state_times[state_key][2] += 1
            edge_time += holding_time * adjacency
            total_holding_time += holding_time

        probabilities = np.asarray(rates, dtype=float) / total_rate
        selected = int(rng.choice(len(moves), p=probabilities))
        move_counts[moves[selected][0]] += 1
        adjacency = proposals[selected]
        current_score = proposal_scores[selected]

    if total_holding_time <= 0.0:
        raise RuntimeError("birth-death sampler recorded no post-burn-in holding time")

    states = [
        (record[0], float(record[1]))
        for record in state_times.values()
        if record[1] > 0.0
    ]
    diagnostics = {
        "enabled": True,
        "steps": int(steps),
        "burn_in": int(burn_in),
        "seed": int(seed),
        "initial_edge_threshold": float(initial_edge_threshold),
        "edge_penalty": float(edge_penalty),
        "temperature": float(temperature),
        "allow_reverse": bool(allow_reverse),
        "unique_post_burn_in_graphs": len(states),
        "total_holding_time": total_holding_time,
        "move_counts": move_counts,
        "posterior_edge_probability": (edge_time / total_holding_time).tolist(),
    }
    return states, diagnostics


def _constraint_support_key(spec):
    if "x_index" not in spec or "y_index" not in spec:
        return None
    pair = tuple(sorted((int(spec["x_index"]), int(spec["y_index"]))))
    conditioning = tuple(sorted(int(index) for index in spec.get("z_indices", [])))
    return pair + (conditioning,)


def posterior_stable_constraints_from_dags(
    states,
    build_constraints,
    *,
    independence_support=0.8,
    dependence_support=0.8,
    max_conflict_support=0.2,
):
    """Aggregate graph-derived constraints using holding-time support."""

    for label, value in (
        ("independence_support", independence_support),
        ("dependence_support", dependence_support),
        ("max_conflict_support", max_conflict_support),
    ):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{label} must lie in [0, 1]")

    total_time = float(sum(float(weight) for _, weight in states))
    if total_time <= 0.0:
        raise ValueError("states must contain positive holding time")

    support = {}
    representatives = {}
    for adjacency, holding_time in states:
        constraints = build_constraints(adjacency)
        present = set()
        for raw_spec in constraints:
            spec = dict(raw_spec)
            relation = str(spec.get("type", "independent"))
            if relation not in {"independent", "dependent"}:
                continue
            base_key = _constraint_support_key(spec)
            if base_key is None:
                continue
            relation_key = (base_key, relation)
            present.add(relation_key)
            representatives.setdefault(relation_key, spec)
        for relation_key in present:
            support[relation_key] = support.get(relation_key, 0.0) + float(
                holding_time
            )

    base_keys = sorted({key[0] for key in support})
    retained = []
    support_records = []
    for base_key in base_keys:
        independent = support.get((base_key, "independent"), 0.0) / total_time
        dependent = support.get((base_key, "dependent"), 0.0) / total_time
        support_records.append(
            {
                "x_index": base_key[0],
                "y_index": base_key[1],
                "z_indices": list(base_key[2]),
                "independence_support": independent,
                "dependence_support": dependent,
            }
        )

        relation = None
        relation_support = 0.0
        opposing_support = 0.0
        if independent >= independence_support and dependent <= max_conflict_support:
            relation = "independent"
            relation_support = independent
            opposing_support = dependent
        elif dependent >= dependence_support and independent <= max_conflict_support:
            relation = "dependent"
            relation_support = dependent
            opposing_support = independent
        if relation is None:
            continue

        spec = dict(representatives[(base_key, relation)])
        spec["base_source"] = spec.get("source", "graph")
        spec["source"] = "birth_death_posterior"
        spec["posterior_support"] = float(relation_support)
        spec["opposing_support"] = float(opposing_support)
        retained.append(spec)

    diagnostics = {
        "independence_support_threshold": float(independence_support),
        "dependence_support_threshold": float(dependence_support),
        "max_conflict_support": float(max_conflict_support),
        "candidate_constraint_keys": len(base_keys),
        "retained_constraints": len(retained),
        "retained_independence_constraints": sum(
            spec.get("type") == "independent" for spec in retained
        ),
        "retained_dependence_constraints": sum(
            spec.get("type") == "dependent" for spec in retained
        ),
        "constraint_support": support_records,
    }
    return retained, diagnostics


def sample_posterior_stable_constraints(
    W,
    graph_data,
    build_constraints,
    *,
    initial_edge_threshold=0.1,
):
    """Run one or more opt-in birth-death chains and aggregate support.

    Each chain contributes equal total posterior mass, irrespective of its raw
    simulated holding time.  This makes support thresholds meaningful when
    ``HC_CE_BD_CHAINS`` is greater than one.  The function records per-chain
    diagnostics but intentionally does not claim an ESS or R-hat diagnostic;
    those require retaining edge traces rather than only aggregate states.
    """

    steps = _environment_int("HC_CE_BD_STEPS", 1000, minimum=1)
    burn_in = _environment_int("HC_CE_BD_BURN_IN", 200, minimum=0)
    chains = _environment_int("HC_CE_BD_CHAINS", 1, minimum=1)
    base_seed = _environment_int("HC_CE_BD_SEED", 20260916, minimum=0)
    seed_stride = _environment_int("HC_CE_BD_CHAIN_SEED_STRIDE", 1009, minimum=1)
    sampler_kwargs = {
        "initial_edge_threshold": _environment_float(
            "HC_CE_BD_INITIAL_THRESHOLD",
            initial_edge_threshold,
            minimum=0.0,
        ),
        "steps": steps,
        "burn_in": burn_in,
        "edge_penalty": _environment_float(
            "HC_CE_BD_EDGE_PENALTY", 1.0, minimum=0.0
        ),
        "temperature": _environment_float(
            "HC_CE_BD_TEMPERATURE", 1.0, minimum=1.0e-12
        ),
        "allow_reverse": _environment_flag("HC_CE_BD_ALLOW_REVERSE", True),
    }

    states = []
    chain_diagnostics = []
    posterior_edge_probabilities = []
    for chain_index in range(chains):
        chain_seed = base_seed + chain_index * seed_stride
        chain_states, diagnostics = birth_death_dag_samples(
            W,
            graph_data,
            seed=chain_seed,
            **sampler_kwargs,
        )
        chain_total_time = float(sum(weight for _, weight in chain_states))
        if chain_total_time <= 0.0:
            raise RuntimeError("birth-death chain recorded no positive post-burn-in holding time")
        # Equalize chains before deriving posterior support so one chain cannot
        # dominate only because its continuous-time clock advanced further.
        states.extend(
            (adjacency, float(weight) / chain_total_time / float(chains))
            for adjacency, weight in chain_states
        )
        diagnostics["chain_index"] = chain_index
        diagnostics["normalized_chain_mass"] = 1.0 / float(chains)
        chain_diagnostics.append(diagnostics)
        posterior_edge_probabilities.append(
            np.asarray(diagnostics["posterior_edge_probability"], dtype=float)
        )

    sampler_diagnostics = {
        "enabled": True,
        "chains": chains,
        "base_seed": base_seed,
        "chain_seed_stride": seed_stride,
        "steps": steps,
        "burn_in": burn_in,
        "initial_edge_threshold": sampler_kwargs["initial_edge_threshold"],
        "edge_penalty": sampler_kwargs["edge_penalty"],
        "temperature": sampler_kwargs["temperature"],
        "allow_reverse": sampler_kwargs["allow_reverse"],
        "unique_post_burn_in_graphs": int(
            sum(item["unique_post_burn_in_graphs"] for item in chain_diagnostics)
        ),
        "total_holding_time": float(
            sum(item["total_holding_time"] for item in chain_diagnostics)
        ),
        "posterior_edge_probability": np.mean(
            np.stack(posterior_edge_probabilities, axis=0), axis=0
        ).tolist(),
        "per_chain": chain_diagnostics,
        "convergence_diagnostic": "not_available_edge_traces_not_retained",
    }
    constraints, support_diagnostics = posterior_stable_constraints_from_dags(
        states,
        build_constraints,
        independence_support=_environment_float(
            "HC_CE_BD_INDEPENDENCE_SUPPORT", 0.8, minimum=0.0, maximum=1.0
        ),
        dependence_support=_environment_float(
            "HC_CE_BD_DEPENDENCE_SUPPORT", 0.8, minimum=0.0, maximum=1.0
        ),
        max_conflict_support=_environment_float(
            "HC_CE_BD_MAX_CONFLICT_SUPPORT", 0.2, minimum=0.0, maximum=1.0
        ),
    )
    return constraints, {**sampler_diagnostics, **support_diagnostics}

try:
    from humancompatible.train.dual_optim import PBM
    from humancompatible.train.dual_optim import pbm as humancompatible_pbm_module
    from humancompatible.train.fairness.utils import BalancedBatchSampler
except ModuleNotFoundError:  # pragma: no cover - keeps local smoke tests importable.
    BalancedBatchSampler = None
    humancompatible_pbm_module = None

    class PBM:
        def __init__(self, m, **kwargs):
            del kwargs
            self.duals = torch.zeros(m)
            self.penalties = torch.ones(m)

        def forward_update(self, loss, constraints):
            del constraints
            return loss


def _quad_log_barrier(t):
    out = torch.empty_like(t)
    mask = t >= -0.5
    out[mask] = t[mask] + 0.5 * torch.pow(t[mask], 2)
    out[~mask] = -0.25 * torch.log(-2 * t[~mask]) - 3.0 / 8.0
    return out


def _quad_log_barrier_derivative(t):
    out = torch.empty_like(t)
    mask = t >= -0.5
    out[mask] = 1.0 + t[mask]
    out[~mask] = -1.0 / (4.0 * t[~mask])
    return out


def _quad_reciprocal_barrier(t):
    out = torch.empty_like(t)
    mask = t >= -1.0 / 3.0
    out[mask] = t[mask] + 0.5 * torch.pow(t[mask], 2)
    out[~mask] = (32.0 / 27.0) / (1.0 - t[~mask]) - 7.0 / 6.0
    return out


def _quad_reciprocal_barrier_derivative(t):
    out = torch.empty_like(t)
    mask = t >= -1.0 / 3.0
    out[mask] = 1.0 + t[mask]
    out[~mask] = (32.0 / 27.0) / torch.square(1.0 - t[~mask])
    return out


_PENALTY_BARRIER_FUNCS = {
    "quadratic_logarithmic": (_quad_log_barrier, _quad_log_barrier_derivative),
    "quadratic_reciprocal": (_quad_reciprocal_barrier, _quad_reciprocal_barrier_derivative),
}


_HUMANCOMPATIBLE_PBM_PENALTY_UPDATES = {
    "const": "_update_penalties_const",
    "dimin": "_update_penalties_dimin",
    "diminish": "_update_penalties_dimin",
    "dimin_dual": "_update_penalties_dimin_dual",
}


def ensure_humancompatible_pbm_compatible(pbm):
    if humancompatible_pbm_module is None or getattr(pbm, "is_local_stochastic_pbm", False):
        return pbm
    if not hasattr(pbm, "param_groups"):
        return pbm

    meta = getattr(pbm, "_codiet_pbm_meta", {})
    penalty_update = str(meta.get("penalty_update", "const")).lower()
    update_fn_name = _HUMANCOMPATIBLE_PBM_PENALTY_UPDATES.get(penalty_update)
    if update_fn_name is None:
        raise ValueError(
            f"humancompatible_pbm does not support penalty_update={penalty_update!r}. "
            "Use one of {'const', 'dimin', 'dimin_dual'} or switch to ce_pbm_backend=stochastic_pbm."
        )
    update_fn = getattr(humancompatible_pbm_module, update_fn_name)

    for group in pbm.param_groups:
        group.setdefault("penalty_update", update_fn)
        group.setdefault("pbf", meta.get("pbf", "quadratic_logarithmic"))
        group.setdefault("mu", float(meta.get("mu", group.get("lr", 0.3))))
        group.setdefault("lr", float(meta.get("lr", group.get("mu", 0.3))))
        group.setdefault("momentum", float(meta.get("momentum", 0.0)))
        group.setdefault("dampening", float(meta.get("dampening", 0.0)))
        if "momentum_buffer" not in group:
            group["momentum_buffer"] = torch.zeros_like(group["params"][0])
    return pbm


class StochasticPBM:
    """Local SPBM-style PBM with explicit dual and penalty updates."""

    def __init__(
        self,
        m,
        mu=0.3,
        lr=0.95,
        penalty_update="const",
        pbf="quadratic_logarithmic",
        init_duals=0.01,
        init_penalties=1.0,
        dual_range=(0.01, 100.0),
        penalty_range=(0.001, 100.0),
        dual_ema_gamma=0.0,
        epoch_len=1,
        adapt_delta=1.5,
        device=None,
    ):
        if pbf not in _PENALTY_BARRIER_FUNCS:
            raise ValueError(
                f"Unknown penalty-barrier function: {pbf!r}. "
                f"Expected one of {sorted(_PENALTY_BARRIER_FUNCS)}."
            )
        self.mu = float(mu)
        self.lr = float(lr)
        self.penalty_update = str(penalty_update)
        self.pbf = str(pbf)
        self.dual_range = tuple(dual_range)
        self.penalty_range = tuple(penalty_range)
        self.dual_ema_gamma = float(dual_ema_gamma)
        self.epoch_len = max(1, int(epoch_len))
        self.adapt_delta = float(adapt_delta)
        self.iter = 0
        self.is_local_stochastic_pbm = True
        self.duals = torch.full((int(m),), float(init_duals), dtype=torch.float32, device=device)
        self.penalties = torch.full((int(m),), float(init_penalties), dtype=torch.float32, device=device)
        self.constraints_epoch = torch.zeros(int(m), dtype=torch.float32, device=device)

    def forward(self, loss, constraints):
        constraints = constraints.to(device=self.duals.device, dtype=self.duals.dtype)
        barrier, _ = _PENALTY_BARRIER_FUNCS[self.pbf]
        scaled_constraints = constraints / self.penalties
        return loss + torch.dot(self.duals * self.penalties, barrier(scaled_constraints))

    def update(self, constraints):
        constraints = constraints.to(device=self.duals.device, dtype=self.duals.dtype)
        _, barrier_derivative = _PENALTY_BARRIER_FUNCS[self.pbf]
        scaled_constraints = constraints / self.penalties
        raw_duals = self.duals * barrier_derivative(scaled_constraints.detach())
        raw_duals = torch.clamp(raw_duals, min=self.dual_range[0], max=self.dual_range[1])
        if 0.0 < self.dual_ema_gamma < 1.0:
            raw_duals = self.dual_ema_gamma * self.duals + (1.0 - self.dual_ema_gamma) * raw_duals
        self.duals.copy_(raw_duals)
        self.update_penalties(constraints)

    def forward_update(self, loss, constraints):
        with torch.no_grad():
            self.update(constraints.detach())
        return self.forward(loss, constraints)

    def update_penalties(self, constraints=None):
        penalty_update = self.penalty_update.lower()
        if penalty_update == "const":
            self.iter += 1
            return
        if penalty_update == "diminish":
            penalty_update = "dimin"
        if penalty_update == "adapt":
            penalty_update = "dimin_adapt"
        if penalty_update == "alm":
            updated = self.mu * self.duals
        elif penalty_update == "dimin":
            updated = self.penalties * self.lr
        elif penalty_update == "dimin_dual":
            updated = torch.minimum(self.penalties * self.lr, self.mu / torch.clamp(self.duals, min=1e-12))
        elif penalty_update == "dimin_adapt":
            if constraints is None:
                updated = self.penalties * self.lr
            else:
                _, barrier_derivative = _PENALTY_BARRIER_FUNCS[self.pbf]
                self.constraints_epoch.add_(constraints.to(self.penalties.device).detach())
                self.iter += 1
                if self.iter % self.epoch_len != 0:
                    return
                constraint_mean = self.constraints_epoch / float(self.epoch_len)
                growth = barrier_derivative(constraint_mean)
                growth = torch.nan_to_num(growth, nan=self.penalty_range[1])
                high_growth = growth > 1.0
                safe_growth = torch.clamp(growth, min=1e-4)
                updated = torch.empty_like(self.penalties)
                updated[high_growth] = (
                    0.1 * self.penalties[high_growth]
                    + 0.9
                    * self.penalties[high_growth]
                    / (self.adapt_delta * safe_growth[high_growth])
                )
                updated[~high_growth] = (
                    0.1 * self.penalties[~high_growth]
                    + 0.9 * self.penalties[~high_growth] / safe_growth[~high_growth]
                )
                self.constraints_epoch.zero_()
        else:
            raise ValueError(
                f"Unknown stochastic PBM penalty_update: {self.penalty_update!r}. "
                "Expected one of {'const', 'alm', 'dimin', 'diminish', 'adapt', 'dimin_dual', 'dimin_adapt'}."
            )
        self.penalties.copy_(
            torch.nan_to_num(
                torch.clamp(updated, min=self.penalty_range[0], max=self.penalty_range[1]),
                nan=self.penalty_range[1],
            )
        )
        if penalty_update != "dimin_adapt":
            self.iter += 1


def _residualize(target, cond, method="linear", *, irls_iterations=12, irls_eps=1e-3):
    method = str(method).strip().lower()
    if cond is None or cond.numel() == 0:
        return target - target.mean()
    if cond.ndim == 1:
        cond = cond.unsqueeze(1)
    cond_aug = torch.cat(
        [cond, torch.ones(cond.shape[0], 1, device=cond.device, dtype=cond.dtype)],
        dim=1,
    )
    beta = torch.linalg.pinv(cond_aug) @ target.unsqueeze(1)
    if method == "linear":
        fitted = (cond_aug @ beta).squeeze(1)
        return target - fitted
    if method != "quantile":
        raise ValueError(
            "ce_residualize_method must be 'linear' or 'quantile', "
            f"got {method!r}."
        )

    # Smoothed-L1 IRLS approximates median regression.  The IRLS weights are
    # detached at each step so the predictor gradient follows the weighted
    # least-squares residualization map, without differentiating through the
    # iterative optimizer itself.  This is experimental, not a general CI test.
    for _ in range(max(1, int(irls_iterations))):
        residual = target - (cond_aug @ beta).squeeze(1)
        weights = torch.rsqrt(residual.detach().square() + float(irls_eps) ** 2)
        sqrt_weights = torch.sqrt(weights)
        weighted_design = cond_aug * sqrt_weights.unsqueeze(1)
        weighted_target = target.unsqueeze(1) * sqrt_weights.unsqueeze(1)
        beta = torch.linalg.pinv(weighted_design) @ weighted_target
    fitted = (cond_aug @ beta).squeeze(1)
    return target - fitted


def conditional_covariance_mean(x_var, y_var, z_var):
    x_res = _residualize(x_var, z_var)
    y_res = _residualize(y_var, z_var)
    return torch.mean(x_res * y_res)


def conditional_partial_correlation(
    x_var,
    y_var,
    z_var,
    *,
    residualize_method="linear",
    residualize_iterations=12,
    residualize_eps=1e-3,
    eps=1e-8,
):
    """Partial correlation after configured residualization of X and Y on Z."""
    x_res = _residualize(
        x_var,
        z_var,
        method=residualize_method,
        irls_iterations=residualize_iterations,
        irls_eps=residualize_eps,
    )
    y_res = _residualize(
        y_var,
        z_var,
        method=residualize_method,
        irls_iterations=residualize_iterations,
        irls_eps=residualize_eps,
    )
    x_std = torch.sqrt(torch.mean(x_res.square())).clamp_min(float(eps))
    y_std = torch.sqrt(torch.mean(y_res.square())).clamp_min(float(eps))
    return torch.mean((x_res / x_std) * (y_res / y_std)).clamp(-1.0, 1.0)


def _ce_residuals_for_spec(X, y, spec, cfg=None):
    if y.ndim < 2:
        y = y.unsqueeze(1)
    XY = torch.cat([X, y], dim=1)
    x_var = _select_column(XY, spec["x_index"])
    y_var = _select_column(XY, spec["y_index"])
    z_var = _select_columns(XY, spec.get("z_indices", []))
    method = str(getattr(cfg, "ce_residualize_method", "linear")) if cfg is not None else "linear"
    iterations = int(getattr(cfg, "ce_residualize_iterations", 12)) if cfg is not None else 12
    residual_eps = float(getattr(cfg, "ce_residualize_eps", 1e-3)) if cfg is not None else 1e-3
    return (
        _residualize(x_var, z_var, method=method, irls_iterations=iterations, irls_eps=residual_eps),
        _residualize(y_var, z_var, method=method, irls_iterations=iterations, irls_eps=residual_eps),
    )


def _ce_window_setting(cfg, canonical, alias, default):
    """Read a CE window setting, accepting the launch-script alias.

    The CE-new launch scripts use ``ce_cross_window_n_windows`` and
    ``ce_cross_window_min_size`` while the implementation originally shipped
    ``ce_window_n_windows`` and ``ce_window_min_size``.  Accept both names so
    the scripts and the code agree instead of failing on an unknown Hydra key.
    """
    if cfg is not None:
        # Prefer the alias when it is explicitly set (default is null); fall
        # back to the canonical key otherwise.
        for name in (alias, canonical):
            value = getattr(cfg, name, None)
            if value is not None:
                return int(value)
    return int(default)


def _ce_window_n_windows(cfg, default=5):
    return _ce_window_setting(cfg, "ce_window_n_windows", "ce_cross_window_n_windows", default)


def _ce_window_min_size(cfg, default=100):
    return _ce_window_setting(cfg, "ce_window_min_size", "ce_cross_window_min_size", default)


def _resolve_ce_se_method(cfg, n):
    method = str(getattr(cfg, "ce_se_method", "auto") if cfg is not None else "hac").strip().lower()
    n_windows = _ce_window_n_windows(cfg)
    min_window_size = _ce_window_min_size(cfg)
    enough_windows = n >= n_windows * min_window_size
    if method == "auto":
        return "window" if enough_windows else "hac"
    if method == "window":
        # Small samples use HAC as a fallback; five tiny windows are not a
        # meaningful stability audit.
        return "window" if enough_windows else "hac"
    if method != "hac":
        raise ValueError("ce_se_method must be one of {'auto', 'window', 'hac'}; iid is not supported.")
    return method


def _partial_correlation_for_spec(X, y, spec, cfg=None):
    x_res, y_res = _ce_residuals_for_spec(X, y, spec, cfg=cfg)
    eps = float(getattr(cfg, "ce_statistic_eps", 1e-8)) if cfg is not None else 1e-8
    x_res = x_res - x_res.mean()
    y_res = y_res - y_res.mean()
    x_std = torch.sqrt(x_res.square().mean()).clamp_min(eps)
    y_std = torch.sqrt(y_res.square().mean()).clamp_min(eps)
    raw = torch.mean((x_res / x_std) * (y_res / y_std)).clamp(-1.0, 1.0)
    shrinkage = float(getattr(cfg, "ce_statistic_shrinkage", 0.0)) if cfg is not None else 0.0
    return (1.0 - shrinkage) * raw


def constraint_window_statistics(X, y, spec, cfg=None):
    """Summarize a constraint statistic over contiguous training-data windows.

    Window spread measures temporal/block instability; it is not a classical
    standard error. Sign-flip rate is descriptive only for independence specs.
    """
    n = int(X.shape[0])
    n_windows = _ce_window_n_windows(cfg)
    min_window_size = _ce_window_min_size(cfg)
    if n_windows < 2 or n < n_windows * min_window_size:
        return {
            "window_count": 0,
            "window_statistics": [],
            "window_mean": None,
            "window_std": None,
            "window_sign_flip_rate": None,
        }
    window_size = n // n_windows
    values = []
    for window_index in range(n_windows):
        start = window_index * window_size
        end = n if window_index == n_windows - 1 else (window_index + 1) * window_size
        values.append(
            float(_partial_correlation_for_spec(X[start:end], y[start:end], spec, cfg=cfg).detach().cpu())
        )
    epsilon = float(getattr(cfg, "ce_window_sign_epsilon", 0.01)) if cfg is not None else 0.01
    positive = sum(value > epsilon for value in values)
    negative = sum(value < -epsilon for value in values)
    nonzero = positive + negative
    sign_flip_rate = min(positive, negative) / float(nonzero) if nonzero else 0.0
    return {
        "window_count": len(values),
        "window_statistics": values,
        "window_mean": float(np.mean(values)),
        "window_std": float(np.std(values, ddof=0)),
        "window_sign_flip_rate": float(sign_flip_rate),
    }


def _partial_correlation_standard_error(X, y, spec, cfg=None):
    """Approximate SE for partial correlation; HAC is used for time series.

    The iid formula is only an approximation.  HAC uses a Newey-West estimate
    of the long-run variance of the partial-correlation influence sequence.
    """
    with torch.no_grad():
        x_res, y_res = _ce_residuals_for_spec(X, y, spec, cfg=cfg)
        n = int(x_res.numel())
        p = len(spec.get("z_indices", []) or [])
        dof = n - p - 2
        if n < 4 or dof <= 0:
            return X.new_tensor(1.0)
        eps = float(getattr(cfg, "ce_statistic_eps", 1e-8)) if cfg is not None else 1e-8
        x_centered = x_res - x_res.mean()
        y_centered = y_res - y_res.mean()
        x_std = torch.sqrt(torch.mean(x_centered.square())).clamp_min(eps)
        y_std = torch.sqrt(torch.mean(y_centered.square())).clamp_min(eps)
        u = x_centered / x_std
        v = y_centered / y_std
        r = torch.mean(u * v).clamp(-1.0, 1.0)
        method = _resolve_ce_se_method(cfg, n)
        shrinkage = float(getattr(cfg, "ce_statistic_shrinkage", 0.0)) if cfg is not None else 0.0
        if method == "window":
            window_summary = constraint_window_statistics(X, y, spec, cfg=cfg)
            if window_summary["window_count"] >= 2:
                # Empirical between-window spread (a stability scale), not an
                # iid standard error or a classical confidence interval.
                return X.new_tensor(window_summary["window_std"])

        influence = u * v - 0.5 * r * (u.square() + v.square())
        influence = influence - influence.mean()
        lag_setting = int(getattr(cfg, "ce_hac_max_lag", 0) or 0) if cfg is not None else 0
        max_lag = lag_setting if lag_setting > 0 else int(math.ceil(n ** (1.0 / 3.0)))
        max_lag = min(max_lag, n - 1)
        long_run_variance = torch.mean(influence.square())
        for lag in range(1, max_lag + 1):
            bartlett = 1.0 - lag / float(max_lag + 1)
            gamma = torch.mean(influence[lag:] * influence[:-lag])
            long_run_variance = long_run_variance + 2.0 * bartlett * gamma
        finite_sample_correction = float(n) / float(dof)
        se = torch.sqrt((long_run_variance.clamp_min(0.0) / float(n)) * finite_sample_correction)
        return (1.0 - shrinkage) * se


def independent_expectation_tolerances(
    X, y, ci_constraints, tolerance=0.0, cfg=None, *, se_X=None, se_y=None
):
    """Return one fixed or SE-based tolerance per independence constraint."""
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)
    base = float(tolerance)
    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation")) if cfg is not None else "conditional_expectation"
    mode = str(getattr(cfg, "ce_tolerance_mode", "fixed") if cfg is not None else "fixed").strip().lower()
    if mode not in {"fixed", "se", "standard_error"}:
        raise ValueError("ce_tolerance_mode must be 'fixed' or 'standard_error'.")
    if mode == "fixed" or _is_discrete_ci_kind(penalty_kind):
        return X.new_full((len(ci_constraints),), base)
    statistic_kind = str(getattr(cfg, "ce_statistic_kind", "partial_correlation") if cfg is not None else "partial_correlation").strip().lower()
    if statistic_kind != "partial_correlation":
        raise ValueError("SE-based tolerances currently require ce_statistic_kind='partial_correlation'.")
    multiplier = float(getattr(cfg, "ce_tolerance_sd_multiplier", 1.96)) if cfg is not None else 1.96
    if multiplier < 0:
        raise ValueError("ce_tolerance_sd_multiplier must be non-negative.")
    variance_X = X if se_X is None else se_X
    variance_y = y if se_y is None else se_y
    ses = [
        _partial_correlation_standard_error(variance_X, variance_y, spec, cfg=cfg)
        for spec in ci_constraints
    ]
    return base + multiplier * torch.stack(ses).to(device=X.device, dtype=X.dtype)


def _hard_quantile_bins(var, n_bins):
    var_flat = var.detach().reshape(-1).cpu()
    n = int(var_flat.numel())
    n_bins = max(1, min(int(n_bins), n))
    order = torch.argsort(var_flat)
    labels = torch.empty(n, dtype=torch.long)
    labels[order] = torch.arange(n, dtype=torch.long) * n_bins // n
    return labels.to(device=var.device)


def _hard_categorical_bins(var, n_bins):
    var_flat = var.detach().reshape(-1).cpu()
    values, labels = torch.unique(var_flat, sorted=True, return_inverse=True)
    if int(values.numel()) <= int(n_bins):
        return labels.to(device=var.device)
    return _hard_quantile_bins(var, n_bins)


def _soft_quantile_membership(var, n_bins, temperature=0.2):
    var_flat = var.reshape(-1)
    n = int(var_flat.numel())
    n_bins = max(1, min(int(n_bins), n))
    if n_bins == 1:
        return torch.ones(n, 1, device=var.device, dtype=var.dtype)

    with torch.no_grad():
        probs = torch.linspace(
            0.0,
            1.0,
            steps=n_bins,
            device=var.device,
            dtype=var.dtype,
        )
        centers = torch.quantile(var_flat.detach(), probs)
        scale = torch.std(var_flat.detach()).clamp_min(1e-6)
    distances = torch.square((var_flat[:, None] - centers[None, :]) / scale)
    return torch.softmax(-distances / max(float(temperature), 1e-6), dim=1)


def _joint_membership(memberships, max_states=256):
    if not memberships:
        return None
    joint = memberships[0]
    for membership in memberships[1:]:
        next_joint = joint[:, :, None] * membership[:, None, :]
        joint = next_joint.reshape(joint.shape[0], -1)
        if joint.shape[1] > int(max_states):
            # Keep the strongest soft conditioning states to avoid exploding
            # high-dimensional Z tables in mini-batches.
            keep = torch.topk(joint.mean(dim=0), k=int(max_states)).indices
            joint = joint.index_select(1, keep)
    return joint


def soft_discrete_conditional_mutual_information(
    x_var,
    y_var,
    z_var=None,
    n_bins=4,
    temperature=0.2,
    max_z_states=256,
    eps=1e-8,
):
    """Differentiable CMI after quantile binning via soft bin memberships."""
    x_membership = _soft_quantile_membership(x_var, n_bins, temperature)
    y_membership = _soft_quantile_membership(y_var, n_bins, temperature)
    if z_var is None or z_var.numel() == 0:
        z_membership = torch.ones(
            x_membership.shape[0],
            1,
            device=x_membership.device,
            dtype=x_membership.dtype,
        )
    else:
        if z_var.ndim == 1:
            z_var = z_var.unsqueeze(1)
        z_membership = _joint_membership(
            [
                _soft_quantile_membership(z_var[:, idx], n_bins, temperature)
                for idx in range(z_var.shape[1])
            ],
            max_states=max_z_states,
        )

    xyz = (
        z_membership[:, :, None, None]
        * x_membership[:, None, :, None]
        * y_membership[:, None, None, :]
    ).mean(dim=0)
    p_z = xyz.sum(dim=(1, 2)).clamp_min(eps)
    p_xz = xyz.sum(dim=2).clamp_min(eps)
    p_yz = xyz.sum(dim=1).clamp_min(eps)
    p_xyz = xyz.clamp_min(eps)
    ratio = p_xyz * p_z[:, None, None] / (p_xz[:, :, None] * p_yz[:, None, :])
    return torch.sum(p_xyz * torch.log(ratio.clamp_min(eps)))


def hard_discrete_conditional_mutual_information(
    x_var,
    y_var,
    z_var=None,
    n_bins=4,
    max_z_states=256,
    eps=1e-12,
):
    """Strict empirical CMI after hard quantile/value binning."""
    x_bins = _hard_categorical_bins(x_var, n_bins)
    y_bins = _hard_categorical_bins(y_var, n_bins)
    n = int(x_bins.numel())
    if z_var is None or z_var.numel() == 0:
        z_codes = torch.zeros(n, dtype=torch.long, device=x_bins.device)
    else:
        if z_var.ndim == 1:
            z_var = z_var.unsqueeze(1)
        z_codes = torch.zeros(n, dtype=torch.long, device=x_bins.device)
        multiplier = 1
        for idx in range(z_var.shape[1]):
            bins = _hard_categorical_bins(z_var[:, idx], n_bins)
            z_codes = z_codes + multiplier * bins
            multiplier *= max(1, min(int(n_bins), n))
        unique_z, z_codes = torch.unique(z_codes, sorted=True, return_inverse=True)
        if int(unique_z.numel()) > int(max_z_states):
            z_codes = z_codes % int(max_z_states)

    cmi = torch.zeros((), device=x_bins.device, dtype=torch.float32)
    for z_code in torch.unique(z_codes):
        mask = z_codes == z_code
        n_z = int(mask.sum().item())
        if n_z <= 0:
            continue
        x_z = x_bins[mask]
        y_z = y_bins[mask]
        p_z = float(n_z) / float(n)
        x_states = torch.unique(x_z)
        y_states = torch.unique(y_z)
        for x_state in x_states:
            px = (x_z == x_state).float().mean()
            if px <= 0:
                continue
            for y_state in y_states:
                py = (y_z == y_state).float().mean()
                pxy = ((x_z == x_state) & (y_z == y_state)).float().mean()
                if pxy <= 0 or py <= 0:
                    continue
                cmi = cmi + p_z * pxy * torch.log((pxy / (px * py)).clamp_min(eps))
    return cmi


def _is_discrete_ci_kind(penalty_kind):
    return penalty_kind in {
        "discrete_conditional_independence",
        "strict_discrete_ci",
        "discrete_ci",
        "quantile_discrete_ci",
    }


def is_expectation_constraint_mode(cfg):
    if not bool(getattr(cfg, "use_ci_penalty", False)):
        return False
    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation"))
    return penalty_kind in {"conditional_expectation", "expectation", "ce"} or _is_discrete_ci_kind(
        penalty_kind
    )


def split_expectation_constraints(ci_constraints):
    independent = []
    dependent = []
    for spec in ci_constraints:
        relation = spec.get("type", "independent")
        if relation == "independent":
            independent.append(spec)
        elif relation == "dependent":
            dependent.append(spec)
    return independent, dependent


def filter_unstable_signed_dependence_constraints(X, y, ci_constraints, cfg=None):
    """Drop directionally unstable dependent constraints using training windows.

    Sign flipping is not used to filter independence constraints: their null
    target is zero, so random sign changes around zero are expected and do not
    indicate a bad constraint.  This filter applies only to signed dependence
    constraints and uses only the current training fold.
    """
    diagnostics = []
    if not ci_constraints or cfg is None:
        return list(ci_constraints or []), diagnostics
    if not bool(getattr(cfg, "ce_window_filter_enabled", False)):
        return list(ci_constraints), diagnostics
    if _is_discrete_ci_kind(str(getattr(cfg, "ci_penalty_kind", ""))):
        return list(ci_constraints), diagnostics
    if str(getattr(cfg, "ce_statistic_kind", "partial_correlation")).strip().lower() != "partial_correlation":
        return list(ci_constraints), diagnostics
    if str(getattr(cfg, "ci_dependent_statistic", "signed")).strip().lower() != "signed":
        return list(ci_constraints), diagnostics

    n = int(X.shape[0])
    if _resolve_ce_se_method(cfg, n) != "window":
        return list(ci_constraints), diagnostics
    n_windows = _ce_window_n_windows(cfg)
    window_size = n // n_windows
    max_flip_rate = float(getattr(cfg, "ce_window_max_sign_flip_rate", 0.30))
    sign_epsilon = float(getattr(cfg, "ce_window_sign_epsilon", 0.01))
    if not 0.0 <= max_flip_rate <= 1.0:
        raise ValueError("ce_window_max_sign_flip_rate must be between 0 and 1.")

    kept = []
    for constraint_index, spec in enumerate(ci_constraints, start=1):
        updated = dict(spec)
        if str(spec.get("type", "independent")) != "dependent":
            kept.append(updated)
            continue
        values = []
        for window_index in range(n_windows):
            start = window_index * window_size
            end = n if window_index == n_windows - 1 else (window_index + 1) * window_size
            values.append(
                float(_partial_correlation_for_spec(X[start:end], y[start:end], spec, cfg=cfg).detach().cpu())
            )
        positive = sum(value > sign_epsilon for value in values)
        negative = sum(value < -sign_epsilon for value in values)
        nonzero = positive + negative
        flip_rate = min(positive, negative) / float(nonzero) if nonzero else 0.0
        drop = nonzero > 0 and flip_rate > max_flip_rate
        updated["window_sign_flip_rate"] = flip_rate
        updated["window_filter_applied"] = True
        updated["window_filter_dropped"] = drop
        diagnostics.append(
            {
                "constraint_index": constraint_index,
                "relation": "dependent",
                "x_index": int(spec["x_index"]),
                "y_index": int(spec["y_index"]),
                "z_indices": ";".join(str(index) for index in spec.get("z_indices", []) or []),
                "window_statistics": ";".join(f"{value:.10g}" for value in values),
                "window_sign_flip_rate": flip_rate,
                "max_sign_flip_rate": max_flip_rate,
                "dropped": bool(drop),
            }
        )
        if not drop:
            kept.append(updated)
    return kept, diagnostics


def _select_column(XY, index):
    return XY.select(1, int(index))


def _select_columns(XY, indices):
    if not indices:
        return None
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=XY.device)
    return XY.index_select(1, index_tensor)


def _constraint_statistic_values(X, y, ci_constraints, cfg=None):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    if y.ndim < 2:
        y = y.unsqueeze(1)
    XY = torch.cat([X, y], dim=1)
    values = []
    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation")) if cfg is not None else "conditional_expectation"
    n_bins = int(getattr(cfg, "ci_discrete_n_bins", getattr(cfg, "ce_sensitive_bins", 4))) if cfg is not None else 4
    temperature = float(getattr(cfg, "ci_discrete_soft_temperature", 0.2)) if cfg is not None else 0.2
    max_z_states = int(getattr(cfg, "ci_discrete_max_z_states", 256)) if cfg is not None else 256
    statistic_kind = str(
        getattr(cfg, "ce_statistic_kind", "covariance")
        if cfg is not None
        else "covariance"
    ).strip().lower()
    shrinkage = float(getattr(cfg, "ce_statistic_shrinkage", 0.0)) if cfg is not None else 0.0
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("ce_statistic_shrinkage must be between 0 and 1.")
    residualize_method = str(getattr(cfg, "ce_residualize_method", "linear")) if cfg is not None else "linear"
    residualize_iterations = int(getattr(cfg, "ce_residualize_iterations", 12)) if cfg is not None else 12
    residualize_eps = float(getattr(cfg, "ce_residualize_eps", 1e-3)) if cfg is not None else 1e-3
    statistic_eps = float(getattr(cfg, "ce_statistic_eps", 1e-8)) if cfg is not None else 1e-8

    for spec in ci_constraints:
        x_var = _select_column(XY, spec["x_index"])
        y_var = _select_column(XY, spec["y_index"])
        z_indices = spec.get("z_indices", [])
        z_var = _select_columns(XY, z_indices)
        if _is_discrete_ci_kind(penalty_kind):
            values.append(
                soft_discrete_conditional_mutual_information(
                    x_var,
                    y_var,
                    z_var,
                    n_bins=n_bins,
                    temperature=temperature,
                    max_z_states=max_z_states,
                )
            )
        else:
            if statistic_kind == "partial_correlation":
                value = conditional_partial_correlation(
                    x_var,
                    y_var,
                    z_var,
                    residualize_method=residualize_method,
                    residualize_iterations=residualize_iterations,
                    residualize_eps=residualize_eps,
                    eps=statistic_eps,
                )
                values.append((1.0 - shrinkage) * value)
            elif statistic_kind == "covariance":
                if residualize_method != "linear":
                    raise ValueError("ce_residualize_method='quantile' is only supported with partial_correlation.")
                values.append(conditional_covariance_mean(x_var, y_var, z_var))
            else:
                raise ValueError(
                    "ce_statistic_kind must be 'partial_correlation' or 'covariance', "
                    f"got {statistic_kind!r}."
                )

    return torch.stack(values)


def signed_expectation_equalities(X, y, ci_constraints, cfg=None):
    return _constraint_statistic_values(X, y, ci_constraints, cfg=cfg)


def independent_expectation_equalities(
    X, y, ci_constraints, tolerance=0.0, cfg=None, *, se_X=None, se_y=None
):
    """ALM residuals with a zero-valued dead zone inside the CI tolerance."""
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)
    statistics = signed_expectation_equalities(X, y, ci_constraints, cfg=cfg)
    tolerances = independent_expectation_tolerances(
        X, y, ci_constraints, tolerance=tolerance, cfg=cfg, se_X=se_X, se_y=se_y
    )
    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation")) if cfg is not None else "conditional_expectation"
    if _is_discrete_ci_kind(penalty_kind):
        return torch.relu(statistics - tolerances)
    return torch.sign(statistics) * torch.relu(torch.abs(statistics) - tolerances)


def dependent_statistic_values(statistics, cfg=None):
    """Return the quantity compared with a dependence margin.

    Conditional residual covariance is signed.  A lower-bound constraint on a
    signed value encodes an arbitrary positive direction, whereas an
    ``absolute`` statistic asks only for detectable conditional association.
    Discrete CMI is already non-negative, so the two choices coincide there.
    """
    mode = str(getattr(cfg, "ci_dependent_statistic", "signed")).strip().lower()
    if mode == "signed":
        return statistics
    if mode == "absolute":
        return torch.abs(statistics)
    raise ValueError(
        "ci_dependent_statistic must be 'signed' or 'absolute', "
        f"got {mode!r}."
    )


def dependent_expectation_inequalities(X, y, ci_constraints, cfg=None):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    statistics = signed_expectation_equalities(X, y, ci_constraints, cfg=cfg)
    dependence_statistics = dependent_statistic_values(statistics, cfg=cfg)
    margins = torch.tensor(
        [float(spec.get("margin", 0.05)) for spec in ci_constraints],
        device=X.device,
        dtype=X.dtype,
    )
    # PBM expects g(theta) <= 0, so dependence becomes margin - statistic <= 0.
    return margins - dependence_statistics


def dependent_expectation_violations(X, y, ci_constraints, cfg=None):
    return torch.relu(dependent_expectation_inequalities(X, y, ci_constraints, cfg=cfg))


def independent_expectation_inequalities(
    X, y, ci_constraints, tolerance=0.0, cfg=None, *, se_X=None, se_y=None
):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    signed_expectations = signed_expectation_equalities(X, y, ci_constraints, cfg=cfg)
    tolerances = independent_expectation_tolerances(
        X, y, ci_constraints, tolerance=tolerance, cfg=cfg, se_X=se_X, se_y=se_y
    )
    if cfg is not None and _is_discrete_ci_kind(str(getattr(cfg, "ci_penalty_kind", ""))):
        return signed_expectations - tolerances
    return torch.abs(signed_expectations) - tolerances


def make_expectation_pbm(cfg, n_constraints, device):
    if n_constraints <= 0:
        return None
    pbm_backend = str(getattr(cfg, "ce_pbm_backend", "humancompatible_pbm"))
    pbm_lr = float(
        getattr(
            cfg,
            "ci_pbm_penalty_mult",
            getattr(cfg, "ci_pbm_lr", 0.95),
        )
    )
    pbm_kwargs = dict(
        m=n_constraints,
        mu=float(getattr(cfg, "ci_pbm_mu", 0.3)),
        lr=pbm_lr,
        penalty_update=str(getattr(cfg, "ci_pbm_penalty_update", "const")),
        init_duals=float(getattr(cfg, "ci_pbm_init_duals", 0.01)),
        init_penalties=float(getattr(cfg, "ci_pbm_init_penalties", 1.0)),
        penalty_range=tuple(getattr(cfg, "ci_pbm_penalty_range", (0.001, 100.0))),
        dual_range=tuple(getattr(cfg, "ci_pbm_dual_range", (0.01, 100.0))),
        device=device,
    )
    if pbm_backend in {"humancompatible_pbm", "pbm"}:
        pbm = PBM(**pbm_kwargs)
        pbm._codiet_pbm_meta = {
            "mu": pbm_kwargs["mu"],
            "lr": pbm_kwargs["lr"],
            "penalty_update": pbm_kwargs["penalty_update"],
            "pbf": str(getattr(cfg, "ci_pbm_pbf", "quadratic_logarithmic")),
            "momentum": float(getattr(cfg, "ci_pbm_momentum", 0.0)),
            "dampening": float(getattr(cfg, "ci_pbm_dampening", 0.0)),
        }
        return ensure_humancompatible_pbm_compatible(pbm)
    if pbm_backend in {"stochastic_pbm", "spbm"}:
        return StochasticPBM(
            **pbm_kwargs,
            pbf=str(getattr(cfg, "ci_pbm_pbf", "quadratic_logarithmic")),
            dual_ema_gamma=float(
                getattr(
                    cfg,
                    "ci_pbm_gamma",
                    getattr(cfg, "ci_pbm_dual_ema_gamma", 0.0),
                )
            ),
            epoch_len=int(getattr(cfg, "ci_pbm_epoch_len", getattr(cfg, "n_inner", 1))),
            adapt_delta=float(getattr(cfg, "ci_pbm_adapt_delta", 1.5)),
        )
    raise ValueError(
        f"Unknown ce_pbm_backend: {pbm_backend!r}. "
        "Expected one of {'humancompatible_pbm', 'stochastic_pbm'}."
    )


def ce_constraint_backend(cfg):
    return str(getattr(cfg, "ce_constraint_backend", "alm_pbm"))


def _target_quantile_group_onehot(y, n_bins):
    y_flat = y.detach().reshape(-1).cpu()
    n = int(y_flat.numel())
    n_bins = max(1, min(int(n_bins), n))
    order = torch.argsort(y_flat)
    labels = torch.empty(n, dtype=torch.long)
    labels[order] = torch.arange(n, dtype=torch.long) * n_bins // n
    group_onehot = torch.zeros(n, n_bins, dtype=torch.long)
    group_onehot[torch.arange(n), labels] = 1
    return group_onehot


def _target_value_group_onehot(y):
    y_flat = y.detach().reshape(-1).cpu()
    _, labels = torch.unique(y_flat, sorted=True, return_inverse=True)
    n_groups = int(labels.max().item()) + 1 if labels.numel() else 0
    group_onehot = torch.zeros(labels.numel(), n_groups, dtype=torch.long)
    if labels.numel():
        group_onehot[torch.arange(labels.numel()), labels] = 1
    return group_onehot


def _target_looks_discrete(y, max_groups):
    y_flat = y.detach().reshape(-1).cpu()
    n = int(y_flat.numel())
    if n == 0:
        return False
    unique_values = torch.unique(y_flat)
    n_unique = int(unique_values.numel())
    if n_unique <= 1 or n_unique > int(max_groups):
        return False
    # Treat small-cardinality targets as discrete groups, matching the fairness notebook sampler setup.
    return n_unique <= max(2, n // 2)


def make_ce_minibatch_loader(X, y, cfg):
    if not is_expectation_constraint_mode(cfg):
        return None
    if not bool(getattr(cfg, "ce_use_balanced_batches", False)):
        return None

    batch_size = int(getattr(cfg, "ce_batch_size", min(128, X.shape[0])))
    batch_size = max(1, min(batch_size, X.shape[0]))
    n_bins = int(getattr(cfg, "ce_sensitive_bins", 4))
    max_discrete_groups = int(getattr(cfg, "ce_discrete_max_groups", 20))
    group_source = str(getattr(cfg, "ce_sensitive_group_source", "auto"))
    if group_source == "auto":
        group_source = "target_value" if _target_looks_discrete(y, max_discrete_groups) else "target_quantile"

    if group_source == "target_quantile":
        group_onehot = _target_quantile_group_onehot(y, n_bins)
    elif group_source in {"target_value", "discrete_target"}:
        group_onehot = _target_value_group_onehot(y)
    else:
        raise ValueError(
            f"Unknown ce_sensitive_group_source: {group_source!r}. "
            "Supported values are 'auto', 'target_quantile', and 'target_value'."
        )

    dataset = TensorDataset(X, y)
    n_groups = int(group_onehot.shape[1]) if group_onehot.ndim == 2 else 0
    if n_groups <= 1:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    batch_size = max(batch_size, min(n_groups, X.shape[0]))
    if BalancedBatchSampler is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    remainder = batch_size % n_groups
    if remainder:
        batch_size -= remainder
    if batch_size < n_groups:
        return DataLoader(dataset, batch_size=min(int(X.shape[0]), max(1, batch_size)), shuffle=True, drop_last=False)

    sampler = BalancedBatchSampler(
        group_onehot=group_onehot,
        batch_size=batch_size,
        drop_last=True,
    )
    return DataLoader(dataset, batch_sampler=sampler)
