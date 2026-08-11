import torch
from torch.utils.data import DataLoader, TensorDataset

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


def _residualize(target, cond):
    if cond is None or cond.numel() == 0:
        return target - target.mean()
    if cond.ndim == 1:
        cond = cond.unsqueeze(1)
    cond_aug = torch.cat(
        [cond, torch.ones(cond.shape[0], 1, device=cond.device, dtype=cond.dtype)],
        dim=1,
    )
    beta = torch.linalg.pinv(cond_aug) @ target.unsqueeze(1)
    fitted = (cond_aug @ beta).squeeze(1)
    return target - fitted


def conditional_covariance_mean(x_var, y_var, z_var):
    x_res = _residualize(x_var, z_var)
    y_res = _residualize(y_var, z_var)
    return torch.mean(x_res * y_res)


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
            values.append(conditional_covariance_mean(x_var, y_var, z_var))

    return torch.stack(values)


def signed_expectation_equalities(X, y, ci_constraints, cfg=None):
    return _constraint_statistic_values(X, y, ci_constraints, cfg=cfg)


def dependent_expectation_inequalities(X, y, ci_constraints, cfg=None):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    signed_expectations = signed_expectation_equalities(X, y, ci_constraints, cfg=cfg)
    margins = torch.tensor(
        [float(spec.get("margin", 0.05)) for spec in ci_constraints],
        device=X.device,
        dtype=X.dtype,
    )
    # PBM expects g(theta) <= 0, so dependence becomes margin - statistic <= 0.
    return margins - signed_expectations


def dependent_expectation_violations(X, y, ci_constraints, cfg=None):
    return torch.relu(dependent_expectation_inequalities(X, y, ci_constraints, cfg=cfg))


def independent_expectation_inequalities(X, y, ci_constraints, tolerance=0.0, cfg=None):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    signed_expectations = signed_expectation_equalities(X, y, ci_constraints, cfg=cfg)
    if cfg is not None and _is_discrete_ci_kind(str(getattr(cfg, "ci_penalty_kind", ""))):
        return signed_expectations - float(tolerance)
    return torch.abs(signed_expectations) - float(tolerance)


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
