import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from humancompatible.train.dual_optim import PBM
    from humancompatible.train.fairness.utils import BalancedBatchSampler
except ModuleNotFoundError:  # pragma: no cover - keeps local smoke tests importable.
    BalancedBatchSampler = None

    class PBM:
        def __init__(self, m, **kwargs):
            del kwargs
            self.duals = torch.zeros(m)
            self.penalties = torch.ones(m)

        def forward_update(self, loss, constraints):
            del constraints
            return loss


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


def is_expectation_constraint_mode(cfg):
    if not bool(getattr(cfg, "use_ci_penalty", False)):
        return False
    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation"))
    return penalty_kind in {"conditional_expectation", "expectation", "ce"}


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


def signed_expectation_equalities(X, y, ci_constraints):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    if y.ndim < 2:
        y = y.unsqueeze(1)
    XY = torch.cat([X, y], dim=1)
    values = []

    for spec in ci_constraints:
        x_var = XY[:, spec["x_index"]]
        y_var = XY[:, spec["y_index"]]
        z_indices = spec.get("z_indices", [])
        z_var = XY[:, z_indices] if z_indices else None
        values.append(conditional_covariance_mean(x_var, y_var, z_var))

    return torch.stack(values)


def dependent_expectation_inequalities(X, y, ci_constraints):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    signed_expectations = signed_expectation_equalities(X, y, ci_constraints)
    margins = torch.tensor(
        [float(spec.get("margin", 0.05)) for spec in ci_constraints],
        device=X.device,
        dtype=X.dtype,
    )
    # PBM expects g(theta) <= 0, so E[X_res Y_res] > margin becomes margin - E[X_res Y_res] <= 0.
    return margins - signed_expectations


def dependent_expectation_violations(X, y, ci_constraints):
    return torch.relu(dependent_expectation_inequalities(X, y, ci_constraints))


def independent_expectation_inequalities(X, y, ci_constraints, tolerance=0.0):
    if not ci_constraints:
        return torch.empty(0, device=X.device, dtype=X.dtype)

    signed_expectations = signed_expectation_equalities(X, y, ci_constraints)
    return torch.abs(signed_expectations) - float(tolerance)


def make_expectation_pbm(cfg, n_constraints, device):
    if n_constraints <= 0:
        return None
    return PBM(
        m=n_constraints,
        mu=float(getattr(cfg, "ci_pbm_mu", 0.3)),
        lr=float(getattr(cfg, "ci_pbm_lr", 0.95)),
        penalty_update=str(getattr(cfg, "ci_pbm_penalty_update", "const")),
        init_duals=float(getattr(cfg, "ci_pbm_init_duals", 0.01)),
        init_penalties=float(getattr(cfg, "ci_pbm_init_penalties", 1.0)),
        penalty_range=tuple(getattr(cfg, "ci_pbm_penalty_range", (0.001, 100.0))),
        dual_range=tuple(getattr(cfg, "ci_pbm_dual_range", (0.01, 100.0))),
        device=device,
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
