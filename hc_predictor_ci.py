import torch

from hc_predictor_ce import is_expectation_constraint_mode


def _safe_std(x, eps=1e-8):
    return torch.sqrt(torch.mean((x - x.mean()) ** 2) + eps)


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


def _corrcoef_1d(x, y, eps=1e-8):
    x = x - x.mean()
    y = y - y.mean()
    return torch.mean(x * y) / (_safe_std(x, eps) * _safe_std(y, eps))


def conditional_correlation_value(x_var, y_var, z_var, eps=1e-8):
    x_res = _residualize(x_var, z_var)
    y_res = _residualize(y_var, z_var)
    return _corrcoef_1d(x_res, y_res, eps=eps)


def conditional_correlation_penalty(X, y, ci_constraints, eps=1e-8):
    if not ci_constraints:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    if y.ndim < 2:
        y = y.unsqueeze(1)
    XY = torch.cat([X, y], dim=1)
    penalties = []

    for spec in ci_constraints:
        x_idx = spec["x_index"]
        y_idx = spec["y_index"]
        z_indices = spec.get("z_indices", [])
        relation = spec.get("type", "independent")
        margin = float(spec.get("margin", 0.05))

        x_var = XY[:, x_idx]
        y_var = XY[:, y_idx]
        z_var = XY[:, z_indices] if z_indices else None

        corr = conditional_correlation_value(x_var, y_var, z_var, eps=eps)

        if relation == "independent":
            penalties.append(corr ** 2)
        elif relation == "dependent":
            penalties.append(torch.relu(margin - torch.abs(corr)) ** 2)
        else:
            raise ValueError(
                f"Unknown CI constraint type: {relation!r}. "
                "Expected 'independent' or 'dependent'."
            )

    if not penalties:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    return torch.stack(penalties).mean()


def apply_ci_penalty(loss, cfg, W, X, y, build_constraints):
    if not bool(getattr(cfg, "use_ci_penalty", False)):
        return loss
    if is_expectation_constraint_mode(cfg):
        return loss

    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_correlation"))
    if penalty_kind not in {"conditional_correlation", "correlation", "legacy"}:
        raise ValueError(
            f"Unknown ci_penalty_kind for hc_predictor_ci: {penalty_kind!r}. "
            "Expected one of {'conditional_correlation', 'correlation', 'legacy'}."
        )

    ci_constraints = build_constraints(cfg, W)
    eps = float(getattr(cfg, "ci_eps", 1e-8))
    penalty = conditional_correlation_penalty(X, y, ci_constraints, eps=eps)
    return loss + float(getattr(cfg, "lambda_ci", 0.0)) * penalty
