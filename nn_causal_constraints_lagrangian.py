import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
import logging
from hc_predictor_ce import (
    ce_constraint_backend,
    conditional_covariance_mean,
    dependent_expectation_violations,
    dependent_expectation_inequalities,
    ensure_humancompatible_pbm_compatible,
    independent_expectation_inequalities,
    is_expectation_constraint_mode,
    hard_discrete_conditional_mutual_information,
    make_expectation_pbm,
    make_ce_minibatch_loader,
    signed_expectation_equalities,
    split_expectation_constraints,
)
from hc_predictor_ci import apply_ci_penalty, conditional_correlation_value

try:
    from humancompatible.train.dual_optim import ALM, MoreauEnvelope
except ModuleNotFoundError:  # pragma: no cover - keeps local smoke tests importable.
    class MoreauEnvelope:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def zero_grad(self):
            return self.optimizer.zero_grad()

        def step(self):
            return self.optimizer.step()

        @property
        def param_groups(self):
            return self.optimizer.param_groups

    class ALM:
        def __init__(self, m, lr=1.0, penalty=1.0, init_duals=1.0, momentum=0.0):
            del lr, penalty, init_duals, momentum
            self.duals = torch.zeros(m)

        def forward(self, loss, constraints):
            del constraints
            return loss

        def update(self, g):
            del g


class StochasticConstrainedOptimizerState:
    """Optional SPBM-style state around the existing ALM/PBM backends."""

    def __init__(self, model, cfg):
        self.enabled = bool(getattr(cfg, "use_stochastic_constrained_optimizer", False))
        self.prox_mu = float(getattr(cfg, "sco_prox_mu", 0.0)) if self.enabled else 0.0
        self.prox_center_decay = float(getattr(cfg, "sco_prox_center_decay", 0.95))
        self.dual_ema_gamma = float(getattr(cfg, "sco_dual_ema_gamma", 0.0))
        self.use_adaptive_penalty = bool(getattr(cfg, "sco_use_adaptive_penalty", False))
        self.penalty_violation_tol = float(getattr(cfg, "sco_penalty_violation_tol", 0.0))
        self.alm_penalty_mult = float(getattr(cfg, "sco_alm_penalty_mult", 1.05))
        self.pbm_penalty_mult = float(getattr(cfg, "sco_pbm_penalty_mult", 0.95))
        self.alm_penalty_range = tuple(getattr(cfg, "sco_alm_penalty_range", (1e-6, 1e6)))
        self.pbm_penalty_range = tuple(getattr(cfg, "sco_pbm_penalty_range", (1e-6, 1e6)))
        self.prox_centers = [
            param.detach().clone()
            for param in model.parameters()
            if param.requires_grad
        ]

    def apply_prox_gradient(self, model):
        if not self.enabled or self.prox_mu <= 0.0:
            return
        with torch.no_grad():
            center_idx = 0
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                if param.grad is not None:
                    param.grad.add_(param.detach() - self.prox_centers[center_idx], alpha=self.prox_mu)
                center_idx += 1

    def update_prox_center(self, model):
        if not self.enabled or self.prox_mu <= 0.0:
            return
        decay = self.prox_center_decay
        with torch.no_grad():
            center_idx = 0
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                self.prox_centers[center_idx].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
                center_idx += 1

    def capture_duals(self, dual_opt):
        if not self.enabled or not hasattr(dual_opt, "duals"):
            return None
        return dual_opt.duals.detach().clone()

    def smooth_duals(self, dual_opt, previous_duals):
        if (
            not self.enabled
            or previous_duals is None
            or self.dual_ema_gamma <= 0.0
            or self.dual_ema_gamma >= 1.0
            or not hasattr(dual_opt, "duals")
        ):
            return
        with torch.no_grad():
            dual_opt.duals.copy_(
                self.dual_ema_gamma * previous_duals
                + (1.0 - self.dual_ema_gamma) * dual_opt.duals
            )

    def update_alm_penalty(self, dual_opt, constraints):
        if not self.enabled or not self.use_adaptive_penalty or constraints is None:
            return
        if not hasattr(dual_opt, "penalty") or constraints.numel() == 0:
            return
        violation = torch.max(torch.abs(constraints.detach())).item()
        if violation <= self.penalty_violation_tol:
            return
        low, high = self.alm_penalty_range
        dual_opt.penalty = float(np.clip(float(dual_opt.penalty) * self.alm_penalty_mult, low, high))

    def update_pbm_penalties(self, dual_opt, constraints):
        if not self.enabled or not self.use_adaptive_penalty or constraints is None:
            return
        if not hasattr(dual_opt, "penalties") or constraints.numel() == 0:
            return
        violation = torch.max(torch.relu(constraints.detach())).item()
        if violation <= self.penalty_violation_tol:
            return
        low, high = self.pbm_penalty_range
        with torch.no_grad():
            dual_opt.penalties.mul_(self.pbm_penalty_mult).clamp_(min=low, max=high)

    def forward_update_pbm(self, dual_opt, loss, constraints):
        if getattr(dual_opt, "is_local_stochastic_pbm", False):
            return dual_opt.forward_update(loss, constraints)
        ensure_humancompatible_pbm_compatible(dual_opt)
        previous_duals = self.capture_duals(dual_opt)
        updated_loss = dual_opt.forward_update(loss, constraints)
        self.smooth_duals(dual_opt, previous_duals)
        self.update_pbm_penalties(dual_opt, constraints)
        return updated_loss

# def compute_predictor_errors(preds, y, y_train_mean):
#     mse = np.mean((preds - y) ** 2)
#     bench = np.mean((y - y_train_mean) ** 2)
#     return mse / bench


# ------------------------------------------------------------
# Simple MLP Regressor
# ------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super().__init__()
        layers = []

        d = input_dim
        for i in range(depth):
            dim = hidden_dim//(i+1)
            layers.append(nn.Linear(d, dim))
            layers.append(nn.Dropout(p=0.15)),
            layers.append(nn.ReLU())
            d = dim

        layers.append(nn.Linear(d, 1))  # final scalar output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict(self, X):
        return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(-1)


# ------------------------------------------------------------
# Augmented Lagrangian Training
# ------------------------------------------------------------

def fit_aug_lagrangian_nn_constraint(
    X, y, W, cfg, verbose=False, device="cpu", X_val=None, y_val=None,
):
    torch_num_threads = int(getattr(cfg, "torch_num_threads", 1))
    previous_num_threads = torch.get_num_threads()
    should_limit_threads = device == "cpu" and torch_num_threads > 0
    if should_limit_threads and previous_num_threads != torch_num_threads:
        torch.set_num_threads(torch_num_threads)
    try:
        return _fit_aug_lagrangian_nn_constraint_impl(
            X,
            y,
            W,
            cfg,
            verbose=verbose,
            device=device,
            X_val=X_val,
            y_val=y_val,
        )
    finally:
        if should_limit_threads and torch.get_num_threads() != previous_num_threads:
            torch.set_num_threads(previous_num_threads)


def _fit_aug_lagrangian_nn_constraint_impl(
    X, y, W, cfg, verbose=False, device="cpu", X_val=None, y_val=None,
):
    torch.set_default_dtype(torch.float32)

    # Convert to tensors
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
    W = torch.tensor(np.asarray(W), dtype=torch.float32, device=device)
    if X_val is not None and y_val is not None:
        X_val = torch.tensor(np.asarray(X_val), dtype=torch.float32, device=device)
        y_val = torch.tensor(np.asarray(y_val), dtype=torch.float32, device=device)
    else:
        X_val = None
        y_val = None

    n, d = X.shape
    assert W.shape == (d + 1, d + 1), "W must be (d+1)x(d+1)"

    # Build model and optimizers
    model = MLPRegressor(
        input_dim=d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
    ).to(device)

    optimizer = MoreauEnvelope(
        optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    )

    ci_constraints = _build_ci_constraints_from_cfg(cfg, W)
    use_expectation_constraints = is_expectation_constraint_mode(cfg)
    use_w_constraints = bool(getattr(cfg, "use_w_constraints", getattr(cfg, "constrained", True)))
    ce_independent_constraints = []
    ce_dependent_constraints = []
    ce_backend = "none"
    if use_expectation_constraints:
        ce_backend = ce_constraint_backend(cfg)
        if ce_backend not in {"alm_pbm", "alm_all", "pbm_all"}:
            raise ValueError(
                f"Unknown ce_constraint_backend: {ce_backend!r}. "
                "Expected one of {'alm_pbm', 'alm_all', 'pbm_all'}."
            )
        ce_independent_constraints, ce_dependent_constraints = (
            split_expectation_constraints(ci_constraints)
        )
    n_ce_alm_constraints = 0
    n_ce_pbm_constraints = 0
    if ce_backend in {"alm_pbm", "alm_all"}:
        n_ce_alm_constraints += len(ce_independent_constraints)
    if ce_backend == "alm_all":
        n_ce_alm_constraints += len(ce_dependent_constraints)
    if ce_backend == "alm_pbm":
        n_ce_pbm_constraints += len(ce_dependent_constraints)
    elif ce_backend == "pbm_all":
        n_ce_pbm_constraints += len(ce_independent_constraints) + len(ce_dependent_constraints)
    n_w_constraints = d + 1 if use_w_constraints else 0

    dual_opt = ALM(
        m=n_w_constraints + n_ce_alm_constraints,
        lr=cfg.lambda_update_rate,
        penalty=cfg.rho0,
        init_duals=cfg.lambda0,
        momentum=float(getattr(cfg, "alm_momentum", 0.0)),
    )
    ce_pbm_dual_opt = make_expectation_pbm(cfg, n_ce_pbm_constraints, device)
    stochastic_opt = StochasticConstrainedOptimizerState(model, cfg)
    if use_expectation_constraints:
        logging.info(
            "Conditional expectation constraints: backend=%s, ALM=%d, PBM=%d, independent=%d, dependent=%d",
            ce_backend,
            n_ce_alm_constraints,
            n_ce_pbm_constraints,
            len(ce_independent_constraints),
            len(ce_dependent_constraints),
        )

    # Precompute constraint components
    M = W - torch.eye(d + 1, device=device)
    muX = X.mean(dim=0)
    g0 = M[:, :-1] @ muX
    v = M[:, -1]

    if torch.allclose(v, torch.zeros_like(v)):
        if use_w_constraints:
            raise ValueError("Constraint does not depend on predictions.")
    if use_w_constraints:
        logging.info(f"Sanity check: GT constraint = {W_constraint(v, g0, y)}")
    else:
        logging.info("W/DAG constraints disabled by use_w_constraints=false.")
    if bool(getattr(cfg, "ci_log_constraints", True)):
        log_active_gurobi_edges(cfg, W)
        log_active_ci_constraints(cfg, W, X, y, stage="Initial")

    loss = MSELoss()
    best_val_loss = float("inf")
    best_state_dict = None
    best_outer = None
    no_improvement = 0
    validation_history = []
    validation_min_delta = float(getattr(cfg, "validation_min_delta", 0.0))
    early_stopping_patience = int(getattr(cfg, "early_stopping_patience", 0) or 0)
    restore_best_validation_model = bool(getattr(cfg, "restore_best_validation_model", True))
    has_ce_constraints = bool(ce_independent_constraints or ce_dependent_constraints)
    if use_expectation_constraints and bool(getattr(cfg, "ce_use_balanced_batches", False)):
        if has_ce_constraints:
            ce_loader = make_ce_minibatch_loader(X, y, cfg)
        else:
            logging.info(
                "Conditional expectation constraints are empty; falling back to full-batch training."
            )
            ce_loader = None
    else:
        ce_loader = None
    ce_loader_iter = iter(ce_loader) if ce_loader is not None else None

    for outer in range(cfg.n_outer):
        g = torch.empty(0, device=device)
        for _ in range(cfg.n_inner):
            if ce_loader_iter is None:
                X_batch = X
                y_batch = y
            else:
                try:
                    X_batch, y_batch = next(ce_loader_iter)
                except StopIteration:
                    ce_loader_iter = iter(ce_loader)
                    X_batch, y_batch = next(ce_loader_iter)
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

            optimizer.zero_grad()
            yhat = model(X_batch)
            mse = loss(yhat, y_batch)
            mse = apply_ci_penalty(mse, cfg, W, X_batch, yhat, _build_ci_constraints_from_cfg)

            if cfg.constrained:
                g_parts = []
                if use_w_constraints:
                    g0_batch = M[:, :-1] @ X_batch.mean(dim=0)
                    g_parts.append(W_constraint(v, g0_batch, yhat))
                if ce_backend in {"alm_pbm", "alm_all"} and ce_independent_constraints:
                    ce_eq = signed_expectation_equalities(
                        X_batch,
                        yhat,
                        ce_independent_constraints,
                        cfg=cfg,
                    )
                    g_parts.append(ce_eq)
                if ce_backend == "alm_all" and ce_dependent_constraints:
                    ce_dep_eq = dependent_expectation_violations(
                        X_batch,
                        yhat,
                        ce_dependent_constraints,
                        cfg=cfg,
                    )
                    g_parts.append(ce_dep_eq)
                g = torch.cat(g_parts) if g_parts else torch.empty(0, device=device, dtype=yhat.dtype)
                aug_loss = dual_opt.forward(loss=mse, constraints=g) if g_parts else mse
                if ce_pbm_dual_opt is not None:
                    ce_ineq_parts = []
                    if ce_backend == "pbm_all" and ce_independent_constraints:
                        ce_ineq_parts.append(
                            independent_expectation_inequalities(
                                X_batch,
                                yhat,
                                ce_independent_constraints,
                                tolerance=float(getattr(cfg, "ce_independence_tolerance", 0.0)),
                                cfg=cfg,
                            )
                        )
                    if ce_backend in {"alm_pbm", "pbm_all"} and ce_dependent_constraints:
                        ce_ineq_parts.append(
                            dependent_expectation_inequalities(
                                X_batch,
                                yhat,
                                ce_dependent_constraints,
                                cfg=cfg,
                            )
                        )
                    if ce_ineq_parts:
                        aug_loss = stochastic_opt.forward_update_pbm(
                            ce_pbm_dual_opt,
                            aug_loss,
                            torch.cat(ce_ineq_parts),
                        )
                aug_loss.backward()
            else:
                if ce_pbm_dual_opt is not None:
                    ce_ineq_parts = []
                    if ce_backend == "pbm_all" and ce_independent_constraints:
                        ce_ineq_parts.append(
                            independent_expectation_inequalities(
                                X_batch,
                                yhat,
                                ce_independent_constraints,
                                tolerance=float(getattr(cfg, "ce_independence_tolerance", 0.0)),
                                cfg=cfg,
                            )
                        )
                    if ce_backend in {"alm_pbm", "pbm_all"} and ce_dependent_constraints:
                        ce_ineq_parts.append(
                            dependent_expectation_inequalities(
                                X_batch,
                                yhat,
                                ce_dependent_constraints,
                                cfg=cfg,
                            )
                        )
                    if ce_ineq_parts:
                        mse = stochastic_opt.forward_update_pbm(
                            ce_pbm_dual_opt,
                            mse,
                            torch.cat(ce_ineq_parts),
                        )
                mse.backward()
            stochastic_opt.apply_prox_gradient(model)
            optimizer.step()
            stochastic_opt.update_prox_center(model)

        if cfg.constrained and len(dual_opt.duals) > 0:
            with torch.no_grad():
                previous_duals = stochastic_opt.capture_duals(dual_opt)
                dual_opt.update(g)
                stochastic_opt.smooth_duals(dual_opt, previous_duals)
                stochastic_opt.update_alm_penalty(dual_opt, g)
            lam = dual_opt.duals.detach().numpy()

        for param_group in optimizer.param_groups:
            param_group['lr'] *= cfg.lr_decay

        lam = dual_opt.duals.detach().numpy()
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                train_eval_loss = loss(model(X), y).item()
                val_loss = loss(model(X_val), y_val).item()
            model.train()
            validation_history.append(
                {
                    "outer": outer,
                    "train_loss": train_eval_loss,
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val_loss - validation_min_delta:
                best_val_loss = val_loss
                best_outer = outer
                best_state_dict = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                no_improvement = 0
            else:
                no_improvement += 1
            if early_stopping_patience > 0 and no_improvement >= early_stopping_patience:
                logging.info(
                    "Early stopping at outer=%d; best outer=%s, best val_loss=%.6g",
                    outer,
                    best_outer,
                    best_val_loss,
                )
                break
        if verbose:
            print(
                f"outer={outer:02d}  "
                f"mean(yhat)={yhat.mean().item():+.6e}  "
                f"MSE={mse.item():+.6e}  "
                f"||g||={np.linalg.norm(g).item():.6e}  "
                f"lambda_norm={np.linalg.norm(lam).item():.6e}"
            )

    if best_state_dict is not None and restore_best_validation_model:
        model.load_state_dict(best_state_dict)
        logging.info(
            "Loaded best validation model from outer=%s with val_loss=%.6g",
            best_outer,
            best_val_loss,
        )
    elif best_state_dict is not None:
        logging.info(
            "Keeping final outer-loop model; best validation checkpoint was outer=%s with val_loss=%.6g",
            best_outer,
            best_val_loss,
        )
    model.validation_history_ = validation_history
    model.best_validation_loss_ = best_val_loss if best_state_dict is not None else None
    model.best_validation_outer_ = best_outer
    model.restore_best_validation_model_ = restore_best_validation_model

    if bool(getattr(cfg, "ci_log_constraints", True)):
        with torch.no_grad():
            final_yhat = model(X)
        log_active_ci_constraints(cfg, W, X, final_yhat, stage="Final")

    return model, lam

def W_constraint(v, g0, y):
    if y.ndim < 2:
        y = y.unsqueeze(1)
    g = g0 + y.mean(axis=0) * v
    return g


def _resolve_named_ci_constraints(ci_constraints, current_feature_names, target_name):
    if not ci_constraints:
        return []

    feature_names = list(current_feature_names or [])
    xy_names = feature_names + ([target_name] if target_name is not None else [])
    index_by_name = {name: idx for idx, name in enumerate(xy_names)}
    resolved = []

    for spec in ci_constraints:
        spec_dict = dict(spec)
        if "x_index" in spec_dict and "y_index" in spec_dict:
            resolved.append(spec_dict)
            continue

        x_name = spec_dict.get("x_name")
        y_name = spec_dict.get("y_name")
        z_names = list(spec_dict.get("z_names", []) or [])

        if x_name is None or y_name is None:
            continue
        if x_name not in index_by_name or y_name not in index_by_name:
            continue
        if any(name not in index_by_name for name in z_names):
            continue

        spec_dict["x_index"] = index_by_name[x_name]
        spec_dict["y_index"] = index_by_name[y_name]
        spec_dict["z_indices"] = [index_by_name[name] for name in z_names]
        resolved.append(spec_dict)

    return resolved


def _ci_variable_names(cfg, n_vars):
    feature_names = [str(name) for name in list(getattr(cfg, "current_feature_names", []) or [])]
    target_name = getattr(cfg, "current_target_name", None)
    names = feature_names + ([str(target_name)] if target_name is not None else [])
    if len(names) < n_vars:
        names.extend([f"x{i}" for i in range(len(names), n_vars)])
    return names[:n_vars]


def _build_ci_constraints_from_cfg(cfg, W):
    use_ci_penalty = getattr(cfg, "use_ci_penalty", False)
    if not use_ci_penalty:
        return []

    current_feature_names = list(getattr(cfg, "current_feature_names", []) or [])
    current_target_name = getattr(cfg, "current_target_name", None)
    ci_dependent_margin = float(getattr(cfg, "ci_dependent_margin", 0.05))
    if bool(getattr(cfg, "ci_use_shielded_collider_limits", False)):
        shielded_margin_cfg = getattr(cfg, "ci_shielded_dependent_margin", None)
        shielded_dependent_margin = (
            ci_dependent_margin if shielded_margin_cfg is None else float(shielded_margin_cfg)
        )
        max_shielded_constraints_per_collider = int(
            getattr(cfg, "ci_max_shielded_constraints_per_collider", 0) or 0
        )
        max_shielded_constraints_per_target = int(
            getattr(cfg, "ci_max_shielded_constraints_per_target", 0) or 0
        )
        shielded_exclude_target_in_pair = bool(
            getattr(cfg, "ci_shielded_exclude_target_in_pair", False)
        )
    else:
        shielded_dependent_margin = ci_dependent_margin
        max_shielded_constraints_per_collider = 0
        max_shielded_constraints_per_target = 0
        shielded_exclude_target_in_pair = False
    ci_mode = getattr(cfg, "ci_mode", "manual")
    if ci_mode == "manual":
        ci_constraints = []
        if bool(getattr(cfg, "ci_manual_from_training_dag", False)):
            ci_constraints.extend(
                conservative_ci_constraints_from_adjacency(
                    W,
                    base_constraints=[],
                    threshold=float(getattr(cfg, "ci_threshold", 1e-8)),
                    skip_if_direct_edge=bool(getattr(cfg, "ci_skip_if_direct_edge", True)),
                    dependent_margin=ci_dependent_margin,
                    add_dsep_independence=bool(getattr(cfg, "ci_add_dsep_independence", True)),
                    add_shielded_collider_dependence=bool(
                        getattr(cfg, "ci_add_shielded_collider_dependence", False)
                    ),
                    add_collider_marginal_independence=bool(
                        getattr(cfg, "ci_add_collider_marginal_independence", False)
                    ),
                    shielded_dependent_margin=shielded_dependent_margin,
                    max_shielded_constraints_per_collider=max_shielded_constraints_per_collider,
                    max_shielded_constraints_per_target=max_shielded_constraints_per_target,
                    shielded_exclude_target_in_pair=shielded_exclude_target_in_pair,
                    current_feature_names=current_feature_names,
                    current_target_name=current_target_name,
                    max_dsep_separator_size=getattr(cfg, "ci_max_dsep_separator_size", None),
                )
            )
    elif ci_mode == "collider_pairs":
        ci_constraints = collider_constraint_pairs_from_adjacency(
            W,
            threshold=float(getattr(cfg, "ci_threshold", 1e-8)),
            skip_if_direct_edge=bool(getattr(cfg, "ci_skip_if_direct_edge", True)),
            dependent_margin=ci_dependent_margin,
        )
        if bool(getattr(cfg, "ci_add_shielded_collider_dependence", False)):
            ci_constraints.extend(
                shielded_collider_dependence_constraints_from_adjacency(
                    W,
                    threshold=float(getattr(cfg, "ci_threshold", 1e-8)),
                    dependent_margin=shielded_dependent_margin,
                    current_feature_names=list(getattr(cfg, "current_feature_names", []) or []),
                    current_target_name=getattr(cfg, "current_target_name", None),
                    max_constraints_per_collider=max_shielded_constraints_per_collider,
                    max_constraints_per_target=max_shielded_constraints_per_target,
                    exclude_target_in_pair=shielded_exclude_target_in_pair,
                )
            )
    elif ci_mode == "conservative":
        ci_constraints = conservative_ci_constraints_from_adjacency(
            W,
            base_constraints=list(getattr(cfg, "ci_constraints", []) or []),
            threshold=float(getattr(cfg, "ci_threshold", 1e-8)),
            skip_if_direct_edge=bool(getattr(cfg, "ci_skip_if_direct_edge", True)),
            dependent_margin=ci_dependent_margin,
            add_dsep_independence=bool(getattr(cfg, "ci_add_dsep_independence", True)),
            add_shielded_collider_dependence=bool(
                getattr(cfg, "ci_add_shielded_collider_dependence", False)
            ),
            add_collider_marginal_independence=bool(
                getattr(cfg, "ci_add_collider_marginal_independence", False)
            ),
            shielded_dependent_margin=shielded_dependent_margin,
            max_shielded_constraints_per_collider=max_shielded_constraints_per_collider,
            max_shielded_constraints_per_target=max_shielded_constraints_per_target,
            shielded_exclude_target_in_pair=shielded_exclude_target_in_pair,
            current_feature_names=current_feature_names,
            current_target_name=current_target_name,
            max_dsep_separator_size=getattr(cfg, "ci_max_dsep_separator_size", None),
        )
    else:
        raise ValueError(
            f"Unknown ci_mode: {ci_mode!r}. "
            "Expected one of {'manual', 'collider_pairs', 'conservative'}."
        )

    return _dedupe_indexed_ci_constraints(
        _resolve_named_ci_constraints(
            ci_constraints,
            getattr(cfg, "current_feature_names", []),
            getattr(cfg, "current_target_name", None),
        )
    )


def _is_discrete_ci_kind(penalty_kind):
    return penalty_kind in {
        "discrete_conditional_independence",
        "strict_discrete_ci",
        "discrete_ci",
        "quantile_discrete_ci",
    }


def _ci_constraint_penalty_values(X, y, ci_constraints, penalty_kind, eps=1e-8, cfg=None):
    if not ci_constraints:
        return []

    if y.ndim < 2:
        y = y.unsqueeze(1)
    XY = torch.cat([X, y], dim=1)
    values = []

    for spec in ci_constraints:
        x_idx = spec["x_index"]
        y_idx = spec["y_index"]
        z_indices = spec.get("z_indices", [])
        relation = spec.get("type", "independent")
        margin = float(spec.get("margin", 0.05))

        x_var = XY[:, x_idx]
        y_var = XY[:, y_idx]
        z_var = XY[:, z_indices] if z_indices else None

        if penalty_kind in {"conditional_expectation", "expectation", "ce"}:
            cond_cov_mean = conditional_covariance_mean(x_var, y_var, z_var)
            if relation == "independent":
                value = cond_cov_mean
            elif relation == "dependent":
                value = margin - cond_cov_mean
            else:
                raise ValueError(f"Unknown CI constraint type: {relation!r}.")
        elif _is_discrete_ci_kind(penalty_kind):
            cmi = hard_discrete_conditional_mutual_information(
                x_var,
                y_var,
                z_var,
                n_bins=int(getattr(cfg, "ci_discrete_n_bins", getattr(cfg, "ce_sensitive_bins", 4))),
                max_z_states=int(getattr(cfg, "ci_discrete_max_z_states", 256)),
                eps=eps,
            ).to(device=X.device, dtype=X.dtype)
            if relation == "independent":
                value = cmi
            elif relation == "dependent":
                value = margin - cmi
            else:
                raise ValueError(f"Unknown CI constraint type: {relation!r}.")
        elif penalty_kind in {"conditional_correlation", "correlation", "legacy"}:
            corr = conditional_correlation_value(x_var, y_var, z_var, eps=eps)
            if relation == "independent":
                value = corr ** 2
            elif relation == "dependent":
                value = torch.relu(margin - torch.abs(corr)) ** 2
            else:
                raise ValueError(f"Unknown CI constraint type: {relation!r}.")
        else:
            raise ValueError(
                f"Unknown ci_penalty_kind: {penalty_kind!r}. "
                "Expected one of {'conditional_expectation', 'discrete_conditional_independence', 'conditional_correlation'}."
            )

        values.append(float(value.detach().cpu()))

    return values


def log_active_gurobi_edges(cfg, W):
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = np.asarray(W)

    threshold = float(getattr(cfg, "nonzero_threshold", getattr(cfg, "ci_threshold", 1e-8)))
    names = _ci_variable_names(cfg, W_np.shape[0])
    edges = []

    for src in range(W_np.shape[0]):
        for dst in range(W_np.shape[1]):
            if src == dst:
                continue
            weight = float(W_np[src, dst])
            if abs(weight) > threshold:
                edges.append((names[src], names[dst], weight))

    logging.info(
        "Gurobi active W edges above %.3g: %d",
        threshold,
        len(edges),
    )
    for src_name, dst_name, weight in edges:
        logging.info("  Gurobi edge: %s -> %s weight=%+.6g", src_name, dst_name, weight)


def log_active_ci_constraints(cfg, W, X, y, stage):
    if not bool(getattr(cfg, "use_ci_penalty", False)):
        return
    if not bool(getattr(cfg, "ci_log_constraints", True)):
        return

    ci_constraints = _build_ci_constraints_from_cfg(cfg, W)
    penalty_kind = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation"))
    eps = float(getattr(cfg, "ci_eps", 1e-8))
    names = _ci_variable_names(cfg, W.shape[0])

    logging.info(
        "%s CI penalty: kind=%s mode=%s lambda_ci=%s constraints=%d",
        stage,
        penalty_kind,
        getattr(cfg, "ci_mode", "manual"),
        getattr(cfg, "lambda_ci", 0.0),
        len(ci_constraints),
    )

    if not ci_constraints:
        logging.warning(
            "\033[91m%s CI/CE penalty is inactive: no d-separation independence or collider dependence constraints were found.\033[0m",
            stage.upper(),
        )
        return

    with torch.no_grad():
        values = _ci_constraint_penalty_values(
            X,
            y,
            ci_constraints,
            penalty_kind,
            eps=eps,
            cfg=cfg,
        )

    for idx, (spec, value) in enumerate(zip(ci_constraints, values), start=1):
        x_name = names[spec["x_index"]]
        y_name = names[spec["y_index"]]
        z_names = [names[z_idx] for z_idx in spec.get("z_indices", [])]
        collider_idx = spec.get("collider_index")
        collider_name = names[collider_idx] if collider_idx is not None and collider_idx < len(names) else None
        z_text = ", ".join(z_names) if z_names else "<empty>"
        logging.info(
            "  CI constraint %02d: relation=%s mode=%s source=%s x=%s y=%s z=[%s] collider=%s penalty=%+.6g",
            idx,
            spec.get("type", "independent"),
            spec.get("mode", "manual"),
            spec.get("source", "manual"),
            x_name,
            y_name,
            z_text,
            collider_name,
            value,
        )


def collider_constraints_from_adjacency(W, threshold=1e-8, skip_if_direct_edge=True):
    """
    Detect local collider patterns X -> Z <- Y from an adjacency/weight matrix W and
    convert them into marginal-independence constraints.

    Returns a list of constraint dicts compatible with
    `conditional_expectation_penalty(...)`, namely
      {
          "x_index": x,
          "y_index": y,
          "z_indices": [],
          "type": "independent",
          "source": "collider",
          "collider_index": z,
      }

    Notes:
    - We interpret W[i, j] != 0 as an edge i -> j.
    - A collider is local structure x -> z <- y with x != y.
    - If `skip_if_direct_edge=True`, we ignore pairs x,y that are directly connected,
      because they would not generally be marginally independent.
    """
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = np.asarray(W)

    d = W_np.shape[0]
    constraints = []
    seen = set()

    for z in range(d):
        parents = [i for i in range(d) if i != z and abs(W_np[i, z]) > threshold]
        if len(parents) < 2:
            continue

        for idx_x in range(len(parents)):
            for idx_y in range(idx_x + 1, len(parents)):
                x = parents[idx_x]
                y = parents[idx_y]
                pair = tuple(sorted((x, y)))

                if skip_if_direct_edge and (
                    abs(W_np[x, y]) > threshold or abs(W_np[y, x]) > threshold
                ):
                    continue

                key = (pair[0], pair[1], z)
                if key in seen:
                    continue
                seen.add(key)

                constraints.append(
                    {
                        "x_index": x,
                        "y_index": y,
                        "z_indices": [],
                        "type": "independent",
                        "source": "collider",
                        "collider_index": z,
                    }
                )

    return constraints


def collider_constraint_pairs_from_adjacency(
    W,
    threshold=1e-8,
    skip_if_direct_edge=True,
    dependent_margin=0.05,
):
    """
    Detect collider structures x -> z <- y from W and generate a pair of
    CI constraints for each collider:

    1. Marginal independence:        x ⟂ y
    2. Conditional dependence:      x ⫫̸ y | z

    Returned items are compatible with `conditional_expectation_penalty(...)`.
    We annotate each pair with `source="collider"` and `mode` in
    {"marginal_independent", "conditional_dependent"}.
    """
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = np.asarray(W)

    d = W_np.shape[0]
    constraints = []
    seen = set()

    for z in range(d):
        parents = [i for i in range(d) if i != z and abs(W_np[i, z]) > threshold]
        if len(parents) < 2:
            continue

        for idx_x in range(len(parents)):
            for idx_y in range(idx_x + 1, len(parents)):
                x = parents[idx_x]
                y = parents[idx_y]
                pair = tuple(sorted((x, y)))

                if skip_if_direct_edge and (
                    abs(W_np[x, y]) > threshold or abs(W_np[y, x]) > threshold
                ):
                    continue

                key = (pair[0], pair[1], z)
                if key in seen:
                    continue
                seen.add(key)

                constraints.append(
                    {
                        "x_index": x,
                        "y_index": y,
                        "z_indices": [],
                        "type": "independent",
                        "source": "collider",
                        "mode": "marginal_independent",
                        "collider_index": z,
                    }
                )
                constraints.append(
                    {
                        "x_index": x,
                        "y_index": y,
                        "z_indices": [z],
                        "type": "dependent",
                        "margin": dependent_margin,
                        "source": "collider",
                        "mode": "conditional_dependent",
                        "collider_index": z,
                    }
                )

    return constraints


def shielded_collider_dependence_constraints_from_adjacency(
    W,
    threshold=1e-8,
    dependent_margin=0.05,
    current_feature_names=None,
    current_target_name=None,
    max_constraints_per_collider=0,
    max_constraints_per_target=0,
    exclude_target_in_pair=False,
):
    """
    Add softer dependence constraints for shielded colliders x -> z <- y where
    x and y are directly connected. We only add dependence, never independence.

    The CE condition set is Pa(z) without x and y; conditioning on x or y itself
    would make residualization degenerate for CE(x, y | Z).
    """
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = np.asarray(W)

    d = W_np.shape[0]
    feature_names = [str(name) for name in list(current_feature_names or [])]
    target_name = str(current_target_name) if current_target_name is not None else None
    target_index = None
    if target_name is not None:
        if target_name in feature_names:
            target_index = feature_names.index(target_name)
        else:
            target_index = len(feature_names) if len(feature_names) < d else None

    constraints = []
    seen = set()
    per_collider_counts = {}
    per_target_counts = {}

    for z in range(d):
        parents = [i for i in range(d) if i != z and abs(W_np[i, z]) > threshold]
        if len(parents) < 2:
            continue

        for idx_x in range(len(parents)):
            for idx_y in range(idx_x + 1, len(parents)):
                x = parents[idx_x]
                y = parents[idx_y]
                if exclude_target_in_pair and target_index is not None and target_index in {x, y}:
                    continue
                has_x_to_y = abs(W_np[x, y]) > threshold
                has_y_to_x = abs(W_np[y, x]) > threshold
                if not (has_x_to_y or has_y_to_x):
                    continue

                pair = tuple(sorted((x, y)))
                z_indices = sorted(int(parent) for parent in parents if parent not in {x, y})
                key = (pair[0], pair[1], tuple(z_indices), z)
                if key in seen:
                    continue
                if max_constraints_per_collider and per_collider_counts.get(z, 0) >= max_constraints_per_collider:
                    continue
                target_involved = (
                    target_index is not None
                    and (
                        x == target_index
                        or y == target_index
                        or z == target_index
                        or target_index in z_indices
                    )
                )
                if (
                    max_constraints_per_target
                    and target_involved
                    and per_target_counts.get(target_index, 0) >= max_constraints_per_target
                ):
                    continue
                seen.add(key)

                if has_x_to_y and has_y_to_x:
                    shielding_edge = "both"
                elif has_x_to_y:
                    shielding_edge = "x_to_y"
                else:
                    shielding_edge = "y_to_x"

                constraints.append(
                    {
                        "x_index": x,
                        "y_index": y,
                        "z_indices": z_indices,
                        "type": "dependent",
                        "margin": dependent_margin,
                        "source": "shielded_collider",
                        "mode": "shielded_parent_dependent",
                        "collider_index": z,
                        "shielded": True,
                        "shielding_edge": shielding_edge,
                    }
                )
                per_collider_counts[z] = per_collider_counts.get(z, 0) + 1
                if target_involved:
                    per_target_counts[target_index] = per_target_counts.get(target_index, 0) + 1

    return constraints


def _dedupe_indexed_ci_constraints(constraints):
    deduped = []
    seen = set()

    for spec in constraints:
        if "x_index" not in spec or "y_index" not in spec:
            deduped.append(spec)
            continue

        pair = tuple(sorted((int(spec["x_index"]), int(spec["y_index"]))))
        z_indices = tuple(sorted(int(z) for z in spec.get("z_indices", []) or []))
        key = (pair, z_indices, spec.get("type", "independent"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)

    return deduped


def d_separated_independence_constraints_from_adjacency(
    W,
    threshold=1e-8,
    skip_if_direct_edge=True,
    max_separator_size=None,
):
    """
    Add independent penalties for pairs that are d-separated in the learned DAG.

    For each unordered pair (x, y), we ask NetworkX for one minimal
    d-separator Z. If it exists, we add x independent y | Z.
    """
    import networkx as nx
    from networkx.algorithms.d_separation import find_minimal_d_separator, is_d_separator

    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = np.asarray(W)

    d = W_np.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    for src in range(d):
        for dst in range(d):
            if src != dst and abs(W_np[src, dst]) > threshold:
                graph.add_edge(src, dst)

    if not nx.is_directed_acyclic_graph(graph):
        logging.warning("Skipping d-separated CI constraints because W is not a DAG.")
        return []

    constraints = []
    restricted_nodes = set(range(d))

    for x in range(d):
        for y in range(x + 1, d):
            if skip_if_direct_edge and (
                abs(W_np[x, y]) > threshold or abs(W_np[y, x]) > threshold
            ):
                continue

            try:
                separator = find_minimal_d_separator(
                    graph,
                    x,
                    y,
                    restricted=restricted_nodes - {x, y},
                )
            except nx.NetworkXException as exc:
                logging.warning("Could not compute d-separator for %s,%s: %s", x, y, exc)
                continue

            if separator is None:
                continue

            z_indices = sorted(int(z) for z in separator)
            if max_separator_size is not None and len(z_indices) > max_separator_size:
                continue
            if not is_d_separator(graph, x, y, set(z_indices)):
                continue

            constraints.append(
                {
                    "x_index": x,
                    "y_index": y,
                    "z_indices": z_indices,
                    "type": "independent",
                    "source": "d_separation",
                    "mode": "d_separated_independent",
                }
            )

    return constraints


def conservative_ci_constraints_from_adjacency(
    W,
    base_constraints=None,
    threshold=1e-8,
    skip_if_direct_edge=True,
    dependent_margin=0.05,
    add_dsep_independence=True,
    add_shielded_collider_dependence=False,
    add_collider_marginal_independence=False,
    shielded_dependent_margin=0.05,
    max_shielded_constraints_per_collider=0,
    max_shielded_constraints_per_target=0,
    shielded_exclude_target_in_pair=False,
    current_feature_names=None,
    current_target_name=None,
    max_dsep_separator_size=None,
):
    """
    Conservative CI strategy:
      1. Keep user/default independence constraints.
      2. Add independence constraints for d-separated pairs in W.
      3. Add dependence constraints only for detected collider-open cases.

    This matches the recommended policy:
      - enforce d-separation as independence
      - only add dependence for colliders
    """
    constraints = []

    if base_constraints:
        constraints.extend(
            [c for c in base_constraints if c.get("type", "independent") == "independent"]
        )

    if add_dsep_independence:
        constraints.extend(
            d_separated_independence_constraints_from_adjacency(
                W,
                threshold=threshold,
                skip_if_direct_edge=skip_if_direct_edge,
                max_separator_size=max_dsep_separator_size,
            )
        )

    collider_pairs = collider_constraint_pairs_from_adjacency(
        W,
        threshold=threshold,
        skip_if_direct_edge=skip_if_direct_edge,
        dependent_margin=dependent_margin,
    )
    if add_collider_marginal_independence:
        constraints.extend(
            [c for c in collider_pairs if c.get("mode") == "marginal_independent"]
        )
    constraints.extend(
        [c for c in collider_pairs if c.get("mode") == "conditional_dependent"]
    )
    if add_shielded_collider_dependence:
        constraints.extend(
            shielded_collider_dependence_constraints_from_adjacency(
                W,
                threshold=threshold,
                dependent_margin=shielded_dependent_margin,
                current_feature_names=current_feature_names,
                current_target_name=current_target_name,
                max_constraints_per_collider=max_shielded_constraints_per_collider,
                max_constraints_per_target=max_shielded_constraints_per_target,
                exclude_target_in_pair=shielded_exclude_target_in_pair,
            )
        )
    return _dedupe_indexed_ci_constraints(constraints)
