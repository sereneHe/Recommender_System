import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from hc_predictor_ce import (
    birth_death_constraint_sampling_enabled,
    ce_constraint_backend,
    conditional_covariance_mean,
    dependent_statistic_values,
    dependent_expectation_violations,
    dependent_expectation_inequalities,
    ensure_humancompatible_pbm_compatible,
    filter_unstable_signed_dependence_constraints,
    independent_expectation_equalities,
    independent_expectation_inequalities,
    is_expectation_constraint_mode,
    hard_discrete_conditional_mutual_information,
    make_expectation_pbm,
    make_ce_minibatch_loader,
    sample_posterior_stable_constraints,
    signed_expectation_equalities,
    split_expectation_constraints,
)
from hc_predictor_ci import apply_ci_penalty, conditional_correlation_value
from nn_lagrangian import make_regression_loss

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
        # A no-constraint (pure-MSE) arm must not inherit proximal
        # regularization, so gate the proximal term on the constraint switch
        # rather than only on the optimizer switch.
        self.prox_active = self.enabled and bool(getattr(cfg, "constrained", True))
        self.prox_mu = float(getattr(cfg, "sco_prox_mu", 0.0)) if self.prox_active else 0.0
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
        if not self.prox_active or self.prox_mu <= 0.0:
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
        if not self.prox_active or self.prox_mu <= 0.0:
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
    def __init__(self, input_dim, hidden_dim, depth, dropout=0.15):
        super().__init__()
        layers = []

        d = input_dim
        for i in range(depth):
            dim = hidden_dim//(i+1)
            layers.append(nn.Linear(d, dim))
            layers.append(nn.Dropout(p=dropout)),
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
        dropout=float(getattr(cfg, "dropout", 0.15)),
    ).to(device)

    optimizer = MoreauEnvelope(
        optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    )

    graph_target = y if y.ndim == 2 else y.unsqueeze(1)
    graph_data = torch.cat([X, graph_target], dim=1)
    ci_constraints, birth_death_diagnostics = _build_ci_constraints_from_cfg(
        cfg,
        W,
        graph_data=graph_data,
        return_diagnostics=True,
    )
    ci_constraints, window_filter_diagnostics = filter_unstable_signed_dependence_constraints(
        X, y, ci_constraints, cfg=cfg
    )
    # Persist the exact resolved set on the fitted network.  The outer-CV
    # caller writes it as an artifact and evaluates it only on held-out rows.
    # Keeping this on the model avoids rebuilding constraints from a different
    # graph or a differently mutated Hydra config during reporting.
    model.ci_constraints_ = [dict(spec) for spec in ci_constraints]
    model.ci_window_filter_diagnostics_ = [dict(row) for row in window_filter_diagnostics]
    model.ci_variable_names_ = _ci_variable_names(cfg, W.shape[0])
    model.ci_penalty_kind_ = str(getattr(cfg, "ci_penalty_kind", "conditional_expectation"))
    model.ci_dependent_statistic_ = str(
        getattr(cfg, "ci_dependent_statistic", "signed")
    )
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
    # W constraint components.  "legacy_global" is the full (W-I)[Xbar, Ybar]
    # mean moment; "target_residual" replaces it with target-parent conditional
    # moments E[phi_k(Pa_Y) * (Yhat - f_W(Pa_Y))] = 0, which vary with the input
    # region instead of only shifting the global mean prediction.
    M = W - torch.eye(d + 1, device=device)
    muX = X.mean(dim=0)
    g0 = M[:, :-1] @ muX
    v = M[:, -1]
    w_mask_eps = float(getattr(cfg, "w_prediction_dependent_eps", 1e-8))
    w_masking = bool(getattr(cfg, "w_prediction_dependent_mask", False))
    w_constraint_mode = str(getattr(cfg, "w_constraint_mode", "legacy_global")).strip().lower()
    if w_constraint_mode not in {"legacy_global", "target_residual"}:
        raise ValueError(
            "w_constraint_mode must be 'legacy_global' or 'target_residual', "
            f"got {w_constraint_mode!r}."
        )
    target_index = d
    target_parents = []
    target_parent_coefs = torch.zeros(0, device=device)
    if use_w_constraints and w_constraint_mode == "target_residual":
        target_parents = [
            i for i in range(d) if abs(float(W[i, target_index])) > w_mask_eps
        ]
        if target_parents:
            target_parent_coefs = W[target_parents, target_index]
        w_rows_mask = torch.ones(d + 1, dtype=torch.bool, device=device)
        n_w_constraints = 1 + len(target_parents)
        logging.info(
            "W target_residual mode: target=%d parents=%s (n_constraints=%d)",
            target_index,
            target_parents,
            n_w_constraints,
        )
    else:
        if use_w_constraints and w_masking:
            w_rows_mask = torch.abs(v) > w_mask_eps
        else:
            w_rows_mask = torch.ones_like(v, dtype=torch.bool)
        if use_w_constraints and not bool(w_rows_mask.any()):
            logging.warning(
                "All W rows have |M[:, target]| <= %s, so the graph moment does not "
                "depend on predictions; disabling the W ALM constraint.",
                w_mask_eps,
            )
            use_w_constraints = False
        if use_w_constraints and w_masking:
            logging.info(
                "W prediction-dependent mask: kept %d/%d rows (eps=%s); dropped rows "
                "(X-only moment residuals): %s",
                int(w_rows_mask.sum().item()),
                d + 1,
                w_mask_eps,
                torch.nonzero(~w_rows_mask).flatten().detach().cpu().tolist(),
            )
        n_w_constraints = int(w_rows_mask.sum().item()) if use_w_constraints else 0
    model.w_constraint_enabled_ = use_w_constraints
    model.w_constraint_matrix_ = W.detach().cpu().numpy().copy()
    model.w_prediction_dependent_mask_ = w_rows_mask.detach().cpu().numpy()
    model.w_constraint_mode_ = w_constraint_mode

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

    if w_constraint_mode == "legacy_global" and torch.allclose(v, torch.zeros_like(v)):
        if use_w_constraints:
            raise ValueError("Constraint does not depend on predictions.")
    if use_w_constraints:
        if w_constraint_mode == "legacy_global":
            logging.info(
                "W constraint diagnostic on model-standardized training data (legacy_global): %s",
                W_constraint(v, g0, y),
            )
        else:
            logging.info(
                "W constraint mode=%s (target=%d parents=%s); the legacy global "
                "vector is not the active constraint.",
                w_constraint_mode,
                target_index,
                target_parents,
            )
    else:
        logging.info("W/DAG constraints disabled by use_w_constraints=false.")
    if bool(getattr(cfg, "ci_log_constraints", True)):
        log_active_gurobi_edges(cfg, W)
        log_active_ci_constraints(
            cfg,
            W,
            X,
            y,
            stage="Initial",
            ci_constraints=ci_constraints,
        )

    loss = make_regression_loss(cfg)
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

    grad_log_interval = int(getattr(cfg, "gradient_log_interval", 0) or 0)
    grad_conflict_history = []
    warmup_fraction = float(getattr(cfg, "w_warmup_fraction", 0.0) or 0.0)
    ramp_fraction = float(getattr(cfg, "w_warmup_ramp_fraction", 0.0) or 0.0)
    grad_ratio_cap = float(getattr(cfg, "w_grad_ratio_cap", 0.0) or 0.0)
    total_steps = max(1, int(cfg.n_outer) * int(cfg.n_inner))
    warmup_steps = int(warmup_fraction * total_steps)
    ramp_steps = max(1, int(ramp_fraction * total_steps))
    if warmup_steps or grad_ratio_cap > 0.0:
        logging.info(
            "Constraint schedule: warmup=%d/%d steps, ramp=%d steps, grad_ratio_cap=%s",
            warmup_steps,
            total_steps,
            ramp_steps if warmup_steps else 0,
            grad_ratio_cap,
        )
    for outer in range(cfg.n_outer):
        g = torch.empty(0, device=device)
        for inner in range(cfg.n_inner):
            step_index = outer * int(cfg.n_inner) + inner
            if step_index < warmup_steps:
                constraint_scale = 0.0
            elif step_index < warmup_steps + ramp_steps:
                constraint_scale = (step_index - warmup_steps) / float(ramp_steps)
            else:
                constraint_scale = 1.0
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
            aug_loss = mse

            if cfg.constrained and constraint_scale > 0.0:
                g_parts = []
                if use_w_constraints:
                    if w_constraint_mode == "target_residual":
                        if target_parents:
                            parent_term = X_batch[:, target_parents] @ target_parent_coefs
                        else:
                            parent_term = torch.zeros_like(yhat)
                        r_y = yhat - parent_term
                        phi = [torch.ones_like(r_y)]
                        phi += [X_batch[:, p] for p in target_parents]
                        g_parts.append(torch.stack([(phi_k * r_y).mean() for phi_k in phi]))
                    else:
                        g0_batch = M[:, :-1] @ X_batch.mean(dim=0)
                        g_parts.append(W_constraint(v, g0_batch, yhat)[w_rows_mask])
                if ce_backend in {"alm_pbm", "alm_all"} and ce_independent_constraints:
                    ce_eq = independent_expectation_equalities(
                        X_batch,
                        yhat,
                        ce_independent_constraints,
                        cfg=cfg,
                        se_X=X,
                        se_y=y,
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
                                se_X=X,
                                se_y=y,
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
                if g_parts and constraint_scale < 1.0:
                    # Warm-up ramp: scale the whole constraint contribution.
                    aug_loss = mse + constraint_scale * (aug_loss - mse)
                if grad_log_interval > 0 and inner % grad_log_interval == 0:
                    cos, ratio, g_norm, applied_scale = _gradient_conflict(
                        model, mse, aug_loss, g, grad_ratio_cap
                    )
                    grad_conflict_history.append(
                        {
                            "outer": outer,
                            "inner": inner,
                            "cos_mse_constraint": cos,
                            "constraint_over_mse": ratio,
                            "constraint_norm": g_norm,
                            "applied_constraint_scale": applied_scale,
                        }
                    )
                if grad_ratio_cap > 0.0 and g_parts:
                    _backward_with_constraint_cap(model, mse, aug_loss, grad_ratio_cap)
                else:
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
                                se_X=X,
                                se_y=y,
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
            step_loss = aug_loss if cfg.constrained else mse
            if not torch.isfinite(step_loss).all():
                # A non-finite loss would poison every weight for the rest of
                # the outer loop; skip this step instead of propagating NaN.
                optimizer.zero_grad()
                continue
            stochastic_opt.apply_prox_gradient(model)
            clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
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
    if bool(getattr(cfg, "w_bias_calibration", False)):
        # No-retrain mean-calibration control: pick the intercept that zeroes the
        # prediction-dependent W moment on the training data.  If this recovers
        # most of the W-ALM effect, then the ALM was mostly a mean shift.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            yhat_mean = model(X).mean()
            g_cur = g0 + v * yhat_mean
            denom = float((v * v).sum())
            if denom > 1e-12:
                delta = -float((v * g_cur).sum()) / denom
                final_layer = model.net[-1]
                if hasattr(final_layer, "bias") and final_layer.bias is not None:
                    final_layer.bias.add_(delta)
                model.w_calibration_delta_ = delta
                post = g0 + v * (yhat_mean + delta)
                logging.info(
                    "W bias calibration: delta=%.6g | ||g|| %.3e -> %.3e",
                    delta,
                    float(torch.linalg.norm(g_cur)),
                    float(torch.linalg.norm(post)),
                )
        model.train(was_training)

    model.validation_history_ = validation_history
    model.gradient_conflict_history_ = grad_conflict_history
    if grad_conflict_history:
        import statistics as _stats

        cos_values = [row["cos_mse_constraint"] for row in grad_conflict_history]
        ratio_values = [row["constraint_over_mse"] for row in grad_conflict_history]
        g_values = [row["constraint_norm"] for row in grad_conflict_history]
        scale_values = [row.get("applied_constraint_scale", 1.0) for row in grad_conflict_history]
        logging.info(
            "Gradient conflict (n=%d): mean cos(MSE, constraint)=%.4f "
            "(min %.4f, max %.4f) | mean raw ||g_constraint||/||g_MSE||=%.4f "
            "| mean applied constraint scale=%.4f | mean ||g||=%.4e",
            len(grad_conflict_history),
            _stats.fmean(cos_values),
            min(cos_values),
            max(cos_values),
            _stats.fmean(ratio_values),
            _stats.fmean(scale_values),
            _stats.fmean(g_values),
        )
    model.best_validation_loss_ = best_val_loss if best_state_dict is not None else None
    model.best_validation_outer_ = best_outer
    model.restore_best_validation_model_ = restore_best_validation_model
    w_array = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.asarray(W)
    edge_threshold = float(
        getattr(cfg, "nonzero_threshold", getattr(cfg, "ci_threshold", 1e-8))
    )
    active_w_edges = int(
        np.sum(np.abs(w_array) > edge_threshold) - np.sum(
            np.abs(np.diag(w_array)) > edge_threshold
        )
    )
    model.constraint_counts_ = {
        "backend": ce_backend,
        "independent_constraints": len(ce_independent_constraints),
        "dependent_constraints": len(ce_dependent_constraints),
        "alm_constraints": n_ce_alm_constraints,
        "pbm_constraints": n_ce_pbm_constraints,
        "w_constraints": n_w_constraints,
        "active_w_edges": active_w_edges,
        "total_constraints": n_w_constraints
        + len(ce_independent_constraints)
        + len(ce_dependent_constraints),
    }
    model.birth_death_diagnostics_ = birth_death_diagnostics

    if bool(getattr(cfg, "ci_log_constraints", True)):
        with torch.no_grad():
            final_yhat = model(X)
        log_active_ci_constraints(
            cfg,
            W,
            X,
            final_yhat,
            stage="Final",
            ci_constraints=ci_constraints,
        )

    return model, lam

def W_constraint(v, g0, y):
    if y.ndim < 2:
        y = y.unsqueeze(1)
    g = g0 + y.mean(axis=0) * v
    return g


def _gradient_conflict(model, mse, aug_loss, constraints, cap=0.0):
    """Cosine, raw norm ratio, ||g|| and the applied constraint-gradient scale.

    Diagnostic only: it does not modify the update.  Pair it with the reported
    constraint violation; a small cosine is a warning, not a feasibility proof.
    ``cap > 0`` reports the scale ``_backward_with_constraint_cap`` would apply.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad_mse = torch.autograd.grad(mse, params, retain_graph=True, allow_unused=True)
    grad_aug = torch.autograd.grad(aug_loss, params, retain_graph=True, allow_unused=True)

    def _flat(grads):
        parts = [g.reshape(-1) for g in grads if g is not None]
        return torch.cat(parts) if parts else torch.zeros(1, device=params[0].device)

    gm = _flat(grad_mse)
    gc = _flat(grad_aug) - gm
    gm_norm = float(gm.norm())
    gc_norm = float(gc.norm())
    cos = (
        float((gm @ gc) / (gm_norm * gc_norm))
        if gm_norm > 1e-12 and gc_norm > 1e-12
        else 0.0
    )
    ratio = gc_norm / gm_norm if gm_norm > 1e-12 else 0.0
    applied_scale = (
        min(1.0, cap * gm_norm / gc_norm)
        if cap > 0.0 and gc_norm > 1e-12 and gm_norm > 1e-12
        else 1.0
    )
    g_norm = float(torch.linalg.norm(constraints)) if constraints.numel() else 0.0
    return cos, ratio, g_norm, applied_scale


def _backward_with_constraint_cap(model, mse, aug_loss, cap):
    """Backprop with the constraint gradient capped at ``cap`` x the MSE norm.

    ``aug_loss = mse + constraint_term``; the constraint direction is
    ``grad(aug_loss) - grad(mse)``.  If its norm exceeds ``cap * ||grad(mse)||``
    it is rescaled, so an informative but dominant constraint cannot swamp the
    prediction objective.  cap <= 0 is treated as no cap.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad_mse = torch.autograd.grad(mse, params, retain_graph=True, allow_unused=True)
    grad_aug = torch.autograd.grad(aug_loss, params, retain_graph=True, allow_unused=True)
    grad_constraint = [
        (a - m) if (a is not None and m is not None) else (a if a is not None else m)
        for a, m in zip(grad_aug, grad_mse)
    ]

    def _norm(grads):
        parts = [g.reshape(-1) for g in grads if g is not None]
        return float(torch.cat(parts).norm()) if parts else 0.0

    mse_norm = _norm(grad_mse)
    constraint_norm = _norm(grad_constraint)
    if cap > 0.0 and constraint_norm > 1e-12 and mse_norm > 1e-12:
        scale = min(1.0, cap * mse_norm / constraint_norm)
    else:
        scale = 1.0
    for param, g_mse, g_constraint in zip(params, grad_mse, grad_constraint):
        total = None
        if g_mse is not None:
            total = g_mse.clone()
        if g_constraint is not None:
            scaled = g_constraint * scale
            total = scaled if total is None else total + scaled
        param.grad = total
    return scale


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


def _build_ci_constraints_from_adjacency(cfg, W):
    use_ci_penalty = getattr(cfg, "use_ci_penalty", False)
    if not use_ci_penalty:
        return []

    current_feature_names = list(getattr(cfg, "current_feature_names", []) or [])
    current_target_name = getattr(cfg, "current_target_name", None)
    ci_dependent_margin = float(getattr(cfg, "ci_dependent_margin", 0.05))
    add_collider_conditional_dependence = bool(
        getattr(cfg, "ci_add_collider_conditional_dependence", True)
    )
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
                    add_collider_conditional_dependence=add_collider_conditional_dependence,
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
                    all_dsep_separators=bool(
                        getattr(cfg, "ci_dsep_all_separators", False)
                    ),
                )
            )
    elif ci_mode == "collider_pairs":
        collider_pairs = collider_constraint_pairs_from_adjacency(
            W,
            threshold=float(getattr(cfg, "ci_threshold", 1e-8)),
            skip_if_direct_edge=bool(getattr(cfg, "ci_skip_if_direct_edge", True)),
            dependent_margin=ci_dependent_margin,
        )
        ci_constraints = []
        if bool(getattr(cfg, "ci_add_collider_marginal_independence", True)):
            ci_constraints.extend(
                [pair for pair in collider_pairs if pair.get("mode") == "marginal_independent"]
            )
        if add_collider_conditional_dependence:
            ci_constraints.extend(
                [pair for pair in collider_pairs if pair.get("mode") == "conditional_dependent"]
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
            add_collider_conditional_dependence=add_collider_conditional_dependence,
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
            all_dsep_separators=bool(getattr(cfg, "ci_dsep_all_separators", False)),
        )
    else:
        raise ValueError(
            f"Unknown ci_mode: {ci_mode!r}. "
            "Expected one of {'manual', 'collider_pairs', 'conservative'}."
        )

    resolved_constraints = _resolve_named_ci_constraints(
        ci_constraints,
        getattr(cfg, "current_feature_names", []),
        getattr(cfg, "current_target_name", None),
    )

    # Optional experiment control: retain constraints that involve the
    # prediction target.  ``endpoint`` is the safe option for a predictor
    # audit: the target must be X or Y in T(X_i, Y | Z), never merely part of
    # Z/a collider.  ``any`` preserves the historical behavior.
    if bool(getattr(cfg, "ci_target_related_only", False)):
        target_role = str(getattr(cfg, "ci_target_constraint_role", "any")).strip().lower()
        if target_role not in {"any", "endpoint"}:
            raise ValueError(
                "ci_target_constraint_role must be 'any' or 'endpoint', "
                f"got {target_role!r}."
            )
        target_index = len(list(getattr(cfg, "current_feature_names", []) or []))
        target_related = []
        for spec in resolved_constraints:
            endpoints = {
                int(spec.get("x_index", -1)),
                int(spec.get("y_index", -1)),
            }
            if target_role == "endpoint":
                if target_index in endpoints:
                    target_related.append(spec)
                continue
            indices = {
                *endpoints,
                *(int(index) for index in spec.get("z_indices", []) or []),
            }
            collider_index = spec.get("collider_index")
            if collider_index is not None:
                indices.add(int(collider_index))
            if target_index in indices:
                target_related.append(spec)
        resolved_constraints = target_related

    resolved_constraints = _dedupe_indexed_ci_constraints(resolved_constraints)
    if bool(getattr(cfg, "ci_prune_redundant", False)):
        before = len(resolved_constraints)
        resolved_constraints = _prune_redundant_independence_constraints(
            resolved_constraints
        )
        logging.info(
            "CI subset-pruning enabled: before=%d after=%d removed=%d",
            before,
            len(resolved_constraints),
            before - len(resolved_constraints),
        )

    return resolved_constraints


def _build_ci_constraints_from_cfg(
    cfg,
    W,
    graph_data=None,
    return_diagnostics=False,
):
    """Build CI constraints from one DAG or an opt-in DAG posterior.

    ``HC_CE_BD_MCMC=1`` activates birth-death/reversal sampling.  With the
    switch absent, this wrapper is intentionally identical to the historical
    single-thresholded-W path.
    """

    if not birth_death_constraint_sampling_enabled():
        constraints = _build_ci_constraints_from_adjacency(cfg, W)
        result = (constraints, None)
        return result if return_diagnostics else constraints

    if graph_data is None:
        logging.warning(
            "HC_CE_BD_MCMC is enabled but graph_data was not supplied; "
            "falling back to the single-DAG constraint set for this call."
        )
        constraints = _build_ci_constraints_from_adjacency(cfg, W)
        diagnostics = {
            "enabled": True,
            "fallback": "missing_graph_data",
        }
        result = (constraints, diagnostics)
        return result if return_diagnostics else constraints

    constraints, diagnostics = sample_posterior_stable_constraints(
        W,
        graph_data,
        lambda adjacency: _build_ci_constraints_from_adjacency(cfg, adjacency),
        initial_edge_threshold=float(getattr(cfg, "ci_threshold", 0.1)),
    )
    logging.info(
        "Birth-death DAG constraint sampling: graphs=%d, candidates=%d, "
        "retained=%d (independent=%d, dependent=%d)",
        diagnostics["unique_post_burn_in_graphs"],
        diagnostics["candidate_constraint_keys"],
        diagnostics["retained_constraints"],
        diagnostics["retained_independence_constraints"],
        diagnostics["retained_dependence_constraints"],
    )
    result = (constraints, diagnostics)
    return result if return_diagnostics else constraints


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
                value = margin - dependent_statistic_values(cond_cov_mean, cfg=cfg)
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


def log_active_ci_constraints(cfg, W, X, y, stage, ci_constraints=None):
    if not bool(getattr(cfg, "use_ci_penalty", False)):
        return
    if not bool(getattr(cfg, "ci_log_constraints", True)):
        return

    if ci_constraints is None:
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
            "  CI constraint %02d: relation=%s mode=%s source=%s x=%s y=%s "
            "z=[%s] collider=%s support=%s opposing=%s penalty=%+.6g",
            idx,
            spec.get("type", "independent"),
            spec.get("mode", "manual"),
            spec.get("source", "manual"),
            x_name,
            y_name,
            z_text,
            collider_name,
            spec.get("posterior_support", "n/a"),
            spec.get("opposing_support", "n/a"),
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


def _prune_redundant_independence_constraints(constraints):
    """Keep only minimal conditioning sets for an endpoint pair.

    This is an optional stability heuristic for estimated-DAG screens.  It is
    deliberately not enabled by default: in general, conditional
    independence is not monotone in the conditioning set, so dropping
    ``X _||_ Y | C,D`` merely because ``X _||_ Y | C`` exists is not a
    theorem.  The phase-1 arm records this as a pruning ablation, not as a
    logically equivalent replacement for the full constraint pool.
    """
    if not constraints:
        return []

    grouped = {}
    passthrough = []
    for spec in constraints:
        relation = spec.get("type", "independent")
        mode = spec.get("mode", "")
        if relation != "independent" or mode != "d_separated_independent":
            passthrough.append(spec)
            continue
        pair = tuple(sorted((int(spec["x_index"]), int(spec["y_index"]))))
        grouped.setdefault(pair, []).append(spec)

    retained = list(passthrough)
    for specs in grouped.values():
        ordered = sorted(
            specs,
            key=lambda item: (
                len(item.get("z_indices", []) or []),
                tuple(sorted(int(z) for z in item.get("z_indices", []) or [])),
            ),
        )
        minimal_sets = []
        for spec in ordered:
            z_set = set(int(z) for z in spec.get("z_indices", []) or [])
            if any(previous <= z_set for previous in minimal_sets):
                continue
            minimal_sets.append(z_set)
            retained.append(spec)

    return _dedupe_indexed_ci_constraints(retained)


def d_separated_independence_constraints_from_adjacency(
    W,
    threshold=1e-8,
    skip_if_direct_edge=True,
    max_separator_size=None,
    all_separators=False,
):
    """
    Add independent penalties for pairs that are d-separated in the learned DAG.

    For each unordered pair (x, y), the default asks NetworkX for one minimal
    d-separator Z.  The opt-in ``all_separators`` mode enumerates every
    conditioning set up to ``max_separator_size``; it exists only for the
    Phase-1 pruning ablation and should not be enabled for high-dimensional
    real data.
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

            candidate_nodes = sorted(restricted_nodes - {x, y})
            if all_separators:
                if max_separator_size is None:
                    raise ValueError(
                        "all_separators requires max_separator_size to bound the candidate pool"
                    )
                from itertools import combinations

                max_size = min(int(max_separator_size), len(candidate_nodes))
                for size in range(max_size + 1):
                    for combination in combinations(candidate_nodes, size):
                        z_indices = list(combination)
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
    add_collider_conditional_dependence=True,
    add_shielded_collider_dependence=False,
    add_collider_marginal_independence=False,
    shielded_dependent_margin=0.05,
    max_shielded_constraints_per_collider=0,
    max_shielded_constraints_per_target=0,
    shielded_exclude_target_in_pair=False,
    current_feature_names=None,
    current_target_name=None,
    max_dsep_separator_size=None,
    all_dsep_separators=False,
):
    """
    Conservative CI strategy:
      1. Keep user/default independence constraints.
      2. Add independence constraints for d-separated pairs in W.
      3. Optionally add dependence constraints for detected collider-open cases.

    This matches the recommended policy:
      - enforce d-separation as independence
      - only add dependence for colliders when explicitly enabled
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
                all_separators=bool(all_dsep_separators),
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
    if add_collider_conditional_dependence:
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
