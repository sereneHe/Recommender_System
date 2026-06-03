import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from humancompatible.train.dual_optim import ALM, MoreauEnvelope
from torch.nn import MSELoss


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super().__init__()
        layers = []

        d = input_dim
        for i in range(depth):
            dim = hidden_dim // (i + 1)
            layers.append(nn.Linear(d, dim))
            layers.append(nn.Dropout(p=0.15))
            layers.append(nn.ReLU())
            d = dim

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, X):
        return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(-1)


def fit_aug_lagrangian_nn_constraint(
    X, y, W, cfg, verbose=False, device="cpu",
):
    torch.set_default_dtype(torch.float32)

    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
    W = torch.tensor(np.asarray(W), dtype=torch.float32, device=device)

    _, d = X.shape
    assert W.shape == (d + 1, d + 1), "W must be (d+1)x(d+1)"

    model = MLPRegressor(
        input_dim=d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
    ).to(device)

    optimizer = MoreauEnvelope(
        optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    )
    dual_opt = ALM(
        m=d + 1,
        lr=cfg.lambda_update_rate,
        penalty=cfg.rho0,
        init_duals=cfg.lambda0,
    )

    M = W - torch.eye(d + 1, device=device)
    muX = X.mean(dim=0)
    g0 = M[:, :-1] @ muX
    v = M[:, -1]

    if torch.allclose(v, torch.zeros_like(v)):
        raise ValueError("Constraint does not depend on predictions.")
    logging.info(f"Sanity check: GT constraint = {W_constraint(v, g0, y)}")

    loss = MSELoss()

    for outer in range(cfg.n_outer):
        for _ in range(cfg.n_inner):
            optimizer.zero_grad()
            yhat = model(X)
            mse = loss(yhat, y)

            if cfg.constrained:
                g = W_constraint(v, g0, yhat)
                aug_loss = dual_opt.forward(loss=mse, constraints=g)
                aug_loss.backward()
            else:
                mse.backward()
            optimizer.step()

        if cfg.constrained:
            with torch.no_grad():
                dual_opt.update(g)
            lam = dual_opt.duals.detach().numpy()

        for param_group in optimizer.param_groups:
            param_group["lr"] *= cfg.lr_decay

        lam = dual_opt.duals.detach().numpy()
        if verbose:
            print(
                f"outer={outer:02d}  "
                f"mean(yhat)={yhat.mean().item():+.6e}  "
                f"MSE={mse.item():+.6e}  "
                f"||g||={np.linalg.norm(g).item():.6e}  "
                f"lambda_norm={np.linalg.norm(lam).item():.6e}"
            )

    return model, lam


def W_constraint(v, g0, y):
    if y.ndim < 2:
        y = y.unsqueeze(1)
    g = g0 + y.mean(axis=0) * v
    return g
