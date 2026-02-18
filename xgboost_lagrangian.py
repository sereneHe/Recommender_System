import numpy as np
np.seterr(invalid="raise")
import xgboost as xgb

def fit_aug_lagrangian_W_constraint(
    X, y, W,
    params=None,
    num_boost_round=300,
    n_outer=10,
    rho0=1.0,
    rho_mult=2.0,
    lam0=None,
    sample_weight=None,
    verbose=False
):
    """
    Squared-error regression with constraint: mean_s (W z_s - z_s) = 0,
    where z_s = [X_s; yhat_s].

    Uses augmented Lagrangian with vector lambda.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    W = np.asarray(W)

    n, d = X.shape
    # print("n:", n)
    # print("d:", d)
    # print("W:", W.shape)
    assert W.shape == (d + 1, d + 1), "W must be (d+1)x(d+1) where d = X.shape[1]."

    base_score = float(np.average(y, weights=sample_weight)) if sample_weight is not None else float(y.mean())


    if params is None:
        params = dict(
            max_depth=3,
            eta=0.1,
            #tree_method="hist",
            seed=42,
        )
    params = dict(params)  # copy
    params["base_score"] = base_score

    M = W - np.eye(d + 1)

    muX = X.mean(axis=0)
    zbar_const = np.concatenate([muX, [0.0]])           # [muX; 0]
    g0 = M @ zbar_const                                  # constant part when mean y = 0
    v = M[:, -1].copy()                                  # last column of M

    # If v is (near) zero, constraint does not depend on predictions at all.
    if np.allclose(v, 0.0):
        raise ValueError("Constraint does not depend on yhat (last column of W-I is ~0). "
                         "Either it's already satisfied/violated by X alone and cannot be fixed by yhat.")

    lam = np.zeros(d + 1) if lam0 is None else np.asarray(lam0, dtype=float).copy()
    rho = float(rho0)

    dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
    booster = None

    for k in range(n_outer):
        if booster is not None and "base_score" in params:
            del params["base_score"]
        # capture lam, rho in closure
        def obj(preds, dtrain):
            y_true = dtrain.get_label()
            nloc = preds.shape[0]

            # base squared error: 0.5*(y - pred)^2
            grad = preds - y_true
            hess = np.ones_like(preds)

            muY = preds.mean()
            try:
                g_ = g0 + muY * v
            except FloatingPointError as e:
                print("Invalid multiply detected")
                print("muY:", muY)
                print("v:", v)
                raise
            # scalar that multiplies d(muY)/d(pred_i)=1/n
            alpha = float(v @ lam + rho * (v @ g_))

            grad = grad + alpha / nloc

            # diagonal Hessian approximation
            hess = hess + rho * float(v @ v) / (nloc * nloc)

            return grad, hess

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            xgb_model=booster,   # keep adding trees across outer iterations
        )

        preds = booster.predict(dtrain)
        muY = float(preds.mean())
        g = g0 + muY * v

        # dual update
        lam = lam + rho * g

        # optional rho schedule
        rho *= rho_mult

        # diagnostics
        if verbose:
            print(
                f"outer={k:02d}  mean(yhat)={muY:+.6e}  ||g||={np.linalg.norm(g):.6e}  "
                f"lambda_norm={np.linalg.norm(lam):.6e}  rho={rho:.3g}"
            )

    return booster, lam