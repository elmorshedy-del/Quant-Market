"""panel_tools: reusable instruments for the return-panel investigation.

What it does
    * load_panel(dir)              -> (assets DataFrame, X ndarray n_assets x T, ids)
    * per_asset_ar1(X)             -> per-asset OLS (c, phi, sigma) of r_t on r_{t-1}
    * simulate_panel(...)          -> synthetic panel from a K-type AR(1) mixture whose type
                                      probabilities may depend on covariates (planted truth)
    * fit_mixture_ar1(X, K, Z)     -> EM for a mixture of AR(1) regressions. Each ASSET has one
                                      latent type, shared by all of its transitions. Optional gating
                                      covariates Z make P(type) a multinomial logit in Z
                                      ("mixture of experts"). Returns params, loglik, BIC, posteriors.
    * type_posterior(X, params, Z) -> posterior type probabilities of assets, including unseen ones
    * predict_next(X, params, Z)   -> one-step-ahead forecast  sum_k p_k (c_k + phi_k r_T)

When to use it
    Use it whenever the question is "does this panel contain distinct AR(1) types, and does an
    observed characteristic predict the type?" Run simulate_panel first as a planted-truth control.

What the output means
    loglik is the conditional log-likelihood of r_2..r_T given r_1, summed over assets.
    BIC = -2 loglik + n_params * log(n_assets). The unit of type assignment is the asset.

Test its answer must pass
    On simulate_panel(K=2, phis=(-0.5,+0.4)) with n=300 and T=40, the fit must recover the phis
    within about 0.05 and classify more than 90% of assets correctly. On a homogeneous (K=1)
    panel, BIC must not prefer K=2 by more than 10. See research/tools/run_planted_truth.py.
"""
import numpy as np
import pandas as pd


def load_panel(d):
    a = pd.read_csv(f"{d}/assets.csv")
    r = pd.read_csv(f"{d}/returns.csv")
    X = r.pivot(index="asset_id", columns="t", values="ret")
    X = X.loc[a.asset_id]
    return a.reset_index(drop=True), X.values.astype(float), a.asset_id.values


def per_asset_ar1(X):
    y, x = X[:, 1:], X[:, :-1]
    xm, ym = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    phi = ((x - xm) * (y - ym)).sum(1) / ((x - xm) ** 2).sum(1)
    c = ym[:, 0] - phi * xm[:, 0]
    res = y - c[:, None] - phi[:, None] * x
    sig = np.sqrt((res ** 2).sum(1) / (y.shape[1] - 2))
    return c, phi, sig


def gate_probs_from(Z, G):
    """Type probabilities softmax([1, Z] @ G.T) for an n x p covariate matrix Z (or None)."""
    n = Z.shape[0] if Z is not None else 1
    Zc = np.ones((n, 1)) if Z is None else np.column_stack([np.ones(n), Z])
    return _gate_probs(Zc, G)


def simulate_panel(n, T, phis, cs, sigs, gate_coef=None, Z=None, probs=None, rng=None, burn=50):
    """Simulate a K-type AR(1) panel. Type probabilities are softmax([1,Z] @ gate_coef.T) if
    gate_coef is given, otherwise the fixed vector probs (uniform by default)."""
    rng = np.random.default_rng(rng)
    K = len(phis)
    if gate_coef is not None:
        P = gate_probs_from(Z, np.asarray(gate_coef, float))
    else:
        p = np.asarray(probs if probs is not None else np.full(K, 1.0 / K))
        P = np.tile(p, (n, 1))
    u = rng.random(n)
    types = (u[:, None] > np.cumsum(P, 1)).sum(1)
    X = np.zeros((n, T + burn))
    phi, c, s = np.asarray(phis)[types], np.asarray(cs)[types], np.asarray(sigs)[types]
    X[:, 0] = c / (1 - phi)
    for t in range(1, T + burn):
        X[:, t] = c + phi * X[:, t - 1] + s * rng.standard_normal(n)
    return X[:, burn:], types


def _comp_loglik(X, c, phi, sig):
    """n x K matrix of per-asset conditional log-likelihoods under each component."""
    y, x = X[:, 1:], X[:, :-1]
    L = np.empty((X.shape[0], len(c)))
    m = y.shape[1]
    for k in range(len(c)):
        e = y - c[k] - phi[k] * x
        L[:, k] = -0.5 * (e ** 2).sum(1) / sig[k] ** 2 - m * np.log(sig[k]) - 0.5 * m * np.log(2 * np.pi)
    return L


def _gate_probs(Zc, G):
    eta = Zc @ G.T
    P = np.exp(eta - eta.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True)


def _fit_gate(Zc, R, G0, iters=10, ridge=1e-3):
    """Multinomial logit (reference = component 0) fitted to soft labels R by Newton steps."""
    G = G0.copy()
    K = R.shape[1]
    for _ in range(iters):
        for k in range(1, K):
            P = _gate_probs(Zc, G)
            w = P[:, k] * (1 - P[:, k])
            g = Zc.T @ (R[:, k] - P[:, k]) - ridge * G[k]
            H = Zc.T @ (Zc * w[:, None]) + ridge * np.eye(Zc.shape[1])
            G[k] += np.linalg.solve(H, g)
    return G


def fit_mixture_ar1(X, K, Z=None, n_starts=6, iters=500, tol=1e-9, rng=0):
    """EM for an asset-level mixture of AR(1) regressions with optional covariate gating.
    Components are returned sorted by phi (ascending)."""
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    Zc = np.ones((n, 1)) if Z is None else np.column_stack([np.ones(n), Z])
    y, x = X[:, 1:], X[:, :-1]
    xx, yy = x.ravel(), y.ravel()
    _, phi0, _ = per_asset_ar1(X)
    best = None
    for s in range(n_starts):
        phi = np.quantile(phi0, (np.arange(K) + 0.5) / K) + (0.1 * rng.standard_normal(K) if s else 0)
        c = np.full(K, X.mean()); sig = np.full(K, X.std())
        G = np.zeros((K, Zc.shape[1]))
        ll_old = -np.inf
        for it in range(iters):
            L = _comp_loglik(X, c, phi, sig) + np.log(_gate_probs(Zc, G) + 1e-300)
            mx = L.max(1, keepdims=True)
            ll = float((mx[:, 0] + np.log(np.exp(L - mx).sum(1))).sum())
            R = np.exp(L - mx); R /= R.sum(1, keepdims=True)
            for k in range(K):
                w = np.repeat(R[:, k:k + 1], y.shape[1], 1).ravel()
                W = w.sum() + 1e-300
                mxw = (w * xx).sum() / W; myw = (w * yy).sum() / W
                phi[k] = (w * (xx - mxw) * (yy - myw)).sum() / ((w * (xx - mxw) ** 2).sum() + 1e-300)
                c[k] = myw - phi[k] * mxw
                sig[k] = max(np.sqrt((w * (yy - c[k] - phi[k] * xx) ** 2).sum() / W), 1e-6)
            if K > 1:
                if Z is None:
                    pk = R.mean(0) + 1e-12
                    G = np.log(pk / pk[0])[:, None]
                else:
                    G = _fit_gate(Zc, R, G)
            if abs(ll - ll_old) < tol * abs(ll):
                break
            ll_old = ll
        if best is None or ll > best["loglik"]:
            o = np.argsort(phi)
            best = dict(c=c[o].copy(), phi=phi[o].copy(), sig=sig[o].copy(), gate=G[o] - G[o][0],
                        loglik=ll, post=R[:, o].copy(), K=K)
    best["n_params"] = 3 * K + (K - 1) * Zc.shape[1]
    best["bic"] = -2 * best["loglik"] + best["n_params"] * np.log(n)
    return best


def type_posterior(X, params, Z=None):
    n = X.shape[0]
    Zc = np.ones((n, 1)) if Z is None else np.column_stack([np.ones(n), Z])
    L = _comp_loglik(X, params["c"], params["phi"], params["sig"]) + np.log(_gate_probs(Zc, params["gate"]) + 1e-300)
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def predict_next(X, params, Z=None):
    P = type_posterior(X, params, Z)
    return (P * (params["c"][None, :] + params["phi"][None, :] * X[:, -1:])).sum(1)
