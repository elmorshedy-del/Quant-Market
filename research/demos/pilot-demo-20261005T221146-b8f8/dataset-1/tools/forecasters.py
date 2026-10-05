"""forecasters: the candidate one-step forecasters, given fitted params, and an evaluation loop.

What it does
    fit_all(X, Z)                      -> dict of fitted parameter sets (pooled AR(1), raw mixture
                                          with and without gate, standardized-likelihood mixture with gate)
    forecast(name, params, Xhist, Z)   -> one-step forecast for each asset from its history Xhist
    rolling_mse(params, X, Z, t_from)  -> MSE of each forecaster for targets t_from..T, using the
                                          history 1..t-1 (assets are UNSEEN by the fit when X is a held-out fold)
When to use it: choosing or validating the predictor. The output is a dict name -> MSE.
Test it must pass: forecaster 'zero' gives mean(r^2), and the planted two-type panel must give
mix_gated MSE < pooled MSE (see research/tools/run_cv.py).
"""
import numpy as np
from panel_tools import fit_mixture_ar1, type_posterior, per_asset_ar1


def standardize(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def fit_all(X, Z):
    p = {}
    p["pooled"] = fit_mixture_ar1(X, 1)
    p["mix"] = fit_mixture_ar1(X, 2)
    p["mix_gated"] = fit_mixture_ar1(X, 2, Z=Z)
    p["mix_std_gated"] = fit_mixture_ar1(standardize(X), 2, Z=Z)
    return p


def _mix_forecast(P, comp, Xh):
    return (P * (comp["c"][None, :] + comp["phi"][None, :] * Xh[:, -1:])).sum(1)


def forecast(name, p, Xh, Z):
    if name == "zero":
        return np.zeros(Xh.shape[0])
    if name == "pooled":
        return p["pooled"]["c"][0] + p["pooled"]["phi"][0] * Xh[:, -1]
    if name == "per_asset_ols":
        c, phi, _ = per_asset_ar1(Xh); return c + phi * Xh[:, -1]
    if name == "mix":
        return _mix_forecast(type_posterior(Xh, p["mix"]), p["mix"], Xh)
    if name == "mix_gated":
        return _mix_forecast(type_posterior(Xh, p["mix_gated"], Z), p["mix_gated"], Xh)
    if name == "mix_std_gated":
        P = type_posterior(standardize(Xh), p["mix_std_gated"], Z)
        return _mix_forecast(P, p["mix_gated"], Xh)
    if name == "gate_only":          # characteristics only, no use of history for typing
        from panel_tools import gate_probs_from
        P = gate_probs_from(Z, p["mix_gated"]["gate"]); return _mix_forecast(P, p["mix_gated"], Xh)
    raise ValueError(name)


NAMES = ["zero", "pooled", "per_asset_ols", "mix", "mix_gated", "mix_std_gated", "gate_only"]


def rolling_mse(p, X, Z, t_from, names=NAMES):
    """Targets are columns t_from..T-1 (0-based). History is X[:, :t]. Returns (mse dict, sq-error dict)."""
    err = {k: [] for k in names}
    for t in range(t_from, X.shape[1]):
        Xh = X[:, :t]
        for k in names:
            err[k].append((X[:, t] - forecast(k, p, Xh, Z)) ** 2)
    se = {k: np.column_stack(v) for k, v in err.items()}
    return {k: float(v.mean()) for k, v in se.items()}, se
