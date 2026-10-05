"""One-step-ahead return predictor (ledger R6; see research/predictor/README.md for the interface).

Model (fitted parameters in research/predictor/params.json):
  each asset has a latent type k in {reversal, momentum}. Its returns follow AR(1):
      r_t = c_k + phi_k r_{t-1} + sig_k e_t.
  Prior P(momentum) = logistic(g0 + g1 * age). The posterior combines that prior with the
  likelihood of the asset's whole history under each type.
  Forecast for t = h+1:  sum_k P(k | history, age) * (c_k + phi_k * r_h).
Robustness: missing age -> age_fill. An asset with no history -> the prior-weighted intercept.
A single observation -> the prior and the last return. NaN returns are dropped (order kept by t).
The script reads only params.json and the given inputs, and writes only --out.

Usage: .venv/bin/python research/predictor/predict.py --assets A.csv --history H.csv --out O.csv
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def predict(assets, history, P):
    c, phi, sig = (np.asarray(P[k], float) for k in ("c", "phi", "sig"))
    g = np.asarray(P["gate"], float)
    age = pd.to_numeric(assets.get("age", pd.Series(np.nan, index=assets.index)), errors="coerce")
    age = age.fillna(P.get("age_fill", 0.0)).values
    eta = g[:, 0][None, :] + g[:, 1][None, :] * age[:, None]
    logprior = eta - np.logaddexp.reduce(eta, axis=1, keepdims=True)
    h = history[["asset_id", "t", "ret"]].copy()
    h["ret"] = pd.to_numeric(h["ret"], errors="coerce")
    h = h.dropna(subset=["ret"]).sort_values(["asset_id", "t"])
    groups = {k: v["ret"].values for k, v in h.groupby("asset_id", sort=False)}
    preds = np.empty(len(assets))
    for i, aid in enumerate(assets["asset_id"].values):
        r = groups.get(aid)
        L = logprior[i].copy()
        if r is None or len(r) == 0:
            w = np.exp(L - L.max()); w /= w.sum()
            preds[i] = float((w * c).sum()); continue
        if len(r) > 1:
            y, x = r[1:], r[:-1]
            for k in range(len(c)):
                e = y - c[k] - phi[k] * x
                L[k] += -0.5 * (e @ e) / sig[k] ** 2 - len(y) * np.log(sig[k])
        w = np.exp(L - L.max()); w /= w.sum()
        preds[i] = float((w * (c + phi * r[-1])).sum())
    return pd.DataFrame({"asset_id": assets["asset_id"].values, "prediction": preds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True); ap.add_argument("--history", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(os.path.join(HERE, "params.json")))
    out = predict(pd.read_csv(a.assets), pd.read_csv(a.history), P)
    out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
