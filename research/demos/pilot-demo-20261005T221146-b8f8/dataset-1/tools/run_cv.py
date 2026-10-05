"""run_cv: asset-level 2-fold cross-validation of the forecasters on data/explore (ledger R4).

Fit on half the assets (all t). Forecast the other half's returns at t=21..40 from their own
history 1..t-1, so the evaluated assets are unseen by the fit, as in the final scoring.
Repeated over 5 random fold splits. Also runs a planted-truth sanity check.
Usage: .venv/bin/python research/tools/run_cv.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from panel_tools import load_panel, simulate_panel
from forecasters import fit_all, rolling_mse, NAMES

a, X, ids = load_panel("data/explore")
Z = a[["age", "size"]].values
n = X.shape[0]
res = {k: [] for k in NAMES}
rng = np.random.default_rng(7)
for rep in range(5):
    perm = rng.permutation(n); folds = [perm[: n // 2], perm[n // 2:]]
    for i in range(2):
        tr, te = folds[i], folds[1 - i]
        p = fit_all(X[tr], Z[tr])
        m, _ = rolling_mse(p, X[te], Z[te], t_from=20)
        for k in NAMES: res[k].append(m[k])
summ = {k: float(np.mean(v)) for k, v in res.items()}
base = summ["pooled"]
for k in NAMES:
    print(f"{k:15s} MSE={summ[k]:.4e}  ratio_vs_pooled={summ[k]/base:.4f}  ratio_vs_zero={summ[k]/summ['zero']:.4f}  sd_over_folds={np.std(res[k]):.2e}")
json.dump(dict(cv_mse=summ, folds=res), open("research/artifacts/cv_explore.json", "w"), indent=1)

# planted-truth sanity: two-type panel, gate on Z
Zs = np.random.default_rng(3).standard_normal((300, 2))
Xs, _ = simulate_panel(300, 40, (-0.45, 0.4), (0.0005, 0.0005), (0.014, 0.014), gate_coef=[[0, 0, 0], [0, 1.5, 0]], Z=Zs, rng=9)
p = fit_all(Xs[:150], Zs[:150]); m, _ = rolling_mse(p, Xs[150:], Zs[150:], 20)
print("planted:", {k: round(v / m['pooled'], 4) for k, v in m.items()}, "PASS" if m["mix_gated"] < m["pooled"] else "FAIL")
