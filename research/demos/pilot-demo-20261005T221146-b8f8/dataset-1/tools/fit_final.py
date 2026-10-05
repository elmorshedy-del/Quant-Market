"""fit_final: fits the final predictor parameters (ledger R6). Run only AFTER R5 is recorded.

The gated K=2 AR(1) mixture is fitted on explore + confirmation (450 assets) with an age-only gate.
Size was dropped because the registered P4 failed in R5. The result is written to
research/predictor/params.json, which predict.py reads.
Also reports the parameters from explore alone, to show how much the refit moved them, and the
age-only versus age+size comparison on the combined data, for information only.
Usage: .venv/bin/python research/tools/fit_final.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from panel_tools import load_panel, fit_mixture_ar1

ae, Xe, _ = load_panel("data/explore"); ac, Xc, _ = load_panel("data/confirmation")
a = pd.concat([ae, ac], ignore_index=True); X = np.vstack([Xe, Xc])
f = fit_mixture_ar1(X, 2, Z=a[["age"]].values, n_starts=8)
f_as = fit_mixture_ar1(X, 2, Z=a[["age", "size"]].values, n_starts=8)
f1 = fit_mixture_ar1(X, 1)
print(f"combined age-gate: phi={f['phi'].round(4)} c={f['c'].round(6)} sig={f['sig'].round(5)} gate={f['gate'].round(3).tolist()} bic={f['bic']:.1f}")
print(f"combined age+size gate (info): gate={f_as['gate'].round(3).tolist()} bic={f_as['bic']:.1f}  K=1 bic={f1['bic']:.1f}")
out = dict(model="asset-level 2-type AR(1) mixture; P(type=momentum)=logistic(g0+g1*age); forecast=sum_k p_k(c_k+phi_k r_T)",
           gate_covariates=["age"], c=f["c"].tolist(), phi=f["phi"].tolist(), sig=f["sig"].tolist(),
           gate=f["gate"].tolist(), age_fill=0.0, fitted_on="data/explore + data/confirmation (450 assets, t=1..40), after R5 was recorded",
           ledger_entry="R6", bic=f["bic"], bic_K1=f1["bic"], bic_age_size=f_as["bic"])
json.dump(out, open("research/predictor/params.json", "w"), indent=1)
print("wrote research/predictor/params.json")
