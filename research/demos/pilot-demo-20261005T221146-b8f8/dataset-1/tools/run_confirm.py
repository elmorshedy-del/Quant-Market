"""run_confirm: single, pre-registered confirmation test of P1–P7 on data/confirmation (ledger R5).

Uses ONLY parameters frozen from data/explore: research/artifacts/explore_fit.json (gated K=2) plus
the K=1 pooled fit and the continuous-loading model, both refit here on explore. Nothing is fitted on
confirmation data except where a prediction itself asks for a refit on confirmation (P2: K=1/2/3 BIC).
Writes research/artifacts/confirm_results.json. Run once.
Usage: .venv/bin/python research/tools/run_confirm.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from panel_tools import load_panel, per_asset_ar1, fit_mixture_ar1, type_posterior, gate_probs_from, _comp_loglik
from forecasters import rolling_mse, standardize

fz = json.load(open("research/artifacts/explore_fit.json"))
frozen = {k: np.array(fz[k]) for k in ("c", "phi", "sig", "gate")}
assert fz["gate_covariates"] == ["size", "age"]
ae, Xe, _ = load_panel("data/explore")
pooled_e = fit_mixture_ar1(Xe, 1)
ac, Xc, _ = load_panel("data/confirmation")
n, T = Xc.shape
print("confirmation:", n, "assets, T =", T, "sectors", ac.sector.value_counts().to_dict())
Zc = ac[["size", "age"]].values
R = {}; pv = {}

# P1 persistence
_, p1, _ = per_asset_ar1(Xc[:, :20]); _, p2, _ = per_asset_ar1(Xc[:, 20:])
r = stats.pearsonr(p1, p2); ci = r.confidence_interval(0.95)
R["P1"] = dict(r=r.statistic, ci=[ci.low, ci.high], p=r.pvalue, pass_=bool(r.statistic > 0.5 and ci.low > 0.3)); pv["P1"] = r.pvalue

# P2 mixture on confirmation (refit is what P2 asks)
f = {K: fit_mixture_ar1(Xc, K) for K in (1, 2)}
S = standardize(Xc); s2 = fit_mixture_ar1(S, 2); s3 = fit_mixture_ar1(S, 3)
d12 = f[1]["bic"] - f[2]["bic"]; d23s = s2["bic"] - s3["bic"]; ph = f[2]["phi"]
R["P2"] = dict(dBIC12=d12, phi=ph.tolist(), c=f[2]["c"].tolist(), sig=f[2]["sig"].tolist(), share_momentum=float(f[2]["post"][:, 1].mean()),
               dBIC23_standardized=d23s,
               pass_=bool(d12 > 10 and -0.55 <= ph[0] <= -0.28 and 0.27 <= ph[1] <= 0.53 and d23s < 10))

# P3/P4 handle: labels from history with frozen components, NO gate
ung = dict(frozen); ung["gate"] = np.zeros((2, 1))
lab = (type_posterior(Xc, ung)[:, 1] > 0.5).astype(int)
lg = sm.Logit(lab, sm.add_constant(ac[["age", "size"]])).fit(disp=0)
lga = sm.Logit(lab, sm.add_constant(ac[["age"]])).fit(disp=0)
gp = gate_probs_from(Zc, frozen["gate"])[:, 1]
auc = roc_auc_score(lab, gp)
R["P3"] = dict(age_coef_univ=lga.params["age"], age_p_univ=lga.pvalues["age"], age_coef=lg.params["age"], age_p=lg.pvalues["age"], auc_frozen_gate=auc,
               share_momentum_label=float(lab.mean()), pass_=bool(lga.params["age"] > 0 and lga.pvalues["age"] < 0.01 and auc > 0.70))
pv["P3"] = lga.pvalues["age"]
p_size_1s = lg.pvalues["size"] / 2 if lg.params["size"] < 0 else 1 - lg.pvalues["size"] / 2
R["P4"] = dict(size_coef=lg.params["size"], size_p_one_sided=p_size_1s, pass_=bool(lg.params["size"] < 0 and p_size_1s < 0.05)); pv["P4"] = p_size_1s

# P5 discrete vs continuous: frozen gated mixture loglik vs frozen continuous-loading model (fit on explore)
ye, xe = Xe[:, 1:].ravel(), Xe[:, :-1].ravel()
def design(a, x):
    ag = np.repeat(a.age.values, T - 1); sz = np.repeat(a["size"].values, T - 1)
    return np.column_stack([np.ones_like(x), x, x * ag, x * sz, ag, sz])
oc = sm.OLS(ye, design(ae, xe)).fit(); sig_c = np.sqrt(oc.ssr / oc.nobs)
yc, xc = Xc[:, 1:].ravel(), Xc[:, :-1].ravel()
ll_cont = float(stats.norm.logpdf(yc - design(ac, xc) @ oc.params, scale=sig_c).sum())
L = _comp_loglik(Xc, frozen["c"], frozen["phi"], frozen["sig"]) + np.log(gate_probs_from(Zc, frozen["gate"]))
mx = L.max(1, keepdims=True); ll_mix = float((mx[:, 0] + np.log(np.exp(L - mx).sum(1))).sum())
_, phic, _ = per_asset_ar1(Xc)
w = [stats.pearsonr(ac.age[lab == k], phic[lab == k]) for k in (0, 1)]
R["P5"] = dict(ll_mix_frozen=ll_mix, ll_cont_frozen=ll_cont, dll_per100=(ll_mix - ll_cont) / n * 100,
               within_type_r=[x.statistic for x in w], within_type_p=[x.pvalue for x in w],
               pass_=bool((ll_mix - ll_cont) / n * 100 > 5 and all(abs(x.statistic) < 0.25 for x in w)))
pv["P5_type0"] = w[0].pvalue; pv["P5_type1"] = w[1].pvalue   # small p would count AGAINST P5

# P6 sector means (null prediction)
from panel_tools import type_posterior as tp
Pc = tp(Xc, {**frozen}, Zc)
y, x = Xc[:, 1:], Xc[:, :-1]
fc = Pc[:, :1] * (frozen["c"][0] + frozen["phi"][0] * x) + Pc[:, 1:] * (frozen["c"][1] + frozen["phi"][1] * x)
rbar = (y - fc).mean(1)
an = stats.f_oneway(*[rbar[ac.sector.values == s] for s in sorted(ac.sector.unique())])
R["P6"] = dict(F=an.statistic, p=an.pvalue, sector_means=pd.Series(rbar).groupby(ac.sector.values).mean().to_dict(), pass_=bool(an.pvalue > 0.05))
pv["P6_effect"] = an.pvalue   # small p would count AGAINST P6

# P7 predictive, frozen explore parameters
params = {"pooled": pooled_e, "mix_gated": dict(frozen)}
names = ["zero", "pooled", "per_asset_ols", "mix_gated"]
m, se = rolling_mse(params, Xc, Zc, t_from=20, names=names)
dif = se["pooled"].mean(1) - se["mix_gated"].mean(1)
tt = stats.ttest_1samp(dif, 0, alternative="greater")
R["P7"] = dict(mse=m, ratio_vs_pooled=m["mix_gated"] / m["pooled"], ratio_vs_zero=m["mix_gated"] / m["zero"],
               ratio_vs_ols=m["mix_gated"] / m["per_asset_ols"], paired_t=tt.statistic, p_one_sided=tt.pvalue,
               pass_=bool(m["mix_gated"] / m["pooled"] < 0.9 and m["mix_gated"] / m["zero"] < 0.9 and m["mix_gated"] / m["per_asset_ols"] < 0.97 and tt.pvalue < 0.01))
pv["P7"] = tt.pvalue

# Holm over the confirmation p-values
keys = list(pv); ps = np.array([pv[k] for k in keys]); o = np.argsort(ps); h = np.empty(len(ps)); run = 0
for rank, j in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[j])); h[j] = run
R["holm"] = {k: dict(p=float(pv[k]), p_holm=float(h[i])) for i, k in enumerate(keys)}
json.dump(R, open("research/artifacts/confirm_results.json", "w"), indent=1, default=float)
for k, v in R.items(): print(k, json.dumps(v, default=lambda z: round(float(z), 6)))
