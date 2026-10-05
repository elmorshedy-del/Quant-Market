"""run_explore: counted exploratory analysis of data/explore (ledger R2). Never reads data/confirmation.

What it does: runs every exploratory test in a fixed list, records each p-value in a running
register, applies Holm correction over the whole register, and writes
research/artifacts/explore_tests.csv and research/artifacts/explore_fit.json (the fitted gated mixture).
Noise floors come from a parametric bootstrap of the fitted single-process (K=1) AR(1) model.
Usage: .venv/bin/python research/tools/run_explore.py
Test it must pass: research/tools/run_planted_truth.py prints PASS. This script uses the same fitter.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from panel_tools import load_panel, per_asset_ar1, simulate_panel, fit_mixture_ar1

t0 = time.time()
os.makedirs("research/artifacts", exist_ok=True)
a, X, ids = load_panel("data/explore")
n, T = X.shape
CH = ["size", "liquidity", "value", "age"]
tests = []                      # running register: (id, description, statistic, p)
def reg(name, stat, p): tests.append((name, float(stat), float(p))); print(f"[{len(tests):2d}] {name}: stat={stat:.4g} p={p:.3g}")

# ---- pooled conditional regressions (clustered by asset) -------------------------------
rows = []
for i in range(n):
    for t in range(5, T):
        rows.append([i, X[i, t]] + [X[i, t - k] for k in range(1, 6)])
D = pd.DataFrame(rows, columns=["i", "y", "l1", "l2", "l3", "l4", "l5"])
f = sm.OLS(D.y, sm.add_constant(D[["l1", "l2", "l3", "l4", "l5"]])).fit(cov_type="cluster", cov_kwds={"groups": D.i})
for k in range(1, 6): reg(f"pooled lag{k} coef={f.params[f'l{k}']:.3f}", f.tvalues[f"l{k}"], f.pvalues[f"l{k}"])

# ---- per-asset statistics vs single-process noise floor --------------------------------
c_i, phi_i, sig_i = per_asset_ar1(X)
mu_i = X.mean(1)
f1 = fit_mixture_ar1(X, 1)
B = 500
sd_phi_b, sd_mu_b, sd_sig_b = [], [], []
for b in range(B):
    Xb, _ = simulate_panel(n, T, f1["phi"], f1["c"], f1["sig"], rng=1000 + b)
    cb, pb, sb = per_asset_ar1(Xb)
    sd_phi_b.append(pb.std()); sd_mu_b.append(Xb.mean(1).std()); sd_sig_b.append(sb.std())
for nm, obs, bs in [("dispersion per-asset phi", phi_i.std(), sd_phi_b),
                    ("dispersion per-asset mean", mu_i.std(), sd_mu_b),
                    ("dispersion per-asset sigma", sig_i.std(), sd_sig_b)]:
    bs = np.array(bs); p = (1 + (bs >= obs).sum()) / (B + 1)
    reg(f"{nm} obs={obs:.4f} floor95={np.quantile(bs, .95):.4f}", obs / bs.mean(), p)
noise = dict(phi_sd_obs=phi_i.std(), phi_sd_floor_mean=float(np.mean(sd_phi_b)), phi_sd_floor_99=float(np.quantile(sd_phi_b, .99)),
             mu_sd_obs=mu_i.std(), mu_sd_floor_mean=float(np.mean(sd_mu_b)),
             sig_sd_obs=sig_i.std(), sig_sd_floor_mean=float(np.mean(sd_sig_b)))
print("bimodality: per-asset phi histogram", np.histogram(phi_i, bins=np.linspace(-1, 1, 11))[0])

# ---- persistence: first half (t=1..20) vs second half (t=21..40) -------------------------
_, p1, _ = per_asset_ar1(X[:, :20]); _, p2, _ = per_asset_ar1(X[:, 20:])
r = stats.pearsonr(p1, p2); ci = r.confidence_interval(0.95)
reg(f"persistence phi halves r={r.statistic:.3f} CI=({ci.low:.3f},{ci.high:.3f})", r.statistic, r.pvalue)
rm = stats.pearsonr(X[:, :20].mean(1), X[:, 20:].mean(1)); cim = rm.confidence_interval(0.95)
reg(f"persistence mean halves r={rm.statistic:.3f} CI=({cim.low:.3f},{cim.high:.3f})", rm.statistic, rm.pvalue)
rs = stats.pearsonr(X[:, :20].std(1), X[:, 20:].std(1)); cis = rs.confidence_interval(0.95)
reg(f"persistence sd halves r={rs.statistic:.3f} CI=({cis.low:.3f},{cis.high:.3f})", rs.statistic, rs.pvalue)
# out-of-half classification: type from first-half sign, AR coefficient in second half
grp = p1 > 0
print("second-half pooled phi by first-half sign: neg=%.3f pos=%.3f" % (np.median(p2[~grp]), np.median(p2[grp])))

# ---- mixture comparison -----------------------------------------------------------------
fits = {K: fit_mixture_ar1(X, K) for K in (1, 2, 3)}
for K in (1, 2, 3): print(f"K={K} phi={fits[K]['phi'].round(3)} c={fits[K]['c'].round(5)} sig={fits[K]['sig'].round(4)} bic={fits[K]['bic']:.1f}")
d12 = fits[1]["bic"] - fits[2]["bic"]; d23 = fits[2]["bic"] - fits[3]["bic"]
# LRT p-values calibrated by parametric bootstrap under the null (small B: each refit is costly)
def boot_lrt(null_fit, Knull, Kalt, obs, Bb=40):
    cnt = 0
    for b in range(Bb):
        if Knull == 1:
            Xb, _ = simulate_panel(n, T, null_fit["phi"], null_fit["c"], null_fit["sig"], rng=5000 + b)
        else:
            Xb, _ = simulate_panel(n, T, null_fit["phi"], null_fit["c"], null_fit["sig"],
                                   probs=np.exp(null_fit["gate"][:, 0]) / np.exp(null_fit["gate"][:, 0]).sum(), rng=6000 + b)
        lr = 2 * (fit_mixture_ar1(Xb, Kalt, n_starts=3)["loglik"] - fit_mixture_ar1(Xb, Knull, n_starts=3)["loglik"])
        cnt += lr >= obs
    return (1 + cnt) / (Bb + 1)
lr12 = 2 * (fits[2]["loglik"] - fits[1]["loglik"]); lr23 = 2 * (fits[3]["loglik"] - fits[2]["loglik"])
reg(f"mixture K=2 vs K=1 LR={lr12:.1f} dBIC={d12:.1f}", lr12, boot_lrt(fits[1], 1, 2, lr12))
reg(f"mixture K=3 vs K=2 LR={lr23:.1f} dBIC={d23:.1f}", lr23, boot_lrt(fits[2], 2, 3, lr23))

# do the two components differ in intercept or volatility?  (LRT with constrained refit by profile)
f2 = fits[2]
P = f2["post"]
y, x = X[:, 1:], X[:, :-1]
def ll_given(c, phi, sig, pi):
    from panel_tools import _comp_loglik
    L = _comp_loglik(X, c, phi, sig) + np.log(pi)
    m = L.max(1, keepdims=True); return float((m[:, 0] + np.log(np.exp(L - m).sum(1))).sum())
pi2 = P.mean(0)
cpool = np.full(2, (P[:, :, None] * 0 + 1).sum() and np.average(np.repeat(y.mean(1, keepdims=True), 1, 1)[:, 0] - 0, weights=None))
# constrained models: equal c (set to weighted mean, refit phi by WLS within each comp) / equal sig
def refit(equal_c=False, equal_sig=False, iters=200):
    c, phi, sig, pi = f2["c"].copy(), f2["phi"].copy(), f2["sig"].copy(), pi2.copy()
    from panel_tools import _comp_loglik
    for _ in range(iters):
        L = _comp_loglik(X, c, phi, sig) + np.log(pi); L -= L.max(1, keepdims=True)
        R = np.exp(L); R /= R.sum(1, keepdims=True); pi = R.mean(0)
        W = np.repeat(R[:, :, None], T - 1, 2)  # n x K x T-1
        if equal_c:
            # coordinate step: phi_k given common c, then c given phis
            for k in range(2):
                w = W[:, k]; phi[k] = (w * x * (y - c[0])).sum() / (w * x * x).sum()
            cc = sum((W[:, k] * (y - phi[k] * x)).sum() for k in range(2)) / W.sum(); c[:] = cc
        else:
            for k in range(2):
                w = W[:, k]; mx_ = (w * x).sum() / w.sum(); my_ = (w * y).sum() / w.sum()
                phi[k] = (w * (x - mx_) * (y - my_)).sum() / (w * (x - mx_) ** 2).sum(); c[k] = my_ - phi[k] * mx_
        ss = np.array([(W[:, k] * (y - c[k] - phi[k] * x) ** 2).sum() for k in range(2)]); ws = np.array([W[:, k].sum() for k in range(2)])
        sig = np.full(2, np.sqrt(ss.sum() / ws.sum())) if equal_sig else np.sqrt(ss / ws)
    return ll_given(c, phi, sig, pi), c, phi, sig
llc, cc_, _, _ = refit(equal_c=True); lls, _, _, ss_ = refit(equal_sig=True)
reg(f"components differ in intercept c={f2['c'].round(5)} (LR)", 2 * (f2["loglik"] - llc), stats.chi2.sf(max(2 * (f2["loglik"] - llc), 0), 1))
reg(f"components differ in sigma sig={f2['sig'].round(4)} (LR)", 2 * (f2["loglik"] - lls), stats.chi2.sf(max(2 * (f2["loglik"] - lls), 0), 1))

# ---- handle search: gating covariates (LR vs ungated K=2, chi2) -------------------------
sect = pd.get_dummies(a.sector, drop_first=True).values.astype(float)
Zs = {c: a[[c]].values for c in CH}
Zs["sector"] = sect
gfits = {}
for nm, Z in Zs.items():
    g = fit_mixture_ar1(X, 2, Z=Z, n_starts=3); gfits[nm] = g
    lr = 2 * (g["loglik"] - f2["loglik"]); dfree = Z.shape[1]
    reg(f"gate on {nm} coef={g['gate'][1, 1:].round(2)}", lr, stats.chi2.sf(max(lr, 0), dfree))
gall = fit_mixture_ar1(X, 2, Z=np.column_stack([a[CH].values, sect]), n_starts=3)
lr = 2 * (gall["loglik"] - f2["loglik"])
reg(f"gate on all chars+sector coef={gall['gate'][1, 1:].round(2)}", lr, stats.chi2.sf(lr, len(CH) + sect.shape[1]))
# final gate: age + size (only if each survives; decided below after Holm)

# ---- within-component check: is phi continuous in age instead of discrete? -------------
hard = P[:, 1] > 0.5
for k, msk in [(0, ~hard), (1, hard)]:
    rr = stats.pearsonr(a.age[msk], phi_i[msk])
    reg(f"within-type-{k} corr(per-asset phi, age) r={rr.statistic:.3f}", rr.statistic, rr.pvalue)

# competing continuous model: single process with phi = b0 + b1*age + b2*size (pooled interactions)
Dc = pd.DataFrame({"y": y.ravel(), "x": x.ravel(), "age": np.repeat(a.age.values, T - 1), "size": np.repeat(a["size"].values, T - 1)})
Dc["xa"] = Dc.x * Dc.age; Dc["xs"] = Dc.x * Dc["size"]
oc = sm.OLS(Dc.y, sm.add_constant(Dc[["x", "xa", "xs", "age", "size"]])).fit()
ll_cont = oc.llf; k_cont = 7
bic_cont = -2 * ll_cont + k_cont * np.log(n)
print(f"continuous-loading single process: params={oc.params.round(4).to_dict()} BIC={bic_cont:.1f}  vs K=2 ungated BIC={f2['bic']:.1f}")

# ---- mean effects (pooled, conditional on own-lag via type posterior) ---------------------
resid = y - (P[:, :1] * (f2["c"][0] + f2["phi"][0] * x) + P[:, 1:] * (f2["c"][1] + f2["phi"][1] * x))
rbar = resid.mean(1)
for c in CH:
    rr = stats.pearsonr(a[c], rbar); reg(f"mean-return effect of {c} r={rr.statistic:.3f}", rr.statistic, rr.pvalue)
fs = stats.f_oneway(*[rbar[a.sector.values == s] for s in sorted(a.sector.unique())])
reg("mean-return differs by sector (ANOVA)", fs.statistic, fs.pvalue)

# ---- Holm over the full register -----------------------------------------------------------
tdf = pd.DataFrame(tests, columns=["test", "stat", "p"])
m = len(tdf); order = np.argsort(tdf.p.values); holm = np.empty(m); run = 0
for rank, j in enumerate(order):
    run = max(run, min(1, (m - rank) * tdf.p.values[j])); holm[j] = run
tdf["p_holm"] = holm
tdf.to_csv("research/artifacts/explore_tests.csv", index=False)
print(tdf.to_string())

# ---- gated final model on explore: covariates surviving Holm among gate tests --------------
keep = [c for c in CH if tdf.loc[tdf.test.str.startswith(f"gate on {c} "), "p_holm"].iloc[0] < 0.05]
if tdf.loc[tdf.test.str.startswith("gate on sector "), "p_holm"].iloc[0] < 0.05: keep.append("sector")
print("gate covariates surviving Holm:", keep)
Zk = np.column_stack([a[c].values for c in keep if c != "sector"]) if keep else None
fg = fit_mixture_ar1(X, 2, Z=Zk, n_starts=6)
print(f"final explore gated model: phi={fg['phi'].round(3)} c={fg['c'].round(5)} sig={fg['sig'].round(4)} gate={fg['gate'].round(3).tolist()} bic={fg['bic']:.1f}")
out = dict(K=2, gate_covariates=[c for c in keep if c != "sector"], phi=fg["phi"].tolist(), c=fg["c"].tolist(),
           sig=fg["sig"].tolist(), gate=fg["gate"].tolist(), bic=fg["bic"], loglik=fg["loglik"],
           bic_K1=fits[1]["bic"], bic_K2=fits[2]["bic"], bic_K3=fits[3]["bic"], bic_continuous=bic_cont,
           type_share_explore=float((fg["post"][:, 1] > 0.5).mean()), noise=noise, n_tests=m,
           fit_on="data/explore only")
json.dump(out, open("research/artifacts/explore_fit.json", "w"), indent=1, default=float)
print("elapsed %.1fs" % (time.time() - t0))
