"""run_explore2: follow-up exploratory tests on data/explore (ledger R3). Never reads confirmation.

Adds tests to the register started by run_explore.py and re-applies Holm over ALL tests (R2+R3):
  * noise floors for per-asset mean/sigma dispersion under the K=2 mixture (the right null;
    R2 used K=1, which understates the mean dispersion because phi=+0.4 assets have larger
    long-run variance)
  * higher-resolution bootstrap for the K=2 vs K=1 likelihood ratio (B=200)
  * "dispersion from scale" check: refit K=2 and K=3 on per-asset standardized returns. If K=3
    no longer wins, the third raw component is a volatility artefact rather than a third dynamic type
  * per-asset volatility vs characteristics
Writes research/artifacts/explore_tests_all.csv.
Usage: .venv/bin/python research/tools/run_explore2.py   (after run_explore.py)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from panel_tools import load_panel, per_asset_ar1, simulate_panel, fit_mixture_ar1

t0 = time.time()
a, X, ids = load_panel("data/explore")
n, T = X.shape
prev = pd.read_csv("research/artifacts/explore_tests.csv")[["test", "stat", "p"]]
tests = []
def reg(name, stat, p): tests.append((name, float(stat), float(p))); print(f"[{len(prev)+len(tests):2d}] {name}: stat={stat:.4g} p={p:.3g}")

f1 = fit_mixture_ar1(X, 1); f2 = fit_mixture_ar1(X, 2)
pi2 = f2["post"].mean(0)
c_i, phi_i, sig_i = per_asset_ar1(X)
B = 500
mu_b, sg_b = [], []
for b in range(B):
    Xb, _ = simulate_panel(n, T, f2["phi"], f2["c"], f2["sig"], probs=pi2, rng=20000 + b)
    mu_b.append(Xb.mean(1).std()); sg_b.append(per_asset_ar1(Xb)[2].std())
mu_b, sg_b = np.array(mu_b), np.array(sg_b)
reg(f"dispersion per-asset mean vs K=2 floor obs={X.mean(1).std():.5f} floor95={np.quantile(mu_b,.95):.5f}",
    X.mean(1).std() / mu_b.mean(), (1 + (mu_b >= X.mean(1).std()).sum()) / (B + 1))
reg(f"dispersion per-asset sigma vs K=2 floor obs={sig_i.std():.5f} floor95={np.quantile(sg_b,.95):.5f}",
    sig_i.std() / sg_b.mean(), (1 + (sg_b >= sig_i.std()).sum()) / (B + 1))

lr12 = 2 * (f2["loglik"] - f1["loglik"]); Bb = 200; null_lr = []
for b in range(Bb):
    Xb, _ = simulate_panel(n, T, f1["phi"], f1["c"], f1["sig"], rng=30000 + b)
    null_lr.append(2 * (fit_mixture_ar1(Xb, 2, n_starts=2)["loglik"] - fit_mixture_ar1(Xb, 1, n_starts=1)["loglik"]))
null_lr = np.array(null_lr)
reg(f"mixture K=2 vs K=1 (B=200) LR={lr12:.1f} null max={null_lr.max():.1f}", lr12, (1 + (null_lr >= lr12).sum()) / (Bb + 1))

# standardized returns: remove per-asset scale (and mean)
S = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
s1, s2, s3, s4 = (fit_mixture_ar1(S, K) for K in (1, 2, 3, 4))
for K, s in zip((1, 2, 3, 4), (s1, s2, s3, s4)): print(f"standardized K={K} phi={s['phi'].round(3)} sig={s['sig'].round(3)} bic={s['bic']:.1f}")
lr23 = 2 * (s3["loglik"] - s2["loglik"]); null23 = []
for b in range(60):
    Sb, _ = simulate_panel(n, T, s2["phi"], s2["c"], s2["sig"], probs=s2["post"].mean(0), rng=40000 + b)
    Sb = (Sb - Sb.mean(1, keepdims=True)) / Sb.std(1, keepdims=True)
    null23.append(2 * (fit_mixture_ar1(Sb, 3, n_starts=2)["loglik"] - fit_mixture_ar1(Sb, 2, n_starts=2)["loglik"]))
null23 = np.array(null23)
reg(f"standardized K=3 vs K=2 LR={lr23:.1f} dBIC={s2['bic']-s3['bic']:.1f} null95={np.quantile(null23,.95):.1f}",
    lr23, (1 + (null23 >= lr23).sum()) / (len(null23) + 1))

# volatility vs characteristics (one joint test) and by type
import statsmodels.api as sm
Zv = sm.add_constant(pd.concat([a[["size", "liquidity", "value", "age"]], pd.get_dummies(a.sector, drop_first=True).astype(float)], axis=1))
ov = sm.OLS(np.log(sig_i), Zv).fit()
reg(f"log per-asset sigma ~ characteristics+sector (F) coefs={ov.params.round(3).to_dict()}", ov.fvalue, ov.f_pvalue)

allt = pd.concat([prev, pd.DataFrame(tests, columns=["test", "stat", "p"])], ignore_index=True)
m = len(allt); order = np.argsort(allt.p.values); holm = np.empty(m); run = 0
for rank, j in enumerate(order):
    run = max(run, min(1, (m - rank) * allt.p.values[j])); holm[j] = run
allt["p_holm"] = holm
allt.to_csv("research/artifacts/explore_tests_all.csv", index=False)
print(allt.tail(len(tests)).to_string()); print("total tests:", m)
print("Holm-adjusted for key tests:"); print(allt[allt.test.str.contains("gate on|sector|K=2 vs K=1|persistence phi|dispersion per-asset phi")].to_string())
print("elapsed %.1fs" % (time.time() - t0))
