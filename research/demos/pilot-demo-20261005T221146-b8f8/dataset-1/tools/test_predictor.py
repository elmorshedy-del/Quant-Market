"""test_predictor: acceptance tests for research/predictor/predict.py (ledger R6).

1. CLI run on explore assets, history t<=39 -> predicts t=40. Checks schema, row count and finiteness.
   This is in-sample, because the params include explore.
2. Equivalence check: predict() with the FROZEN explore params (age+size gate) on confirmation
   targets t=21..40 must reproduce the R5 mix_gated MSE.
3. Synthetic panel, 1,000 unseen assets, h=60, drawn from the fitted model: CLI timing < 60 s, and
   MSE compared with the oracle (true type known), pooled and zero forecasts.
4. A file-modification check: hashes of every file under research/ and data/ before and after the CLI runs.
Usage: .venv/bin/python research/tools/test_predictor.py
"""
import sys, os, json, time, hashlib, subprocess, glob
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "research/predictor")
import numpy as np, pandas as pd
from panel_tools import load_panel, simulate_panel
from predict import predict

TD = "research/artifacts/predictor_test"; os.makedirs(TD, exist_ok=True)
def hashes():
    fs = [f for f in glob.glob("research/**/*", recursive=True) + glob.glob("data/**/*", recursive=True)
          if os.path.isfile(f) and not f.startswith(TD) and "__pycache__" not in f]
    return {f: hashlib.md5(open(f, "rb").read()).hexdigest() for f in fs}
def cli(a, h, o):
    t = time.time()
    subprocess.run([".venv/bin/python", "research/predictor/predict.py", "--assets", a, "--history", h, "--out", o], check=True)
    return time.time() - t

H0 = hashes()
# 1
r = pd.read_csv("data/explore/returns.csv"); r[r.t <= 39].to_csv(f"{TD}/hist39.csv", index=False)
dt1 = cli("data/explore/assets.csv", f"{TD}/hist39.csv", f"{TD}/out39.csv")
o = pd.read_csv(f"{TD}/out39.csv"); y = r[r.t == 40].set_index("asset_id").ret.loc[o.asset_id].values
ok1 = list(o.columns) == ["asset_id", "prediction"] and len(o) == 300 and np.isfinite(o.prediction).all()
print(f"[1] explore t=40 (in-sample params): rows={len(o)} cols={list(o.columns)} finite={np.isfinite(o.prediction).all()} "
      f"MSE={np.mean((y-o.prediction)**2):.4e} zeroMSE={np.mean(y**2):.4e} time={dt1:.2f}s {'OK' if ok1 else 'FAIL'}")
# 2
fz = json.load(open("research/artifacts/explore_fit.json"))
ac, Xc, ids = load_panel("data/confirmation")
se = []
for t in range(20, 40):
    hist = pd.DataFrame({"asset_id": np.repeat(ids, t), "t": np.tile(np.arange(1, t + 1), len(ids)), "ret": Xc[:, :t].ravel()})
    P = dict(c=fz["c"], phi=fz["phi"], sig=fz["sig"], gate=[[g[0], g[2]] for g in fz["gate"]], age_fill=0.0)
    # the frozen gate has a size term, which this age-only predict() ignores, so emulate it via an adjusted intercept
    pr = []
    for i, aid in enumerate(ids):
        g = np.array(fz["gate"]); Pi = dict(P); Pi["gate"] = [[g[0, 0] + g[0, 1] * ac["size"][i], g[0, 2]], [g[1, 0] + g[1, 1] * ac["size"][i], g[1, 2]]]
        pr.append(predict(ac.iloc[[i]], hist[hist.asset_id == aid], Pi).prediction.values[0])
    se.append((Xc[:, t] - np.array(pr)) ** 2)
mse2 = float(np.mean(se)); R5 = json.load(open("research/artifacts/confirm_results.json"))["P7"]["mse"]["mix_gated"]
print(f"[2] frozen-explore params on confirmation t=21..40: MSE={mse2:.6e} vs R5 {R5:.6e} {'OK' if abs(mse2-R5)/R5 < 1e-6 else 'MISMATCH'}")
# 3
Pf = json.load(open("research/predictor/params.json"))
rng = np.random.default_rng(11); n, h = 1000, 60
age = rng.standard_normal(n)
X, typ = simulate_panel(n, h + 1, Pf["phi"], Pf["c"], Pf["sig"], gate_coef=Pf["gate"], Z=age[:, None], rng=12)
assets = pd.DataFrame({"asset_id": [f"S{i:04d}" for i in range(n)], "sector": rng.choice(list("ABCDEF"), n),
                       "size": rng.standard_normal(n), "liquidity": rng.standard_normal(n), "value": rng.standard_normal(n), "age": age})
assets.to_csv(f"{TD}/syn_assets.csv", index=False)
pd.DataFrame({"asset_id": np.repeat(assets.asset_id, h), "t": np.tile(np.arange(1, h + 1), n), "ret": X[:, :h].ravel()}).to_csv(f"{TD}/syn_hist.csv", index=False)
dt3 = cli(f"{TD}/syn_assets.csv", f"{TD}/syn_hist.csv", f"{TD}/syn_out.csv")
o3 = pd.read_csv(f"{TD}/syn_out.csv"); yt = X[:, h]
c, phi = np.array(Pf["c"]), np.array(Pf["phi"])
orc = c[typ] + phi[typ] * X[:, h - 1]
print(f"[3] synthetic 1000 assets h=60: time={dt3:.2f}s rows={len(o3)} MSE={np.mean((yt-o3.prediction)**2):.4e} oracle={np.mean((yt-orc)**2):.4e} "
      f"zero={np.mean(yt**2):.4e} corr(pred,oracle)={np.corrcoef(o3.prediction, orc)[0,1]:.3f} {'OK' if dt3 < 60 and len(o3)==n else 'FAIL'}")
# 4
H1 = hashes()
print(f"[4] files modified by CLI runs: {[f for f in H0 if H0[f] != H1.get(f)]} new files outside test dir: {sorted(set(H1)-set(H0))}")
