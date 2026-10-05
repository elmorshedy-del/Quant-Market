"""run_planted_truth: planted-truth and negative controls for panel_tools.fit_mixture_ar1 (R1).

What it does: simulates panels the same shape as data/explore (n=300, T=40):
  (A) two AR(1) types, phi=(-0.5,+0.4), with type probability logistic in one covariate (slope 1.5);
  (B) one homogeneous AR(1), phi=0, so no types exist (negative control).
It fits K=1,2,3 with and without gating, and reports the recovered parameters, classification
accuracy and BIC differences.
Pass criteria: (A) phis recovered within 0.05, accuracy > 0.9, dBIC(1->2) > 10, gate slope sign
correct. (B) dBIC(1->2) < 10, meaning no spurious split.
Usage: .venv/bin/python research/tools/run_planted_truth.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from panel_tools import simulate_panel, fit_mixture_ar1

t0 = time.time()
rng = np.random.default_rng(123)
n, T = 300, 40
Z = rng.standard_normal((n, 1))
XA, typA = simulate_panel(n, T, phis=(-0.5, 0.4), cs=(0.0005, 0.0005), sigs=(0.014, 0.014),
                          gate_coef=[[0, 0], [0, 1.5]], Z=Z, rng=1)
fA = {K: fit_mixture_ar1(XA, K) for K in (1, 2, 3)}
fAg = fit_mixture_ar1(XA, 2, Z=Z)
acc = ((fA[2]["post"][:, 1] > 0.5) == (typA == 1)).mean()
print("A: phis K=2", fA[2]["phi"].round(3), "sig", fA[2]["sig"].round(4), "acc", round(acc, 3))
print("A: BIC K1,K2,K3", [round(fA[K]["bic"], 1) for K in (1, 2, 3)],
      "dBIC12", round(fA[1]["bic"] - fA[2]["bic"], 1), "dBIC23", round(fA[2]["bic"] - fA[3]["bic"], 1))
print("A: gated gate", fAg["gate"].round(2), "dBIC(gate vs nogate)", round(fA[2]["bic"] - fAg["bic"], 1))

XB, _ = simulate_panel(n, T, phis=(0.0,), cs=(0.0005,), sigs=(0.015,), rng=2)
fB = {K: fit_mixture_ar1(XB, K) for K in (1, 2)}
print("B (homogeneous): phis K=2", fB[2]["phi"].round(3), "dBIC12", round(fB[1]["bic"] - fB[2]["bic"], 1))

ok = (np.abs(fA[2]["phi"] - np.array([-0.5, 0.4])).max() < 0.05 and acc > 0.9 and
      fA[1]["bic"] - fA[2]["bic"] > 10 and fAg["gate"][1, 1] > 0 and fB[1]["bic"] - fB[2]["bic"] < 10)
print("PASS" if ok else "FAIL", "elapsed %.1fs" % (time.time() - t0))
