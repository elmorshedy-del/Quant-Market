"""Planted-truth tests for the demo generator and evaluator.

The evaluator must (a) reward a predictor that recovers the hidden split,
(b) fail the pooled "overall average" model on the split case, (c) fail an
overfitting per-asset model on the control case, and (d) fail broken predictors.
If any of these did not hold, a passing demo would mean nothing.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scientist.demo import evaluate, generator

SEEDS = [101, 202, 303]

POOLED = """
import argparse, pandas as pd
p = argparse.ArgumentParser(); p.add_argument('--assets'); p.add_argument('--history'); p.add_argument('--out'); a = p.parse_args()
ex = pd.read_csv('data/explore/returns.csv').sort_values(['asset_id','t'])
m = ex.ret.mean(); ex['lag'] = ex.groupby('asset_id').ret.shift(1); ex = ex.dropna()
phi = ((ex.lag-m)*(ex.ret-m)).sum()/((ex.lag-m)**2).sum()
h = pd.read_csv(a.history).sort_values(['asset_id','t']); last = h.groupby('asset_id').ret.last()
pd.DataFrame({'asset_id': last.index, 'prediction': m + phi*(last.values-m)}).to_csv(a.out, index=False)
"""

PER_ASSET_RAW = """
import argparse, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument('--assets'); p.add_argument('--history'); p.add_argument('--out'); a = p.parse_args()
h = pd.read_csv(a.history).sort_values(['asset_id','t'])
rows = []
for aid, g in h.groupby('asset_id'):
    r = g.ret.to_numpy(); m = r.mean(); x = r[:-1]-m; y = r[1:]-m
    phi = (x@y)/(x@x); rows.append((aid, m + phi*(r[-1]-m)))
pd.DataFrame(rows, columns=['asset_id','prediction']).to_csv(a.out, index=False)
"""

TWO_GROUP = """
import argparse, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument('--assets'); p.add_argument('--history'); p.add_argument('--out'); a = p.parse_args()
def acf(r):
    m = r.mean(); x = r[:-1]-m; y = r[1:]-m; return (x@y)/(x@x)
ex = pd.read_csv('data/explore/returns.csv').sort_values(['asset_id','t'])
coefs = ex.groupby('asset_id').ret.apply(lambda s: acf(s.to_numpy()))
pos, neg = coefs[coefs > 0].mean(), coefs[coefs <= 0].mean()
h = pd.read_csv(a.history).sort_values(['asset_id','t'])
rows = []
for aid, g in h.groupby('asset_id'):
    r = g.ret.to_numpy(); m = r.mean(); phi = pos if acf(r) > 0 else neg
    rows.append((aid, m + phi*(r[-1]-m)))
pd.DataFrame(rows, columns=['asset_id','prediction']).to_csv(a.out, index=False)
"""

CRASH = "raise SystemExit('boom')\n"


def make_workspace(tmp_path: Path, key: dict, label: str, predictor: str, claim: str) -> Path:
    ws = tmp_path / label
    (ws / "data" / "explore").mkdir(parents=True)
    explore = generator.generate_part(key, label, "explore")
    explore["assets"].to_csv(ws / "data" / "explore" / "assets.csv", index=False)
    explore["returns"].to_csv(ws / "data" / "explore" / "returns.csv", index=False)
    (ws / "research" / "predictor").mkdir(parents=True)
    (ws / "research" / "predictor" / "predict.py").write_text(textwrap.dedent(predictor))
    (ws / "research" / "conclusions.json").write_text(json.dumps({"heterogeneity": claim}))
    os.symlink(Path(__file__).resolve().parents[2] / ".venv", ws / ".venv")
    return ws


def labels_by_kind(key: dict) -> dict[str, str]:
    return {case["kind"]: label for label, case in key["cases"].items()}


@pytest.mark.parametrize("seed", SEEDS)
def test_generator_hides_split_in_the_average(seed):
    key = generator.make_answer_key(seed)
    labels = labels_by_kind(key)
    stats = {}
    for kind, label in labels.items():
        part = generator.generate_part(key, label, "explore")
        returns = part["returns"]
        _, phi_pool = evaluate.pooled_ar(returns)
        per_asset = returns.groupby("asset_id")["ret"].apply(
            lambda s: np.corrcoef(s.to_numpy()[:-1], s.to_numpy()[1:])[0, 1])
        stats[kind] = {"pooled": phi_pool, "dispersion": float(per_asset.std()), "vol": float(returns["ret"].std()),
                       "feature_sd": part["assets"][list(generator.FEATURES)].std().round(2).tolist()}
    noise_floor = 1 / np.sqrt(generator.T_PERIODS)
    assert abs(stats["split"]["pooled"]) < 0.12          # the average hides the two processes
    assert stats["split"]["dispersion"] > 2 * noise_floor  # but individuals do not
    assert stats["single"]["dispersion"] < 1.3 * noise_floor
    assert 0.85 < stats["split"]["vol"] / stats["single"]["vol"] < 1.15  # no trivial tell
    for kind in stats:  # proxy keeps unit variance, so feature scales give nothing away
        assert all(0.85 < sd < 1.15 for sd in stats[kind]["feature_sd"])


def test_answer_key_randomizes_assignment():
    kinds = {generator.make_answer_key(seed)["cases"]["dataset-1"]["kind"] for seed in range(20)}
    assert kinds == {"split", "single"}


@pytest.mark.parametrize("seed", SEEDS)
def test_evaluator_planted_truth(seed, tmp_path):
    key = generator.make_answer_key(seed)
    labels = labels_by_kind(key)
    split, single = labels["split"], labels["single"]

    def run(label, predictor, claim, sub):
        ws = make_workspace(tmp_path / sub, key, label, predictor, claim)
        return evaluate.evaluate_case(key, label, ws, sensitive=["/sealed-root-that-does-not-exist"])

    good_split = run(split, TWO_GROUP, "multiple_processes", "a")
    assert good_split["verdict"]["pass"], good_split["verdict"]
    assert good_split["scores"]["captured_gain"] > 0.7

    pooled_split = run(split, POOLED, "multiple_processes", "b")
    assert not pooled_split["verdict"]["pass"]
    assert abs(pooled_split["scores"]["captured_gain"]) < 0.15

    wrong_claim = run(split, TWO_GROUP, "single_process", "c")
    assert not wrong_claim["verdict"]["pass"] and not wrong_claim["verdict"]["claim_ok"]

    pooled_single = run(single, POOLED, "single_process", "d")
    assert pooled_single["verdict"]["pass"], pooled_single["verdict"]

    overfit_single = run(single, PER_ASSET_RAW, "single_process", "e")
    assert not overfit_single["verdict"]["pass"]
    assert overfit_single["scores"]["mse_ratio_vs_pooled"] > 1.01

    false_split = run(single, TWO_GROUP, "multiple_processes", "f")
    assert not false_split["verdict"]["pass"]

    crashed = run(split, CRASH, "multiple_processes", "g")
    assert not crashed["verdict"]["pass"] and crashed["predictor_errors"]


def test_leakage_audit_detects_canary(tmp_path):
    key = generator.make_answer_key(5)
    label = next(iter(key["cases"]))
    ws = make_workspace(tmp_path, key, label, POOLED, "single_process")
    clean = evaluate.leakage_audit(ws, canary=key["canary"], sensitive=["/secret/root"])
    assert not clean["leak_detected"]
    (ws / "research" / "notes.md").write_text(f"I found {key['canary']}")
    leaked = evaluate.leakage_audit(ws, canary=key["canary"], sensitive=["/secret/root"])
    assert leaked["leak_detected"] and leaked["canary_found_in"] == ["research/notes.md"]
