"""Final pre-registered test of the maker filters on v2 tables (no outcome-dependent end of sample).

Usage: python3 maker_final2.py SPECS.json [discovery|holdout|fresh ...]
Writes SPECS_final2.json and prints one block per spec.

Per spec and set (cents per contract, contract-weighted, 95% bootstrap over games):
  net_mo300     5-minute markout at the mid, minus maker fee
  net_mo60      1-minute markout at the mid, minus maker fee
  exit_bid      sell by hitting the bid 5 minutes later (maker fee in + taker fee out): realistic scalp
  net_settle    held to settlement, minus maker fee
  sweep_mo      net_mo300 on swept fills (a new order at the back of the queue); sweep_cap200 caps each
                fill at 200 contracts (a small bot cannot absorb a whale sweep)
  pos_games     share of games with positive net_mo300; med_game = median game net_mo300 (cents)
  drop10        net_mo300 without the 10 games that contribute most profit
  cap20         net_mo300 with each markout capped at +20c (limits single comeback windows)
Primary success rule (pre-registered in MAKER_PREREG.md): P1 net_mo300 95% range above zero on BOTH the
sealed holdout and the fresh sample.
"""
import json
import sys

import numpy as np
import pandas as pd

import maker_eval as E


def ci(g, val, n=2000, seed=5):
    pg = g.assign(w=val * g["count"]).groupby("event").agg(w=("w", "sum"), c=("count", "sum"))
    if len(pg) == 0:
        return (np.nan, np.nan, np.nan), pg
    idx = np.random.default_rng(seed).integers(0, len(pg), size=(n, len(pg)))
    w, c = pg["w"].to_numpy(), pg["c"].to_numpy()
    r = w[idx].sum(1) / c[idx].sum(1)
    return (float(w.sum() / c.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))), pg


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"


def score(spec, df):
    g = E.apply_filters(df, spec)
    if g.empty:
        return {"games": 0}
    mo = g["markout_300"] - g["fee"]
    r_mo, pg = ci(g, mo)
    r_60, _ = ci(g, g["markout_60"] - g["fee"])
    r_ex, _ = ci(g, g["exit_bid_300"])
    r_st, _ = ci(g, g["settle"] - g["fee"])
    sw = g[g["sweep"]]
    r_sw, _ = ci(sw, sw["markout_300"] - sw["fee"]) if len(sw) else ((np.nan,) * 3, None)
    swc = sw.assign(count=sw["count"].clip(upper=200))
    r_swc, _ = ci(swc, swc["markout_300"] - swc["fee"]) if len(sw) else ((np.nan,) * 3, None)
    per_game = pg["w"] / pg["c"]
    top = pg["w"].sort_values(ascending=False).index[:10]
    rest = g[~g["event"].isin(top)]
    r_d10, _ = ci(rest, rest["markout_300"] - rest["fee"])
    r_c20, _ = ci(g, g["markout_300"].clip(upper=20) - g["fee"])
    return {"games": int(g["event"].nunique()), "contracts_M": round(g["count"].sum() / 1e6, 2),
            "avg_c": round(float(np.average(g["c"], weights=g["count"])), 1),
            "net_mo300": fmt(r_mo), "net_mo60": fmt(r_60), "exit_bid": fmt(r_ex), "net_settle": fmt(r_st),
            "sweep_mo": fmt(r_sw), "sweep_cap200": fmt(r_swc),
            "pos_games": round(float((per_game > 0).mean()), 2), "med_game": round(float(per_game.median()), 2),
            "drop10": fmt(r_d10), "cap20": fmt(r_c20), "_mo": r_mo, "_ex": r_ex}


def main():
    path = sys.argv[1]
    periods = sys.argv[2:] or ["discovery", "holdout", "fresh"]
    specs = json.load(open(path))
    sets = {p: E.load(p) for p in periods}
    out = []
    for sp in specs:
        print(f"\n=== {sp['name']}  ({sp['role']})  {json.dumps(sp['spec'])}")
        for p, df in sets.items():
            r = score(sp["spec"], df)
            per = []
            for s in sp["spec"].get("series", E.SERIES):
                g = E.apply_filters(df, {**sp["spec"], "series": [s]})
                if len(g):
                    v = float(np.average(g["markout_300"] - g["fee"], weights=g["count"]))
                    per.append(f'{s.replace("KX", "").replace("GAME", "").replace("MATCH", "")}:{v:+.2f}')
            r["per_series_mo300"] = " ".join(per)
            print(f"  [{p:9s}] " + " | ".join(f"{k}={v}" for k, v in r.items() if not k.startswith("_")))
            out.append({"name": sp["name"], "role": sp["role"], "set": p,
                        **{k: v for k, v in r.items() if not k.startswith("_")}})
    json.dump(out, open(path.replace(".json", "_final2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
