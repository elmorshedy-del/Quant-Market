"""Final pre-registered test of the maker filters (v2.1 tables; see MAKER_PREREG.md and its addendum).

Usage: python3 maker_final2.py SPECS.json discovery          # discovery only (allowed any time)
       python3 maker_final2.py SPECS.json holdout fresh      # THE scoring run (once)
Writes SPECS_final2_<sets>.json and prints one block per spec.

Per spec and set (cents per contract, 95% game-bootstrap ranges, 2,000 draws):
  net_mo300   markout to the corrected mark (two-sided mid; bid if no ask; 0 if no bid) at the first closed
              minute >= 300 s after the fill, minus maker fee. PRIMARY METRIC (contract-weighted).
  mo300_mid   same at the raw mid (the v2.0 definition; one-sided books at (bid+ask)/2)
  net_mo60    1-minute version of net_mo300
  exit_bid    sell 5 minutes later by hitting the bid (maker fee in, taker fee out)
  net_settle  held to settlement
  fillw10     each fill's count capped at 10 (small fixed-size bot at the front of the queue)
  gamew       equal weight per game (the typical game)
  sweep_mo / sweep_cap200   swept fills only (order-free flag); cap200 caps each fill at 200 contracts
  bot_Q*_w*   back-of-queue bot: Q contracts at the first print of each (ticker, side, price, minute) cell,
              filled only by later prints strictly through the price within w seconds; markout from the
              first through print (only for specs inside c <= 20)
  pos_games / med_game / cap20 / drop10 / trim1   with "null" = the same statistic after shifting every
              contract's net markout so the set's mean is exactly zero (what zero edge looks like)
  lb_bonf     lower bound of the one-sided 99.6875% range (Bonferroni over 8 secondary specs)
"""
import json
import sys

import numpy as np
import pandas as pd

import maker_eval as E

EXPECTED = {"discovery": 10, "holdout": 10, "fresh": 9}
BOT_Q, BOT_W = (100, 500), (60, 300)


def boot(g, val, n=2000, seed=5, extra=()):
    pg = g.assign(w=val * g["count"]).groupby("event").agg(w=("w", "sum"), c=("count", "sum"))
    if len(pg) == 0:
        return (np.nan,) * (3 + len(extra)), pg
    idx = np.random.default_rng(seed).integers(0, len(pg), size=(n, len(pg)))
    w, c = pg["w"].to_numpy(), pg["c"].to_numpy()
    r = w[idx].sum(1) / c[idx].sum(1)
    q = [2.5, 97.5, *extra]
    return (float(w.sum() / c.sum()), *[float(np.percentile(r, x)) for x in q]), pg


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]" if np.isfinite(t[0]) else "n/a"


def shape_stats(g, net):
    """pos_games, med_game, cap20, drop10, trim1 for a vector of per-contract net markouts."""
    pg = g.assign(w=net * g["count"]).groupby("event").agg(w=("w", "sum"), c=("count", "sum"))
    per = pg["w"] / pg["c"]
    order = pg["w"].sort_values(ascending=False).index
    k = max(1, int(round(0.01 * len(pg))))
    d10 = pg.drop(order[:10])
    tr = pg.drop(list(order[:k]) + list(order[-k:]))
    gross = net + g["fee"]
    cap = (gross.clip(upper=20) - g["fee"]) * g["count"]
    return {"pos_games": float((per > 0).mean()), "med_game": float(per.median()),
            "cap20": float(cap.sum() / g["count"].sum()),
            "drop10": float(d10["w"].sum() / d10["c"].sum()), "trim1": float(tr["w"].sum() / tr["c"].sum())}


def bot_columns(df, max_c=20):
    """For every fill at c <= max_c: through-volume and the first through print's mark/bid within w s."""
    res = {}
    mark = (df["markout_300"] + df["c"]).to_numpy()
    bid = df["bid_300"].to_numpy()
    for w in BOT_W:
        res[w] = (np.zeros(len(df)), np.full(len(df), np.nan), np.full(len(df), np.nan))
    for (tk, side), t in df.groupby(["ticker", "maker_long_yes"], sort=False):
        t = t.sort_values("ts", kind="stable")
        ts, c, n = t["ts"].to_numpy(), t["c"].to_numpy(), t["count"].to_numpy()
        pos = t.index.to_numpy()
        cand = np.flatnonzero(c <= max_c)
        if not len(cand):
            continue
        j0 = np.searchsorted(ts, ts, side="left")
        for w in BOT_W:
            je = np.searchsorted(ts, ts + w, side="right")
            vol, mk_, bd = res[w]
            for a in cand:
                seg = slice(j0[a], je[a])
                m = c[seg] < c[a]
                if m.any():
                    k = j0[a] + int(np.argmax(m))
                    vol[pos[a]] = n[seg][m].sum()
                    mk_[pos[a]], bd[pos[a]] = mark[pos[k]], bid[pos[k]]
    for w in BOT_W:
        df[f"bvol{w}"], df[f"bmark{w}"], df[f"bbid{w}"] = res[w]
    return df


def bot_score(g):
    out = {}
    if "bvol60" not in g or g["c"].max() > 20:
        return out
    cell = (g["ticker"] + "|" + g["maker_long_yes"].astype(str) + "|" + g["c"].astype(int).astype(str) + "|"
            + (g["ts"] // 60).astype(int).astype(str))
    first = g.assign(cell=cell).sort_values("ts", kind="stable").groupby("cell").head(1)
    for w in BOT_W:
        f = first[first[f"bvol{w}"] > 0]
        for Q in BOT_Q:
            b = f.assign(count=np.minimum(Q, f[f"bvol{w}"]))
            r, _ = boot(b, b[f"bmark{w}"] - b["c"] - b["fee"])
            bb = b[f"bbid{w}"].clip(lower=0)
            fm = b["fee"] / (0.0175 * b["c"] * (100 - b["c"]) / 100)
            ex, _ = boot(b, bb - b["c"] - b["fee"] - fm * 0.07 * bb * (100 - bb) / 100)
            out[f"bot_Q{Q}_w{w}"] = f"{fmt(r)} exit {ex[0]:+.2f} ({b['count'].sum() / 1e6:.2f}M ct)"
    return out


def score(spec, df):
    g = E.apply_filters(df, spec)
    if g.empty:
        return {"games": 0}
    net = g["markout_300"] - g["fee"]
    r_mo, pg = boot(g, net, extra=(0.3125,))
    r_mid, _ = boot(g, g["markout_300_mid"] - g["fee"])
    r_60, _ = boot(g, g["markout_60"] - g["fee"])
    r_ex, _ = boot(g, g["exit_bid_300"])
    r_st, _ = boot(g, g["settle"] - g["fee"])
    g10 = g.assign(count=g["count"].clip(upper=10))
    r_f10, _ = boot(g10, g10["markout_300"] - g10["fee"])
    per = pg["w"] / pg["c"]
    sw = g[g["sweep"]]
    r_sw, _ = boot(sw, sw["markout_300"] - sw["fee"])
    swc = sw.assign(count=sw["count"].clip(upper=200))
    r_swc, _ = boot(swc, swc["markout_300"] - swc["fee"])
    sh = shape_stats(g, net)
    null = shape_stats(g, net - r_mo[0])
    out = {"games": int(g["event"].nunique()), "contracts_M": round(g["count"].sum() / 1e6, 2),
           "avg_c": round(float(np.average(g["c"], weights=g["count"])), 1),
           "net_mo300": fmt(r_mo), "lb_bonf": round(r_mo[3], 2), "mo300_mid": fmt(r_mid), "net_mo60": fmt(r_60),
           "exit_bid": fmt(r_ex), "net_settle": fmt(r_st), "fillw10": fmt(r_f10),
           "gamew": f"{per.mean():+.2f}", "sweep_mo": fmt(r_sw), "sweep_cap200": fmt(r_swc)}
    for k in sh:
        out[k] = f"{sh[k]:+.2f} (null {null[k]:+.2f})"
    out.update(bot_score(g))
    out["_mo"], out["_ex"] = r_mo, r_ex
    return out


def main():
    path, periods = sys.argv[1], sys.argv[2:]
    specs = json.load(open(path))
    sets = {}
    for p in periods:
        df = E.load(p)
        ns = df["series"].nunique()
        assert ns == EXPECTED[p], f"{p}: {ns} series loaded, expected {EXPECTED[p]}"
        sets[p] = bot_columns(df.reset_index(drop=True))
        print(f"{p}: {df['event'].nunique()} games, {ns} series, {len(df):,} fills", flush=True)
    if {"holdout", "fresh"} <= set(sets):
        sets["holdout_exNFL"] = sets["holdout"][sets["holdout"]["series"] != "KXNFLGAME"]
        sets["pooled_hold+fresh"] = pd.concat([sets["holdout"], sets["fresh"]], ignore_index=True)
    out = []
    for sp in specs:
        print(f"\n=== {sp['name']}  ({sp['role']})  {json.dumps(sp['spec'])}", flush=True)
        for p, df in sets.items():
            r = score(sp["spec"], df)
            per, share = [], []
            g = E.apply_filters(df, sp["spec"])
            tot = g["count"].sum()
            for s in sp["spec"].get("series", E.SERIES):
                gs = g[g["series"] == s]
                if len(gs):
                    v = float(np.average(gs["markout_300"] - gs["fee"], weights=gs["count"]))
                    nm = s.replace("KX", "").replace("GAME", "").replace("MATCH", "")
                    per.append(f"{nm}:{v:+.2f}")
                    share.append(f"{nm}:{gs['count'].sum() / tot:.0%}")
            r["per_series_mo300"], r["contract_share"] = " ".join(per), " ".join(share)
            print(f"  [{p}]\n    " + "\n    ".join(f"{k} = {v}" for k, v in r.items() if not k.startswith("_")),
                  flush=True)
            out.append({"name": sp["name"], "role": sp["role"], "set": p,
                        **{k: v for k, v in r.items() if not k.startswith("_")}})
        if sp["role"] == "primary":
            for p, df in sets.items():
                g = E.apply_filters(df, sp["spec"])
                q = pd.to_datetime(g["ts"], unit="s").dt.to_period("Q").astype(str)
                rows = [f"{k}: {np.average(x['markout_300'] - x['fee'], weights=x['count']):+.2f} "
                        f"({x['event'].nunique()} g)" for k, x in g.groupby(q)]
                print(f"  [{p}] by quarter: " + " | ".join(rows))
    json.dump(out, open(path.replace(".json", f"_final2_{'_'.join(periods)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
