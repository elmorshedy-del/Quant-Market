"""Evaluate resting-order filters on the maker-edge data (every historical trade = a maker fill).

    import maker_eval as E
    df = E.load("discovery")                    # first 60% of games per series (by kickoff); holdout is sealed
    E.evaluate({"series": ["KXATPMATCH"], "filters": [["c", "<=", 10], ["phase", "between", [0.5, 1.0]]]}, df)

Filters may only use what a bot knows when it places the order:
    c              price paid for the contract (cents; the maker's side: YES price, or 100 - YES price for NO)
    mid_before     mid of that contract in the last minute before the fill (cents)
    depth          mid_before - c (how far below the mid the order rests; <0 means above mid)
    spread_before  ask - bid in the last minute before the fill (cents)
    max_bid_before highest bid on any leg of the event in the last closed minute (>= 95: game ~decided)
    minute         minutes since the listed start
    hour_utc, weekday   time of the fill
    vol10          sum of |mid changes| over the previous 10 minutes (news intensity proxy, cents)
    move2          change of the contract's mid over the last 2 minutes (cents; <0 = it just fell)
    tv10           contracts traded in this market over the previous 10 minutes (liquidity/crowding)
Outcome-side columns (markout_*, settle, bias, sweep, count, bid_300, exit_bid_300) are never filters, and
neither is phase (scaled by decided_ts, which uses the outcome).
Version 2 tables (VERSION = 2) have no outcome-dependent end of sample; see maker_edge.py.
Scores (cents per contract, contract-weighted, 95% bootstrap over games):
    net_mo300  = 5-minute markout - fee   (low-noise microstructure edge)
    net_settle = settlement P&L - fee     (what holding to settlement paid; noisy)
    sweep_mo   = net_mo300 on swept fills only (price traded through the level: conservative for a
                 new order at the back of the queue)
Profitable resting environment <=> net edge > 0 with the range above zero.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
SEALED = Path(__file__).parent.parent / "holdout_sealed" / "maker"
FRESH = SEALED.parent / "fresh"
VERSION = 2
SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXNBAGAME", "KXNHLGAME", "KXMLBGAME", "KXWNBAGAME", "KXNFLGAME",
          "KXEPLGAME", "KXLALIGAGAME", "KXMLSGAME"]
ALLOWED = {"series", "c", "mid_before", "depth", "spread_before", "max_bid_before", "minute", "hour_utc",
           "weekday", "vol10", "move2", "tv10"}


def load(period="discovery", series=None):
    parts = []
    for s in series or SERIES:
        v = "mk_edge2" if VERSION == 2 else "mk_edge"
        p = {"discovery": DATA / f"{v}_disc_{s}.parquet", "holdout": SEALED / f"{v}_hold_{s}.parquet",
             "fresh": FRESH / f"{v}_fresh_{s}.parquet"}[period]
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df["depth"] = df["mid_before"] - df["c"]
    dt = pd.to_datetime(df["ts"], unit="s")
    df["hour_utc"], df["weekday"] = dt.dt.hour, dt.dt.weekday
    return df


def apply_filters(df, spec):
    m = df["series"].isin(spec.get("series", SERIES))
    for feat, op, val in spec.get("filters", []):
        if feat not in ALLOWED:
            raise ValueError(f"{feat} is not available when placing an order")
        x = df[feat]
        ops = {"==": lambda: x == val, "!=": lambda: x != val, "<": lambda: x < val, "<=": lambda: x <= val,
               ">": lambda: x > val, ">=": lambda: x >= val, "in": lambda: x.isin(val),
               "between": lambda: (x >= val[0]) & (x <= val[1])}
        m &= ops[op]()          # lazy: only the chosen comparison is evaluated
    return df[m]


def _ci(g, col, n=2000, seed=5):
    pg = g.assign(w=g[col] * g["count"]).groupby("event").agg(w=("w", "sum"), c=("count", "sum"))
    if len(pg) == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(pg), size=(n, len(pg)))
    w, c = pg["w"].to_numpy(), pg["c"].to_numpy()
    r = w[idx].sum(1) / c[idx].sum(1)
    return float(w.sum() / c.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def evaluate(spec, df):
    g = apply_filters(df, spec)
    if g.empty:
        return {"games": 0}
    g = g.assign(net_mo=g["markout_300"] - g["fee"], net_settle=g["settle"] - g["fee"])
    mo = _ci(g, "net_mo")
    st = _ci(g, "net_settle")
    sw = _ci(g[g["sweep"]], "net_mo") if g["sweep"].any() else (np.nan,) * 3
    return {"games": int(g["event"].nunique()), "fills": int(len(g)), "contracts": int(g["count"].sum()),
            "net_mo300": round(mo[0], 2), "mo_95": [round(mo[1], 2), round(mo[2], 2)],
            "net_settle": round(st[0], 2), "settle_95": [round(st[1], 2), round(st[2], 2)],
            "sweep_mo": round(sw[0], 2), "sweep_95": [round(sw[1], 2), round(sw[2], 2)]}
