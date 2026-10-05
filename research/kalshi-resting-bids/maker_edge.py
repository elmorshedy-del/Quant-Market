"""Maker edge on every historical trade: who rested, what they got, and why.

For each in-play trade the resting side (maker) bought a contract at price c (cents):
  taker_side "no"  -> taker sold YES -> maker bought YES at c = yes_price
  taker_side "yes" -> taker bought YES -> maker sold YES = bought NO at c = 100 - yes_price
Per contract, in cents, from the maker's side:
  markout_h   = mid(t+h) - c        mid of the maker's contract h seconds after the fill (60 s, 300 s)
  settle      = 100*W - c           held to settlement (W = maker's contract won)
  bias        = settle - markout_300   mispricing left after 5 minutes
  fee         = 0.0175 * c * (100 - c) / 100   (Kalshi maker fee, before per-order rounding)
Profitable resting environment <=> contract-weighted E[settle] > E[fee]
(equivalently E[markout] + E[bias] > E[fee]).

v2 (2026-10-05): every in-play trade from the listed start to the market close is kept. v1 cut each game at
decided_ts (first minute after which the WINNER's bid stays >= 95), which used the outcome to choose which
fills enter the sample (a cheap fill after the favourite reached 95 survived only if the game swung back).
The live, ex-ante version of that rule is the column max_bid_before (highest bid on any leg of the event in
the last closed minute); a bot can stop quoting when it is >= 95. phase is still scaled by decided_ts and is
descriptive only, never a filter.
Extra columns: max_bid_before, bid_300 (bid of the maker's contract 300 s later) and exit_bid_300 =
bid_300 - c - maker fee - taker fee at bid_300 (scalp out by hitting the bid 5 minutes later; 0 if no bid).

v2.1 (2026-10-05, after the pre-scoring audit): markouts use mark() - the mid of a two-sided book, the bid
when there is no ask, 0 when there is no bid (markout_300_mid keeps the raw mid); fills before the market's
first candle are dropped (their features came from a later candle); the sweep flag is order-free across
trades sharing a timestamp; tv10 counts strictly earlier trades; fees use the series fee multiplier (MLB 0.5).
The markout quote is the first closed minute ending at or after t + h (about h + 30 s on average).

Usage: python3 maker_edge.py SERIES [SERIES ...]   -> data/mk_edge_<SERIES>.parquet, printed summary
"""
import json
import sys

import numpy as np
import pandas as pd

import legs as L

DATA = L.DATA


FEES = json.load(open(DATA / "series_fees.json")) if (DATA / "series_fees.json").exists() else {}


def mark(bid, ask):
    """Value of a contract from its book: the mid when two-sided; the bid when there is no ask; 0 when
    there is no bid (v2.1 - a no-bid contract marked at ask/2 almost never wins)."""
    return np.where(bid <= 0, 0.0, np.where(ask >= 100, bid, (bid + ask) / 2))


def swept_flags(tt, x, long_yes):
    """Order-free sweep flag: a print of the same taker side within [t, t + 1 s], same timestamp included,
    traded at a price further through this level (trades sharing a timestamp come in arbitrary order)."""
    out = np.zeros(len(tt), bool)
    t0 = tt.max() + 10.0
    for side in (True, False):
        idx = np.flatnonzero(long_yes == side)
        if len(idx) < 2:
            continue
        # long_yes maker = taker sold YES, prices walk down (look for a lower print); otherwise up
        g = pd.Series(x[idx]).groupby(tt[idx])
        per_t = g.min() if side else g.max()                    # extreme per exact timestamp
        ut = per_t.index.to_numpy()
        ser = pd.Series(per_t.to_numpy()[::-1], index=pd.to_datetime(t0 - ut[::-1], unit="s"))
        roll = ser.rolling("1s", closed="both")                 # reversed time: covers [t, t + 1 s]
        ext = (roll.min() if side else roll.max()).to_numpy()[::-1]
        e = ext[np.searchsorted(ut, tt[idx])]
        out[idx] = e < x[idx] if side else e > x[idx]
    return out


def build(series, src_dir=None, out_path=None):
    src_dir = src_dir or DATA
    fm = FEES.get(series, {}).get("fee_multiplier", 1)
    tr = pd.read_parquet(src_dir / f"mk_trades_{series}.parquet").sort_values(["ticker", "ts_exact"], kind="stable")
    mk = pd.read_parquet(DATA / f"markets_{series}.parquet").set_index("ticker")
    allc = L.load_series(series)                    # minute candles with live flag, decided_ts, start_ts
    # ex-ante "game effectively decided" signal: highest bid on any leg of the event, per closed minute
    lb = allc[allc["live"]].groupby(["event_ticker", "ts"])["bid_c"].max().reset_index()
    lead = {e: (g["ts"].to_numpy(), g["bid_c"].to_numpy()) for e, g in lb.groupby("event_ticker")}
    cd = allc[allc["leg"].str.endswith(":yes")]
    out, dropped = [], 0
    for ticker, t in tr.groupby("ticker"):
        c = cd[cd["ticker"] == ticker].sort_values("ts")
        if c.empty or ticker not in mk.index:
            continue
        start, decided = c["start_ts"].iat[0], c["decided_ts"].iat[0]
        live = c[c["live"]]     # empty books (bid 0 and ask 100) excluded
        ts = live["ts"].to_numpy()                  # candle ts = END of the minute
        if len(ts) == 0:
            continue
        bid_y, ask_y = live["bid_c"].to_numpy(), live["ask_c"].to_numpy()
        mid = (bid_y + ask_y) / 2                   # raw mid: ex-ante features and the v2.0 markout
        spr = ask_y - bid_y
        mk_yes, mk_no = mark(bid_y, ask_y), mark(100 - ask_y, 100 - bid_y)
        t = t[t["ts_exact"] >= start]               # v2: no outcome-dependent end of sample
        # v2.1: a fill needs a closed candle before it (otherwise every feature would come from the future)
        keep = t["ts_exact"].to_numpy() >= ts[0]
        dropped += int((~keep).sum())
        t = t[keep]
        if t.empty:
            continue
        x = t["yes_price"].to_numpy().astype(float)
        tt = t["ts_exact"].to_numpy()
        long_yes = (t["taker_side"] == "no").to_numpy()
        win_yes = mk.at[ticker, "result"] == "yes"
        i_now = np.searchsorted(ts, tt, side="right") - 1          # last closed minute before the fill (>= 0)
        i_10 = np.clip(np.searchsorted(ts, tt - 600, side="right") - 1, 0, len(ts) - 1)
        i_2 = np.clip(np.searchsorted(ts, tt - 120, side="right") - 1, 0, len(ts) - 1)

        def after(h):                               # first closed minute ending at or after t + h
            return np.clip(np.searchsorted(ts, tt + h, side="left"), 0, len(ts) - 1)

        i60, i300 = after(60), after(300)
        m0, s0 = mid[i_now], spr[i_now]
        absd = np.concatenate([[0.0], np.abs(np.diff(mid))])
        cum = np.cumsum(absd)
        vol10 = cum[i_now] - cum[i_10]
        move2_yes = mid[i_now] - mid[i_2]
        cnt = t["count"].to_numpy()
        ccum = np.concatenate([[0.0], np.cumsum(cnt)])
        j_now = np.searchsorted(tt, tt, side="left")               # strictly earlier trades only
        j_10 = np.searchsorted(tt, tt - 600, side="left")
        tv10 = ccum[j_now] - ccum[j_10]
        c = np.where(long_yes, x, 100 - x)
        mark60 = np.where(long_yes, mk_yes[i60], mk_no[i60])
        mark300 = np.where(long_yes, mk_yes[i300], mk_no[i300])
        mid300 = np.where(long_yes, mid[i300], 100 - mid[i300])
        b300 = np.where(long_yes, bid_y[i300], 100 - ask_y[i300]).astype(float)
        lt, lbid = lead.get(mk.at[ticker, "event_ticker"], (ts, np.maximum(bid_y, 100 - ask_y)))
        il = np.searchsorted(lt, tt, side="right") - 1
        max_bid = np.where(il >= 0, lbid[np.clip(il, 0, len(lt) - 1)], np.nan)
        cw = np.where(long_yes, win_yes, not win_yes).astype(float)
        dur = max(decided - start, 60)
        out.append(pd.DataFrame({
            "series": series, "event": t["event"].to_numpy(), "ticker": ticker, "ts": tt,
            "count": cnt, "maker_long_yes": long_yes, "c": c,
            "mid_before": np.where(long_yes, m0, 100 - m0), "spread_before": s0,
            "markout_60": mark60 - c, "markout_300": mark300 - c, "markout_300_mid": mid300 - c,
            "settle": 100 * cw - c, "fee": fm * 0.0175 * c * (100 - c) / 100,
            "phase": (tt - start) / dur, "minute": (tt - start) / 60, "sweep": swept_flags(tt, x, long_yes),
            "vol10": vol10, "move2": np.where(long_yes, move2_yes, -move2_yes), "tv10": tv10,
            "kickoff": start, "max_bid_before": max_bid, "bid_300": b300,
        }))
    df = pd.concat(out, ignore_index=True)
    df["bias"] = df["settle"] - df["markout_300"]
    b = df["bid_300"].clip(lower=0)
    df["exit_bid_300"] = b - df["c"] - df["fee"] - fm * 0.07 * b * (100 - b) / 100
    df.attrs["dropped_before_first_candle"] = dropped
    df.to_parquet(out_path or DATA / f"mk_edge_{series}.parquet", index=False)
    print(f"  {series}: dropped {dropped} trades before the market's first candle", flush=True)
    return df


def wmean(g, col):
    return float(np.average(g[col], weights=g["count"])) if len(g) else np.nan


def cluster_ci(df, col, n=2000, seed=3):
    """Contract-weighted mean of col with a bootstrap over games."""
    pg = df.assign(w=df[col] * df["count"]).groupby("event").agg(w=("w", "sum"), c=("count", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pg), size=(n, len(pg)))
    w, cc = pg["w"].to_numpy(), pg["c"].to_numpy()
    r = w[idx].sum(1) / cc[idx].sum(1)
    return float(w.sum() / cc.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def summary(df, by):
    rows = []
    for key, g in df.groupby(by, observed=True):
        g = g.assign(net_mo=g["markout_300"] - g["fee"], net_settle=g["settle"] - g["fee"])
        mo, mlo, mhi = cluster_ci(g, "net_mo")
        st, slo, shi = cluster_ci(g, "net_settle")
        rows.append({**dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,))),
                     "games": g["event"].nunique(), "fills": len(g), "contracts": int(g["count"].sum()),
                     "mo60": round(wmean(g, "markout_60"), 2), "mo300": round(wmean(g, "markout_300"), 2),
                     "fee": round(wmean(g, "fee"), 2),
                     "net_mo300": round(mo, 2), "mo_95": f"[{mlo:+.2f},{mhi:+.2f}]",
                     "net_settle": round(st, 2), "settle_95": f"[{slo:+.2f},{shi:+.2f}]"})
    return pd.DataFrame(rows)


def load_edge(series):
    df = pd.read_parquet(DATA / f"mk_edge_{series}.parquet")
    df["price_bucket"] = pd.cut(df["c"], [0, 10, 30, 70, 90, 100], labels=["1-10", "11-30", "31-70", "71-90", "91-99"])
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for s in sys.argv[1:]:
        build(s)
        df = load_edge(s)
        print(f"\n===== {s}: {df.event.nunique()} games, {len(df)} maker fills")
        print(summary(df.assign(all="all"), ["all"]).to_string(index=False))
        print(summary(df, ["price_bucket"]).to_string(index=False))
        print(summary(df, ["sweep"]).to_string(index=False))
