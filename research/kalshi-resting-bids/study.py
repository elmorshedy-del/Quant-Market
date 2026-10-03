"""Research tables for the cheap-leg resting-order idea.

overview    per sport: how often legs get cheap in play, how often they come back, liquidity
calibrate   when a resting bid at b would have been filled in play, how often did the leg win
grid        soccer strategy grid: start minute x bid distance x exit x fill rule, P&L per game

Usage: python3 study.py overview|calibrate|grid SERIES [SERIES ...]
"""
import math
import sys

import numpy as np
import pandas as pd

import legs as L

Q = 100                 # contracts per resting order
MAKER_RATE = 0.0175     # Kalshi maker fee rate (quadratic_with_maker_fees series)
RNG = np.random.default_rng(7)


def maker_fee_per_contract(price_c, q=Q, mult=1.0):
    """Kalshi rounds the fee for an order up to the next cent."""
    p = price_c / 100.0
    return math.ceil(MAKER_RATE * mult * q * p * (1 - p) * 100) / q   # cents per contract


def inplay(df):
    return df[df["live"] & (df["t_min"] >= 0) & (df["ts"] < df["decided_ts"])]


def boot_ci(values, n=4000):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (np.nan, np.nan)
    means = RNG.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ---------------------------------------------------------------- overview
def overview(series_list):
    rows = []
    for s in series_list:
        df = L.load_series(s)
        live = df[df["live"] & (df["t_min"] >= 0)]
        per_leg = []
        for leg, d in live.groupby("leg", sort=False):
            d = d.sort_values("ts")
            inp = (d["ts"] < d["decided_ts"]).to_numpy()
            ask_l, bid_h = d["ask_l"].to_numpy(), d["bid_h"].to_numpy()
            cheap_vol = d.loc[inp & (d["px_c"] <= 10).to_numpy(), "volume"].sum()
            touch = np.flatnonzero(inp & (ask_l <= 10))
            ev, win = d["event_ticker"].iat[0], bool(d["win"].iat[0])
            if len(touch) == 0:
                per_leg.append((ev, False, False, win, np.nan, cheap_vol))
                continue
            back = bool((bid_h[touch[0] + 1:] >= 30).any())
            per_leg.append((ev, True, back, win, ask_l[touch[0]], cheap_vol))
        pl = pd.DataFrame(per_leg, columns=["event", "cheap", "back30", "win", "first_ask", "cheap_vol"])
        games = pl["event"].nunique()
        cheap = pl[pl["cheap"]]
        rows.append({
            "series": s, "games": games,
            "games_with_cheap_leg": f"{pl.groupby('event')['cheap'].any().mean():.0%}",
            "cheap_legs": len(cheap),
            "came_back_to_30c": f"{cheap['back30'].mean():.1%}",
            "won": f"{cheap['win'].mean():.1%}",
            "median_cheap_volume_per_game": int(pl.groupby("event")["cheap_vol"].sum().median()),
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- calibration
def phase_label(series, t_min):
    if series in L.SOCCER:
        m = float(L.match_minute(t_min))
        if np.isnan(m):
            return "HT"
        return "<45" if m < 45 else "45-60" if m < 60 else "60-75" if m < 75 else "75-85" if m < 85 else "85+"
    return "all"


LEVELS = (2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 80, 90, 95, 97)


def calibrate(series_list, levels=LEVELS):
    """A resting bid at b, placed in play while the leg's best bid is above b, held to settlement."""
    out = []
    for s in series_list:
        df = L.load_series(s)
        ip = inplay(df)
        for leg, d in ip.groupby("leg", sort=False):
            d = d.sort_values("ts")
            ev, win, k0 = d["event_ticker"].iat[0], bool(d["win"].iat[0]), d["start_ts"].iat[0]
            ask_l, px_l, bid_c = d["ask_l"].to_numpy(), d["px_l"].to_numpy(), d["bid_c"].to_numpy()
            t = d["t_min"].to_numpy()
            for b in levels:
                above = np.flatnonzero(bid_c > b)
                if len(above) == 0:
                    continue
                p = above[0]                                    # placement minute
                for rule, hit in (("either", (ask_l <= b) | (px_l <= b - 1)), ("through", px_l <= b - 1)):
                    idx = np.flatnonzero(hit[p + 1:])
                    if len(idx):
                        f = p + 1 + idx[0]
                        out.append((s, ev, k0, leg, b, rule, phase_label(s, t[f]), win))
    c = pd.DataFrame(out, columns=["series", "event", "kickoff", "leg", "b", "rule", "phase", "win"])
    tag = "_".join(sorted({x.replace("KX", "").replace("GAME", "").replace("MATCH", "") for x in series_list}))
    c.to_parquet(L.DATA / f"calibration_{tag[:60]}.parquet", index=False)
    recent = c["kickoff"] >= pd.Timestamp("2026-01-01").timestamp()
    print("ALL GAMES")
    print(calibration_table(c).to_string(index=False))
    print("\nGAMES SINCE 2026-01-01 (liquid era)")
    print(calibration_table(c[recent]).to_string(index=False))
    return c


def ratio_ci(pnl, cost, n=4000):
    """Bootstrap over games for total P&L / total cost."""
    pnl, cost = np.asarray(pnl, float), np.asarray(cost, float)
    if len(pnl) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(pnl), size=(n, len(pnl)))
    r = pnl[idx].sum(axis=1) / cost[idx].sum(axis=1)
    return tuple(np.percentile(r, [2.5, 97.5]))


def calibration_table(c, by=("rule", "b")):
    rows = []
    for key, g in c.groupby(list(by)):
        key = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        b = key["b"]
        fee = maker_fee_per_contract(b)
        pg = g.groupby("event").agg(wins=("win", "sum"), n=("win", "size"))
        pnl = pg["wins"] * 100 - pg["n"] * (b + fee)
        cost = pg["n"] * (b + fee)
        lo, hi = ratio_ci(pnl, cost)
        rows.append({**key, "fills": len(g), "games": len(pg),
                     "win_rate": f"{g['win'].mean():.1%}", "breakeven": f"{(b + fee) / 100:.1%}",
                     "return_per_$": f"{pnl.sum() / cost.sum():+.0%}", "95%_range": f"[{lo:+.0%}, {hi:+.0%}]"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- strategy grid (soccer)
STARTS = (0, 60, 75, 85)                    # approx match minute when bids are placed
LEVELS_GRID = (3, 5, 7, 10)                 # resting bid price in cents
EXITS = ("hold", "tp50", "tp5x")            # hold to settlement, or rest a sell at 50c / 5x the bid
RULES = ("either", "through")


def wall_minute(match_min):
    return match_min if match_min < 45 else match_min + 18


def grid(series_list):
    recs = []
    for s in series_list:
        if s not in L.SOCCER:
            continue
        df = L.load_series(s)
        df = df[df["live"] & df["start_ts"].notna()]
        for leg, d in df.groupby("leg", sort=False):
            d = d.sort_values("ts")
            ev, win, k0 = d["event_ticker"].iat[0], bool(d["win"].iat[0]), d["start_ts"].iat[0]
            t = d["t_min"].to_numpy()
            bid_c, ask_l, bid_h = d["bid_c"].to_numpy(), d["ask_l"].to_numpy(), d["bid_h"].to_numpy()
            px_l, px_h = d["px_l"].to_numpy(), d["px_h"].to_numpy()
            for start in STARTS:
                before = np.flatnonzero(t <= wall_minute(start))
                if len(before) == 0 or t[before[-1]] < wall_minute(start) - 3:
                    continue
                i0 = before[-1]
                for b in LEVELS_GRID:
                    if bid_c[i0] <= b:                  # must rest below the market when placed
                        continue
                    fee_in = maker_fee_per_contract(b)
                    for rule in RULES:
                        hit = (px_l <= b - 1) if rule == "through" else ((ask_l <= b) | (px_l <= b - 1))
                        hit[: i0 + 1] = False
                        fills = np.flatnonzero(hit)
                        if len(fills) == 0:
                            continue
                        f = fills[0]
                        for ex in EXITS:
                            pnl, how = (100 if win else 0) - b - fee_in, "settle"
                            if ex != "hold":
                                tp = 50 if ex == "tp50" else min(95, 5 * b)
                                if rule == "through":
                                    out_hit = px_h >= tp + 1
                                else:
                                    out_hit = (bid_h >= tp) | (px_h >= tp + 1)
                                out_hit[: f + 1] = False
                                if out_hit.any():
                                    pnl, how = tp - b - fee_in - maker_fee_per_contract(tp), "tp"
                            recs.append((s, ev, k0, leg, start, b, rule, ex, how, win,
                                         phase_label(s, t[f]), pnl * Q / 100.0, b * Q / 100.0))
    r = pd.DataFrame(recs, columns=["series", "event", "kickoff", "leg", "start", "b", "rule", "exit",
                                    "how", "win", "fill_phase", "pnl_usd", "cost_usd"])
    r.to_parquet(L.DATA / "grid_fills.parquet", index=False)
    summarize_grid(r)
    return r


def summarize_grid(r):
    cut = r.drop_duplicates("event").sort_values("kickoff")["kickoff"].quantile(0.6)
    rows = []
    for key, g in r.groupby(["start", "b", "exit", "rule"]):
        pg = g.groupby("event").agg(pnl=("pnl_usd", "sum"), cost=("cost_usd", "sum"), k=("kickoff", "first"))
        lo, hi = ratio_ci(pg["pnl"], pg["cost"])
        early, late = pg[pg["k"] <= cut], pg[pg["k"] > cut]
        llo, lhi = ratio_ci(late["pnl"], late["cost"])
        rows.append({"start": key[0], "bid_c": key[1], "exit": key[2], "fill": key[3],
                     "games": len(pg), "fills": len(g), "pnl/game": round(pg["pnl"].mean(), 2),
                     "return": f"{pg['pnl'].sum() / pg['cost'].sum():+.0%}", "95%": f"[{lo:+.0%},{hi:+.0%}]",
                     "early_ret": f"{early['pnl'].sum() / early['cost'].sum():+.0%}",
                     "late_ret": f"{late['pnl'].sum() / late['cost'].sum():+.0%}",
                     "late_95%": f"[{llo:+.0%},{lhi:+.0%}]"})
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- pre-registered confirmation test
# Fixed before looking at the confirmation leagues (2026-10-03): bids placed at ~75', cancelled at
# ~85', levels 3/5/7/10c, either-fill rule, held to settlement, maker fees, 100 contracts per order.
DISCOVERY = {"KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME"}
PERIODS = [("2025", "2025-01-01", "2026-01-01"), ("2026 H1", "2026-01-01", "2026-07-01"),
           ("2026 Jul-Sep", "2026-07-01", "2027-01-01")]


def run_window(series_list, start=75, end=85, levels=(3, 5, 7, 10), rule="either"):
    recs = []
    for s in series_list:
        df = L.load_series(s)
        df = df[df["live"] & df["start_ts"].notna()]
        for leg, d in df.groupby("leg", sort=False):
            d = d.sort_values("ts")
            ev, win, k0 = d["event_ticker"].iat[0], bool(d["win"].iat[0]), d["start_ts"].iat[0]
            t, bid_c = d["t_min"].to_numpy(), d["bid_c"].to_numpy()
            ask_l, px_l = d["ask_l"].to_numpy(), d["px_l"].to_numpy()
            before = np.flatnonzero(t <= wall_minute(start))
            if len(before) == 0 or t[before[-1]] < wall_minute(start) - 3:
                continue
            i0 = before[-1]
            open_until = t < wall_minute(end) if end is not None else np.ones(len(t), bool)
            for b in levels:
                if bid_c[i0] <= b:
                    continue
                hit = (px_l <= b - 1) if rule == "through" else ((ask_l <= b) | (px_l <= b - 1))
                hit &= open_until
                hit[: i0 + 1] = False
                if hit.any():
                    pnl = (100 if win else 0) - b - maker_fee_per_contract(b)
                    recs.append((s, ev, k0, leg, b, win, pnl * Q / 100.0, b * Q / 100.0))
    return pd.DataFrame(recs, columns=["series", "event", "kickoff", "leg", "b", "win", "pnl_usd", "cost_usd"])


def confirm(series_list):
    r = run_window(series_list)
    r.to_parquet(L.DATA / "confirm_fills.parquet", index=False)
    r["set"] = np.where(r["series"].isin(DISCOVERY), "discovery (EPL/LaLiga/SerieA)", "confirmation (6 other leagues)")
    rows = []
    for st, g0 in r.groupby("set"):
        for name, a, b in PERIODS + [("all", "2000-01-01", "2100-01-01")]:
            g = g0[(g0["kickoff"] >= pd.Timestamp(a).timestamp()) & (g0["kickoff"] < pd.Timestamp(b).timestamp())]
            if g.empty:
                continue
            pg = g.groupby("event").agg(pnl=("pnl_usd", "sum"), cost=("cost_usd", "sum"))
            lo, hi = ratio_ci(pg["pnl"], pg["cost"])
            rows.append({"set": st, "period": name, "games": len(pg), "fills": len(g),
                         "win%": round(100 * g["win"].mean(), 1), "pnl/game": round(pg["pnl"].mean(), 2),
                         "return": f"{pg['pnl'].sum() / pg['cost'].sum():+.0%}", "95%": f"[{lo:+.0%},{hi:+.0%}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    return r


# ---------------------------------------------------------------- idea B: follow the shock (taker)
TAKER_RATE = 0.07


def taker_fee_per_contract(price_c, q=Q):
    p = price_c / 100.0
    return math.ceil(TAKER_RATE * q * p * (1 - p) * 100) / q


def shock(series_list, jumps=(10, 20), max_entry=90):
    """After a leg's mid price jumps by >= J cents in one minute, buy at the ask and either hold
    to settlement or sell at the bid 5 minutes later. 'fast' buys at the end of the jump minute,
    'slow' one minute later (a realistic reaction delay)."""
    recs = []
    for s in series_list:
        df = L.load_series(s)
        df = df[df["live"] & df["start_ts"].notna()]
        for leg, d in df.groupby("leg", sort=False):
            d = d.sort_values("ts")
            inp = ((d["t_min"] >= 0) & (d["ts"] < d["decided_ts"])).to_numpy()
            ev, win, k0 = d["event_ticker"].iat[0], bool(d["win"].iat[0]), d["start_ts"].iat[0]
            ask, bid = d["ask_c"].to_numpy(), d["bid_c"].to_numpy()
            mid, t = (ask + bid) / 2.0, d["t_min"].to_numpy()
            for J in jumps:
                last = -10
                for i in np.flatnonzero(inp[1:] & (np.diff(mid) >= J)) + 1:
                    if i - last < 5:
                        continue
                    last = i
                    for speed, e in (("fast", i), ("slow", i + 1)):
                        if e >= len(ask) or not (1 <= ask[e] <= max_entry):
                            continue
                        fee = taker_fee_per_contract(ask[e])
                        hold = (100 if win else 0) - ask[e] - fee
                        x = min(e + 5, len(bid) - 1)
                        flip5 = bid[x] - ask[e] - fee - taker_fee_per_contract(max(bid[x], 1))
                        recs.append((s, ev, k0, leg, J, speed, phase_label(s, t[i]), ask[e], win,
                                     hold * Q / 100, flip5 * Q / 100, ask[e] * Q / 100))
    r = pd.DataFrame(recs, columns=["series", "event", "kickoff", "leg", "J", "speed", "phase", "entry",
                                    "win", "pnl_hold", "pnl_5min", "cost"])
    r.to_parquet(L.DATA / "shock_trades.parquet", index=False)
    rows = []
    for (J, speed), g in r.groupby(["J", "speed"]):
        for exit_col in ("pnl_hold", "pnl_5min"):
            pg = g.groupby("event").agg(pnl=(exit_col, "sum"), cost=("cost", "sum"))
            lo, hi = ratio_ci(pg["pnl"], pg["cost"])
            rows.append({"jump>=": J, "entry": speed, "exit": exit_col.replace("pnl_", ""),
                         "trades": len(g), "games": len(pg), "pnl/trade": round(g[exit_col].mean(), 2),
                         "return": f"{pg['pnl'].sum() / pg['cost'].sum():+.1%}",
                         "95%": f"[{lo:+.1%},{hi:+.1%}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    return r


if __name__ == "__main__":
    cmd, series = sys.argv[1], sys.argv[2:]
    {"overview": overview, "calibrate": calibrate, "grid": grid, "shock": shock, "confirm": confirm}[cmd](series)
