"""Shared simulator for MLS end-game strategies (entry at ~82:00 match clock, trade-level fills).

    import mls_sim as S
    legs, book = S.load()                       # discovery games only (kickoff before 2026-07-01)
    res = S.simulate(spec, legs, book)          # one row per leg that traded
    print(S.summarize(res))                     # games, fills, win rate, return, 95% range (bootstrap by game)

A spec is a dict:
    {"filters": [["role", "in", ["leader"]], ["goal_diff", "==", 1], ["ask_E", "between", [70, 95]]],
     "order":   {"type": "rest", "level": {"mode": "bid_minus", "value": 2},
                "cancel_after_s": 600},                                       # optional; or {"type": "taker"}
     "exit":    {"type": "hold"},                                             # or {"type": "tp", "value": 60}
     "fill":    {"queue_share": 0.0},                                         # optional
     "contract": "yes"}                                                       # or "no" (bet against the leg)

Order level modes (resting buy price b, cents):
    abs        b = value
    bid        b = bid_E (join the best bid)
    bid_minus  b = bid_E - value
    frac_mid   b = round(mid_E * value)
A resting buy must be below ask_E at entry (otherwise skipped) and >= 1.
Fills (conservative): only trades after entry printed strictly below b fill it, up to Q contracts; trades
exactly at b add queue_share * their size. Take-profit sells fill only on trades strictly above the price.
Taker orders buy Q at ask_E. Fees: Kalshi maker 1.75% / taker 7% of C*P*(1-P), rounded up per fill.
Unfilled orders are cancelled at the end. Held contracts settle at 100 (leg wins) or 0.
contract "no" buys the NO side of the leg: its bid/ask are 100-ask_E / 100-bid_E, its trade prices are
100-yes_price, and it pays 100 when the leg does NOT win. Level modes then refer to the NO side's quotes.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
SEALED = HERE.parent / "holdout_sealed"
Q = 100
RNG = np.random.default_rng(11)

FEATURES = """
event, ticker, code, side (home/away/draw), role (leader, trailer, tied_team, draw_tied, draw_vs_lead),
kickoff, entry_ts (unix s, ~82:00 match clock), clock_fitted (bool), entry_capped (bool),
score_home, score_away, goal_diff, total_goals, goals_for, goals_against (team legs),
red_for, red_against (red cards up to entry, team legs), goals_last15 (goals by this team in the
15 min before entry; for draw legs: all goals in that span), pre_bid/pre_ask/pre_mid (this leg near
kickoff), pre_fav (1 if this team was the pre-game favourite, draw = NaN), bid_E/ask_E/mid_E (best
bid/ask at entry, cents), spread_E (ask_E - bid_E), last_E (last trade before entry), vol_10m_before, trades_after, vol_after,
late_goals / late_goal_list (OUTCOME - never use as a filter), final_home/final_away (OUTCOME),
win (OUTCOME: leg settled YES)
"""
OUTCOME_COLUMNS = {"late_goals", "late_goal_list", "final_home", "final_away", "win",
                   "trades_after", "vol_after"}


def load(period="discovery"):
    base = DATA if period == "discovery" else SEALED
    legs = pd.read_parquet(base / "mls_late_legs.parquet")
    legs["spread_E"] = legs["ask_E"] - legs["bid_E"]
    tr = pd.read_parquet(base / "mls_late_legtrades.parquet").sort_values("ts_exact")
    book = {t: (g["ts_exact"].to_numpy(), g["yes_price"].to_numpy(), g["count"].to_numpy())
            for t, g in tr.groupby("ticker")}
    return legs, book


def _fee(n, price_c, rate):
    p = price_c / 100.0
    return math.ceil(rate * n * p * (1 - p) * 100) / 100.0      # dollars


def apply_filters(legs, filters):
    m = pd.Series(True, index=legs.index)
    for feat, op, val in filters or []:
        if feat in OUTCOME_COLUMNS:
            raise ValueError(f"{feat} is an outcome column and cannot be used as a filter")
        x = legs[feat]
        if op == "==":
            m &= x == val
        elif op == "!=":
            m &= x != val
        elif op == "<":
            m &= x < val
        elif op == "<=":
            m &= x <= val
        elif op == ">":
            m &= x > val
        elif op == ">=":
            m &= x >= val
        elif op == "in":
            m &= x.isin(val)
        elif op == "between":
            m &= (x >= val[0]) & (x <= val[1])
        else:
            raise ValueError(op)
    return legs[m]


def _level(row, lv):
    mode, v = lv["mode"], lv.get("value", 0)
    if mode == "abs":
        return int(v)
    if mode == "bid":
        return int(row.bid_E)
    if mode == "bid_minus":
        return int(row.bid_E - v)
    if mode == "frac_mid":
        return int(round(row.mid_E * v))
    raise ValueError(mode)


def simulate(spec, legs, book):
    sel = apply_filters(legs, spec.get("filters"))
    order, exit_ = spec["order"], spec.get("exit", {"type": "hold"})
    qs = spec.get("fill", {}).get("queue_share", 0.0)
    no = spec.get("contract", "yes") == "no"
    if no:
        sel = sel.assign(bid_E=100 - sel["ask_E"], ask_E=100 - sel["bid_E"], mid_E=100 - sel["mid_E"],
                         win=~sel["win"].astype(bool))
    out = []
    for row in sel.itertuples():
        t, p, n = book.get(row.ticker, (np.array([]), np.array([]), np.array([])))
        if no:
            p = 100 - p
        after = t >= row.entry_ts
        execs = []
        if order["type"] == "taker":
            if not (1 <= row.ask_E <= 99):
                continue
            b, kind = int(row.ask_E), "taker"
            execs = [(row.entry_ts, Q)]
            fees = _fee(Q, b, 0.07)
        else:
            b, kind = _level(row, order["level"]), "maker"
            if b < 1 or b >= row.ask_E:
                continue
            rem = Q
            live = after & (t < row.entry_ts + order.get("cancel_after_s", 1e9))
            for i in np.flatnonzero(live & (p <= b)):
                take = n[i] if p[i] < b else n[i] * qs
                take = min(rem, take)
                if take > 0:
                    execs.append((t[i], take))
                    rem -= take
                if rem <= 0:
                    break
            fees = sum(_fee(q, b, 0.0175) for _, q in execs)
        qty = sum(q for _, q in execs)
        if qty <= 0:
            continue
        payout = 100 if row.win else 0
        sold, s = 0.0, None
        if exit_["type"] in ("tp", "tp_mult"):
            s = int(exit_["value"]) if exit_["type"] == "tp" else int(min(99, round(exit_["value"] * b)))
            if s > b:
                t_first = execs[0][0]
                rem = qty
                for i in np.flatnonzero((t > t_first) & (p > s)):
                    take = min(rem, n[i])
                    sold += take
                    fees += _fee(take, s, 0.0175)
                    rem -= take
                    if rem <= 0:
                        break
        pnl_c = sold * (s - b if s else 0) + (qty - sold) * (payout - b)
        out.append({"event": row.event, "ticker": row.ticker, "role": row.role, "kickoff": row.kickoff,
                    "contract": "no" if no else "yes",
                    "kind": kind, "price": b, "qty": qty, "sold_tp": sold, "win": row.win,
                    "pnl_usd": pnl_c / 100.0 - fees, "cost_usd": qty * b / 100.0, "fees_usd": fees})
    return pd.DataFrame(out)


def summarize(res, n_boot=4000):
    if res is None or len(res) == 0:
        return {"games": 0, "fills": 0}
    pg = res.groupby("event").agg(pnl=("pnl_usd", "sum"), cost=("cost_usd", "sum"))
    idx = RNG.integers(0, len(pg), size=(n_boot, len(pg)))
    pnl, cost = pg["pnl"].to_numpy(), pg["cost"].to_numpy()
    r = pnl[idx].sum(axis=1) / cost[idx].sum(axis=1)
    return {"games": int(len(pg)), "fills": int(len(res)), "win_rate": round(float(res["win"].mean()), 3),
            "avg_price": round(float(res["price"].mean()), 1), "avg_qty": round(float(res["qty"].mean()), 1),
            "pnl_usd": round(float(pnl.sum()), 2), "pnl_per_game": round(float(pnl.mean()), 2),
            "return": round(float(pnl.sum() / cost.sum()), 4),
            "ci95": [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]}


def evaluate(spec, period="discovery"):
    legs, book = load(period)
    return summarize(simulate(spec, legs, book))
