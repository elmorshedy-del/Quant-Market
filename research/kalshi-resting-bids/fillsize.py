"""Compare candle-based fills (assumes all 100 contracts) with fills sized from real trades."""
import math
import numpy as np
import pandas as pd
import study

Q = 100
tr = pd.read_parquet("data/trades_sample.parquet")
tr = tr[~tr["block"]].sort_values("ts")
bf = pd.read_parquet("data/broad_fills.parquet")
bf["ticker"] = bf["leg"].str.replace(":yes", "", regex=False)
bf = bf[bf["ticker"].isin(tr["ticker"].unique())].copy()

def fee_exec(n, b):
    p = b / 100.0
    return math.ceil(0.0175 * n * p * (1 - p) * 100) / 100.0

by_ticker = {k: g for k, g in tr.groupby("ticker")}
rows = []
for r in bf.itertuples():
    g = by_ticker[r.ticker]
    g = g[g["ts"] > r.kickoff]
    below = g[g["yes_price"] < r.b]
    at = g[g["yes_price"] == r.b]
    qty, fees, remaining = 0.0, 0.0, Q
    for n in below["count"].to_numpy():
        take = min(remaining, n)
        qty += take; fees += fee_exec(take, r.b); remaining -= take
        if remaining <= 0:
            break
    payoff = 1.0 if r.win else 0.0
    rows.append({"event": r.event, "kickoff": r.kickoff, "b": r.b, "win": r.win,
                 "qty": qty, "vol_below": below["count"].sum(), "vol_at": at["count"].sum(),
                 "pnl_candle": r.pnl_usd, "cost_candle": r.cost_usd,
                 "pnl_trades": qty * (payoff - r.b / 100.0) - fees, "cost_trades": qty * r.b / 100.0})
x = pd.DataFrame(rows)
x["period"] = pd.cut(pd.to_datetime(x["kickoff"], unit="s"),
                     [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01"), pd.Timestamp("2027-01-01")],
                     labels=["2025", "2026 H1", "2026 Jul-Sep"])
print("fill size from real trades, share of candle fills:")
x["size_bucket"] = pd.cut(x["qty"], [-1, 0, 9, 49, 99, 100], labels=["0 (quote-only)", "1-9", "10-49", "50-99", "100 (full)"])
print((x.groupby("period", observed=True)["size_bucket"].value_counts(normalize=True).unstack().round(2)).to_string())
print()
rows = []
for per, g in x.groupby("period", observed=True):
    pg = g.groupby("event").agg(pc=("pnl_candle", "sum"), cc=("cost_candle", "sum"), pt=("pnl_trades", "sum"), ct=("cost_trades", "sum"))
    lo_c, hi_c = study.ratio_ci(pg.pc, pg.cc)
    lo_t, hi_t = study.ratio_ci(pg.pt, pg.ct)
    rows.append({"period": per, "games": len(pg), "fills": len(g),
                 "candle_return": f"{pg.pc.sum()/pg.cc.sum():+.0%} [{lo_c:+.0%},{hi_c:+.0%}]",
                 "trade_sized_return": f"{pg.pt.sum()/pg.ct.sum():+.0%} [{lo_t:+.0%},{hi_t:+.0%}]",
                 "avg_contracts_filled": round(g["qty"].mean(), 1),
                 "trade_sized_pnl/game_$": round(pg.pt.mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nmedian contracts traded below the bid price, per filled order:", x.groupby("period", observed=True)["vol_below"].median().to_dict())

# --- winners vs losers: how full are the fills? (strictly below b, and including trades at b)
def qty_upto(g, b, inclusive):
    sel = g[g["yes_price"] <= b] if inclusive else g[g["yes_price"] < b]
    return min(Q, sel["count"].sum())
extra = []
for r in bf.itertuples():
    g = by_ticker[r.ticker]
    g = g[g["ts"] > r.kickoff]
    extra.append((qty_upto(g, r.b, False), qty_upto(g, r.b, True)))
x["qty_strict"], x["qty_incl_at_b"] = zip(*extra)
print("\nmean contracts filled (of 100):")
print(x.groupby("win")[["qty_strict", "qty_incl_at_b"]].mean().round(1).rename(index={False: "losing legs", True: "winning legs"}).to_string())
print("\nshare of winning fills that were partial (<100 strictly below):",
      f"{(x[x.win]['qty_strict'] < 100).mean():.0%}", " | losing:", f"{(x[~x.win]['qty_strict'] < 100).mean():.0%}")
print("winning fills with zero trades strictly below the bid:", int((x[x.win]['qty_strict'] == 0).sum()), "of", int(x.win.sum()))
