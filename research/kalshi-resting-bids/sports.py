"""Apply the soccer-derived rules, unchanged, to other sports (out-of-sample by sport).

Strategy A: resting bids at 3/5/7/10c placed at the start of play on every leg whose best bid is
above the level, held to settlement (maker fees, 100 contracts per order).
Strategy B: buy at the ask right after a >= 10c / 20c one-minute jump (taker), hold or exit at 5 min.
Winning fills are also shown at 91% / 96% of size: in the soccer trade sample, winning legs filled
91 of 100 contracts on average from trades strictly below the bid (96 counting trades at the bid).

Usage: python3 sports.py SERIES [SERIES ...]
"""
import sys

import numpy as np
import pandas as pd

import study


def period_table(r, pnl_col="pnl_usd", cost_col="cost_usd", haircuts=(1.0, 0.964, 0.91)):
    rows = []
    for f in haircuts:
        rr = r.copy()
        if f != 1.0:
            rr.loc[rr["win"], pnl_col] *= f
            rr.loc[rr["win"], cost_col] *= f
        for name, a, b in study.PERIODS + [("all", "2000-01-01", "2100-01-01")]:
            g = rr[(rr["kickoff"] >= pd.Timestamp(a).timestamp()) & (rr["kickoff"] < pd.Timestamp(b).timestamp())]
            if g.empty:
                continue
            pg = g.groupby("event").agg(pnl=(pnl_col, "sum"), cost=(cost_col, "sum"))
            lo, hi = study.ratio_ci(pg["pnl"], pg["cost"])
            rows.append({"winner_fill": f"{f:.0%}", "period": name, "games": len(pg), "fills": len(g),
                         "return": f"{pg['pnl'].sum() / pg['cost'].sum():+.0%}", "95%": f"[{lo:+.0%},{hi:+.0%}]",
                         "pnl/game $": round(pg["pnl"].mean(), 2)})
    return pd.DataFrame(rows)


def main(series_list):
    for s in series_list:
        print(f"\n=================== {s}")
        r = study.run_window([s], start=0, end=None)
        r.to_parquet(study.L.DATA / f"broad_{s}.parquet", index=False)
        print("A. resting bids 3-10c from start of play, hold to settlement")
        print(period_table(r).to_string(index=False))
        by_b = r.groupby("b").apply(lambda g: f"{g['pnl_usd'].sum() / g['cost_usd'].sum():+.0%}", include_groups=False)
        print("   by bid level (full fills):", by_b.to_dict())
        print("B. buy after a jump (taker)")
        study.shock([s])


if __name__ == "__main__":
    main(sys.argv[1:])
