"""Download in-play trades for a sample of 2026 games per series (maker-edge study).

Usage: python3 maker_trades.py N_PER_SERIES SERIES [SERIES ...]
Output: data/mk_trades_<SERIES>.parquet (event, ticker, ts_exact, yes_price, count, taker_side)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import download as dl

FROM = pd.Timestamp("2026-01-01").timestamp()


def leg_trades(row):
    path = "/historical/trades" if row.archived else "/markets/trades"
    out, cursor = [], None
    while True:
        params = dict(ticker=row.ticker, min_ts=int(row.start_ts), max_ts=int(row.close_ts) + 120, limit=1000)
        if cursor:
            params["cursor"] = cursor
        d = dl.get(path, **params) or {}
        for t in d.get("trades", []):
            if t.get("is_block_trade"):
                continue
            out.append((row.event_ticker, row.ticker, pd.Timestamp(t["created_time"]).value / 1e9,
                        round(float(t["yes_price_dollars"]) * 100), float(t["count_fp"]), t.get("taker_side")))
        cursor = d.get("cursor")
        if not cursor or not d.get("trades"):
            return out


def main():
    dl.LIMIT = dl.RateLimiter(14)
    n = int(sys.argv[1])
    for series in sys.argv[2:]:
        t0 = time.time()
        mk = pd.read_parquet(dl.DATA / f"markets_{series}.parquet")
        gp = dl.DATA / f"games_{series}.parquet"
        gm = pd.read_parquet(gp)[["event_ticker", "start_ts"]]
        mk = mk.merge(gm, on="event_ticker").dropna(subset=["start_ts"])
        mk = mk[mk["start_ts"] >= FROM]
        events = mk["event_ticker"].drop_duplicates().sample(min(n, mk["event_ticker"].nunique()), random_state=7)
        sel = mk[mk["event_ticker"].isin(events)]
        with ThreadPoolExecutor(6) as pool:
            res = list(pool.map(leg_trades, sel.itertuples()))
        tr = pd.DataFrame([r for leg in res for r in leg],
                          columns=["event", "ticker", "ts_exact", "yes_price", "count", "taker_side"])
        tr.to_parquet(dl.DATA / f"mk_trades_{series}.parquet", index=False)
        print(f"{series}: {len(events)} games, {sel.ticker.nunique()} markets, {len(tr)} trades, "
              f"{tr['count'].sum():.0f} contracts, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
