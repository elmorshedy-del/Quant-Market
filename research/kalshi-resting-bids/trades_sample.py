"""Download in-play trades for a stratified sample of soccer games (fill-size validation).

Usage: RATE=4 python3 trades_sample.py [games_per_period]
Output: data/trades_sample.parquet
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import download as dl

LEAGUES = ["KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME"]
PERIODS = [("2025-05-01", "2025-12-31"), ("2026-01-01", "2026-06-30"), ("2026-07-01", "2026-12-31")]


def leg_trades(row):
    path = "/historical/trades" if row.archived else "/markets/trades"
    out, cursor = [], None
    while True:
        params = dict(ticker=row.ticker, min_ts=int(row.start_ts), max_ts=int(row.close_ts) + 300, limit=1000)
        if cursor:
            params["cursor"] = cursor
        d = dl.get(path, **params) or {}
        for t in d.get("trades", []):
            out.append((row.event_ticker, row.ticker, dl.ts(t["created_time"]),
                        round(float(t["yes_price_dollars"]) * 100), float(t["count_fp"]),
                        t.get("taker_side"), bool(t.get("is_block_trade"))))
        cursor = d.get("cursor")
        if not cursor or not d.get("trades"):
            return out


def main():
    dl.LIMIT = dl.RateLimiter(float(os.environ.get("RATE", 4)))
    per_period = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = []
    for lg in LEAGUES:
        mk = pd.read_parquet(dl.DATA / f"markets_{lg}.parquet")
        gm = pd.read_parquet(dl.DATA / f"games_{lg}.parquet")[["event_ticker", "start_ts"]]
        rows.append(mk.merge(gm, on="event_ticker"))
    mk = pd.concat(rows).dropna(subset=["start_ts"])
    mk["date"] = pd.to_datetime(mk["start_ts"], unit="s")
    events = mk.drop_duplicates("event_ticker")
    chosen = []
    for a, b in PERIODS:
        ev = events[(events["date"] >= a) & (events["date"] <= b)]
        chosen += list(ev.sample(min(per_period, len(ev)), random_state=1)["event_ticker"])
    sel = mk[mk["event_ticker"].isin(chosen)]
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(leg_trades, sel.itertuples()))
    tr = pd.DataFrame([r for leg in res for r in leg],
                      columns=["event_ticker", "ticker", "ts", "yes_price", "count", "taker_side", "block"])
    tr.to_parquet(dl.DATA / "trades_sample.parquet", index=False)
    print(f"{len(chosen)} games, {sel['ticker'].nunique()} legs, {len(tr)} trades, "
          f"{tr['block'].sum()} block trades, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
