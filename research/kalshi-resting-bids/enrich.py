"""Add kickoff times (Kalshi milestones) and series fee schedules to downloaded markets.

Usage: python3 enrich.py SERIES [SERIES ...]
Output: data/games_<SERIES>.parquet (one row per event), data/series_fees.json
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import download as dl


def milestone(event_ticker):
    d = dl.get("/milestones", limit=5, related_event_ticker=event_ticker) or {}
    ms = d.get("milestones", [])
    if not ms:
        return event_ticker, None, None, None
    m = ms[0]
    return event_ticker, dl.ts(m.get("start_date")), dl.ts(m.get("end_date")), m.get("type")


def series_fee(series):
    d = dl.get(f"/series/{series}") or {}
    s = d.get("series", {})
    return {k: s.get(k) for k in ("fee_type", "fee_multiplier", "title")}


def main():
    dl.LIMIT = dl.RateLimiter(float(os.environ.get("RATE", 15)))
    fees_path = dl.DATA / "series_fees.json"
    fees = json.loads(fees_path.read_text()) if fees_path.exists() else {}
    for series in sys.argv[1:]:
        t0 = time.time()
        mk = pd.read_parquet(dl.DATA / f"markets_{series}.parquet")
        events = sorted(mk.event_ticker.unique())
        with ThreadPoolExecutor(6) as pool:
            rows = list(pool.map(milestone, events))
        gm = pd.DataFrame(rows, columns=["event_ticker", "start_ts", "end_ts", "milestone_type"])
        first = mk.groupby("event_ticker").agg(close_ts=("close_ts", "max"), archived=("archived", "first"))
        gm = gm.merge(first, on="event_ticker", how="left")
        gm["series"] = series
        gm.to_parquet(dl.DATA / f"games_{series}.parquet", index=False)
        fees[series] = series_fee(series)
        print(f"{series}: {len(gm)} games, {gm.start_ts.notna().mean():.0%} with kickoff, "
              f"fees={fees[series]}, {time.time() - t0:.0f}s", flush=True)
    fees_path.write_text(json.dumps(fees, indent=1))


if __name__ == "__main__":
    main()
