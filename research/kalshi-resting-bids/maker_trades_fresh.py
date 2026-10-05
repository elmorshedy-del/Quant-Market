"""Fresh, never-used sample of 2026 games per series (final out-of-sample test for the maker-edge study).

Usage: python3 maker_trades_fresh.py N SERIES [SERIES ...] -> ../holdout_sealed/fresh/mk_trades_<SERIES>.parquet
Excludes every game already present in ../holdout_sealed/maker/mk_trades_<SERIES>.parquet.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import download as dl
from maker_trades import leg_trades, FROM

OUT = Path(__file__).parent.parent / "holdout_sealed" / "fresh"
USED = Path(__file__).parent.parent / "holdout_sealed" / "maker"


def main():
    dl.LIMIT = dl.RateLimiter(14)
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[1])
    for series in sys.argv[2:]:
        t0 = time.time()
        used = set(pd.read_parquet(USED / f"mk_trades_{series}.parquet", columns=["event"])["event"].unique())
        mk = pd.read_parquet(dl.DATA / f"markets_{series}.parquet")
        gm = pd.read_parquet(dl.DATA / f"games_{series}.parquet")[["event_ticker", "start_ts"]]
        mk = mk.merge(gm, on="event_ticker").dropna(subset=["start_ts"])
        mk = mk[(mk["start_ts"] >= FROM) & ~mk["event_ticker"].isin(used)]
        events = mk["event_ticker"].drop_duplicates()
        events = events.sample(min(n, len(events)), random_state=99)
        sel = mk[mk["event_ticker"].isin(events)]
        with ThreadPoolExecutor(6) as pool:
            res = list(pool.map(leg_trades, sel.itertuples()))
        tr = pd.DataFrame([r for leg in res for r in leg],
                          columns=["event", "ticker", "ts_exact", "yes_price", "count", "taker_side"])
        tr.to_parquet(OUT / f"mk_trades_{series}.parquet", index=False)
        print(f"{series}: {len(events)} fresh games, {len(tr)} trades, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
