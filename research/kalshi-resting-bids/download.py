"""Download settled Kalshi game markets and their 1-minute bid/ask candles.

Read-only public API. Recent markets come from /markets and /series/.../candlesticks;
markets settled before the archive cutoff come from /historical/....

Usage: python3 download.py SERIES [SERIES ...] [--limit N] [--window-hours H]
Output: data/markets_<SERIES>.parquet, data/candles_<SERIES>.parquet, data/failures.log
"""
import argparse
import datetime as dt
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://api.elections.kalshi.com/trade-api/v2"
_CA = Path("/root/.ccr/ca-bundle.crt")  # this sandbox's proxy CA; elsewhere use system certs
CTX = ssl.create_default_context(cafile=str(_CA)) if _CA.exists() else ssl.create_default_context()
DATA = Path(__file__).parent / "data"
TWO_OUTCOME = {"KXATPMATCH", "KXWTAMATCH", "KXMLBGAME", "KXNBAGAME", "KXNHLGAME",
               "KXNFLGAME", "KXWNBAGAME", "KXNCAAFGAME"}


class RateLimiter:
    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval


LIMIT = RateLimiter(15)


def get(path, **params):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(6):
        LIMIT.wait()
        try:
            with urllib.request.urlopen(url, context=CTX, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(30, 2 ** attempt))
        except Exception:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"failed after retries: {url}")


def ts(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()) if s else None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def list_markets(series):
    rows = {}
    for path, extra in (("/markets", {"status": "settled"}), ("/historical/markets", {})):
        cursor = None
        while True:
            params = dict(series_ticker=series, limit=1000, **extra)
            if cursor:
                params["cursor"] = cursor
            d = get(path, **params) or {}
            for m in d.get("markets", []):
                rows[m["ticker"]] = {
                    "series": series,
                    "ticker": m["ticker"],
                    "event_ticker": m.get("event_ticker"),
                    "open_ts": ts(m.get("open_time")),
                    "close_ts": ts(m.get("close_time")),
                    "result": m.get("result"),
                    "volume": num(m.get("volume_fp", m.get("volume"))),
                    "archived": path.startswith("/historical"),
                }
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    return pd.DataFrame(rows.values())


def cents(block, key):
    if not block:
        return None
    v = block.get(key + "_dollars", block.get(key))
    v = num(v)
    return None if v is None else round(v * 100)


def fetch_candles(row, window_s):
    end = int(row.close_ts) + 300
    start = int(row.close_ts) - window_s
    if row.archived:
        path = f"/historical/markets/{row.ticker}/candlesticks"
    else:
        path = f"/series/{row.series}/markets/{row.ticker}/candlesticks"
    d = get(path, start_ts=start, end_ts=end, period_interval=1)
    if d is None:
        return None
    out = []
    for c in d.get("candlesticks", []):
        yb, ya, pr = c.get("yes_bid"), c.get("yes_ask"), c.get("price")
        out.append((
            row.ticker, c["end_period_ts"],
            cents(yb, "open"), cents(yb, "high"), cents(yb, "low"), cents(yb, "close"),
            cents(ya, "open"), cents(ya, "high"), cents(ya, "low"), cents(ya, "close"),
            cents(pr, "high"), cents(pr, "low"), cents(pr, "close"),
            num(c.get("volume_fp", c.get("volume"))),
        ))
    return out


COLS = ["ticker", "ts", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c",
        "px_h", "px_l", "px_c", "volume"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="max events per series (0 = all)")
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    fail_log = open(DATA / "failures.log", "a")

    for series in args.series:
        t0 = time.time()
        mk = list_markets(series)
        if mk.empty:
            print(f"{series}: no markets", flush=True)
            continue
        mk = mk[mk.result.isin(["yes", "no"])].sort_values(["close_ts", "ticker"])
        if series in TWO_OUTCOME:
            # One market per game is enough: its NO side is the other player/team.
            mk = mk.groupby("event_ticker", as_index=False).first()
        if args.limit:
            keep = mk.event_ticker.drop_duplicates().head(args.limit)
            mk = mk[mk.event_ticker.isin(keep)]
        mk.to_parquet(DATA / f"markets_{series}.parquet", index=False)

        rows, n_fail = [], 0
        with ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(fetch_candles, r, int(args.window_hours * 3600)): r.ticker
                    for r in mk.itertuples()}
            for f in as_completed(futs):
                try:
                    res = f.result()
                except Exception as e:
                    res, err = None, repr(e)
                else:
                    err = "not found"
                if res is None:
                    n_fail += 1
                    fail_log.write(f"{series}\t{futs[f]}\t{err}\n")
                else:
                    rows.extend(res)
        cd = pd.DataFrame(rows, columns=COLS)
        cd.to_parquet(DATA / f"candles_{series}.parquet", index=False)
        print(f"{series}: {mk.event_ticker.nunique()} games, {len(mk)} markets, "
              f"{len(cd)} candles, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)
    fail_log.close()


if __name__ == "__main__":
    sys.exit(main())
