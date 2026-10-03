"""MLS end-game dataset: true match events (goals, cards) and late-game trades.

Usage: python3 mls_data.py events   -> data/mls_events.parquet, data/mls_games.parquet
       python3 mls_data.py trades   -> data/mls_late_trades.parquet  (wall minute >= 88 after kickoff)
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import download as dl

SERIES = "KXMLSGAME"
CLOCK = re.compile(r"(\d+)(?:\+(\d+))?")


def parse_minute(text):
    m = CLOCK.match(text or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2) or 0)


def game_events(event_ticker):
    ms = (dl.get("/milestones", limit=5, related_event_ticker=event_ticker) or {}).get("milestones", [])
    if not ms:
        return event_ticker, None, []
    m = ms[0]
    d = dl.get("/live_data/batch", milestone_ids=m["id"]) or {}
    ld = (d.get("live_datas") or [{}])[0]
    det = ld.get("details") or {}
    info = {
        "event_ticker": event_ticker, "milestone_id": m["id"], "start_ts": dl.ts(m.get("start_date")),
        "home_score": det.get("home_same_game_score"), "away_score": det.get("away_same_game_score"),
        "winner": det.get("winner"), "status": det.get("status"), "status_text": det.get("status_text"),
        "ht_home": None, "ht_away": None,
    }
    for p in det.get("period_scores") or []:
        if p.get("number") == 1:
            info["ht_home"], info["ht_away"] = p.get("home_score"), p.get("away_score")
    events = []
    for side in ("home", "away"):
        for e in det.get(f"{side}_significant_events") or []:
            minute, extra = parse_minute(e.get("time"))
            events.append({"event_ticker": event_ticker, "side": side, "type": e.get("event_type"),
                           "minute": minute, "extra": extra, "time_text": e.get("time"),
                           "player": e.get("player")})
    return event_ticker, info, events


def events_main():
    mk = pd.read_parquet(dl.DATA / f"markets_{SERIES}.parquet")
    evs = sorted(mk.event_ticker.unique())
    t0 = time.time()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(game_events, evs))
    games = pd.DataFrame([r[1] for r in res if r[1]])
    events = pd.DataFrame([e for r in res for e in r[2]])
    games.to_parquet(dl.DATA / "mls_games.parquet", index=False)
    events.to_parquet(dl.DATA / "mls_events.parquet", index=False)
    print(f"{len(games)}/{len(evs)} games with live data, {len(events)} events, "
          f"types={events['type'].value_counts().to_dict()}, {time.time() - t0:.0f}s")


def leg_trades(row):
    path = "/historical/trades" if row.archived else "/markets/trades"
    out, cursor = [], None
    lo, hi = int(row.start_ts + 88 * 60), int(row.close_ts + 300)
    while True:
        params = dict(ticker=row.ticker, min_ts=lo, max_ts=hi, limit=1000)
        if cursor:
            params["cursor"] = cursor
        d = dl.get(path, **params) or {}
        for t in d.get("trades", []):
            out.append((row.event_ticker, row.ticker, dl.ts(t["created_time"]),
                        float(pd.Timestamp(t["created_time"]).value / 1e9),
                        round(float(t["yes_price_dollars"]) * 100), float(t["count_fp"]),
                        t.get("taker_side"), bool(t.get("is_block_trade"))))
        cursor = d.get("cursor")
        if not cursor or not d.get("trades"):
            return out


def trades_main():
    mk = pd.read_parquet(dl.DATA / f"markets_{SERIES}.parquet")
    gm = pd.read_parquet(dl.DATA / "mls_games.parquet")[["event_ticker", "start_ts"]]
    mk = mk.merge(gm, on="event_ticker").dropna(subset=["start_ts"])
    t0 = time.time()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(leg_trades, mk.itertuples()))
    tr = pd.DataFrame([r for leg in res for r in leg],
                      columns=["event_ticker", "ticker", "ts", "ts_exact", "yes_price", "count", "taker_side", "block"])
    tr.to_parquet(dl.DATA / "mls_late_trades.parquet", index=False)
    print(f"{mk.ticker.nunique()} legs, {len(tr)} trades, {tr['count'].sum():.0f} contracts, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    dl.LIMIT = dl.RateLimiter(12)
    {"events": events_main, "trades": trades_main}[sys.argv[1]]()
