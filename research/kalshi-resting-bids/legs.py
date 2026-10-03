"""Turn downloaded minute candles into per-leg price paths with a game clock.

A "leg" is one outcome you can buy: each soccer market (home / draw / away), or each side
(YES and NO) of a two-outcome market. All prices are in cents from that leg's point of view:
ask = cost to buy it now, bid = what you could sell it for, px = trade prices.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
SOCCER = {"KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME",
          "KXMLSGAME", "KXUCLGAME", "KXLIGAMXGAME", "KXBRASILEIROGAME"}
QUOTES = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]


def minute_grid(c):
    """Reindex one market's candles to every minute; a missing minute means nothing changed."""
    c = c.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    full = np.arange(c.index[0], c.index[-1] + 60, 60)
    c = c.reindex(full)
    for side in ("bid", "ask"):
        c[f"{side}_c"] = c[f"{side}_c"].ffill()
        for k in ("o", "h", "l"):
            c[f"{side}_{k}"] = c[f"{side}_{k}"].fillna(c[f"{side}_c"])
    c["volume"] = c["volume"].fillna(0.0)
    c["ticker"] = c["ticker"].ffill()
    return c.rename_axis("ts").reset_index()


def flip(c):
    """The NO side of a market seen as its own leg."""
    f = c[["ticker", "ts", "volume"]].copy()
    f["bid_o"], f["bid_c"] = 100 - c["ask_o"], 100 - c["ask_c"]
    f["bid_h"], f["bid_l"] = 100 - c["ask_l"], 100 - c["ask_h"]
    f["ask_o"], f["ask_c"] = 100 - c["bid_o"], 100 - c["bid_c"]
    f["ask_h"], f["ask_l"] = 100 - c["bid_l"], 100 - c["bid_h"]
    f["px_h"], f["px_l"], f["px_c"] = 100 - c["px_l"], 100 - c["px_h"], 100 - c["px_c"]
    return f


def load_series(series):
    mk = pd.read_parquet(DATA / f"markets_{series}.parquet")
    cd = pd.read_parquet(DATA / f"candles_{series}.parquet")
    gp = DATA / f"games_{series}.parquet"
    games = pd.read_parquet(gp) if gp.exists() else pd.DataFrame(columns=["event_ticker", "start_ts"])
    meta = mk.set_index("ticker")
    legs = []
    for ticker, c in cd.groupby("ticker", sort=False):
        if ticker not in meta.index or len(c) < 5:
            continue
        m = meta.loc[ticker]
        g = minute_grid(c)
        sides = [("yes", g)] if series in SOCCER else [("yes", g), ("no", flip(g))]
        for side, frame in sides:
            frame = frame.copy()
            frame["leg"] = f"{ticker}:{side}"
            frame["event_ticker"] = m.event_ticker
            frame["win"] = (m.result == side)
            legs.append(frame)
    df = pd.concat(legs, ignore_index=True)
    df["series"] = series
    df = df.merge(games[["event_ticker", "start_ts"]], on="event_ticker", how="left")
    df["t_min"] = (df["ts"] - df["start_ts"]) / 60.0
    df = add_decided(df)
    return df


def add_decided(df):
    """decided_ts: first minute from which the winning leg's bid stays >= 95 to the end.

    Minutes with an empty book (bid 0 and ask 100, e.g. after the market closes) are ignored.
    """
    df["live"] = ~((df["bid_c"] <= 0) & (df["ask_c"] >= 100))
    w = df[df["win"] & df["live"]][["event_ticker", "ts", "bid_c"]].sort_values(["event_ticker", "ts"])
    w["below"] = (w["bid_c"] < 95).astype(int)
    # last minute that was still below 95; decided is the minute after it
    last_below = w[w["below"] == 1].groupby("event_ticker")["ts"].max()
    first_ts = w.groupby("event_ticker")["ts"].min()
    decided = (last_below + 60).reindex(first_ts.index).fillna(first_ts)
    return df.merge(decided.rename("decided_ts"), left_on="event_ticker", right_index=True, how="left")


def match_minute(t_min):
    """Approximate soccer match minute from wall minutes since kickoff (~18 min for half time)."""
    t = np.asarray(t_min, dtype=float)
    return np.where(t < 48, t, np.where(t < 63, np.nan, t - 18))
