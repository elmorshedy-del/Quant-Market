"""Soccer kickoff estimate = Kalshi expected_expiration_time - 3h (checked against milestones)."""
import sys
import pandas as pd
import download as dl

def expected_times(series):
    rows = {}
    for path, extra in (("/markets", {"status": "settled"}), ("/historical/markets", {})):
        cursor = None
        while True:
            p = dict(series_ticker=series, limit=1000, **extra)
            if cursor: p["cursor"] = cursor
            d = dl.get(path, **p) or {}
            for m in d.get("markets", []):
                rows[m["event_ticker"]] = dl.ts(m.get("expected_expiration_time") or m.get("occurrence_datetime"))
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"): break
    return pd.Series(rows, name="expected_ts")

if __name__ == "__main__":
    dl.LIMIT = dl.RateLimiter(3)
    mode, series = sys.argv[1], sys.argv[2:]
    for s in series:
        e = expected_times(s)
        est = (e - 3 * 3600).rename("start_est")
        if mode == "check":
            g = pd.read_parquet(dl.DATA / f"games_{s}.parquet").set_index("event_ticker")
            diff = ((est.reindex(g.index) - g["start_ts"]) / 60).dropna()
            print(s, f"n={len(diff)} exact={(diff == 0).mean():.1%} within5min={(diff.abs() <= 5).mean():.1%} "
                     f"median diff={diff.median():.0f} min p5/p95={diff.quantile(.05):.0f}/{diff.quantile(.95):.0f}")
        else:
            mk = pd.read_parquet(dl.DATA / f"markets_{s}.parquet")
            g = mk.groupby("event_ticker").agg(close_ts=("close_ts", "max"), archived=("archived", "first")).reset_index()
            g["start_ts"] = g["event_ticker"].map(est)
            g["end_ts"], g["milestone_type"], g["series"] = None, "estimated", s
            g.to_parquet(dl.DATA / f"games_{s}.parquet", index=False)
            print(s, f"{len(g)} games, {g.start_ts.notna().mean():.0%} with kickoff estimate")
