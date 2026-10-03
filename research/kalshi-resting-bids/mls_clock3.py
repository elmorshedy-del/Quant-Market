"""Per-game second-half clock offset: one offset that explains the most goal->price jumps.

offset O means: displayed match clock c (minutes, 2nd half) happens at wall time start + (c + O) minutes,
with the goal itself somewhere inside displayed minute c (i.e. clock c-1 .. c).
"""
import numpy as np
import pandas as pd
import legs as L
from mls_state import team_codes

DATA = L.DATA

def jumps(tr_leg, min_jump=8):
    """Sustained price jumps in one leg: list of (time, direction)."""
    t, p = tr_leg.ts_exact.to_numpy(), tr_leg.yes_price.to_numpy().astype(float)
    out, last = [], -1e18
    for i in range(len(t)):
        if t[i] - last < 60:
            continue
        prior = p[(t >= t[i] - 150) & (t < t[i] - 10)]
        after = p[(t > t[i]) & (t <= t[i] + 60)]
        if len(prior) == 0 or len(after) == 0:
            continue
        d = p[i] - np.median(prior)
        if abs(d) >= min_jump and np.sign(np.median(after) - np.median(prior)) == np.sign(d) \
                and abs(np.median(after) - np.median(prior)) >= 0.75 * min_jump:
            out.append((t[i], int(np.sign(d))))
            last = t[i]
    return out

def main():
    tr = pd.read_parquet(DATA / "mls_late_trades.parquet")
    tr = tr[~tr.block].sort_values("ts_exact")
    tr["code"] = tr.ticker.str.split("-").str[-1]
    games = pd.read_parquet(DATA / "mls_games.parquet").set_index("event_ticker")
    ev = pd.read_parquet(DATA / "mls_events.parquet")
    g2 = ev[(ev.type == "score_change") & (ev.minute >= 62) & (ev.minute <= 84)]
    rows, gt = [], []
    for event, g in tr.groupby("event_ticker"):
        if event not in games.index:
            continue
        start = games.at[event, "start_ts"]
        home, away = team_codes(event, g.ticker.unique().tolist() + [event + "-TIE"])
        if home is None:
            continue
        J = {c: jumps(d) for c, d in g.groupby("code")}
        gl = g2[g2.event_ticker == event]
        cands = []          # per goal: list of jump times consistent with direction
        for r in gl.itertuples():
            s, o = (home, away) if r.side == "home" else (away, home)
            ts = [t for t, dr in J.get(s, []) if dr > 0] + [t for t, dr in J.get(o, []) if dr < 0]
            cands.append((r.minute + r.extra, sorted(ts), r))
        best = None
        for O in np.arange(20, 42.01, 0.25):
            hits, resid, used = 0, 0.0, set()
            for clock, ts, r in cands:
                lo, hi = start + (clock - 1 + O) * 60 - 20, start + (clock + O) * 60 + 75
                m = [t for t in ts if lo <= t <= hi and t not in used]
                if m:
                    hits += 1; used.add(m[0]); resid += abs((m[0] - start) / 60 - (clock - 0.5 + O))
            key = (hits, -resid)
            if best is None or key > best[0]:
                best = (key, O)
        n_goals = len(cands)
        if best and best[0][0] > 0:
            rows.append((event, best[1], best[0][0], n_goals))
            O = best[1]
            for clock, ts, r in cands:
                lo, hi = start + (clock - 1 + O) * 60 - 20, start + (clock + O) * 60 + 75
                m = [t for t in ts if lo <= t <= hi]
                gt.append((event, r.side, r.minute, r.extra, m[0] if m else np.nan))
        else:
            rows.append((event, np.nan, 0, n_goals))
    x = pd.DataFrame(rows, columns=["event", "offset", "goals_explained", "goals_2h"])
    x.to_parquet(DATA / "mls_offsets.parquet", index=False)
    pd.DataFrame(gt, columns=["event", "side", "minute", "extra", "t_jump"]).to_parquet(DATA / "mls_goal_times.parquet", index=False)
    has = x[x.goals_2h > 0]
    print("games with 2nd-half goals:", len(has), " offset fitted:", has.offset.notna().sum(),
          " all 2H goals explained:", (has.goals_explained == has.goals_2h).sum())
    multi = has[has.goals_2h >= 2]
    print("games with >=2 2H goals:", len(multi), " share fully explained:", round((multi.goals_explained == multi.goals_2h).mean(), 2))
    print("fitted offset p10/25/50/75/90:", np.nanpercentile(x.offset, [10, 25, 50, 75, 90]).round(2))

if __name__ == "__main__":
    main()
