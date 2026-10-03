"""Build the MLS end-game analysis dataset (entry at ~82:00 match clock) and split discovery/holdout.

Outputs
  data/mls_late_legs.parquet          one row per leg: state at entry, prices, features, outcome (discovery)
  data/mls_late_legtrades.parquet     trades after entry for those legs (discovery)
  ../holdout_sealed/...               the same for games kicked off on/after 2026-07-01
"""
from pathlib import Path

import numpy as np
import pandas as pd

import legs as L
from mls_clock3 import jumps
from mls_state import team_codes

DATA = L.DATA
SEALED = Path(__file__).parent.parent / "holdout_sealed"
HOLDOUT_FROM = pd.Timestamp("2026-07-01").timestamp()
ENTRY_CLOCK = 82


def main():
    games = pd.read_parquet(DATA / "mls_games.parquet").set_index("event_ticker")
    off = pd.read_parquet(DATA / "mls_offsets.parquet").set_index("event")
    med = float(off["offset"].median())
    ev = pd.read_parquet(DATA / "mls_events.parquet")
    goals, reds = ev[ev["type"] == "score_change"], ev[ev["type"] == "red_card"]
    tr = pd.read_parquet(DATA / "mls_late_trades.parquet")
    tr = tr[~tr["block"]].sort_values("ts_exact")
    tr["code"] = tr["ticker"].str.split("-").str[-1]
    cd = L.load_series("KXMLSGAME")
    cd = cd[cd["live"]]

    leg_rows, trade_parts = [], []
    for event, g in tr.groupby("event_ticker"):
        if event not in games.index:
            continue
        info = games.loc[event]
        home, away = team_codes(event, g["ticker"].unique().tolist() + [event + "-TIE"])
        if home is None:
            continue
        start = info["start_ts"]
        O = off["offset"].get(event, np.nan)
        fitted = O == O
        O = O if fitted else med
        E = start + (ENTRY_CLOCK + O) * 60

        gl = goals[goals["event_ticker"] == event]
        late = gl[gl["minute"] > ENTRY_CLOCK]
        # emulate a true clock: never enter after a late goal's price reaction
        J = {c: jumps(d) for c, d in g.groupby("code")}
        capped = False
        for r in late.itertuples():
            s, o = (home, away) if r.side == "home" else (away, home)
            ts = sorted([t for t, dr in J.get(s, []) if dr > 0] + [t for t, dr in J.get(o, []) if dr < 0])
            clock = r.minute + r.extra
            m = [t for t in ts if start + (clock - 1 + O) * 60 - 300 <= t <= start + (clock + O) * 60 + 300]
            if m and m[0] - 60 < E:
                E, capped = m[0] - 60, True

        early = gl[gl["minute"] <= ENTRY_CLOCK]
        h, a = int((early["side"] == "home").sum()), int((early["side"] == "away").sum())
        rr = reds[(reds["event_ticker"] == event) & (reds["minute"] <= ENTRY_CLOCK)]
        last15 = early[early["minute"] > ENTRY_CLOCK - 15]
        late_list = [(r.side, int(r.minute), int(r.extra)) for r in late.sort_values(["minute", "extra"]).itertuples()]
        cg = cd[cd["event_ticker"] == event]

        for code, d in g.groupby("code"):
            side = "draw" if code == "TIE" else ("home" if code == home else "away")
            ticker = d["ticker"].iat[0]
            c = cg[cg["ticker"] == ticker].sort_values("ts")
            before = c[c["ts"] <= E]
            pre = c[c["ts"] <= start + 9 * 60]                 # ~actual kickoff for MLS
            if before.empty or pre.empty:
                continue
            q = before.iloc[-1]
            dbe = d[d["ts_exact"] < E]
            dafter = d[d["ts_exact"] >= E]
            if side == "draw":
                role = "draw_tied" if h == a else "draw_vs_lead"
                goals_for = goals_against = None
                red_for = red_against = None
                mom = len(last15)
            else:
                mine, theirs = (h, a) if side == "home" else (a, h)
                role = "tied_team" if mine == theirs else ("leader" if mine > theirs else "trailer")
                goals_for, goals_against = mine, theirs
                red_for = int((rr["side"] == side).sum())
                red_against = int((rr["side"] != side).sum())
                mom = int((last15["side"] == side).sum())
            leg_rows.append({
                "event": event, "ticker": ticker, "code": code, "side": side, "role": role,
                "kickoff": start, "entry_ts": E, "clock_fitted": fitted, "entry_capped": capped,
                "score_home": h, "score_away": a, "goal_diff": abs(h - a), "total_goals": h + a,
                "goals_for": goals_for, "goals_against": goals_against,
                "red_for": red_for, "red_against": red_against, "goals_last15": mom,
                "pre_bid": pre["bid_c"].iat[-1], "pre_ask": pre["ask_c"].iat[-1],
                "pre_mid": (pre["bid_c"].iat[-1] + pre["ask_c"].iat[-1]) / 2,
                "bid_E": q["bid_c"], "ask_E": q["ask_c"], "mid_E": (q["bid_c"] + q["ask_c"]) / 2,
                "last_E": dbe["yes_price"].iat[-1] if len(dbe) else np.nan,
                "vol_10m_before": dbe[dbe["ts_exact"] >= E - 600]["count"].sum(),
                "trades_after": len(dafter), "vol_after": dafter["count"].sum(),
                "late_goals": len(late_list), "late_goal_list": str(late_list),
                "final_home": info["home_score"], "final_away": info["away_score"],
                "win": bool(c["win"].iat[0]),
            })
            trade_parts.append(dafter[["ticker", "ts_exact", "yes_price", "count", "taker_side"]])

    legs = pd.DataFrame(leg_rows)
    # pre-game favourite flag per game (highest pre-game mid among the two team legs)
    teams = legs[legs["side"] != "draw"]
    fav = teams.loc[teams.groupby("event")["pre_mid"].idxmax(), ["event", "code"]].rename(columns={"code": "fav_code"})
    legs = legs.merge(fav, on="event", how="left")
    legs["pre_fav"] = np.where(legs["side"] == "draw", np.nan, (legs["code"] == legs["fav_code"]).astype(float))
    trades = pd.concat(trade_parts, ignore_index=True)

    hold = legs["kickoff"] >= HOLDOUT_FROM
    SEALED.mkdir(exist_ok=True)
    legs[~hold].to_parquet(DATA / "mls_late_legs.parquet", index=False)
    trades[trades["ticker"].isin(legs[~hold]["ticker"])].to_parquet(DATA / "mls_late_legtrades.parquet", index=False)
    legs[hold].to_parquet(SEALED / "mls_late_legs.parquet", index=False)
    trades[trades["ticker"].isin(legs[hold]["ticker"])].to_parquet(SEALED / "mls_late_legtrades.parquet", index=False)
    print(f"discovery: {legs[~hold].event.nunique()} games / {(~hold).sum()} legs; "
          f"holdout (sealed): {legs[hold].event.nunique()} games / {hold.sum()} legs; "
          f"clock fitted {legs.drop_duplicates('event').clock_fitted.mean():.0%}, entry capped {legs.drop_duplicates('event').entry_capped.sum()} games")


if __name__ == "__main__":
    main()
