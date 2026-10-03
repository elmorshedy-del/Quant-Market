"""Per-game MLS end-game state from true match events + minute candles.

For each game: home/away/draw legs, wall-clock offset of the second half (from goal-driven price
jumps), score and red cards at match minute 85:00, goals after 85', prices of every leg at 85:00.

Usage: python3 mls_state.py  -> data/mls_state.parquet (one row per leg), prints a summary
"""
import re

import numpy as np
import pandas as pd

import legs as L

DATA = L.DATA
WINDOW_MIN = 85          # late window starts at match clock 85:00 (goals shown 86' or later are "late")


def team_codes(event_ticker, tickers):
    rest = re.sub(r"^KXMLSGAME-\d{2}[A-Z]{3}\d{2}(\d{4})?", "", event_ticker)
    codes = [t.split("-")[-1] for t in tickers if not t.endswith("-TIE")]
    home = [c for c in codes if rest.startswith(c)]
    away = [c for c in codes if rest.endswith(c)]
    if len(home) == 1 and len(away) == 1 and home[0] != away[0]:
        return home[0], away[0]
    return None, None


def second_half_offset(game_df, goal_minutes):
    """Wall minutes after kickoff minus displayed match minute, estimated from 2nd-half goals."""
    mid = game_df.assign(mid=(game_df["bid_c"] + game_df["ask_c"]) / 2)
    piv = mid.pivot_table(index="t_min", columns="leg", values="mid").sort_index()
    intensity = piv.diff().abs().sum(axis=1)
    offs = []
    for m in goal_minutes:
        lo, hi = m + 12, m + 27
        seg = intensity[(intensity.index >= lo) & (intensity.index <= hi)]
        if len(seg) and seg.max() >= 20:          # a goal moves the triplet by >= 20c in total
            offs.append(seg.idxmax() - m)
    return float(np.median(offs)) if offs else np.nan


def main():
    df = L.load_series("KXMLSGAME")
    df = df[df["live"]]
    games = pd.read_parquet(DATA / "mls_games.parquet")
    ev = pd.read_parquet(DATA / "mls_events.parquet")
    goals = ev[ev["type"] == "score_change"]
    reds = ev[ev["type"] == "red_card"]

    rows, offsets = [], {}
    for event, g in df.groupby("event_ticker"):
        tickers = sorted(g["ticker"].unique())
        home, away = team_codes(event, tickers)
        if home is None:
            continue
        gg = goals[goals["event_ticker"] == event]
        sh = gg[(gg["minute"] > 45) & (gg["minute"] <= 90) & (gg["extra"] == 0)]["minute"].tolist()
        offsets[event] = second_half_offset(g, sh)
    med = float(np.nanmedian(list(offsets.values())))

    for event, g in df.groupby("event_ticker"):
        tickers = sorted(g["ticker"].unique())
        home, away = team_codes(event, tickers)
        info = games[games["event_ticker"] == event]
        if home is None or info.empty:
            continue
        info = info.iloc[0]
        off = offsets.get(event)
        off_used = off if off == off and 14 <= off <= 26 else med
        wall85 = WINDOW_MIN + off_used
        gg = goals[goals["event_ticker"] == event]
        early = gg[gg["minute"] <= WINDOW_MIN]
        late = gg[gg["minute"] > WINDOW_MIN]
        h85 = int((early["side"] == "home").sum())
        a85 = int((early["side"] == "away").sum())
        rr = reds[(reds["event_ticker"] == event) & (reds["minute"] <= WINDOW_MIN)]
        late_list = [(r.side, int(r.minute), int(r.extra)) for r in late.sort_values(["minute", "extra"]).itertuples()]
        for leg, d in g.groupby("leg"):
            code = d["ticker"].iat[0].split("-")[-1]
            role_side = "draw" if code == "TIE" else ("home" if code == home else "away")
            d = d.sort_values("t_min")
            at = d[d["t_min"] <= wall85]
            if at.empty or at["t_min"].iat[-1] < wall85 - 3:
                continue
            row = at.iloc[-1]
            diff = h85 - a85
            if role_side == "draw":
                role = "draw_tied" if diff == 0 else "draw_vs_lead"
            else:
                mine = diff if role_side == "home" else -diff
                role = "tied_team" if mine == 0 else ("leader" if mine > 0 else "trailer")
            rows.append({
                "event_ticker": event, "leg": leg, "ticker": d["ticker"].iat[0], "side": role_side,
                "role": role, "kickoff": info["start_ts"], "offset": off, "offset_used": off_used,
                "wall85": wall85, "score_home_85": h85, "score_away_85": a85, "goal_diff_85": abs(h85 - a85),
                "red_home_85": int((rr["side"] == "home").sum()), "red_away_85": int((rr["side"] == "away").sum()),
                "late_goals": len(late_list), "late_goal_list": str(late_list),
                "final_home": info["home_score"], "final_away": info["away_score"],
                "bid85": row["bid_c"], "ask85": row["ask_c"], "mid85": (row["bid_c"] + row["ask_c"]) / 2,
                "win": bool(d["win"].iat[0]),
            })
    st = pd.DataFrame(rows)
    st.to_parquet(DATA / "mls_state.parquet", index=False)
    o = pd.Series(offsets)
    print(f"games with state: {st.event_ticker.nunique()}; 2nd-half offset from goals: "
          f"n={o.notna().sum()}, median={med:.1f}, p10/p90={o.quantile(.1):.1f}/{o.quantile(.9):.1f}")
    # consistency: does the true final result match the settled leg?
    chk = st.groupby("event_ticker").apply(lambda x: (
        x.loc[x["win"], "side"].iat[0] if x["win"].any() else None), include_groups=False)
    fin = st.drop_duplicates("event_ticker").set_index("event_ticker")
    truth = np.where(fin["final_home"] > fin["final_away"], "home", np.where(fin["final_home"] < fin["final_away"], "away", "draw"))
    agree = (chk.reindex(fin.index).values == truth)
    print(f"settled leg agrees with true final score: {agree.mean():.1%} of {len(agree)} games")


if __name__ == "__main__":
    main()
