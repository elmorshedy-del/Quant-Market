"""Evaluate frozen rule specs on discovery and on the sealed holdout (games from 2026-07-01).

Usage: python3 mls_holdout.py rules.json     (a list of {"name": ..., "spec": {...}} objects)
"""
import json
import sys

import pandas as pd

import mls_sim as S


def split_summary(res, cut="2026-01-01"):
    if res is None or len(res) == 0:
        return {}, {}
    t = pd.Timestamp(cut).timestamp()
    return S.summarize(res[res["kickoff"] < t]), S.summarize(res[res["kickoff"] >= t])


def main(path):
    rules = json.load(open(path))
    disc_legs, disc_book = S.load("discovery")
    hold_legs, hold_book = S.load("holdout")
    rows = []
    for r in rules:
        d = S.simulate(r["spec"], disc_legs, disc_book)
        h = S.simulate(r["spec"], hold_legs, hold_book)
        ds, hs = S.summarize(d), S.summarize(h)
        d25, d26 = split_summary(d)
        rows.append({"rule": r["name"],
                     "disc_games": ds.get("games", 0), "disc_return": ds.get("return"), "disc_ci95": ds.get("ci95"),
                     "disc_2025": d25.get("return"), "disc_2026H1": d26.get("return"),
                     "hold_games": hs.get("games", 0), "hold_return": hs.get("return"), "hold_ci95": hs.get("ci95"),
                     "hold_pnl_per_game": hs.get("pnl_per_game")})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    out.to_json(path.replace(".json", "_results.json"), orient="records", indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
