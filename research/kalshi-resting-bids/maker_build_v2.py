"""Rebuild the maker-edge tables with maker_edge v2 (no outcome-dependent end of sample).

Usage: python3 maker_build_v2.py main  [SERIES ...]   # 150-game sample -> discovery (first 60% by kickoff) + sealed
       python3 maker_build_v2.py fresh [SERIES ...]   # fresh never-used sample
Outputs: data/mk_edge2_disc_<S>.parquet, ../holdout_sealed/maker/mk_edge2_hold_<S>.parquet,
         ../holdout_sealed/fresh/mk_edge2_fresh_<S>.parquet
The discovery/sealed split is the same rule as v1 (first 60% of games by kickoff per series); the script
checks that the event sets match v1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import maker_edge as M
import maker_eval as E

SEALED = E.SEALED
FRESH = SEALED.parent / "fresh"


def main():
    mode, series = sys.argv[1], sys.argv[2:] or E.SERIES
    for s in series:
        if mode == "fresh":
            src = FRESH / f"mk_trades_{s}.parquet"
            if not src.exists() or not len(pd.read_parquet(src)):
                continue
            df = M.build(s, src_dir=FRESH, out_path=FRESH / f"mk_edge2_fresh_{s}.parquet")
            print(s, "fresh", df.event.nunique(), "games", len(df), "fills", flush=True)
            continue
        tmp = SEALED / f"mk_edge2_{s}.parquet"
        df = M.build(s, src_dir=SEALED, out_path=tmp)
        ko = df.groupby("event")["kickoff"].first().sort_values(kind="stable")
        n_disc = int(np.floor(len(ko) * 0.6))
        disc_ev = set(ko.index[:n_disc])
        v1_disc = set(pd.read_parquet(E.DATA / f"mk_edge_disc_{s}.parquet", columns=["event"])["event"])
        v1_hold = set(pd.read_parquet(SEALED / f"mk_edge_hold_{s}.parquet", columns=["event"])["event"])
        # keep v1 membership where known (identical rule); new events (none expected) follow the rule
        disc_mask = df["event"].isin(v1_disc) | (~df["event"].isin(v1_hold) & df["event"].isin(disc_ev))
        df[disc_mask].to_parquet(E.DATA / f"mk_edge2_disc_{s}.parquet", index=False)
        df[~disc_mask].to_parquet(SEALED / f"mk_edge2_hold_{s}.parquet", index=False)
        tmp.unlink()
        print(s, "disc", df[disc_mask].event.nunique(), "hold", df[~disc_mask].event.nunique(),
              "| rule vs v1 disc mismatches:", len(disc_ev ^ v1_disc), flush=True)


if __name__ == "__main__":
    main()
