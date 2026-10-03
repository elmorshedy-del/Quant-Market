import copy, json
import numpy as np, pandas as pd
import mls_sim as S

legs, book = S.load()
rules = json.load(open("rules_round2.json"))

def fmt(s):
    if not s or not s.get("games"): return "n/a"
    return f'{s["return"]:+.0%} [{s["ci95"][0]:+.0%},{s["ci95"][1]:+.0%}] g={s["games"]} f={s["fills"]} win={s["win_rate"]:.0%}'

def run(spec, sub=None):
    return S.simulate(spec, legs if sub is None else sub, book)

def periods(res):
    out = {}
    for name, a, b in [("2025", "2025-01-01", "2026-01-01"), ("2026Q1", "2026-01-01", "2026-04-01"), ("2026Q2", "2026-04-01", "2026-07-01")]:
        r = res[(res.kickoff >= pd.Timestamp(a).timestamp()) & (res.kickoff < pd.Timestamp(b).timestamp())]
        out[name] = fmt(S.summarize(r))
    return out

def drop_top(res, k=5):
    pg = res.groupby("event").pnl_usd.sum().sort_values(ascending=False)
    return S.summarize(res[~res.event.isin(pg.index[:k])])

def variant(spec, path, value):
    s = copy.deepcopy(spec); d = s
    for key in path[:-1]: d = d[key]
    d[path[-1]] = value
    return s

for r in rules:
    spec = r["spec"]; print("\n=====", r["name"])
    res = run(spec)
    print(" base         :", fmt(S.summarize(res)))
    for k, v in periods(res).items(): print(f"   {k:12s}:", v)
    print(" queue 0.5    :", fmt(S.summarize(run(variant(spec, ["fill", "queue_share"], 0.5)))))
    print(" clock_fitted :", fmt(S.summarize(run(spec, legs[legs.clock_fitted]))))
    print(" not capped   :", fmt(S.summarize(run(spec, legs[~legs.entry_capped]))))
    print(" drop top 5   :", fmt(drop_top(res)))
    print(" drop top 10  :", fmt(drop_top(res, 10)))
    lv = spec["order"]["level"]
    vals = [2, 3, 4, 5, 6] if lv["mode"] == "abs" else [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    for v in vals:
        print(f"   level {v:<5}  :", fmt(S.summarize(run(variant(spec, ["order", "level", "value"], v)))))
    if "cancel_after_s" in spec["order"]:
        for v in [600, 900, 1500, 99999]:
            print(f"   cancel {v:<6}:", fmt(S.summarize(run(variant(spec, ["order", "cancel_after_s"], v)))))

# consistency of 'tied' state vs prices at entry (clock-error check)
t = legs[legs.role == "tied_team"]; d = legs[legs.role == "draw_tied"]
print("\ntied_team legs: mid_E p5/p50/p95", np.percentile(t.mid_E, [5, 50, 95]), " share mid_E<8 (contradicts tied):", round((t.mid_E < 8).mean(), 3))
print("draw_tied legs: mid_E p5/p50/p95", np.percentile(d.mid_E, [5, 50, 95]), " share mid_E<30:", round((d.mid_E < 30).mean(), 3))
