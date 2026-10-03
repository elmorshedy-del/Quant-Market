# Resting bids on cheap in-play legs (Kalshi): research log

Status: round 1 complete (2026-10-03): broad, no-selection baselines across 9 soccer leagues + NBA,
NFL, WNBA, NHL, MLB, ATP, WTA (about 21,700 games). Round 2 (thesis-driven MLS end-game study) in
`MLS_ENDGAME.md`.

## Corrections (read first)

1. **The "buy after a jump" test is NOT a test of Football-Bot.** It used 1-minute candles, any
   >=10c/20c one-minute move at any time in the game, and entry up to a minute late. Football-Bot
   trades sub-second sweeps with sibling confirmation, late in the game, with tuned entry/exit. These
   results say nothing about that strategy; the earlier wording ("this is the Football-Bot approach",
   "would not take Football-Bot live") is withdrawn.
2. **Late-window clock error for MLS (and possibly other leagues with delayed kickoffs).** The match
   minute was computed as kickoff + minute (+18 in the second half) from the listed start time. True
   goal times show EPL/La Liga kick off on time (offset ~1-2 min), but MLS kicks off ~9-10 min after
   the listed start, so the "75'-85'" window was really ~65'-75' for MLS. Section 5 below is
   therefore unreliable for MLS (one of the six confirmation leagues). Broad kickoff-to-settlement
   results are unaffected (orders placed before kickoff simply rest).
3. **Regime caveat.** 2026 includes the World Cup period, with unusually heavy, bot-driven soccer
   volume; the decay in soccer results may partly reflect that regime rather than a permanent change.
4. These round-1 tests are deliberately broad baselines (no game selection, no state conditioning,
   no dissection of winners vs losers). They answer "does the naive version work", not "can a
   selective version work".

## Question

When a Kalshi sports leg is cheap during a game, does a resting buy order (placed below the
market, filled when sellers come to it) win more often than its price implies, after fees?
Compared against: buying right after a big one-minute jump (a crude taker proxy), and resting
bids on favourites.

## Data

- Kalshi public API: every settled game market since spring 2025 (live + archive endpoints).
- 1-minute best bid / best ask candles (open/high/low/close) plus trade-price high/low and volume.
- Kickoff: Kalshi milestones (EPL, La Liga, Serie A, Bundesliga, Ligue 1); for MLS, UCL, Liga MX,
  Brasileirao, `expected_expiration_time - 3h`, which matches the milestone kickoff exactly in
  95-98% of games where both exist.
- 9 soccer leagues, 3,681 games.

## Simulation rules

- One resting buy of 100 contracts per price level per leg; it must sit below the best bid when
  placed (a real resting order, not an immediate buy).
- Fill ("either" rule): a later minute where the best ask reached our price, or a trade printed
  below our price (that means the whole level was taken, so our order would have been too).
  "through" rule = trade-below only. They give nearly identical results at 5c and up.
- Fees: Kalshi maker fee 1.75% x C x P x (1-P), rounded up to the cent per order (all nine
  leagues are `quadratic_with_maker_fees`, multiplier 1). Taker fee 7% for the shock test.
- Unit of statistics: the game (all fills in one game share one outcome). 95% ranges are bootstrap
  over games.

## Findings: soccer

### 1. Buying right after a jump (taker) loses
EPL, 413 games: -9% to -20% per dollar for every variant (jump >= 10c or 20c, fast or slow entry,
hold or exit after 5 min), all ranges below zero.

### 2. Resting bids on favourites (30-97c) lose
EPL: -1% to -34% per dollar. A favourite's in-play dip is usually real news.

### 3. Resting bids on cheap legs (3-10c), placed at kickoff, held to settlement
Discovery leagues (EPL, La Liga, Serie A) vs six leagues not used to design the test:

| Period | 6 unseen leagues (2,285 games) | 3 discovery leagues (1,311 games) |
|---|---|---|
| 2025 | +50% [+35%, +66%] | +54% [+35%, +74%] |
| 2026 Jan-Jun | +28% [+14%, +42%] | +35% [+17%, +52%] |
| 2026 Jul-Sep | **+9% [-9%, +26%]** | **-6% [-37%, +25%]** |
| All | +32% [+23%, +41%] | +38% [+26%, +51%] |

Every league and every level is positive over the whole period. The edge shrinks steadily and is
not distinguishable from zero in the latest quarter. Over the same time EPL in-play volume per game
rose from ~0.6M (Aug 2025) to ~2.5M contracts (Aug-Sep 2026) and spreads on 3-30c legs tightened
from ~2-3c to ~1c: more makers competing for the same seller flow.

### 4. Exits
Selling at 3x, 5x or 50c does worse than holding to settlement in every cell tested.

### 5. Pre-registered late window: not confirmed
Bids at ~75', cancelled at ~85' (the strongest slice in the discovery data):
+88% in discovery (selected, so biased up) but **+21% [-4%, +49%] on the six unseen leagues**.
Treat as unconfirmed.

### 6. Fill sizes (trade-level sample: 180 games, 540 legs, 1.25M trades, no block trades)
- 98-100% of candle fills would have filled all 100 contracts from trades strictly below the bid
  (median volume below the bid per filled order: 3.6k contracts in 2025, 73k in Jul-Sep 2026).
- But fills are lopsided: losing legs always fill completely (they get dumped as they die);
  winning legs fill only 91 of 100 contracts on average (96 counting trades at the bid price),
  and 15% of winning fills are partial. Brief dips that later win have less volume below the bid.
- Applying that to all 3,596 soccer games:

| Period | Full fills | Winners 96% | Winners 91% |
|---|---|---|---|
| 2025 | +52% [+40%, +64%] | +47% | +40% [+28%, +51%] |
| 2026 Jan-Jun | +31% [+20%, +42%] | +27% | +20% [+10%, +30%] |
| 2026 Jul-Sep | +6% [-9%, +21%] | +2% | **-3% [-18%, +11%]** |

Soccer verdict: a real edge in 2025 and early 2026 that has been competed away; the latest
quarter is indistinguishable from zero after realistic fills.

## Other sports (soccer rules applied unchanged = out-of-sample by sport)

Strategy A = resting bids at 3/5/7/10c from the start of play, held to settlement.
"Latest" = Jul-Sep 2026, except NBA/NHL (season ended in June, so Jan-Jun 2026).
Realistic = winning fills at 91% of size (from the soccer trade sample).

| Sport | Games | A, all periods (full fills) | A, 2025 (full fills) | A, latest (realistic) | Buy after jump (B) |
|---|---|---|---|---|---|
| Soccer (9 leagues) | 3,596 | +34% [+26%, +41%] | +52% | -3% [-18%, +11%] | -9% to -20% (EPL) |
| MLB | 4,608 | +9% [+1%, +18%] | +19% | -14% [-29%, +1%] | -5% to -20% |
| ATP | 4,252 | +2% [-8%, +12%] | +36% | -17% [-33%, +0%] | -8% to -28% |
| WTA | 4,246 | +6% [-3%, +16%] | +49% | -20% [-36%, -5%] | -9% to -32% |
| NHL | 1,606 | +9% [-6%, +25%] | +22% | -10% [-28%, +10%] | -5% to -12% |
| NBA | 1,436 | -3% [-19%, +12%] | +14% | -24% [-41%, -7%] | -2% to -8% |
| WNBA | 647 | +18% [-6%, +42%] | +27% | +1% [-40%, +43%] | -7% to -21% |
| NFL | 423 | +11% [-20%, +42%] | +17% | -17% [-65%, +37%] | -3% to -12% |

## Conclusions

1. **The crude minute-level "buy after a jump" proxy loses in every sport.** It is not
   Football-Bot's strategy (see Corrections) and says nothing about sub-second sweep trading.
2. **Resting bids on cheap legs had a real edge in 2025 in every sport (+14% to +52%).** It has
   decayed everywhere as Kalshi liquidity grew; with realistic fills the latest period is zero or
   negative in every sport, and significantly negative in WTA and NBA. Tennis lost it first
   (by early 2026), soccer last (gone by Jul-Sep 2026).
3. **Resting bids on favourites lose; far take-profits do worse than holding.**
4. The simple, rule-based version of the Reddit strategy is not profitable today on this data. If
   that trader is really still profitable, the edge must come from things not modelled here:
   hand-picked games ("x factor"), Kalshi liquidity rewards for resting orders (he mentions them;
   not included here), faster cancel/re-pricing, or luck over seven months of fat-tailed returns.

## Robustness of "the edge is gone" (all sports pooled, strategy A)

| Quarter | Games | Return, winners 91% filled |
|---|---|---|
| 2025 Q2 | 1,395 | +36% [+20%, +52%] |
| 2025 Q3 | 3,217 | +22% [+12%, +32%] |
| 2025 Q4 | 2,827 | +19% [+9%, +29%] |
| 2026 Q1 | 4,337 | -3% [-11%, +5%] |
| 2026 Q2 | 4,718 | -3% [-10%, +5%] |
| 2026 Q3 | 4,231 | -12% [-19%, -5%] |

- Jul-Sep 2026 pooled: -4% [-12%, +4%] even assuming every fill is complete; -13% [-20%, -5%] realistic.
- Every bid level (3/5/7/10c) is negative in Jul-Sep 2026; dropping any one sport leaves it negative.

## Caveats

- 1-minute candles: fills are judged per minute, not per message; queue position at the exact bid
  price is not modelled (the 91%/96% fill factors bracket it).
- Fill-size factors come from a soccer sample and are applied to all sports.
- Maker fee multiplier 1 everywhere (MLB's series lists 0.5, so MLB fees are slightly overstated;
  negligible at 3-10c).
- Start times: Kalshi milestones (soccer partly via expected expiration - 3h).

## Files

`download.py` (markets + candles), `enrich.py` (milestone kickoffs, fee schedules),
`kickoff_est.py` (soccer kickoff estimate), `legs.py` (minute grid, legs, clock),
`study.py` (overview, calibrate, grid, confirm, shock), `trades_sample.py` (trade-level sample),
`fillsize.py` (fill sizes from trades), `sports.py` (same rules on other sports).
Data is not committed; `python3 download.py SERIES...` rebuilds it.
