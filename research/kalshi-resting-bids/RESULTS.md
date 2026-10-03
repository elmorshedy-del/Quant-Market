# Resting bids on cheap in-play legs (Kalshi): research log

Status: **interim** (2026-10-03). Soccer results below; fill-size check and other sports in progress.

## Question

When a Kalshi sports leg is cheap during a game, does a resting buy order (placed below the
market, filled when sellers come to it) win more often than its price implies, after fees?
Compared against: buying right after a big jump (taker, the Football-Bot approach), and resting
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

## Findings so far (soccer)

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

## Open checks

- Fill size: candles cannot show how many contracts the sellers dumped below our price; a
  trade-level sample is being downloaded to measure realistic fill sizes.
- Tennis, MLB, NBA, NHL, NFL, WNBA (downloading).

## Files

`download.py` (markets + candles), `enrich.py` (milestone kickoffs, fee schedules),
`kickoff_est.py` (soccer kickoff estimate), `legs.py` (minute grid, legs, clock),
`study.py` (overview, calibrate, grid, confirm, shock), `trades_sample.py` (trade-level sample).
Data is not committed; `python3 download.py SERIES...` rebuilds it.
