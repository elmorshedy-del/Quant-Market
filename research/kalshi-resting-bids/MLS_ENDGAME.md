# MLS end-game resting orders (round 2): research log

Status: in progress (2026-10-03). Resting (maker) orders only, per the brief.

## Setup

- 691 MLS games (May 2025 - Sep 2026) with true match events from Kalshi live data (every goal and
  card with its match minute, incl. stoppage; settled results match the true scores in 100% of games)
  and 2.2M trades with second-level timestamps for the last ~30 minutes of every game.
- Entry at ~82:00 match clock. MLS kicks off ~9-10 min after the listed start; the second-half clock
  offset is fitted per game from price jumps at goals between 62' and 84' only (never from late goals,
  to avoid hindsight). Games where a late goal's price reaction precedes entry have entry moved to just
  before it (32 games). Leak check: 4% of goals at 85'+ would otherwise leak.
- Discovery: 505 games (kickoff before 2026-07-01). Holdout: 186 games (Jul-Sep 2026), sealed;
  rules are frozen before the holdout is run.
- Simulator (`mls_sim.py`): resting buys fill only on later trades strictly through the price
  (optionally a share of trades at the price); YES or NO side; cancel time, delayed activation,
  cancel-on-jump with reaction latency (>= 2 s); Kalshi maker fee rounded up per fill; 100 contracts.

## End-game map (discovery, at ~82')

| State | Games | Realized | Market mid |
|---|---|---|---|
| 1-goal leader wins | 205 | 76.1% | 72.9c |
| 1-goal game ends drawn | 205 | 16.6% | 19.2c |
| 1-goal trailer wins | 205 | 7.3% | 9.0c |
| Tied game ends drawn | 126 | 57.9% | 54.9c |
| 2+ goal leader wins | 174 | 93.1% | 88.2c |

42% of games have a goal after 82'; half of those in 90+ stoppage. Favourites are 3-5 points cheap.

## Resting-order findings (discovery)

- Joining the bid loses for every role (adverse selection): filled YES leader bids win 64% vs 84%
  for all leaders (-15%, strict fills). Deeper YES bids on favourites are worse (bid-5: filled
  leaders win 45% at 64c).
- Cancel-on-jump (2 s latency) lifts the best touch setups to about break-even, never clearly above:
  the first sweep print that triggers the cancel is itself the toxic fill.
- NO side (resting offers of longshots) is near zero: the longshot premium is real but small.
- Pressure covariates (pre-game favourite, home/away, momentum, red cards, scoring level) shift
  prices but no resting rule survived; the few nominal passes were small (<= 34 games) and fragile.
- One family looked strong: deep resting YES bids in games tied at ~82' (team legs at 3c, or all
  tied legs at 0.2 x mid), filled by the clock and by thin-book "air-pocket" sweeps rather than by
  news, while decisive stoppage-time goals were underpriced.

## Holdout (frozen rules, Jul-Sep 2026)

| Rule | Discovery | Holdout |
|---|---|---|
| YES 3c on tied team legs | +109% [+11%, +226%] (117 games) | **-102%** (53 games, 0 winners) |
| YES 0.2 x mid on tied draw | +112% [+15%, +221%] (47) | +24% [-102%, +154%] (19, 3 winners) |
| YES 0.2 x mid on all tied legs | +84% [+27%, +142%] (98) | **-1% [-67%, +69%]** (55, 6 winners) |

Not confirmed. Discovery profits depended on a handful of games (dropping the top 10 flips two of
the three rules to about -60%), and the market changed: volume in the 10 minutes before entry rose
from ~175 contracts (2025 Q2) to ~12,800 (2026 Q2) and ~30,700 (Jul-Sep 2026); spreads 4c -> 1c.

## Open

Three theses (stoppage-time decay, liquidity regime, favourite-side resting orders) are being re-run;
their first run was cut off by a usage limit.
