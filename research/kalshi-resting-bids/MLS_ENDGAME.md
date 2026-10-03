# MLS end-game resting orders (round 2): research log

Status: complete for this round (2026-10-03). Resting (maker) orders only, per the brief.
6 theses, ~2,700 simulator variants, 4 frozen rules, all tested on the sealed holdout.

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

## Stoppage, liquidity and favourite theses (second run)

- Stoppage-time decay (129 variants, 0 rules): late-goal risk stays flat or rises into stoppage, and
  in 1-goal games stoppage goals come mostly from the trailer; draws in 1-goal games look 2-4c rich and
  leaders 2-4c cheap from 82' into stoppage, but not significantly on liquid books. A fixed-price
  resting order cannot collect the decay: YES buys on decaying legs are filled by the decay (-37% to
  -48%); NO offers go stale above the market and only fill on goals (-1% to -16%). Only quiet fills in
  the first ~60 s earn a small premium; short time-to-live helps but never reaches significance.
- Liquidity (1,400 variants, 0 rules): liquid 2026 books cut maker losses 3-4x vs thin 2025 books, but
  only to break-even; the queue-position assumption moves results more than any placement choice.
- Favourite side (255 variants, 1 rule): resting NO at the 1c touch on trailers 2+ goals down: +1.2%
  with no losses in discovery (101 games) and holdout (53 games). Both reviewers rejected it: every
  fill earns ~1c, so the range is an artifact of zero observed comebacks; one comeback erases the whole
  sample and 0/154 cannot exclude a comeback rate above the ~1.2% break-even; the 1c queue is the most
  crowded on the venue, so queue_share 0.5 is optimistic. Classified as unproven tail-risk carry.
- Informational, not frozen: the best near-miss (NO on the 1-goal draw one tick inside the spread,
  cancelled after 60 s, strict fills) was +9% [-4%, +20%] in discovery and -17% [-44%, +7%] in holdout.

## Data-quality caveat

About 10% of games have a state at entry that contradicts prices (e.g. labelled 0-2 but priced as
tied), most likely from the estimated match clock (entry earlier than 82' in real time). Analysts
re-ran key results without those legs; conclusions did not change.

## Conclusion

Static resting orders in the MLS end game have no robust edge on this data. The reason is adverse
selection: a fixed-price order is left behind on the quiet path and swept when the game turns, and
the 2-5c favourite-longshot gap at 82' is too small to pay for that. The one family that looked
strong in discovery (deep bids in tied games) failed on the holdout as liquidity grew ~25x.

## Most promising lead (not testable with this simulator)

The trade tape shows passive sellers of the draw in 1-goal games and of tied team legs earning about
+3.5 to +8c per contract on trades before any goal; the loss comes from fills at goal time. Capturing
that needs dynamic quoting (re-pricing the offer every few seconds as the leg decays, cancelling
within well under a second on goals), i.e. market making, which matches the Reddit trader's profile
(maker-only, ~350k small trades, liquidity rewards). Testing it needs order-book data with queue
modelling (e.g. Football-Bot's raw L2 recordings since Aug 2026, which have gaps) and the details of
Kalshi's liquidity reward programme.
