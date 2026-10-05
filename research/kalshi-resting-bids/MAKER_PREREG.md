# Maker-edge filters: pre-registration of the final test

Written and committed on 2026-10-05, before any v2 table of the sealed holdout or the fresh sample was scored.
Nothing below changes after the results are seen.

## Question

When does a resting (maker) order make money? Take any fill at price c (cents) on the contract the maker
bought. Then

    E[profit per contract] = E[mid(t+h) - c]  +  E[100*W - mid(t+h)]  -  fee
                             (markout)            (mispricing left)

The markout is positive when most fills are noise: an impatient taker pays up and the price comes back.
It is negative when most fills are news: the price moves through the order and keeps going. The
filters try to pick, using only what a bot knows before it places the order, the market states where
noise fills outweigh news fills by more than the fee.

## Data

- Every in-play trade on Kalshi for a random sample of 2026 games in 10 series: ATP, WTA, NBA, NHL, MLB,
  WNBA, NFL, EPL, La Liga, MLS. Each trade is one maker fill. A taker who bought YES filled a resting
  seller of YES, which is the same as a resting buyer of NO at 100 - price.
- Main sample: 150 games per series (random_state=7). Discovery is the first 60% of each series by
  kickoff (881 games). The sealed holdout is the last 40% (597 games, out of time).
- Fresh sample: 120 more games per series (random_state=99), none of them in the main sample. It covers
  the same months, so it tests new games, not a later period. NFL has no unused 2026 games, so it is absent.

## Correction made before this test (v2 tables)

The v1 tables ended each game at decided_ts, the first minute after which the winner's bid stays at 95c
or more. That rule uses the outcome. A cheap fill taken after the favourite reached 95c stayed in the
sample only if the game later swung back, which inflates exactly the cheap-side edge under test. A
reviewer in the filter search found this.

The v2 tables keep every in-play trade from the listed start to the market close. They add an ex-ante
column, max_bid_before: the highest bid on any leg of the event in the last closed minute. A bot can
stop quoting when it reaches 95. `phase` was also scaled by decided_ts, so it is no longer an allowed
filter. Code: maker_edge.py (v2), maker_build_v2.py, maker_eval.py, maker_final2.py.

## Frozen specs (maker_specs_final.json)

The primary test was fixed before any v2 numbers, including discovery:

- **P1**: c <= 10 and max_bid_before < 95. Rest on the cheap side of a game that is not yet decided.

Secondary specs are the filters frozen by the filter-search workflow (5 lenses, discovery only, v1 tables):

| Spec | Filters |
|---|---|
| S1 | c <= 10 (the v1 discovery headline) |
| S2 | c <= 10, mid_before >= 6 (lens price_depth) |
| S3 | c <= 10, move2 > -10, 10 <= vol10 <= 50 (lens news, F1) |
| S4 | c <= 10, move2 > -10, vol10 <= 50 (lens news, F2) |
| S5 | c <= 10, spread_before <= 2 (lens crowding) |
| S6 | c <= 10, mid_before >= 5.5 (lens queue, F1) |
| S7 | 6 <= c <= 20, move2 < 0 (lens queue, F2) |
| S8 | soccer only, minute 50-62 after listed start, vol10 <= 4, 6 <= c <= 94 (lens sport, halftime) |

Two controls, where theory says the result should be zero or negative:

| Spec | Filters |
|---|---|
| C1 | 90 <= c, max_bid_before < 95 (favourites before the game is decided) |
| C2 | 30 <= c <= 70 (mid-range; the fee exceeds the spread capture) |

## Success rule

- **Primary.** P1's 5-minute net markout (net_mo300) has a 95% game-bootstrap range above zero on both
  the sealed holdout and the fresh sample.
- **Tradable.** On top of the primary rule, P1 has a positive point estimate on both sets for
  exit_bid: buy as a maker, then sell 5 minutes later by hitting the bid and paying the taker fee.
- Secondary specs are reported with the same numbers. They are 8 extra tests, so a single secondary
  pass is not treated as a discovery.
- Also reported, because the discovery edge was lottery-like and driven by a few comeback games:
  - the share of games that are positive and the median game,
  - the result without the 10 most profitable games,
  - markouts capped at +20c,
  - swept fills with each fill capped at 200 contracts, as a stand-in for an order at the back of the queue.

## What the filter search found on discovery (v1, for the record)

The only edge that survived was the cheap side, c <= 10: net_mo300 +1.10c [+0.60, +1.62]. Reviewers
rejected every refinement as adding nothing robust. Their caveats:

- Only about a third of games are positive.
- The top 10 of 878 games carry most of the profit.
- The swept-fill edge mostly disappears once fills are capped at a realistic size.
- About +0.3 to +0.4c of the edge is structural: the 1c tick gives spread capture above the fee only at
  extreme prices. The rest is underdog drift in this sample, which may not repeat.

Reviewers' out-of-sample expectation for the cheap side: about +0.3c per contract at 5 minutes
(range -0.3 to +0.8).
