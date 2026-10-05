# When does a resting order make money? Maker-edge study (round 3)

Status: complete (2026-10-05). The test was pre-registered in MAKER_PREREG.md, including the addendum,
and committed before the out-of-sample sets were scored. Logs and per-spec numbers are in results/.

## Answer

The cheap-side resting edge did not hold out of sample. The test asked whether, with filters chosen in
advance, a bot that only rests orders can be profitable on Kalshi sports markets. No filter passed, the
pre-registered primary one included.

| P1: rest on contracts at 10c or less while no leg's bid is 95c or more | Discovery (882 g) | Sealed holdout (588 g) | Fresh games (1,052 g) | Holdout + fresh |
|---|---|---|---|---|
| 5-min net markout (primary) | +0.76 [+0.20, +1.29] | +0.15 [-0.47, +0.81] | -0.20 [-0.60, +0.21] | -0.03 [-0.40, +0.35] |
| Sell at the bid after 5 min | -0.03 | -0.56 | -0.89 | -0.73 [-1.09, -0.36] |
| Back-of-queue bot, 100 lot, 60 s | +0.73 | -0.37 | -0.23 | -0.29 [-0.77, +0.19] |
| Back-of-queue bot, 500 lot, 300 s | +0.12 | -0.75 | -0.61 | -0.66 [-1.07, -0.23] |
| Held to settlement | +4.75 | -1.26 | -0.42 | -0.83 [-2.09, +0.55] |
| 1-min net markout | +0.65 | +0.21 [-0.06, +0.50] | +0.21 [+0.03, +0.41] | +0.21 [+0.05, +0.37] |

All figures are cents per contract after fees, with 95% game-bootstrap ranges.

- **The primary rule fails.** It needed the range above zero on both sets; the lower bounds are -0.47 and
  -0.60. The power analysis said a fail was likely even if a small edge were real (1.7% chance of passing
  at +0.3c). So the result caps the 5-minute edge at about +0.35c measured at the mid. It does not prove
  the edge is exactly zero.
- **A bot cannot capture what is left.** Exiting by hitting the bid loses on both sets, so the
  pre-registered "tradable" rule fails clearly. A new order at the back of the queue loses in every
  configuration. With 500 contracts per price level its loss is clearly negative.
- **No secondary filter passes.** S1-S8 all have a Bonferroni lower bound below zero on both sets (-0.45
  to -0.90), and every fresh point estimate is zero or negative. The soccer halftime filter, the only one
  with a tight discovery range (+0.19 [+0.10, +0.27]), came in at -0.01 and -0.04.
- **Controls behave as the theory says.** Mid-range contracts (30-70c) lose on every set because the fee
  is larger than the spread captured (pooled -0.26 [-0.46, -0.05]). Favourites at 90-94c are at or below
  zero at 5 minutes.

## Why discovery looked good

- **A handful of comeback games.**
  - On discovery, cheap contracts priced at 6.3c on average won 11.2% of the time. Out of sample they
    won 5.0% (holdout) and 5.8% (fresh) at 6.1c, i.e. close to fair.
  - Discovery's +0.76 falls to +0.13 without its top 10 of 878 games. Its +4.75 at settlement falls to
    -0.73.
  - The excess sits mostly in April-June 2026: discovery Q2 was +2.17, against -0.42 for fresh games
    from the same quarter. Like for like (same series and month) the gap is about 0.9c.
- **Not a data difference between the sets.** A parity check rebuilt every table and compared the three
  sets on 20+ dimensions: candle coverage, marking, dropped trades, trade order, fill sizes, pre-game
  prices, volume, kickoff hours and both random draws. Nothing differs by more than about 0.03c of P1.
  The mid-range control is the same on discovery and fresh (-0.18 vs -0.17).
- **The settlement drop is suggestive, not significant.** Point estimates reversed out of sample: P1
  settles below zero and the favourite control above. But on P1 the reversal is not significant.

## What the profitability definition got right

For a maker fill at price c:

    E[profit] = E[mid(t+h) - c]  +  E[100*W - mid(t+h)]  -  fee
                (noise vs news fills)   (mispricing left)

- **The fee only leaves room at the edges.** Kalshi's fee is proportional to c(100-c) and the tick is
  1c. Half-spread capture beats the fee only at extreme prices. That is why the mid-range control loses
  and why the cheap side shows +0.21c at 1 minute.
- **Out of sample, adverse selection eats about half of that.** The 1-minute gross markout is about
  0.3c, roughly half of the entry half-spread (about 0.6c). The rest goes to fills where the price kept
  moving through the order.
- **The capture cannot be cashed one-sided.** It exists only at the mid. A bot that buys at the bid and
  later sells by crossing the spread pays the spread back and loses.

## The one lead left (a hypothesis, not a result)

A two-sided market maker would buy the cheap contract at the bid and sell it at the ask, both as maker
fills. It would collect the spread on both legs instead of paying it on the exit. An indicative
diagnostic on the out-of-sample sets estimates this. It was not pre-registered, so it is a hypothesis
for new data:

- **Matched fills, 1 minute.** Pairing cheap-side buys with sells in the same market and state gives a
  combined 1-minute markout of +0.14 to +0.32c per round trip after two maker fees.
- **Unmatched, 1 minute.** Simply adding the two legs' averages gives only +0.06 [-0.11, +0.23].
- **5 minutes.** About -0.1c.
- **Not modelled:** queue position, inventory risk on the 35-54% of volume that finds no opposite
  fill, and liquidity rewards.

This matches the profile of the Reddit trader: maker-only, about 350k small trades, constant
re-quoting. It also matches the round-2 finding in MLS (MLS_ENDGAME.md) that passive sellers earned
money before goals and lost it at goal time. Testing it needs order-book data with queue modelling.
Public trades and 1-minute candles are not enough, and those are the limits of this study.

## Corrections made along the way (all before scoring)

- **v1 tables cut each game at a time chosen with the outcome.** The cut was the minute after which the
  winner's bid stays at 95c or more. A cheap fill taken after the favourite reached 95c survived only if
  the game swung back, which inflated the cheap-side edge. The v2 tables keep every in-play trade and use
  the ex-ante rule "max bid on any leg < 95". `phase` was scaled by the same cut and is no longer allowed
  as a filter.
- **v2.1 fixes from the pre-scoring audit:**
  - Contracts with no bid were marked at ask/2. That inflated P1 by about 0.15c.
  - Fills before a market's first candle took their features from a later candle.
  - The sweep flag depended on the order of trades sharing a timestamp.
  - MLB's 0.5 fee multiplier was not applied.
- **The scoring run ran out of memory** on its last descriptive block (C2, pooled). The verifier
  recomputed it with the frozen code: -0.26 [-0.46, -0.05].
- **Corrected counts.** The pre-registration listed 881 discovery and 597 holdout games. The tables hold
  882 and 588.

## Files

- **Pipeline:** maker_trades.py and maker_trades_fresh.py (download), maker_edge.py (v2.1 per-trade maker
  fills), maker_build_v2.py, maker_eval.py, maker_final2.py.
- **Specs and logs:** maker_specs_final.json, results/final2_discovery.log, results/final2_scoring.log,
  results/*.json.
- **Pre-registration and audit:** MAKER_PREREG.md.
