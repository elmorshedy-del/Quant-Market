---
name: shared-mechanisms
description: Test whether apparently different phenomena are one mechanism, by using one to predict the other quantitatively (Newton's Moon test, Maxwell's computation of the speed of light from electrical constants). Use when separate effects have suspiciously similar signatures, when two literatures or datasets describe parallel regularities, or when a cheaper or better-measured phenomenon could constrain a harder one.
---

# shared-mechanisms — one cause, many effects, tested by numbers

**Move:** Propose that phenomena A and B arise from one mechanism M. Fit M using A only. Then
predict B **quantitatively, with no free parameters left to tune**. A shared
mechanism earns credence by connecting things that were measured separately.

## When to consider it

- Two regularities have the same functional form, timescale or sign pattern.
- An effect appears in several assets, periods, markets or datasets, and each one has
  its own ad hoc explanation.
- One phenomenon is well measured, and a related one is noisy or hard to observe.
- A proposed mechanism makes claims beyond the dataset where it was found.

## Actions

1. State M precisely enough that it fixes a number in B once its parameters are
   estimated from A.
2. Estimate M's parameters from A alone. Freeze them, and record the frozen values in the ledger.
3. Derive B's predicted value, with uncertainty propagated from the fit to A.
4. Compare with B, measured independently (`independence-bookkeeping`: no value
   from B may have touched the fit to A).
5. Write down the rival: "A and B are unrelated, or share only a common driver C".
   Check what C predicts, and whether the data can tell M from C.

## Required outputs

- The cross-prediction: parameters from A, predicted B with an interval, observed B.
- Provenance showing that B did not feed the fit.
- Verdict: shared mechanism supported, rejected, or indistinguishable from the rival C.

## Failure checks

- Was any parameter tuned after seeing B? If so, the test is spent; find a new B.
- Is the agreement within an interval so wide that any value would have passed?
- Do A and B share a measurement artifact (the same data vendor, the same
  normalisation) that could create agreement without a shared mechanism?
- Could a common driver C produce both without M?

## Worked example — Isaac Newton (1687) and James Clerk Maxwell (1862–65)

Newton proposed that the force making an apple fall also holds the Moon in orbit,
weakening with the inverse square of distance. The Moon is about 60 Earth radii away,
so its acceleration toward Earth should be about g/60² ≈ 1/3600 of an apple's.
Computed from the Moon's orbital period and distance, it is. One mechanism, fixed by
terrestrial measurement, predicted a celestial number.

Maxwell's equations implied electromagnetic waves travelling at a speed set by
two constants measured in electrical laboratories with coils and capacitors, which had
nothing to do with light. From Weber and Kohlrausch's values he computed about
310,000 km/s, close to Fizeau's measured speed of light (about 315,000 km/s). He
concluded that light is an electromagnetic wave, a prediction later confirmed when
Hertz produced and detected radio waves.
