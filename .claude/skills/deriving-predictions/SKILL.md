---
name: deriving-predictions
description: Derive new, risky, checkable predictions from constraints or from a proposed explanation, then test them on data that played no part in forming the explanation (Mendeleev's predicted elements). Use when a hypothesis currently only explains data already seen, when choosing between explanations that fit equally well, or when an explanation needs to be made falsifiable.
---

# deriving-predictions — make the explanation say something new

**Move:** Ask: *if this explanation (or this constraint) is right, what else must be true
that we have not looked at yet?* Prefer predictions that would be surprising if
the explanation were false. Test them on fresh or held-out data.

## When to consider it

- An explanation fits the existing data, but so do its rivals.
- An explanation has been tuned to the data, and you need to know whether it captured
  structure or noise.
- There are hard constraints (conservation, accounting identities, symmetry,
  no-arbitrage, bounded ranges, the timing of information) that any explanation must respect.
- You are about to declare a result. The acceptance gate needs a held-out check.

## Actions

1. Write the explanation as a generative statement: what produces the data.
2. Derive consequences at **places the explanation was not fitted to**: other
   subsets, other horizons, other variables, extreme conditions, or relations between
   quantities. Constraints (identities, bounds) often give consequences that hold whatever the parameter values.
3. Rank consequences by **risk**: how unlikely they are under the best rival. Pick one or
   two that discriminate.
4. **Register the prediction before looking**: the value, its interval, the data it will
   be tested on, and the pass/fail threshold. Put it in the ledger's *Predictions*
   section with the entry id.
5. Test. Report the result against the registered threshold, whichever way it falls.

## Required outputs

- The prediction as registered (date, entry id, threshold, data).
- The test result and a pass/fail against the threshold that was registered.
- The consequence for belief: which explanations were weakened or strengthened, and by how much.

## Failure checks

- Was the prediction written down before the data were inspected? If not, it is a
  postdiction; label it so.
- Would the main rival make the same prediction? Then passing it does not discriminate.
- Is the threshold loose enough that failure was impossible?
- Did you test on data that shaped the explanation? Check provenance
  (`independence-bookkeeping`).

## Worked example — Dmitri Mendeleev (1869–75)

Mendeleev arranged the elements so that their properties recurred periodically with
atomic weight, and he left gaps where the pattern demanded elements nobody had found.
From the constraint of periodicity he predicted their properties in advance. For
"eka-aluminium" he predicted an atomic weight near 68 and a density near 5.9 g/cm³. When
Lecoq de Boisbaudran discovered gallium in 1875, he first reported a density of
about 4.7. Mendeleev wrote that the sample must be impure. Re-measured on purified
metal, it came out at about 5.9. His predictions for "eka-silicon" (germanium, found
in 1886) matched comparably closely. The pattern earned belief because it predicted
data that played no part in building it, including a correction to a measurement.
