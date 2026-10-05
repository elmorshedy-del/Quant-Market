---
name: selection-and-missing-data
description: Model how units got into the data and what is missing (Wald's analysis of damage on returning aircraft). Use when the sample was filtered by survival, success, availability, or the outcome itself, when missingness might depend on the value that is missing, or when a conclusion depends on units you never observed.
---

# selection-and-missing-data — reason about the units you cannot see

**Move:** Before interpreting a pattern, write down the process that decided which
units, periods and values appear in the data. Then ask what the pattern would look
like even if your hypothesis were false, given that filter.

## When to consider it

- The dataset contains only survivors, completers, listed or current members,
  winners, or "clean" records.
- Missing values, dropped rows or delisted units could be related to the outcome.
- A strategy, model or subgroup was *chosen* because it did well (selection on the
  outcome, including your own earlier choices).
- The effect is concentrated where observation is easiest.

## Actions

1. **Draw the selection funnel.** Starting population → each filter → the analysed sample.
   Record row counts at every stage as observations, and the rule for each filter
   as an assumption.
2. **Classify missingness.** Is it independent of everything, dependent on observed
   variables, or dependent on the missing value itself? State the evidence for each.
3. **Simulate the filter.** Generate data under the null (no effect). Apply the same
   selection. Measure the spurious effect the filter alone creates. That is your bias floor.
4. **Recover what you can.** Find data on the excluded units, model the selection
   explicitly (inverse-probability weights, bounds, Heckman-type corrections), or
   restrict the claim to the selected population.
5. **Report bounds** when selection cannot be modelled: give the range of effects
   consistent with plausible values for the missing units.

## Required outputs

- Selection funnel with counts and the rule at each stage.
- The bias floor from the simulated filter, compared with the observed effect.
- A conclusion phrased for the population it actually applies to.

## Failure checks

- Did the analysis condition on a variable affected by the outcome (a collider)?
- Is "no missing values" true because missing rows were silently dropped upstream?
- Did your own sequence of analyses select the best-looking result? Count the attempts.
- Would the conclusion reverse if the excluded units behaved like the worst
  included ones? If so, it is not robust to selection.

## Worked example — Abraham Wald (1943)

During the Second World War, Wald, at Columbia's Statistical Research Group, analysed
damage on bombers that *returned* from missions. Hits were common on the fuselage and wings
and rarer on the engines. The tempting reading was to armour where the holes were. Wald's
point was that the sample was filtered by survival. Aircraft hit in the engines
tended not to come back, so the sparse engine hits among returners were evidence that
engine hits were *deadly*, not that they were rare. His memoranda gave a method to
estimate how vulnerable each area was from survivors' damage alone, by modelling the
missing aircraft explicitly. (The popular picture of a plane covered in red dots is a modern
illustration, not his figure.)
