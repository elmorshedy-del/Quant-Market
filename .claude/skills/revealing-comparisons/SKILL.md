---
name: revealing-comparisons
description: Design the comparison that isolates an effect by holding confounders fixed by construction (Snow's water-company comparison, Leavitt's same-distance stars). Use when an apparent effect could be produced by something else that varies along with it, or when a raw difference between groups is being read as causal.
---

# revealing-comparisons — make the confounder unable to vary

**Move:** Don't adjust for a confounder after the fact. Find or build a comparison in
which it *cannot* differ between the groups being compared. Then any remaining
difference has fewer candidate explanations.

## When to consider it

- Two groups differ in outcome, and they also differ in other ways (size, period,
  liquidity, selection, measurement) that could produce the same difference.
- A regression "controls for" variables, but you cannot list what it fails to control.
- Someone proposes a mechanism, and a rival mechanism predicts the same raw comparison.
- You can find natural pairings: same units at different times, the same time across
  different units, or units mixed together that differ only in the candidate cause.

## Actions

1. Write the causal claim as "A → Y". List every rival "B → Y" that would produce
   the same raw difference. Label each as an *assumption* or an *observation*.
2. For each rival B, look for a subset or design in which B is constant or balanced
   **by construction**: matched pairs, within-unit changes, the same calendar dates,
   interleaved assignment, or a group whose members share B.
3. Before computing anything, state what the comparison should show if A matters and what it
   should show if only B matters.
4. Run the comparison. Report the effect with an interval and the sample sizes.
5. Run a **placebo comparison**: the same design on an outcome or period where A
   cannot act. It must show no effect.

## Required outputs

- Ledger entry with: claim, rival explanations, the comparison chosen and why it
  neutralises each rival, the prediction under A and under B, result, placebo result.
- A residual list of rivals that the design does **not** neutralise.

## Failure checks

- Is the "matched" variable actually equal across groups? Show the balance table.
- Did the matching itself select on the outcome (e.g., matching on post-treatment data)?
- Did you try several comparison designs and report only the best? Count them all.
- Does the placebo comparison come out null? If not, the design leaks a confounder.
- Is the conclusion causal? A clean comparison supports a causal claim only for
  the rivals it removed.

## Worked example — John Snow (1854) and Henrietta Leavitt (1912)

Snow suspected that cholera spread through water. "Bad air" in poor districts explained
the same geography. In south London two companies piped water to the *same streets*:
Southwark & Vauxhall drew from the sewage-polluted Thames, and Lambeth had moved its
intake upstream. Neighbours shared air, poverty and drains but not water. In Snow's
1855 tabulation, houses supplied by Southwark & Vauxhall suffered about 315 cholera deaths per
10,000, against about 37 per 10,000 for Lambeth. Miasma could not produce that
difference, because it was held fixed by the street layout.

Leavitt wanted to know whether a variable star's period related to its true brightness.
Apparent brightness also depends on distance, which was unknown. She studied Cepheids
in the Small Magellanic Cloud, which are all at essentially the same distance, so that
apparent and true brightness differ only by one shared constant. The
period–luminosity relation appeared cleanly. The design removed the confounder rather
than estimating it.
