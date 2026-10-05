---
name: hidden-states-and-trajectories
description: Look beneath an average for distinct latent types, states, or paths (Mendel's hidden factors and breeding-true tests, McClintock's tracking of individual kernels). Use when a pooled statistic is weak, near zero, or unstable, when individuals behave consistently but differently, or when a population might be a mixture of processes. Includes the multiple-testing and fresh-validation rules subgroup searches require.
---

# hidden-states-and-trajectories — the average may be a mixture

**Move:** Stop summarising the population. Follow individual units over time or across
repetitions. Ask whether they fall into types that behave differently. A weak or null
average can hide strong effects that differ in size or sign. A noisy average can be
several clean processes mixed together. It can also be one process plus noise, and
telling these apart is the whole job.

## When to consider it

- The pooled effect is small, but individual units look strongly patterned.
- Dispersion across units is larger than sampling noise alone predicts.
- The same unit behaves consistently across separate time windows ("breeds true").
- A model fits the average poorly in a structured way: bimodal residuals, or a
  residual sign that persists within units.

## Actions

1. **Predict the noise floor first.** Under "one homogeneous process", how much should
   per-unit statistics vary from sampling noise alone? Compute this analytically or by
   simulating the pooled model (a `planted-truth` control).
2. **Look at individuals.** Estimate the statistic per unit. Compare its spread to the
   noise floor. Check for multimodality, for example with a mixture model compared to a
   single-component fit by BIC, or with a likelihood-ratio test calibrated by parametric bootstrap.
3. **Test persistence ("breeding true").** Split each unit's history into two halves.
   If types are real, a unit's first-half estimate predicts its second-half estimate
   beyond what the pooled model predicts. This is the most direct test that a state is a
   property of the unit and not of noise.
4. **Find a visible handle, if any.** Check whether observed covariates predict the
   latent type. Every covariate and threshold you try is a test, so count them.
5. **Correct and confirm.** Apply a multiple-testing correction to the full search
   (Holm/FDR, or permute the whole search procedure). Then confirm the chosen
   split on data that played no part in finding it, with the prediction written down first.
6. **State the mechanism separately.** "Units fall into two predictive types" is a
   predictive claim. *Why* they differ is a further, causal question.

## Required outputs

- Noise-floor calculation and the observed dispersion, with numbers.
- Persistence result: first-half statistic versus second-half statistic, with an interval.
- If a split is claimed: the number of candidate splits examined, the corrected
  p-value or FDR, and a confirmation-set result tested against a pre-registered
  prediction.
- A clear verdict: *single process*, *multiple processes*, or *insufficient evidence*.

## Failure checks

- **Regression to the mean:** extreme units selected on noisy estimates look
  different in-sample and revert out of sample. The persistence test catches this.
  Never classify units and evaluate them on the same data.
- **Forking paths:** you tried 20 covariates × 3 thresholds and reported the best.
  Count everything you tried.
- **Overfitting units:** per-unit models beat the pooled model in-sample by
  construction. Compare out of sample only.
- **Dispersion from scale, not type:** heteroskedasticity or unequal sample lengths
  can widen the spread of per-unit estimates without distinct types existing. Standardise first.
- **Shrinkage:** with noisy units, shrunken (partially pooled) estimates usually
  predict better than raw per-unit estimates. Compare against them.

## Worked example — Gregor Mendel (1866) and Barbara McClintock (1950s)

Mendel crossed round-seeded and wrinkled-seeded peas. In the first generation,
wrinkledness vanished. In the second it returned at about 3:1 (5,474 round to 1,850
wrinkled). The visible "round" class hid two states. He tested this by letting each round
second-generation plant self-fertilise and following its offspring. Of 565 such plants, 193 bred
true and 372 produced both kinds again, close to the 1:2 that a hidden-factor model
predicts. The prediction concerned individual trajectories, which the average could not reveal.

McClintock tracked the colour patterns of individual maize kernels and plants
across generations. She did not average over them. The timing and placement of
spots revealed genetic elements that moved and switched genes on and off during
development. That mechanism was invisible in population averages.
