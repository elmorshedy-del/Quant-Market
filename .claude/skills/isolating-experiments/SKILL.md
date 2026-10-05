---
name: isolating-experiments
description: Simplify the system until only one mechanism can operate, keeping the critics' favourite alternative present while removing the candidate cause (Pasteur's swan-neck flasks, Nirenberg and Matthaei's cell-free poly-U experiment). Use when many causes are tangled in observational data, when you can run controlled simulations or ablations, or when a mechanism claim needs evidence stronger than correlation.
---

# isolating-experiments — strip the system down to one mechanism

**Move:** Build the smallest system in which the proposed mechanism can act, and remove or
fix everything else. Design it so that the main rival explanation is still allowed to
operate. If the effect appears only when the candidate mechanism is present, the rival is ruled out.

## When to consider it

- Observational data mix many causes and no comparison can separate them.
- You can simulate, ablate, intervene, or construct a synthetic version of the
  system (e.g., a planted-truth dataset, a stripped-down backtest, a toy model).
- A predictive model exists, but whether its mechanism is real remains open.
- Critics have a specific alternative ("it's just X"), and you can keep X present.

## Actions

1. Name the candidate mechanism M and the strongest rival R.
2. Design the minimal system: which components are removed, which fixed, and which
   remain. **R must remain able to act.**
3. Pre-register the outcome under M-present / M-absent, and what R alone would produce.
4. Run M-present and M-absent arms that are identical in everything else, with replication
   and a seed or randomisation record.
5. Run a **positive control** (a system known to show the effect) and a **negative
   control** (a system that should not). Without these, a null result means nothing.
6. Afterwards, check that the simplification did not remove the very thing that made the
   effect occur in the real system (external validity).

## Required outputs

- The design: what was removed, fixed and kept, and why R stays possible.
- Results for M-present, M-absent and both controls, with uncertainty.
- A statement of what the isolated result does *not* show about the full system.

## Failure checks

- Did the simplification also remove R's pathway? Then the experiment is not discriminating.
- Do the controls behave as expected? If the negative control shows the effect,
  something else is producing it.
- Are the arms identical except for M? Diff the configurations.
- Is the isolated effect large enough to matter in the full system?

## Worked example — Louis Pasteur (1859–62) and Marshall Nirenberg & Heinrich Matthaei (1961)

Supporters of spontaneous generation argued that sealed, boiled broth stayed sterile only
because sealing excluded a vital property of air. Pasteur boiled broth in flasks whose
long necks he drew into an S-curve and left **open**. Air still reached the broth, but dust
and microbes settled in the bend. The broth stayed clear. When a flask was tilted so that
the broth touched the trapped dust, it soon teemed with life. The rival explanation
(air) was kept, and only the candidate cause (germs carried in dust) was removed.

To learn how RNA encodes protein, Nirenberg and Matthaei discarded the living cell. They
used a cell-free extract of *E. coli* that could still make protein and supplied a
synthetic RNA made only of uracil (poly-U). The system produced a protein made only of
phenylalanine: the first codon assignment (UUU → Phe), read directly from a system
simple enough that only one message could be translated.
