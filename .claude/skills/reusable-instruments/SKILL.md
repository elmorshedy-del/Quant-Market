---
name: reusable-instruments
description: Build a validated, documented, reusable tool instead of repeating expensive manual work (Hollerith's punched-card tabulator for the 1890 US census). Use when you are about to run the same analysis by hand a second time, when each new question requires re-deriving the same quantities, or when a slow step dominates the research loop. Complements tool-stewardship.
---

# reusable-instruments — automate the repeated measurement, then trust it carefully

**Move:** When the same expensive operation recurs, invest once in an instrument that
performs it: a function, a script with a fixed interface, or a cached dataset. Validate it
on known answers, document it, and reuse it. The next question then starts where the last one finished.

## When to consider it

- You (or an earlier round) wrote nearly the same analysis code before.
- Each hypothesis test requires the same data preparation, estimator or evaluation.
- The research loop is slow because one step is manual or recomputed from scratch.
- Several hypotheses will be compared on one standard metric, and the metric must be
  computed identically for each.

## Actions

1. Check `research/tools/` (and the project's `app/` code) for an existing instrument
   first (`tool-stewardship`). Patch it rather than fork it.
2. Define the interface: inputs, outputs, units, and what failure looks like.
3. **Validate on known answers before use:** a planted-truth case where the right
   output is known, and a corrupted input it must reject (`planted-truth`).
4. Time it on a small case before scaling (`timing-discipline`).
5. Document it in the file header: what it does, when to use it, what its output means, and
   what test its answer must pass. Record it in the ledger entry that created it.
6. Use it for every later comparison, so that differences between hypotheses come from the
   hypotheses and not from the measuring code.

## Required outputs

- The tool in `research/tools/<name>.py` with the four-question header.
- Its validation results (known-answer test, corrupted-input test, timing).
- Ledger entry naming the tool and the analyses that now depend on it.

## Failure checks

- Has the tool ever failed a test? A tool that has never been seen to fail is untested.
- Does the tool silently drop rows, fill missing values or change units? Assert counts
  in and out.
- Is the tool now so trusted that nobody checks its output? Keep the known-answer test
  in the loop and re-run it after every change.
- Was the tool tuned on the data it will be used to evaluate?

## Worked example — Herman Hollerith (1890 US census)

Tabulating the 1880 US census by hand took most of a decade, and the growing
population threatened to make the 1890 count obsolete before it was finished. Hollerith
built an instrument: each person's answers were punched into a card, and an electric
tabulator read the holes through pins dipping into mercury, advancing counters as it went.
The same machine counted any combination of characteristics, so new questions needed new card sorts,
not new clerical campaigns. The 1890 headline population count came out within weeks,
and the full tabulation took a fraction of the time the 1880 census had needed. The machine's counts were checked
against control totals, an early known-answer test for an instrument.
