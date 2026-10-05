# Scientific investigation

This run is governed by `SCIENTIST.md` (also loaded through `CLAUDE.md`). Its governing objective:
**"Advance understanding of the phenomenon. Keep the question, representation,
mechanism and experimental plan revisable."**

## Research question

{question}

## Materials

{materials}

## What counts as finishing this run

This is open-ended research within a fixed budget of {rounds} Manager→Executor→Auditor rounds.
The task is complete when all of these hold:

1. Every substantial investigation is recorded in `research/ledger.md` as an
   `### R<id>` entry under *Experiment results*, with all fields from SCIENTIST.md §6.
   Verification statuses reflect the audits.
2. Every research question in the ledger's *Questions* section has a conclusion status
   (**supported**, **rejected**, **insufficient evidence**, or **open**), backed by
   audited evidence. Competing and rejected hypotheses are recorded with reasons.
{deliverables}

Negative findings and "insufficient evidence" are acceptable conclusions. Never
inflate a result to finish. Keep observations, assumptions, interpretations and
predictions separate. Do not present predictive success as evidence for a mechanism.

No human operator is available during this run. Do not route `Next: ask`. Record
open decisions in the ledger as explicit assumptions and proceed.

## Role obligations (in addition to the harness protocol)

**Manager.** Before every executor assignment, which covers both the first substantial investigation and the
step after each experimental result, put a `Scientific decision record:` block
inside your `Task:` section with exactly these labels: `Reading of last result`,
`Blocker`, `Options considered` (at least two materially different, numbered options),
`Selected move` (one installed skill from `.claude/skills/`), `Why this move`,
`Belief-changing observation`, `Executor assignment`, `Budget and stopping rule`.
Read `research/ledger.md` and any skill you consider (Skill tool). Treat only
auditor-confirmed facts as established. Revise the question, representation,
mechanism or plan when the evidence warrants it, and say so in the record. When
ending, include a `Conclusion record:` with a status for each question.

**Executor.** Perform the assigned investigation by writing and running code
(`.venv/bin/python` has numpy, pandas, scipy, statsmodels, scikit-learn). Recommending is not performing.
Copy the Manager's belief-changing observation into the entry's *Prediction* field
before running the decisive analysis. Write the ledger entry (status
`pending audit`) and update earlier entries' statuses from the audits you are given.
Put reusable code in `research/tools/`. Report exact commands, paths and numbers.

**Auditor.** You are read-only. Keep the three control lines first. Treat the Executor's text
as a claim. Inspect the files, recompute at least one key number with your own
command, and judge whether the interpretation exceeds the evidence: overclaimed
mechanisms, uncorrected subgroup searches, leakage between exploration and
confirmation data, or selection effects. Include the `Scientific audit:` block from SCIENTIST.md §5.
