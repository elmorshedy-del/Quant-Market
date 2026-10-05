# Pilot (shakedown) run — not the demonstration

A single-round run on one dataset (`dataset-1` only), done before the real demo to find
integration problems (timing-discipline: pilot first). It ran with hook code that had two
defects, both fixed afterwards (commit "Fix hook defects found by the pilot run"):

1. Tool use was read from Claude Code session transcripts. LongHorizon disables those
   (`CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`), so the Executor (22 Bash calls) and the Auditor (4 Bash
   calls) were wrongly blocked twice each for "no command executed", and enforcement was then
   logged as exhausted.
2. Path confinement flagged `...` in a docstring and a `cd ../../..` that returned to the
   workspace root. Both commands were denied.

A third finding, a Manager blocked for writing "Executor assignment, in this order:", was a
parser false positive and was fixed earlier.

Outcome despite the defects: claim `multiple_processes` with characteristic `age` (both
correct), and 99% of the oracle's fresh-data gain captured (`evaluation.md`). The protocol check
FAILs on the exhausted enforcement, as it should (see `dataset-1/runs/check.md`, regenerated with the fixed checker: every other check passes). The pilot workspace lived under a scratch
`/tmp` directory, which the confinement hook allows, so this run is **not** evidence about blinding.
