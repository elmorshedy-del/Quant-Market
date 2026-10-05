# AGENTS.md

All agents doing research in this repository follow **[SCIENTIST.md](SCIENTIST.md)**.
Read it in full before starting. It defines the governing objective, how to label
observations, assumptions, interpretations and predictions, the Manager/Executor/Auditor
obligations, and the ledger format.

- Durable research state: `research/ledger.md`. Read it first. Rejected hypotheses are never deleted.
- Reasoning-move skills (Agent Skills format): `.claude/skills/<name>/SKILL.md`.
  Runtimes that do not discover that directory can read a skill directly, e.g.
  "read `.claude/skills/hidden-states-and-trajectories/SKILL.md` and apply it".
- BootLoops protocols, installed at project scope in the same directory: `acceptance-gate`,
  `planted-truth`, `independence-bookkeeping`, `timing-discipline`, `tool-stewardship`.
- Launch and resume autonomous runs with `scientist/bin/scientist` (see `scientist/README.md`).
- Python for analysis: `.venv/bin/python`.
