# CLAUDE.md

Shared scientific instructions for all research work in this repository:

@SCIENTIST.md

## Project notes for Claude Code

- The app (FastAPI quant-strategy tournament) lives in `app/`; see `README.md`.
- Research state lives in `research/ledger.md`. Read it before starting research work.
- Reasoning-move skills and the BootLoops protocols are in `.claude/skills/`.
- Autonomous research runs go through LongHorizon-Harness via `scientist/bin/scientist`
  (see `scientist/README.md`). In those runs, `LH_HARNESS_CLAUDE_ROLE` identifies your
  role, and `.claude/hooks/scientist_gate.py` enforces parts of SCIENTIST.md.
- Use `.venv/bin/python` for analysis (numpy, pandas, scipy, statsmodels, scikit-learn).
