# scientist selftest — 2026-10-05 22:28:40 UTC

Model: `claude-opus-5-5`

| Probe | Result | Evidence |
|---|---|---|
| instruction loading (CLAUDE.md -> SCIENTIST.md) | PASS | repo answer contains objective: True; control without CLAUDE.md lacks it: True |
| skill access (Skill tool) | PASS | Skill calls: ['hidden-states-and-trajectories']; answer: '193' |
| tool execution (Bash in research venv) | PASS | Bash calls: 1; answer: '42' |
| hook enforcement (Manager decision record) | PASS | Stop decisions: ['block', 'pass']; revised output has decision record: True |
| executor enforcement (no transcript, as under LongHorizon) | PASS | real work: ['pass']; recommend-only control: ['block', 'block', 'exhausted'] |
