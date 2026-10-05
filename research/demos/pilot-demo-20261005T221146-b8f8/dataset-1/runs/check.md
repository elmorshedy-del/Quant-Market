# Scientist run check — `demo-20261005T221146-b8f8-dataset-1`

Overall: **FAIL**

| Check | Result | How | Detail |
|---|---|---|---|
| run records exist | PASS | observed | 1 rounds in rounds.jsonl; report.json present |
| round limit respected | PASS | enforced | 1 rounds run, budget 1 (sum of launcher budgets [1]); LongHorizon abort_reason='max_rounds_exhausted' |
| no extra rounds granted at gates | PASS | enforced | 1 gate decisions |
| manager decision record in every routed round | PASS | enforced | rounds [1] carry complete records; moves: {1: 'hidden-states-and-trajectories'} |
| executor executed commands (performed, not recommended) | PASS | enforced | Bash calls per round: {1: 22} |
| auditor scientific-audit block in every round | PASS | enforced | rounds [1] |
| auditor recomputed independently (ran its own commands) | PASS | enforced | auditor Bash calls per round: {1: 4} |
| auditor stayed read-only (LongHorizon mutation guard) | PASS | enforced | mutation_detected per round: {1: False} |
| auditor model differs from executor model | PASS | observed | manager=['claude-opus-5-5'] executor=['claude-opus-5-5'] auditor=['claude-sonnet-5-5'] |
| skills consulted via the Skill tool | PASS | observed | {'manager:hidden-states-and-trajectories': 1} |
| ledger structure and entries complete | PASS | enforced | 6 entries: ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'] |
| rejected hypotheses and entries preserved across the run | PASS | enforced | 4 snapshots compared; rejected: ['H1', 'H2', 'H3', 'H6'] |
| audit log persisted to ledger and research/audits/ | PASS | enforced | 1 audit-log rows for this run; files: 1 |
| enforcement hook ran for every role | PASS | observed | roles with SessionStart events: ['cli_auditor', 'cli_executor', 'final_response', 'manager'] |
| no hook errors or exhausted enforcement | FAIL | observed | 0 errors, 2 exhausted; blocks=5 |

## Facts

```json
{
  "workspace": "/tmp/claude-0/-home-user-Quant-Market/df163b49-c1ec-5647-8cab-004a08ce7907/scratchpad/pilot-demos/demo-20261005T221146-b8f8/dataset-1",
  "rounds_recorded": 1,
  "lh_status": "incomplete",
  "lh_abort_reason": "max_rounds_exhausted",
  "round_budgets_requested": [
    1
  ],
  "unattended_gate_decisions": [
    {
      "ts": 1791239208.7135339,
      "trigger": "max_rounds",
      "action": "stop",
      "note": "unattended policy for gate 'max_rounds'"
    }
  ],
  "skill_tool_calls": {
    "manager:hidden-states-and-trajectories": 1
  },
  "selected_moves": {
    "1": "hidden-states-and-trajectories"
  },
  "ledger_entries": [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6"
  ],
  "enforcement_log": {
    "manager:SessionStart:ok": 1,
    "manager:Stop:block": 1,
    "manager:Stop:pass": 1,
    "cli_executor:SessionStart:ok": 1,
    "cli_executor:PreToolUse:deny": 2,
    "cli_executor:PreToolUse:allow_gated": 1,
    "cli_executor:Stop:block": 2,
    "cli_executor:Stop:exhausted": 1,
    "cli_auditor:SessionStart:ok": 1,
    "cli_auditor:Stop:block": 2,
    "cli_auditor:Stop:exhausted": 1,
    "final_response:SessionStart:ok": 1
  }
}
```
