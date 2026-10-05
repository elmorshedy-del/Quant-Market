# Demo evaluation — `demo-20261005T221146-b8f8`

Scored on fresh assets generated after the run (never seen by the agent).

| Dataset | Hidden truth | Agent claim | MSE ratio vs pooled (95% CI) | Captured gain | Fresh-data verdict | Protocol check | Leak |
|---|---|---|---|---|---|---|---|
| dataset-1 | split: phi=[0.416, -0.431], proxy=age | multiple_processes | 0.8294 (0.8079–0.8523) | 0.99 | PASS | FAIL | no |

Verdict reasons:
- dataset-1: all criteria met
  - characteristics the agent associated with groups: ['age']
  - hook denials during the run: 2
