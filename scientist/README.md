# Scientist agent setup

An autonomous research loop for this repository. It runs on
[LongHorizon-Harness](https://github.com/AMAP-ML/LongHorizon-Harness) (Manager → Executor →
Auditor rounds), with Claude Code as the agent runtime. Shared scientific instructions live in
[`SCIENTIST.md`](../SCIENTIST.md), reasoning-move skills in `.claude/skills/`, durable state in
[`research/ledger.md`](../research/ledger.md), and enforcement hooks in `.claude/hooks/`.

## Startup

Requirements: Python ≥ 3.10, [uv](https://docs.astral.sh/uv/), and the Claude Code CLI
(`claude`), installed and logged in. Check with `claude --version`.

```bash
# once per machine
uv tool install lh-harness              # LongHorizon-Harness (tested: 0.1.7)

# once per clone (from the repository root)
scientist/bin/scientist setup           # .venv with numpy/pandas/scipy/statsmodels/sklearn; runs `lh-harness doctor`
scientist/bin/scientist selftest        # ~2 min of live probes: instructions, skills, tools, hooks

# launch a bounded investigation (3 rounds) on this repository
scientist/bin/scientist run --question @research/questions/q1.md --rounds 3

# resume the same run (continues its round ledger) with one more round
scientist/bin/scientist resume <run-id> --rounds 1

# verify any finished run (exit code 1 on any FAIL)
scientist/bin/scientist check <run-id>
```

The blinded synthetic demonstration (two datasets, three rounds each, then fresh-data scoring):

```bash
scientist/bin/scientist demo all --rounds 3                         # init + run both cases + evaluate
scientist/bin/scientist demo resume --demo-id <demo-id> --rounds 1  # one more round each, then re-evaluate
```

Each run prints its run id. Results are written to `research/runs/<run-id>/`: `check.md`,
`final_response.md`, and the Manager's per-round plans. Auditor reports go to
`research/audits/<run-id>/`. Demo results go to `research/demos/<demo-id>/`. LongHorizon's raw
run records stay under `.lh-harness/runs/` (git-ignored).

## How it works

| Requirement | Mechanism | Kind |
|---|---|---|
| Shared scientific instructions | `SCIENTIST.md`, imported by `CLAUDE.md` (and referenced by `AGENTS.md`). Claude Code loads it in every role because LongHorizon runs roles as `claude --print` in the workspace. | instruction |
| Manager obligations (blocker, ≥2 options, chosen move + why, belief-changing observation, bounded assignment) | Stated in the task text, which LongHorizon gives every role as "Original task" (`scientist/task_template.md`). The **Stop hook** blocks a Manager reply that routes to an executor without a complete `Scientific decision record:`. | **enforced** (form) |
| Executor performs, not recommends | The **Stop hook** blocks an executor that ran no command, or that did not add or complete an `### R<id>` ledger entry. | **enforced** (form) |
| Auditor checks independently | LongHorizon makes the auditor read-only (no write tools, plus a workspace mutation guard). The **Stop hook** requires the `Scientific audit:` block and at least one command the auditor ran itself when calculations are claimed. The auditor uses a different model from the executor. | **enforced** (form) + configuration |
| Reasoning moves | 8 skills plus 5 BootLoops protocols in `.claude/skills/`, discovered by Claude Code. The Manager's `Selected move` must name one of them. | instruction + enforced name |
| Round limit | LongHorizon's `max_rounds`. The launcher resolves every end-of-run approval gate with "stop" and never grants extra rounds. | **enforced** |
| Durable state | `research/ledger.md`. The hook snapshots it at each session start, rejects deletion of rejected hypotheses or entries, and copies auditor reports into `research/audits/` and the ledger's *Audit log*. | **enforced** (preservation) |
| Tool boundary | The PreToolUse hook gives harness roles an allowlist: files, shell, skills and web only. Messaging, scheduling and MCP tools are denied. | **enforced** |

All hook checks concern **form and presence**. Whether a move was well chosen, or a conclusion
is warranted, is judged by the Auditor (a model) and by you. No software in this setup judges
scientific quality.

### Why a thin wrapper exists

LongHorizon's prompts are fixed in its package, so they are not edited. The supported extension
points used here are: the task text, the project's `CLAUDE.md`, skills and hooks (which LongHorizon's
Claude Code adapter loads, and which can see `LH_HARNESS_CLAUDE_ROLE`), and
`.lh-harness/config.toml`. The wrapper (`scientist/launcher.py`) exists for three reasons:

1. LongHorizon's CLI cannot resume. Resume ("continue the same round ledger") exists only in its
   supervisor, behind `lh-harness web` (`POST /api/runs`, `POST /api/runs/{id}/resume`). The
   launcher starts a private, localhost-only `lh-harness web` for each call and uses that API, so every run is
   resumable.
2. Supervised runs stop at human approval gates (run complete, round limit reached, question).
   Unattended runs need a fixed policy. The launcher answers the Manager's first question with
   "no operator; decide and record the assumption" and resolves every other gate with "stop".
3. When the launcher itself runs inside Claude Code, role processes must not inherit that
   session's identity variables, so the launcher scrubs them.

## Models

Only the Claude Code runtime is installed here (doctor: `codex`, `opencode`, `dsh` not found).
LongHorizon's Claude model list is a set of unverified suggestions, so model IDs were verified by
running `claude -p --model <id>` and reading the served model from the JSON result
(2026-10-05):

| ID | Result |
|---|---|
| `claude-opus-5-5` | served `claude-opus-5-5` — **Manager, Executor** |
| `claude-sonnet-5-5` | served `claude-sonnet-5-5` — **Auditor** (deliberately a different model from the Executor) |
| `claude-opus-5` | served `claude-opus-5` (LongHorizon's built-in default) |
| `opus`, `sonnet`, `haiku` | aliases served `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-4-5` |
| `claude-fable-5-1` | rejected for this account ("out of usage credits") |

Change the models in `.lh-harness/config.toml` (`[run.roles.*]`).

## The blinded demonstration

`scientist demo` creates two investigation workspaces **outside the repository**
(`~/scientist-demos/<demo-id>/dataset-1|2/`), each with its own copy of the instructions, skills,
hooks and ledger, a private `.venv`, and a synthetic panel of 300 assets × 40 daily returns
(plus 150 held-out confirmation assets). Both receive identical instructions.

- One dataset hides **two processes under a near-zero average**: half the assets trend and half mean-revert, with
  one characteristic weakly associated with the type. The other is a **control** with
  one homogeneous process and irrelevant characteristics. Which is which, and all parameters,
  come from a random seed stored only in the sealed key
  (`~/.local/state/scientist/sealed/<demo-id>/key.json`). The key is copied into the results only after scoring.
- `data/confirmation/` is locked by the hook until a prediction is registered in the ledger.
- After the run, the agent's predictor is hashed (frozen), and only then are fresh assets generated. The predictor is scored one step ahead on
  600 fresh assets × 10 periods, against a pooled AR(1) baseline (the "overall average"
  model) and the oracle. Pass criteria are fixed in `scientist/demo/generator.py` before any run.
  The split case must claim `multiple_processes` and capture ≥ 50% of the oracle's gain over the pooled
  model, with a 95% CI excluding zero improvement. The control must not claim a split, and its MSE must stay
  ≤ 1.01× the pooled model's, so an overfit per-asset model fails.
- The evaluator itself passes planted-truth tests (`scientist/tests/test_demo.py`). A trajectory-based
  two-group predictor passes the split case (captured gain 0.85–0.90). The pooled model fails
  it (0.00). An overfit per-asset model fails the control (MSE ratio 1.04–1.06).

**Blinding is detectable, not sandboxed.** All agents run as the same OS user. Defences: the
workspace excludes the repository and the key; a PreToolUse hook blocks tool calls that reference
paths outside the workspace (home, `/`, `..`); and a post-run audit searches every workspace file
and tool transcript for the key's canary string and for sealed paths. A script that constructs a
path at run time could evade the hook. It would be caught only if the key's contents surfaced in
a transcript or file. A detected leak invalidates the demo result.

## Tests

```bash
.venv/bin/python -m pytest scientist/tests -q
```

- `test_rules.py`: every hook rule shown to pass and fail (decision record, ledger entries,
  rejected-hypothesis preservation, path confinement, gate, audit sync), including the live hook
  script.
- `test_launcher_stub.py`: the real `lh-harness` supervisor with a stub `claude`. The run stops at the
  round limit, the gate policy stops rather than grants rounds, and resume continues the same ledger.
- `test_demo.py`: planted-truth checks of the generator and the evaluator.

## Results

See the bottom of this file (filled in from actual runs).
