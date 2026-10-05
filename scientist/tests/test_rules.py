"""Known-answer tests for the enforcement rules (each check is shown to pass AND fail)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".claude" / "hooks"))
import scientist_rules as rules  # noqa: E402

SKILLS = {"hidden-states-and-trajectories", "revealing-comparisons", "planted-truth"}

GOOD_PLAN = """\
Current task state:
- Completed: none.
Task contract:
- Target: explain the data.
Dependency assessment:
- Routing rationale: CLI analysis.
Next: cli
Task:
Scientific decision record:
- Reading of last result: round_001 showed the pooled statistic is near zero (audited).
- Blocker: we do not know whether the pooled average hides heterogeneity.
- Options considered:
  1. Per-unit trajectories with a split-half persistence test — high information, cheap.
  2. Selection audit of how units entered the data — medium information.
- Selected move: hidden-states-and-trajectories
- Why this move: dispersion exceeds the noise floor; persistence directly tests types.
- Belief-changing observation: split-half correlation > 0.3 supports types; < 0.1 supports one process.
- Executor assignment: compute per-unit statistics, noise floor, persistence; write ledger entry R2.
- Budget and stopping rule: 20 minutes; stop after the persistence test.
Acceptance criteria:
- Ledger entry R2 complete.
Related audit reports:
- round_001: pooled result.
Boundaries:
- Do not touch data/confirmation.
"""


def test_manager_good_plan_passes():
    assert rules.manager_route(GOOD_PLAN) == "cli"
    record = rules.decision_record(GOOD_PLAN)
    assert set(record) == set(rules.MANAGER_FIELDS)
    assert rules.count_options(record["Options considered"]) == 2
    assert rules.manager_problems(GOOD_PLAN, round_index=2, skills=SKILLS) == []


def test_manager_labels_with_qualifiers_parse():
    plan = GOOD_PLAN.replace("- Executor assignment:", "- Executor assignment, in this order:").replace(
        "- Selected move:", "- **Selected move (primary):**")
    assert rules.manager_problems(plan, round_index=2, skills=SKILLS) == []


def test_manager_missing_record_fails():
    plan = GOOD_PLAN.split("Scientific decision record:")[0] + "Run some analysis.\n"
    problems = rules.manager_problems(plan, round_index=1, skills=SKILLS)
    assert problems and "Scientific decision record" in problems[0]


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (("  2. Selection audit of how units entered the data — medium information.\n", ""), "at least two"),
        (("- Selected move: hidden-states-and-trajectories", "- Selected move: intuition"), "installed skills"),
        (("- Blocker: we do not know whether the pooled average hides heterogeneity.", "- Blocker: TBD"), "'Blocker'"),
        (
            ("- Reading of last result: round_001 showed the pooled statistic is near zero (audited).",
             "- Reading of last result: none yet"),
            "Reading of last result",
        ),
    ],
)
def test_manager_field_mutations_fail(mutation, expected):
    plan = GOOD_PLAN.replace(*mutation)
    problems = rules.manager_problems(plan, round_index=2, skills=SKILLS)
    assert any(expected in problem for problem in problems), problems


def test_manager_round_one_allows_none_yet():
    plan = GOOD_PLAN.replace(
        "- Reading of last result: round_001 showed the pooled statistic is near zero (audited).",
        "- Reading of last result: none yet",
    )
    assert rules.manager_problems(plan, round_index=1, skills=SKILLS) == []


def test_manager_done_requires_conclusion_record():
    done = "Current task state:\n- all audited\nNext: done\nAll requirements met per round_003."
    assert rules.manager_problems(done, round_index=3, skills=SKILLS)
    done_ok = done + "\nConclusion record:\n- Q1: insufficient evidence (round_003) — needs more units."
    assert rules.manager_problems(done_ok, round_index=3, skills=SKILLS) == []


GOOD_AUDIT = """\
Status: complete
Integrity: clean
Contract audit: aligned

Audit facts: the files exist.
Scientific audit:
- Evidence check: read research/ledger.md and analysis/out.csv; 300 rows as claimed.
- Calculation check: recomputed the pooled mean with my own pandas one-liner: 0.0041 vs claimed 0.004.
- Interpretation check: the claim is predictive only; no causal language. Multiple testing: 1 test.
- Ledger check: entry R1 complete and faithful.
- Verdict on claims: pooled mean — confirmed.
Acceptance-constraint backcheck:
- Contract conclusion: aligned
State update for manager:
- R1 confirmed.
"""


def test_auditor_good_report_passes():
    assert rules.auditor_header(GOOD_AUDIT) == {
        "status": "complete", "integrity": "clean", "contract_audit": "aligned",
    }
    assert rules.auditor_problems(GOOD_AUDIT) == []
    assert rules.audit_record(GOOD_AUDIT)["Verdict on claims"].startswith("pooled mean")


def test_auditor_missing_block_or_header_fails():
    assert any("Scientific audit" in p for p in rules.auditor_problems(GOOD_AUDIT.split("Scientific audit:")[0]))
    assert any("control header" in p for p in rules.auditor_problems("Looks fine.\n" + GOOD_AUDIT))
    no_calc = GOOD_AUDIT.replace(
        "- Calculation check: recomputed the pooled mean with my own pandas one-liner: 0.0041 vs claimed 0.004.\n", ""
    )
    assert any("Calculation check" in p for p in rules.auditor_problems(no_calc))


LEDGER_OK = (REPO / "research" / "ledger.md").read_text(encoding="utf-8")

ENTRY = """\
### R1 — Pooled statistics
- **Question:** Q1
- **Reasoning move:** planted-truth
- **Justification:** check the pipeline before real data.
- **Prediction (registered before running):** recovery within 5%.
- **Procedure:** ran research/tools/x.py
- **Actual result:** recovered 0.49 vs planted 0.50.
- **Verification status:** pending audit
- **Change in belief:** pipeline trusted for pooled estimates.
- **Artifacts:** research/tools/x.py
"""


def with_entry(text: str, entry: str) -> str:
    return text.replace("## Rejected explanations", entry + "\n## Rejected explanations")


def test_project_ledger_has_all_sections_and_no_entries():
    assert rules.ledger_problems(LEDGER_OK) == []
    assert rules.parse_entries(LEDGER_OK) == []  # the template entry is inside an HTML comment


def test_entry_complete_and_incomplete():
    ledger = with_entry(LEDGER_OK, ENTRY)
    entries = rules.parse_entries(ledger)
    assert [e["id"] for e in entries] == ["R1"]
    assert rules.ledger_problems(ledger) == []
    broken = with_entry(LEDGER_OK, ENTRY.replace("recovered 0.49 vs planted 0.50.", "TBD"))
    assert any("Actual result" in p for p in rules.ledger_problems(broken))
    bad_status = with_entry(LEDGER_OK, ENTRY.replace("pending audit", "verified by me"))
    assert any("Verification status" in p for p in rules.ledger_problems(bad_status))


def test_changed_entries_detects_new_entry():
    after = with_entry(LEDGER_OK, ENTRY)
    assert [e["id"] for e in rules.changed_entries(LEDGER_OK, after)] == ["R1"]
    assert rules.changed_entries(after, after) == []


def test_rejected_hypotheses_must_be_preserved():
    rejected = LEDGER_OK.replace(
        "_None yet. Use `- **H<n>** rejected",
        "- **H1** rejected in R1: no persistence. Revisit only if: new data.\n_None yet. Use `- **H<n>** rejected",
    )
    assert rules.rejected_ids(rejected) == {"H1"}
    assert rules.preservation_problems(rejected, LEDGER_OK)  # H1 silently removed
    reopened = LEDGER_OK.replace(
        "_None registered yet. Use `- **H<n>** [active]",
        "- **H1** [active, reopened after R4: new evidence]\n_None registered yet. Use `- **H<n>** [active]",
    )
    assert rules.preservation_problems(rejected, reopened) == []
    with_r1 = with_entry(LEDGER_OK, ENTRY)
    assert any("R1" in p for p in rules.preservation_problems(with_r1, LEDGER_OK))


@pytest.mark.parametrize(
    "command, bad",
    [
        ("python analysis/run.py data/explore.csv", False),
        ("cat /usr/lib/python3/os.py", False),
        ("python -c \"print(1/2)\"", False),
        ("ls /tmp/scratch", False),
        ("curl https://example.com/a/b", False),
        ("cat /root/.local/state/secret.json", True),
        ("find / -name '*.json'", True),
        ("ls ~", True),
        ("cat ../../other/file", True),
        ("python -c \"open('/home/user/x')\"", True),
    ],
)
def test_path_confinement(command, bad, tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    violations = rules.path_violations(command, workspace=ws, cwd=ws)
    assert bool(violations) is bad, violations
    inside = rules.path_violations(f"cat {ws}/data/x.csv", workspace=ws, cwd=ws)
    assert inside == []


def run_hook(payload: dict, env: dict, cwd: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(REPO / ".claude" / "hooks" / "scientist_gate.py")],
        input=json.dumps(payload), capture_output=True, text=True, cwd=cwd,
        env={**os.environ, **env}, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "research").mkdir(parents=True)
    (ws / "research" / "ledger.md").write_text(LEDGER_OK, encoding="utf-8")
    skills = ws / ".claude" / "skills"
    for name in SKILLS:
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
    return ws


def transcript(path: Path, tool_names: list[str], prompt: str = "Current management round: 2") -> str:
    lines = [{"type": "user", "message": {"role": "user", "content": prompt}}]
    for name in tool_names:
        lines.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": {"command": "x"}}]}})
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return str(path)


def test_hook_noop_without_role(tmp_path):
    ws = make_workspace(tmp_path)
    out = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "last_assistant_message": "hi"}, {"LH_HARNESS_CLAUDE_ROLE": ""}, ws)
    assert out == {}


def test_hook_manager_block_then_pass(tmp_path):
    ws = make_workspace(tmp_path)
    env = {"LH_HARNESS_CLAUDE_ROLE": "manager"}
    tpath = transcript(tmp_path / "t.jsonl", [])
    run_hook({"hook_event_name": "SessionStart", "cwd": str(ws), "session_id": "s1"}, env, ws)
    bad = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "session_id": "s1", "transcript_path": tpath,
                    "last_assistant_message": "Current task state:\nNext: cli\nTask:\nDo analysis."}, env, ws)
    assert bad.get("decision") == "block"
    good = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "session_id": "s1", "transcript_path": tpath,
                     "last_assistant_message": GOOD_PLAN}, env, ws)
    assert good == {}
    log = [json.loads(l) for l in (ws / ".lh-harness" / "scientist" / "enforcement.jsonl").read_text().splitlines()]
    assert [r.get("decision") for r in log if r["event"] == "Stop"] == ["block", "pass"]


def test_hook_block_limit(tmp_path):
    ws = make_workspace(tmp_path)
    env = {"LH_HARNESS_CLAUDE_ROLE": "manager"}
    tpath = transcript(tmp_path / "t.jsonl", [])
    run_hook({"hook_event_name": "SessionStart", "cwd": str(ws), "session_id": "s2"}, env, ws)
    payload = {"hook_event_name": "Stop", "cwd": str(ws), "session_id": "s2", "transcript_path": tpath,
               "last_assistant_message": "Next: cli\nTask: x"}
    decisions = [run_hook(payload, env, ws).get("decision") for _ in range(3)]
    assert decisions == ["block", "block", None]
    log = (ws / ".lh-harness" / "scientist" / "enforcement.jsonl").read_text()
    assert '"exhausted"' in log


def test_hook_executor_requires_commands_and_entry(tmp_path):
    ws = make_workspace(tmp_path)
    env = {"LH_HARNESS_CLAUDE_ROLE": "cli_executor"}
    run_hook({"hook_event_name": "SessionStart", "cwd": str(ws), "session_id": "e1"}, env, ws)
    no_work = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "session_id": "e1",
                        "transcript_path": transcript(tmp_path / "e.jsonl", ["Read"]),
                        "last_assistant_message": "I recommend running a regression."}, env, ws)
    assert no_work.get("decision") == "block"
    assert "no command was executed" in no_work["reason"] and "not updated" in no_work["reason"]
    (ws / "research" / "ledger.md").write_text(with_entry(LEDGER_OK, ENTRY), encoding="utf-8")
    done = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "session_id": "e1",
                     "transcript_path": transcript(tmp_path / "e2.jsonl", ["Bash", "Write"]),
                     "last_assistant_message": "Ran it; see R1."}, env, ws)
    assert done == {}


def test_hook_auditor_requires_recomputation(tmp_path):
    ws = make_workspace(tmp_path)
    env = {"LH_HARNESS_CLAUDE_ROLE": "cli_auditor"}
    run_hook({"hook_event_name": "SessionStart", "cwd": str(ws), "session_id": "a1"}, env, ws)
    read_only = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "session_id": "a1",
                          "transcript_path": transcript(tmp_path / "a.jsonl", ["Read"]),
                          "last_assistant_message": GOOD_AUDIT}, env, ws)
    assert read_only.get("decision") == "block" and "recomputing" in read_only["reason"]
    ok = run_hook({"hook_event_name": "Stop", "cwd": str(ws), "session_id": "a1",
                   "transcript_path": transcript(tmp_path / "a2.jsonl", ["Read", "Bash"]),
                   "last_assistant_message": GOOD_AUDIT}, env, ws)
    assert ok == {}


def test_hook_pretooluse_allowlist_and_blinding(tmp_path):
    ws = make_workspace(tmp_path)
    env = {"LH_HARNESS_CLAUDE_ROLE": "cli_executor"}
    base = {"hook_event_name": "PreToolUse", "cwd": str(ws), "session_id": "p1"}
    assert run_hook({**base, "tool_name": "Bash", "tool_input": {"command": "cat /root/x"}}, env, ws) == {}
    denied = run_hook({**base, "tool_name": "SendUserFile", "tool_input": {}}, env, ws)
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    (ws / ".scientist").mkdir()
    (ws / ".scientist" / "blinding.json").write_text(json.dumps({
        "confine_to_workspace": True, "allow_web": False,
        "gates": [{"path": "data/confirmation", "unlock_file": "research/preregistration.md", "unlock_pattern": "\\*\\*P\\d+"}],
    }))
    outside = run_hook({**base, "tool_name": "Bash", "tool_input": {"command": "cat /root/x"}}, env, ws)
    assert outside["hookSpecificOutput"]["permissionDecision"] == "deny"
    web = run_hook({**base, "tool_name": "WebSearch", "tool_input": {"query": "x"}}, env, ws)
    assert web["hookSpecificOutput"]["permissionDecision"] == "deny"
    locked = run_hook({**base, "tool_name": "Read", "tool_input": {"file_path": f"{ws}/data/confirmation/a.csv"}}, env, ws)
    assert locked["hookSpecificOutput"]["permissionDecision"] == "deny"
    (ws / "research" / "preregistration.md").write_text("- **P1** registered prediction\n")
    unlocked = run_hook({**base, "tool_name": "Read", "tool_input": {"file_path": f"{ws}/data/confirmation/a.csv"}}, env, ws)
    assert unlocked == {}


def test_hook_syncs_audits_into_ledger(tmp_path):
    ws = make_workspace(tmp_path)
    run_dir = ws / ".lh-harness" / "runs" / "run1"
    rounds = run_dir / "lh_harness" / "role_orchestration"
    rounds.mkdir(parents=True)
    (rounds / "rounds.jsonl").write_text(json.dumps({"round_index": 1, "auditor_report": GOOD_AUDIT}) + "\n")
    env = {"LH_HARNESS_CLAUDE_ROLE": "manager", "SCIENTIST_RUN_ID": "run1"}
    out = run_hook({"hook_event_name": "SessionStart", "cwd": str(ws), "session_id": "m1"}, env, ws)
    assert "Manager role" in out["hookSpecificOutput"]["additionalContext"]
    assert (ws / "research" / "audits" / "run1" / "round_001.md").exists()
    ledger = (ws / "research" / "ledger.md").read_text()
    assert "| run1 | 1 | complete | clean | aligned |" in ledger
    run_hook({"hook_event_name": "SessionStart", "cwd": str(ws), "session_id": "m2"}, env, ws)
    assert (ws / "research" / "ledger.md").read_text().count("| run1 | 1 |") == 1  # idempotent
    assert rules.ledger_problems((ws / "research" / "ledger.md").read_text()) == []
