#!/usr/bin/env python3
"""Role-aware enforcement hook for scientist runs under LongHorizon-Harness.

LongHorizon runs every role as `claude --print` in the workspace and exports
LH_HARNESS_CLAUDE_ROLE (manager, cli_executor, gui_executor, cli_auditor,
gui_auditor, auditor_format_repair, final_response). This hook acts only when
that variable is set, so interactive Claude Code sessions in this repository are
unaffected.

Events handled (configured in .claude/settings.json):
  SessionStart  record a ledger snapshot for the session, sync auditor reports into
                research/audits/ and the ledger's Audit log, and inject a short
                role reminder as context.
  PostToolUse   record each completed tool call in an append-only per-session log.
                (LongHorizon sets CLAUDE_CODE_SKIP_PROMPT_HISTORY=1, so Claude Code
                writes no session transcript; this log is how the Stop check knows
                which tools actually ran.)
  PreToolUse    tool allowlist for harness roles; in blinded workspaces
                (.scientist/blinding.json) also confine paths to the workspace and
                keep gated data locked until a prediction is registered.
  Stop          block the role from finishing until its output has the required
                form (SCIENTIST.md §5): the Manager's decision record, the Executor's
                executed commands and complete ledger entry, and the Auditor's
                scientific-audit block plus an independent inspection.

Every decision is appended to .lh-harness/scientist/enforcement.jsonl, which
`scientist check` reads. All checks concern form and presence, not quality.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scientist_rules as rules  # noqa: E402

MAX_BLOCKS_PER_SESSION = 2
EXECUTOR_ROLES = {"cli_executor", "gui_executor"}
AUDITOR_ROLES = {"cli_auditor", "gui_auditor"}
SYNC_ROLES = {"manager", "cli_executor", "gui_executor"}


# ----------------------------------------------------------------------- helpers

def workspace_of(data: dict) -> Path:
    return Path(data.get("cwd") or os.getcwd()).resolve()


def state_dir(ws: Path) -> Path:
    path = Path(os.environ.get("SCIENTIST_STATE_DIR") or ws / ".lh-harness" / "scientist")
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(ws: Path, record: dict) -> None:
    record = {"ts": time.time(), "role": os.environ.get("LH_HARNESS_CLAUDE_ROLE", ""), **record}
    try:
        with open(state_dir(ws) / "enforcement.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def session_file(ws: Path, session_id: str) -> Path:
    path = state_dir(ws) / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id or 'unknown')}.json"


def load_session(ws: Path, session_id: str) -> dict:
    try:
        return json.loads(session_file(ws, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_session(ws: Path, session_id: str, state: dict) -> None:
    session_file(ws, session_id).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def tools_log(ws: Path, session_id: str) -> Path:
    return session_file(ws, session_id).with_suffix(".tools.jsonl")


def record_tool(ws: Path, session_id: str, tool: str) -> None:
    # One small O_APPEND write per call, so parallel tool calls do not clobber each other.
    with open(tools_log(ws, session_id), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": time.time(), "tool": tool}) + "\n")


def tools_used(ws: Path, session_id: str, transcript_path: str | None) -> list[str]:
    names = [str(item.get("tool")) for item in rules.read_jsonl(tools_log(ws, session_id))]
    if not names and transcript_path:
        names = [use["name"] for use in rules.tool_uses(rules.read_jsonl(transcript_path))]
    return names


def rounds_recorded(run_dir: Path | None) -> int | None:
    if run_dir is None:
        return None
    return len(rules.read_jsonl(run_dir / "lh_harness" / "role_orchestration" / "rounds.jsonl"))


def read_ledger(ws: Path) -> str:
    try:
        return (ws / rules.LEDGER_RELPATH).read_text(encoding="utf-8")
    except OSError:
        return ""


def blinding_config(ws: Path) -> dict:
    try:
        return json.loads((ws / ".scientist" / "blinding.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload))


# ------------------------------------------------------------------ audit syncing

def current_run_dir(ws: Path) -> Path | None:
    runs_root = Path(os.environ.get("SCIENTIST_RUNS_ROOT") or ws / ".lh-harness" / "runs")
    run_id = os.environ.get("SCIENTIST_RUN_ID", "")
    if run_id:
        candidate = runs_root / run_id
        return candidate if candidate.is_dir() else None
    candidates = sorted(
        runs_root.glob("*/lh_harness/role_orchestration/rounds.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parents[2] if candidates else None


def sync_audits(ws: Path, run_dir: Path | None) -> int:
    """Copy auditor reports into research/audits/<run>/ and the ledger's Audit log.

    Idempotent. Returns the number of newly written report files.
    """
    if run_dir is None:
        return 0
    rounds_path = run_dir / "lh_harness" / "role_orchestration" / "rounds.jsonl"
    rounds = [r for r in rules.read_jsonl(rounds_path) if str(r.get("auditor_report") or "").strip()]
    if not rounds:
        return 0
    out_dir = ws / "research" / "audits" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    rows: dict[tuple[str, int], str] = {}
    for record in rounds:
        index = int(record.get("round_index") or 0)
        round_dir = run_dir / "lh_harness" / "role_orchestration" / "rounds" / f"round_{index:03d}"
        report = rules.full_auditor_report(round_dir, str(record["auditor_report"])).strip()
        header = rules.auditor_header(report)
        target = out_dir / f"round_{index:03d}.md"
        body = (
            f"# Audit — run {run_dir.name}, round {index}\n\n"
            "_Copied by software from the LongHorizon auditor report; do not edit._\n\n"
            f"{report}\n"
        )
        if not target.exists() or target.read_text(encoding="utf-8") != body:
            target.write_text(body, encoding="utf-8")
            written += 1
        rel = target.relative_to(ws).as_posix()
        rows[(run_dir.name, index)] = (
            f"| {run_dir.name} | {index} | {header.get('status', '?')} | {header.get('integrity', '?')} "
            f"| {header.get('contract_audit', '?')} | `{rel}` |"
        )
    update_audit_log(ws, rows)
    return written


def update_audit_log(ws: Path, new_rows: dict[tuple[str, int], str]) -> None:
    ledger_path = ws / rules.LEDGER_RELPATH
    text = read_ledger(ws)
    if not text:
        return
    match = re.search(r"(?m)^## Audit log[^\n]*\n", text)
    if not match:
        text = text.rstrip("\n") + "\n\n## Audit log\n"
        match = re.search(r"(?m)^## Audit log[^\n]*\n", text)
    start = match.end()
    next_heading = re.search(r"(?m)^## ", text[start:])
    end = start + next_heading.start() if next_heading else len(text)
    existing: dict[tuple[str, int], str] = {}
    for line in text[start:end].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 6 and cells[1].isdigit():
            existing[(cells[0], int(cells[1]))] = line.strip()
    existing.update(new_rows)
    table = [
        "",
        "_Software-maintained from auditor reports; do not edit by hand._",
        "",
        "| Run | Round | Status | Integrity | Contract audit | Report |",
        "|---|---|---|---|---|---|",
        *[existing[key] for key in sorted(existing)],
        "",
    ]
    new_text = text[:start] + "\n".join(table) + ("\n" + text[end:] if next_heading else "")
    if new_text != text:
        ledger_path.write_text(new_text, encoding="utf-8")


# --------------------------------------------------------------------- reminders

REMINDERS = {
    "manager": (
        "Scientist run, Manager role. Follow SCIENTIST.md §5. Inside your route's `Task:` section, "
        "include 'Scientific decision record:' with the labels: Reading of last result, Blocker, "
        "Options considered (>=2 numbered, materially different), Selected move (an installed skill "
        "name), Why this move, Belief-changing observation, Executor assignment, Budget and stopping "
        "rule. When ending, include 'Conclusion record:' with a status per question (supported / "
        "rejected / insufficient evidence / open). A hook checks these fields before you may finish. "
        "Read research/ledger.md before deciding."
    ),
    "executor": (
        "Scientist run, Executor role. Follow SCIENTIST.md §5–6. Perform the investigation by running "
        "commands (use .venv/bin/python); do not just recommend it. Add or update an entry under "
        "'## Experiment results' in research/ledger.md with every field filled (Reasoning move, "
        "Justification, Prediction (registered before running), Actual result, Verification status = "
        "'pending audit', Change in belief). A hook checks that you ran commands and that the entry is "
        "complete before you may finish."
    ),
    "auditor": (
        "Scientist run, Auditor role. Follow SCIENTIST.md §5. Keep the three harness control lines "
        "first. Independently inspect files and recompute at least one key number with your own command. "
        "Include 'Scientific audit:' with Evidence check, Calculation check, Interpretation check, Ledger "
        "check, Verdict on claims. A hook checks these before you may finish."
    ),
}


def reminder_for(role: str) -> str:
    if role == "manager":
        return REMINDERS["manager"]
    if role in EXECUTOR_ROLES:
        return REMINDERS["executor"]
    if role in AUDITOR_ROLES:
        return REMINDERS["auditor"]
    return ""


# ----------------------------------------------------------------------- events

def on_session_start(data: dict, role: str, ws: Path) -> None:
    session_id = str(data.get("session_id") or "")
    ledger = read_ledger(ws)
    synced = 0
    if role in SYNC_ROLES:
        try:
            synced = sync_audits(ws, current_run_dir(ws))
        except Exception as exc:  # never break a role over bookkeeping
            log(ws, {"event": "SessionStart", "warning": f"audit sync failed: {exc!r}"})
        ledger = read_ledger(ws)
    snapshots = state_dir(ws) / "ledger_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots / f"{int(time.time() * 1000)}_{role}.md"
    snapshot_path.write_text(ledger, encoding="utf-8")
    round_index = None
    if role == "manager":
        done = rounds_recorded(current_run_dir(ws))
        round_index = None if done is None else done + 1
    save_session(ws, session_id, {
        "role": role,
        "round_index": round_index,
        "started": time.time(),
        "ledger_sha": rules.sha256_text(ledger),
        "ledger_snapshot": str(snapshot_path),
        "blocks": 0,
    })
    log(ws, {"event": "SessionStart", "session_id": session_id, "audits_synced": synced})
    context = reminder_for(role)
    if context:
        emit({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}})


def deny(ws: Path, tool: str, reason: str) -> None:
    log(ws, {"event": "PreToolUse", "decision": "deny", "tool": tool, "reason": reason})
    emit({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    })


def on_pre_tool_use(data: dict, role: str, ws: Path) -> None:
    tool = str(data.get("tool_name") or "")
    tool_input = data.get("tool_input") or {}
    config = blinding_config(ws)
    if tool.startswith("mcp__"):
        if os.environ.get("SCIENTIST_ALLOW_MCP") != "1":
            deny(ws, tool, "MCP tools are disabled for scientist harness roles (set SCIENTIST_ALLOW_MCP=1 to allow).")
        return
    if tool not in rules.HARNESS_ALLOWED_TOOLS:
        deny(ws, tool, f"Tool '{tool}' is outside the scientist harness allowlist; use files, shell and skills only.")
        return
    if not config:
        return
    if tool in rules.WEB_TOOLS and not config.get("allow_web", False):
        deny(ws, tool, "Web access is disabled in this blinded investigation; use only the workspace data.")
        return
    text = rules.tool_input_text(tool_input)
    if config.get("confine_to_workspace", False):
        violations = rules.path_violations(
            text,
            workspace=str(ws),
            cwd=str(Path(data.get("cwd") or ws)),
            extra_allowed=config.get("extra_allowed_paths", []),
        )
        if violations:
            deny(ws, tool, "This blinded investigation is confined to the workspace " f"({ws}); disallowed path references: {', '.join(violations[:5])}. Use relative paths inside the workspace.")
            return
    for gate in config.get("gates", []):
        gated = str(gate.get("path", "")).strip("/")
        if not gated or gated not in text:
            continue
        unlock_file = ws / str(gate.get("unlock_file", ""))
        pattern = str(gate.get("unlock_pattern", ".+"))
        try:
            unlocked = bool(re.search(pattern, unlock_file.read_text(encoding="utf-8")))
        except OSError:
            unlocked = False
        if not unlocked:
            deny(ws, tool, gate.get("message") or f"'{gated}' is locked until {gate.get('unlock_file')} registers a prediction.")
            return
        log(ws, {"event": "PreToolUse", "decision": "allow_gated", "tool": tool, "gate": gated})


def on_post_tool_use(data: dict, role: str, ws: Path) -> None:
    record_tool(ws, str(data.get("session_id") or ""), str(data.get("tool_name") or ""))


def stop_problems(data: dict, role: str, ws: Path, state: dict) -> list[str]:
    records = rules.read_jsonl(data.get("transcript_path") or "")
    text = str(data.get("last_assistant_message") or "") or rules.last_assistant_text(records)
    used = tools_used(ws, str(data.get("session_id") or ""), data.get("transcript_path"))
    if role == "manager":
        round_index = state.get("round_index") or rules.round_index_from_prompt(rules.first_user_text(records))
        return rules.manager_problems(text, round_index=round_index, skills=rules.known_skills(ws))
    if role in EXECUTOR_ROLES:
        problems: list[str] = []
        if "Bash" not in used:
            problems.append(
                "no command was executed: the Executor must perform the investigation (run code), not only describe or recommend it"
            )
        ledger = read_ledger(ws)
        if rules.sha256_text(ledger) == state.get("ledger_sha"):
            problems.append(f"{rules.LEDGER_RELPATH} was not updated in this episode; record the investigation as an entry under '## Experiment results'")
        else:
            try:
                before = Path(state.get("ledger_snapshot", "")).read_text(encoding="utf-8")
            except OSError:
                before = ""
            changed = rules.changed_entries(before, ledger)
            if not changed:
                problems.append("the ledger changed but no '### R<id>' entry under '## Experiment results' was added or updated")
            for entry in changed:
                problems.extend(rules.entry_problems(entry))
            problems.extend(rules.preservation_problems(before, ledger))
        return problems
    if role in AUDITOR_ROLES:
        problems = rules.auditor_problems(text)
        if not any(name in {"Read", "Grep", "Glob", "Bash", "LS"} for name in used):
            problems.append("no independent inspection was made: read the files or run a command before judging the claims")
        elif not rules.calculation_not_applicable(text) and "Bash" not in used:
            problems.append("'Calculation check' requires recomputing at least one number with your own command (Bash), or stating 'not applicable' when no calculation was claimed")
        return problems
    return []


def on_stop(data: dict, role: str, ws: Path) -> None:
    if role not in {"manager"} | EXECUTOR_ROLES | AUDITOR_ROLES:
        return
    session_id = str(data.get("session_id") or "")
    state = load_session(ws, session_id)
    problems = stop_problems(data, role, ws, state)
    if not problems:
        log(ws, {"event": "Stop", "session_id": session_id, "decision": "pass", "blocks_before_pass": state.get("blocks", 0)})
        return
    blocks = int(state.get("blocks", 0))
    if blocks >= MAX_BLOCKS_PER_SESSION:
        log(ws, {"event": "Stop", "session_id": session_id, "decision": "exhausted", "problems": problems})
        return
    state["blocks"] = blocks + 1
    save_session(ws, session_id, state)
    log(ws, {"event": "Stop", "session_id": session_id, "decision": "block", "problems": problems})
    advice = {
        "manager": "Re-emit your COMPLETE management result (all protocol sections and exactly one route) with the decision record inside `Task:`; only your final message is used.",
        "executor": "Fix this now (run the work and/or complete the ledger entry), then end with your complete executor report; only your final message is used.",
        "auditor": "Re-emit your COMPLETE audit report, starting with the three control lines; only your final message is used.",
    }["manager" if role == "manager" else "executor" if role in EXECUTOR_ROLES else "auditor"]
    emit({
        "decision": "block",
        "reason": "Scientist protocol check failed (SCIENTIST.md §5):\n- " + "\n- ".join(problems) + "\n" + advice,
    })


def main() -> int:
    role = os.environ.get("LH_HARNESS_CLAUDE_ROLE", "").strip()
    if not role:
        return 0
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    ws = workspace_of(data)
    event = data.get("hook_event_name")
    try:
        if event == "SessionStart":
            on_session_start(data, role, ws)
        elif event == "PreToolUse":
            on_pre_tool_use(data, role, ws)
        elif event == "PostToolUse":
            on_post_tool_use(data, role, ws)
        elif event == "Stop":
            on_stop(data, role, ws)
    except Exception as exc:  # a hook crash must never wedge a role; record it instead
        log(ws, {"event": str(event), "error": repr(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
