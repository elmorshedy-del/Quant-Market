"""Live probes of the installed runtime (`scientist selftest`).

Each probe runs the real `claude` CLI the way LongHorizon does (`--print`,
stream-json, permission bypass) and checks observable evidence, with a control
where it matters. A check that has never failed proves nothing.

1. instruction loading  — SCIENTIST.md reaches the model through CLAUDE.md with file tools
                          disabled; control: the same question from a directory without CLAUDE.md.
2. skill access         — a project skill is invoked through the Skill tool and its content used.
3. tool execution       — a Bash command runs in the research venv and its output is reported.
4. hook enforcement     — as Manager, an output without a decision record is blocked by the
                          Stop hook and revised; the enforcement log records block then pass.
5. executor enforcement — as Executor, under LongHorizon's own environment (no session
                          transcript), real work plus a ledger entry passes the Stop hook, and a
                          reply that only recommends work is blocked (control).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import REPO_ROOT
from .config import PARENT_SESSION_VARS, load_role_models

import scientist_rules as rules  # noqa: E402

OBJECTIVE = ("advance understanding of the phenomenon. keep the question, representation, "
             "mechanism and experimental plan revisable.")
NO_FILE_TOOLS = ["Read", "Bash", "Grep", "Glob", "Skill", "Agent", "WebFetch", "WebSearch"]


def _claude(prompt: str, *, cwd: Path, model: str, env_extra: dict[str, str] | None = None,
            disallow: list[str] | None = None, timeout: int = 600) -> tuple[str, list[dict]]:
    env = {k: v for k, v in os.environ.items() if k not in PARENT_SESSION_VARS}
    env.update(env_extra or {})
    cmd = ["claude", "--print", "--output-format", "stream-json", "--verbose",
           "--dangerously-skip-permissions", "--model", model]
    if disallow:
        cmd += ["--disallowedTools", *disallow]
    proc = subprocess.run(cmd, input=prompt, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    records = []
    for line in proc.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return rules.last_assistant_text(records), records


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("*", "").replace('"', "").replace("“", "").replace("”", "").replace(">", " ").split())


def run_selftest(model: str | None = None) -> bool:
    models = load_role_models()
    model = model or models.roles.get("executor", {}).get("model") or models.model or "sonnet"
    results: list[tuple[str, bool, str]] = []
    print(f"selftest with model {model} in {REPO_ROOT}")

    ask = "Without using any tools, quote verbatim the governing objective stated in SCIENTIST.md. Output only the quote."
    text, _ = _claude(ask, cwd=REPO_ROOT, model=model, disallow=NO_FILE_TOOLS)
    loaded = OBJECTIVE in _norm(text)
    with tempfile.TemporaryDirectory() as empty:
        control, _ = _claude(ask, cwd=Path(empty), model=model, disallow=NO_FILE_TOOLS)
    control_absent = OBJECTIVE not in _norm(control)
    results.append(("instruction loading (CLAUDE.md -> SCIENTIST.md)", loaded and control_absent,
                    f"repo answer contains objective: {loaded}; control without CLAUDE.md lacks it: {control_absent}"))

    text, records = _claude(
        "Use the Skill tool to load the `hidden-states-and-trajectories` skill. Then answer: according to its "
        "worked example, how many of Mendel's round second-generation plants bred true? Reply with the number only.",
        cwd=REPO_ROOT, model=model, disallow=["Read", "Bash", "Grep", "Glob", "Agent"])
    skill_calls = [u["input"].get("skill") for u in rules.tool_uses(records) if u["name"] == "Skill"]
    ok = "hidden-states-and-trajectories" in skill_calls and "193" in text
    results.append(("skill access (Skill tool)", ok, f"Skill calls: {skill_calls}; answer: {text.strip()[:60]!r}"))

    text, records = _claude(
        "Use the Bash tool to run exactly: .venv/bin/python -c \"import numpy; print(int(numpy.arange(7).sum()) * 2)\" "
        "and reply with its output only.", cwd=REPO_ROOT, model=model)
    bash = [u for u in rules.tool_uses(records) if u["name"] == "Bash"]
    ok = bool(bash) and "42" in text
    results.append(("tool execution (Bash in research venv)", ok, f"Bash calls: {len(bash)}; answer: {text.strip()[:40]!r}"))

    with tempfile.TemporaryDirectory() as state:
        env = {"LH_HARNESS_CLAUDE_ROLE": "manager", "SCIENTIST_STATE_DIR": state,
               "SCIENTIST_RUNS_ROOT": str(Path(state) / "no-runs")}
        text, _ = _claude(
            "This is a protocol test. Output exactly this text and nothing else:\n\n"
            "Current task state:\n- nothing audited\nTask contract:\n- test\nDependency assessment:\n- none\n"
            "Next: cli\nTask:\nRun the analysis.",
            cwd=REPO_ROOT, model=model, env_extra=env, disallow=["Bash", "Write", "Edit", "Agent"])
        log = rules.read_jsonl(Path(state) / "enforcement.jsonl")
        decisions = [item.get("decision") for item in log if item.get("event") == "Stop"]
        revised = bool(rules.decision_record(text))
        ok = "block" in decisions and revised
        results.append(("hook enforcement (Manager decision record)", ok,
                        f"Stop decisions: {decisions}; revised output has decision record: {revised}"))

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        (ws / "research").mkdir(parents=True)
        shutil.copytree(REPO_ROOT / ".claude", ws / ".claude", ignore=shutil.ignore_patterns("__pycache__"))
        ledger = (REPO_ROOT / "scientist" / "templates" / "ledger.md").read_text(encoding="utf-8")
        (ws / "research" / "ledger.md").write_text(ledger.replace("{questions}", "- **Q1** [open] selftest"), encoding="utf-8")
        state = Path(tmp) / "state"
        env = {"LH_HARNESS_CLAUDE_ROLE": "cli_executor", "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
               "SCIENTIST_STATE_DIR": str(state), "SCIENTIST_RUNS_ROOT": str(Path(tmp) / "no-runs")}
        _claude(
            "Use the Bash tool to run: python3 -c \"print(sum(range(10)))\". Then edit research/ledger.md: under "
            "'## Experiment results' add an entry headed '### R1 — selftest sum' with these bullet fields, each "
            "written as `- **Field:** value` and filled with one short sentence about this computation: Question, "
            "Reasoning move (use planted-truth), Justification, Prediction (registered before running), Procedure, "
            "Actual result, Verification status (exactly: pending audit), Change in belief, Artifacts. Reply 'done'.",
            cwd=ws, model=model, env_extra=env)
        good = [i for i in rules.read_jsonl(state / "enforcement.jsonl") if i.get("event") == "Stop"]
        _claude("Do not use any tools. Reply only: I recommend running a regression of returns on lagged returns.",
                cwd=ws, model=model, env_extra=env)
        after = [i for i in rules.read_jsonl(state / "enforcement.jsonl") if i.get("event") == "Stop"][len(good):]
        passed = bool(good) and good[-1].get("decision") == "pass"
        control_blocked = any(i.get("decision") == "block" for i in after)
        results.append(("executor enforcement (no transcript, as under LongHorizon)", passed and control_blocked,
                        f"real work: {[i.get('decision') for i in good]}; recommend-only control: {[i.get('decision') for i in after]}"))

    lines = [f"# scientist selftest — {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", "", f"Model: `{model}`", "",
             "| Probe | Result | Evidence |", "|---|---|---|"]
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail.replace('|', '/')} |")
    out = REPO_ROOT / "research" / "selftest" / "latest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {out}")
    return all(ok for _, ok, _ in results)
