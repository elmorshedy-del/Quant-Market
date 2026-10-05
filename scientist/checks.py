"""Post-run verification of a scientist run (`scientist check`).

Reads only run records (LongHorizon's rounds/trajectories/metadata, the hook's
enforcement log, ledger snapshots) and the workspace ledger. Each check states
whether the property was ENFORCED by software during the run, or is only
OBSERVED afterwards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import HOOKS_DIR  # noqa: F401  (puts scientist_rules on sys.path)
import scientist_rules as rules  # noqa: E402

from .launcher import RunPaths


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    kind: str  # "enforced" (software blocked violations during the run) or "observed"


@dataclass
class Report:
    run_id: str
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str, kind: str) -> None:
        self.checks.append(Check(name, bool(ok), detail, kind))

    def markdown(self) -> str:
        lines = [f"# Scientist run check — `{self.run_id}`", "", f"Overall: **{'PASS' if self.ok else 'FAIL'}**", "",
                 "| Check | Result | How | Detail |", "|---|---|---|---|"]
        for check in self.checks:
            detail = check.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {check.name} | {'PASS' if check.ok else 'FAIL'} | {check.kind} | {detail} |")
        lines += ["", "## Facts", "", "```json", json.dumps(self.facts, indent=2, default=str), "```", ""]
        return "\n".join(lines)


def _round_dir(paths: RunPaths, index: int) -> Path:
    return paths.run_dir / "lh_harness" / "role_orchestration" / "rounds" / f"round_{index:03d}"


def _model_of(trajectory: list[dict[str, Any]]) -> str:
    for record in trajectory:
        if record.get("type") == "system" and record.get("subtype") == "init":
            return str(record.get("model") or "")
    return ""


def _hook_events(trajectory: list[dict[str, Any]]) -> list[str]:
    return [
        str(record.get("hook_event") or record.get("hook_name") or "")
        for record in trajectory
        if record.get("type") == "system" and str(record.get("subtype", "")).startswith("hook")
    ]


def check_run(workspace: Path, run_id: str) -> Report:
    paths = RunPaths(workspace, run_id)
    report = Report(run_id)
    rounds = rules.read_jsonl(paths.rounds_jsonl)
    try:
        lh_report = json.loads(paths.report_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        lh_report = {}
    try:
        launcher = json.loads(paths.launcher_record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        launcher = {}
    report.facts.update({
        "workspace": str(workspace),
        "rounds_recorded": len(rounds),
        "lh_status": lh_report.get("status"),
        "lh_abort_reason": lh_report.get("abort_reason"),
        "round_budgets_requested": launcher.get("budgets", []),
        "unattended_gate_decisions": launcher.get("gates", []),
    })
    report.add("run records exist", bool(rounds) and bool(lh_report),
               f"{len(rounds)} rounds in rounds.jsonl; report.json {'present' if lh_report else 'missing'}", "observed")

    # Round limit: LongHorizon enforces max_rounds; the launcher never grants extra rounds at gates.
    budget = sum(int(b) for b in launcher.get("budgets", [])) or int(lh_report.get("max_rounds") or 0)
    indices = sorted(int(r.get("round_index") or 0) for r in rounds)
    report.add("round limit respected", bool(budget) and len(rounds) <= budget and (not indices or indices[-1] <= budget),
               f"{len(rounds)} rounds run, budget {budget} (sum of launcher budgets {launcher.get('budgets', [])}); "
               f"LongHorizon abort_reason={lh_report.get('abort_reason')!r}", "enforced")
    extra_granted = [g for g in launcher.get("gates", []) if g.get("action") == "continue" and g.get("trigger") != "needs_input"]
    report.add("no extra rounds granted at gates", not extra_granted, f"{len(launcher.get('gates', []))} gate decisions", "enforced")

    skills = rules.known_skills(workspace)
    decision_problems: dict[int, list[str]] = {}
    moves: dict[int, str] = {}
    executor_bash: dict[int, int] = {}
    auditor_problems: dict[int, list[str]] = {}
    auditor_bash: dict[int, int] = {}
    auditor_mutation: dict[int, Any] = {}
    skill_calls: dict[str, int] = {}
    models: dict[str, set[str]] = {"manager": set(), "executor": set(), "auditor": set()}
    hook_events_seen: dict[str, int] = {}
    for record in rounds:
        index = int(record.get("round_index") or 0)
        route = str(record.get("next_step") or "")
        rdir = _round_dir(paths, index)
        manager_traj = rules.read_jsonl(rdir / "manager_raw_trajectory.jsonl")
        models["manager"].add(_model_of(manager_traj))
        for use in rules.tool_uses(manager_traj):
            if use["name"] == "Skill":
                key = f"manager:{use['input'].get('skill') or use['input'].get('name')}"
                skill_calls[key] = skill_calls.get(key, 0) + 1
        for event in _hook_events(manager_traj):
            hook_events_seen[event] = hook_events_seen.get(event, 0) + 1
        plan = str(record.get("plan_text") or "")
        decision_problems[index] = rules.manager_problems(plan, round_index=index, skills=skills)
        moves[index] = rules.decision_record(plan).get("Selected move", "").splitlines()[0][:80] if rules.decision_record(plan) else ""
        if route in {"cli", "gui"}:
            executor_traj = rules.read_jsonl(rdir / "executor_raw_trajectory.jsonl")
            models["executor"].add(_model_of(executor_traj))
            uses = rules.tool_uses(executor_traj)
            executor_bash[index] = sum(1 for use in uses if use["name"] == "Bash")
            for use in uses:
                if use["name"] == "Skill":
                    key = f"executor:{use['input'].get('skill') or use['input'].get('name')}"
                    skill_calls[key] = skill_calls.get(key, 0) + 1
            auditor_traj = rules.read_jsonl(rdir / "auditor_raw_trajectory.jsonl")
            models["auditor"].add(_model_of(auditor_traj))
            audit_text = str(record.get("auditor_report") or "")
            auditor_problems[index] = rules.auditor_problems(audit_text)
            auditor_bash[index] = sum(1 for use in rules.tool_uses(auditor_traj) if use["name"] == "Bash")
            try:
                meta = json.loads((rdir / "auditor_metadata.json").read_text(encoding="utf-8")).get("metadata", {})
            except (OSError, ValueError):
                meta = {}
            auditor_mutation[index] = meta.get("verifier_workspace_mutation_detected")

    routed = [i for i, r in ((int(r.get("round_index") or 0), r) for r in rounds) if r.get("next_step") in {"cli", "gui"}]
    bad_decisions = {i: p for i, p in decision_problems.items() if p}
    report.add("manager decision record in every routed round", not bad_decisions,
               "; ".join(f"round {i}: {p[0]}" for i, p in bad_decisions.items()) or
               f"rounds {routed} carry complete records; moves: {moves}", "enforced")
    no_exec = [i for i in routed if executor_bash.get(i, 0) == 0]
    report.add("executor executed commands (performed, not recommended)", bool(routed) and not no_exec,
               f"Bash calls per round: {executor_bash}", "enforced")
    bad_audits = {i: p for i, p in auditor_problems.items() if p}
    report.add("auditor scientific-audit block in every round", bool(routed) and not bad_audits,
               "; ".join(f"round {i}: {p[0]}" for i, p in bad_audits.items()) or f"rounds {routed}", "enforced")
    report.add("auditor recomputed independently (ran its own commands)", bool(routed) and all(auditor_bash.get(i, 0) > 0 for i in routed),
               f"auditor Bash calls per round: {auditor_bash}", "enforced")
    report.add("auditor stayed read-only (LongHorizon mutation guard)", all(v is False for v in auditor_mutation.values()) and bool(auditor_mutation),
               f"mutation_detected per round: {auditor_mutation}", "enforced")
    report.add("auditor model differs from executor model", bool(models["auditor"] - {""}) and not (models["auditor"] & models["executor"] - {""}),
               f"manager={sorted(models['manager'])} executor={sorted(models['executor'])} auditor={sorted(models['auditor'])}", "observed")
    report.facts["skill_tool_calls"] = skill_calls
    report.facts["selected_moves"] = moves
    report.add("skills consulted via the Skill tool", bool(skill_calls), f"{skill_calls or 'none'}", "observed")

    ledger_text = (workspace / rules.LEDGER_RELPATH).read_text(encoding="utf-8") if (workspace / rules.LEDGER_RELPATH).exists() else ""
    entries = rules.parse_entries(ledger_text)
    report.facts["ledger_entries"] = [entry["id"] for entry in entries]
    report.add("ledger structure and entries complete", bool(ledger_text) and not rules.ledger_problems(ledger_text) and bool(entries),
               "; ".join(rules.ledger_problems(ledger_text)[:3]) or f"{len(entries)} entries: {[e['id'] for e in entries]}", "enforced")
    snapshots = sorted((paths.state_dir / "ledger_snapshots").glob("*.md"))
    texts = [s.read_text(encoding="utf-8") for s in snapshots] + [ledger_text]
    preservation = [p for a, b in zip(texts, texts[1:]) for p in rules.preservation_problems(a, b)]
    report.add("rejected hypotheses and entries preserved across the run", not preservation,
               "; ".join(preservation[:3]) or f"{len(snapshots)} snapshots compared; rejected: {sorted(rules.rejected_ids(ledger_text))}", "enforced")
    audit_rows = len(re.findall(rf"(?m)^\| {re.escape(run_id)} \| \d+ \|", ledger_text))
    report.add("audit log persisted to ledger and research/audits/", audit_rows >= len(routed) and len(routed) > 0,
               f"{audit_rows} audit-log rows for this run; files: {len(list((workspace / 'research' / 'audits' / run_id).glob('*.md')))}", "enforced")

    enforcement = rules.read_jsonl(paths.state_dir / "enforcement.jsonl")
    summary: dict[str, int] = {}
    for item in enforcement:
        key = f"{item.get('role')}:{item.get('event')}:{item.get('decision') or ('error' if item.get('error') else 'ok')}"
        summary[key] = summary.get(key, 0) + 1
    report.facts["enforcement_log"] = summary
    errors = [item for item in enforcement if item.get("error")]
    exhausted = [item for item in enforcement if item.get("decision") == "exhausted"]
    roles_started = {item.get("role") for item in enforcement if item.get("event") == "SessionStart"}
    report.add("enforcement hook ran for every role", {"manager", "cli_executor", "cli_auditor"} <= roles_started,
               f"roles with SessionStart events: {sorted(r for r in roles_started if r)}", "observed")
    report.add("no hook errors or exhausted enforcement", not errors and not exhausted,
               f"{len(errors)} errors, {len(exhausted)} exhausted; blocks={sum(1 for i in enforcement if i.get('decision') == 'block')}", "observed")
    return report
