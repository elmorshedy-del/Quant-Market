"""`scientist` command line: launch, resume and verify scientist runs.

    scientist setup                      create .venv with the research stack, check prerequisites
    scientist selftest                   live checks: instruction loading, skill access, hooks, tool use
    scientist run --question @q.md       start a bounded run (default 3 rounds) on this repository
    scientist resume <run-id> --rounds N continue the same run's round ledger
    scientist check <run-id>             verify a finished run (exit 1 on any FAIL)
    scientist status                     list runs in a workspace
    scientist demo all                   blinded synthetic demo: split case + control, 3 rounds each
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import HOOKS_DIR, REPO_ROOT
from .config import load_role_models
from .launcher import RunPaths, resume_run, start_run

import scientist_gate  # noqa: E402  (from HOOKS_DIR)
import scientist_rules as rules  # noqa: E402

TEMPLATE = REPO_ROOT / "scientist" / "task_template.md"
LEDGER_TEMPLATE = REPO_ROOT / "scientist" / "templates" / "ledger.md"


def _text_arg(value: str | None) -> str:
    if not value:
        return ""
    if value.startswith("@"):
        return Path(value[1:]).expanduser().read_text(encoding="utf-8").strip()
    return value.strip()


def build_task(*, question: str, materials: str, deliverables: str, rounds: int) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    extra = f"3. {deliverables.strip()}" if deliverables.strip() else ""
    return (
        template.replace("{question}", question.strip())
        .replace("{materials}", materials.strip() or "The repository in the workspace. Read README.md and docs/ first.")
        .replace("{deliverables}", extra)
        .replace("{rounds}", str(rounds))
    )


def new_run_id(prefix: str = "sci") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


def persist_run_records(workspace: Path, run_id: str) -> Path:
    """Copy decision records and the final reply into research/runs/<run-id>/."""
    paths = RunPaths(workspace, run_id)
    scientist_gate.sync_audits(workspace, paths.run_dir)
    out = workspace / "research" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    for record in rules.read_jsonl(paths.rounds_jsonl):
        index = int(record.get("round_index") or 0)
        (out / f"round_{index:03d}_manager_plan.md").write_text(
            f"# Manager plan — run {run_id}, round {index} (route: {record.get('next_step')})\n\n"
            "_Copied by software from LongHorizon's round record._\n\n" + str(record.get("plan_text") or ""),
            encoding="utf-8",
        )
    try:
        reply = json.loads(paths.report_json.read_text(encoding="utf-8")).get("final_response") or ""
    except (OSError, ValueError):
        reply = ""
    (out / "final_response.md").write_text(f"# Final reply — run {run_id}\n\n{reply}\n", encoding="utf-8")
    return out


def run_check(workspace: Path, run_id: str) -> bool:
    from .checks import check_run

    report = check_run(workspace, run_id)
    out = workspace / "research" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.md").write_text(report.markdown(), encoding="utf-8")
    for check in report.checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] ({check.kind}) {check.name}: {check.detail}")
    print(f"overall: {'PASS' if report.ok else 'FAIL'}  (report: {out / 'check.md'})")
    return report.ok


def cmd_run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    question = _text_arg(args.question)
    if not question:
        raise SystemExit("--question is required (text or @file)")
    ledger = workspace / rules.LEDGER_RELPATH
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(LEDGER_TEMPLATE.read_text(encoding="utf-8").replace("{questions}", f"- **Q1** [open] {question}"), encoding="utf-8")
    task = build_task(question=question, materials=_text_arg(args.materials), deliverables=_text_arg(args.deliverables), rounds=args.rounds)
    run_id = args.run_id or new_run_id()
    status = start_run(workspace=workspace, task=task, rounds=args.rounds, run_id=run_id,
                       models=load_role_models(), max_hours=args.max_hours)
    print(f"run {run_id} finished with status: {status}")
    persist_run_records(workspace, run_id)
    ok = run_check(workspace, run_id)
    print(f"resume with: scientist/bin/scientist resume {run_id} --rounds 1" + (f" --workspace {workspace}" if workspace != REPO_ROOT else ""))
    return 0 if ok else 1


def cmd_resume(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    status = resume_run(workspace=workspace, run_id=args.run_id, rounds=args.rounds, max_hours=args.max_hours)
    print(f"run {args.run_id} finished with status: {status}")
    persist_run_records(workspace, args.run_id)
    return 0 if run_check(workspace, args.run_id) else 1


def cmd_check(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    persist_run_records(workspace, args.run_id)
    return 0 if run_check(workspace, args.run_id) else 1


def cmd_status(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    runs_root = workspace / ".lh-harness" / "runs"
    for run_dir in sorted(runs_root.glob("*")) if runs_root.exists() else []:
        try:
            report = json.loads((run_dir / "lh_harness" / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        rounds = len(rules.read_jsonl(run_dir / "lh_harness" / "role_orchestration" / "rounds.jsonl"))
        print(f"{run_dir.name}\tstatus={report.get('status', '?')}\trounds={rounds}\tabort={report.get('abort_reason', '')}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    venv = REPO_ROOT / ".venv"
    requirements = REPO_ROOT / "scientist" / "requirements-research.txt"
    if not (venv / "bin" / "python").exists():
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", str(venv), "--python", "3.11"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "-r", str(requirements)], check=True)
    else:
        subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements)], check=True)
    for tool in ("claude", "lh-harness"):
        print(f"{tool}: {shutil.which(tool) or 'MISSING'}")
    if shutil.which("lh-harness"):
        subprocess.run(["lh-harness", "doctor"], cwd=REPO_ROOT)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    return 0 if run_selftest(model=args.model) else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import runner

    return runner.main(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scientist", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="create .venv with the research stack and check prerequisites")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("selftest", help="live checks of instruction loading, skills, hooks and tool use")
    p.add_argument("--model", default=None, help="model for the probes (default: the configured executor model)")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("run", help="start a bounded scientist run")
    p.add_argument("--question", required=True, help="research question, or @file")
    p.add_argument("--materials", default="", help="description of the data/materials, or @file")
    p.add_argument("--deliverables", default="", help="extra required deliverables, or @file")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--run-id", default=None)
    p.add_argument("--workspace", default=str(REPO_ROOT))
    p.add_argument("--max-hours", type=float, default=4.0, help="wall-clock safety limit")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue an existing run's round ledger")
    p.add_argument("run_id")
    p.add_argument("--rounds", type=int, default=1, help="additional rounds")
    p.add_argument("--workspace", default=str(REPO_ROOT))
    p.add_argument("--max-hours", type=float, default=4.0)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("check", help="verify a finished run")
    p.add_argument("run_id")
    p.add_argument("--workspace", default=str(REPO_ROOT))
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("status", help="list runs in a workspace")
    p.add_argument("--workspace", default=str(REPO_ROOT))
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("demo", help="blinded synthetic demonstration")
    p.add_argument("action", choices=["init", "run", "evaluate", "all", "resume"])
    p.add_argument("--demo-id", default=None, help="existing demo id (for run/evaluate/resume)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--cases", default="all", help="'all' or a comma list of case labels (dataset-1,dataset-2)")
    p.add_argument("--seed", type=int, default=None, help="answer-key seed (default: random, kept sealed)")
    p.add_argument("--max-hours", type=float, default=4.0)
    p.add_argument("--sequential", action="store_true", help="run cases one after another instead of in parallel")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
