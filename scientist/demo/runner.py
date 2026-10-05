"""Blinded synthetic demonstration: `scientist demo init|run|evaluate|all|resume`.

Layout (defaults; override with SCIENTIST_DEMO_ROOT / SCIENTIST_SEALED_ROOT):
  ~/scientist-demos/<demo-id>/dataset-1/          investigation workspace (agent sees this)
  ~/scientist-demos/<demo-id>/dataset-2/          investigation workspace (agent sees this)
  ~/.local/state/scientist/sealed/<demo-id>/      answer key (agent never sees this)
  <repo>/research/demos/<demo-id>/                results copied back after evaluation

The workspaces live outside the repository so the generator source and this file
are not inside the agents' working tree. Which dataset hides a split is decided
by a random seed stored only in the sealed key; the two workspaces receive
identical instructions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .. import REPO_ROOT
from ..cli import build_task, persist_run_records, run_check
from ..config import load_role_models
from ..launcher import resume_run, start_run
from . import evaluate, generator

TEMPLATE_DIR = Path(__file__).resolve().parent / "workspace_template"
DEMO_ROOT = Path(os.environ.get("SCIENTIST_DEMO_ROOT") or Path.home() / "scientist-demos")
SEALED_ROOT = Path(os.environ.get("SCIENTIST_SEALED_ROOT") or Path.home() / ".local" / "state" / "scientist" / "sealed")
RESULTS_ROOT = REPO_ROOT / "research" / "demos"

QUESTIONS = [
    "What process generates each asset's next-period return in this dataset, and how does it "
    "depend on the information available (the asset's own past returns and its characteristics)?",
    "Do all assets share one such process, or are there distinct groups of assets whose "
    "next-period returns are generated differently? Conclude: single process, multiple processes, "
    "or insufficient evidence.",
]

MATERIALS = (
    "A synthetic daily-return panel described in `data/README.md`. `data/explore/` is open. "
    "`data/confirmation/` holds different assets in the same format. Software keeps it locked "
    "until at least one prediction is registered in the ledger's Predictions section "
    "(`- **P1** ...`); use it to confirm, not to explore. All work stays inside this workspace, "
    "and tool calls that reference paths outside it are blocked. After the run, a frozen copy of "
    "your predictor will be scored on fresh assets that do not exist yet."
)

DELIVERABLES = (
    "`research/predictor/predict.py` implementing the interface in `research/predictor/README.md`. "
    "It will be scored after the run on fresh assets from the same source.\n"
    "4. `research/conclusions.json` with keys: `heterogeneity` (one of `single_process`, "
    "`multiple_processes`, `insufficient_evidence`), `n_processes` (integer or null), "
    "`description` (the process(es) you infer, with parameter estimates), "
    "`characteristics_associated` (list of characteristics you found associated with any grouping, "
    "or []), and `evidence_entries` (ledger entry ids)."
)

PREDICTION_GATE = r"(?m)^## Predictions[^\n]*\n(?:(?!## )[^\n]*\n)*?[^\n]*\*\*P\d+\*\*"


def demo_task(rounds: int) -> str:
    question = "\n".join(f"- **Q{i}.** {q}" for i, q in enumerate(QUESTIONS, 1))
    return build_task(question=question, materials=MATERIALS, deliverables=DELIVERABLES, rounds=rounds)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _scientist_md_for_workspace() -> str:
    text = (REPO_ROOT / "SCIENTIST.md").read_text(encoding="utf-8")
    # §9 describes the Quant-Market repository, which is not part of a demo workspace.
    return re.split(r"(?m)^## 9\. Project context", text)[0].rstrip() + "\n"


def make_workspace(key: dict[str, Any], label: str, workspace: Path, *, say=print) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "SCIENTIST.md").write_text(_scientist_md_for_workspace(), encoding="utf-8")
    shutil.copy2(REPO_ROOT / "AGENTS.md", workspace / "AGENTS.md")
    _copy_tree(TEMPLATE_DIR, workspace)
    for part in ("skills", "hooks"):
        _copy_tree(REPO_ROOT / ".claude" / part, workspace / ".claude" / part)
    shutil.copy2(REPO_ROOT / ".claude" / "settings.json", workspace / ".claude" / "settings.json")
    (workspace / ".lh-harness").mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / ".lh-harness" / "config.toml", workspace / ".lh-harness" / "config.toml")
    (workspace / ".scientist").mkdir(exist_ok=True)
    (workspace / ".scientist" / "blinding.json").write_text(json.dumps({
        "confine_to_workspace": True,
        "allow_web": False,
        "gates": [{
            "path": "data/confirmation",
            "unlock_file": "research/ledger.md",
            "unlock_pattern": PREDICTION_GATE,
            "message": "data/confirmation is locked until the ledger's Predictions section registers a prediction (`- **P1** ...`).",
        }],
    }, indent=2), encoding="utf-8")
    for part, folder in (("explore", "explore"), ("confirm", "confirmation")):
        sample = generator.generate_part(key, label, part)
        target = workspace / "data" / folder
        target.mkdir(parents=True, exist_ok=True)
        sample["assets"].to_csv(target / "assets.csv", index=False)
        sample["returns"].to_csv(target / "returns.csv", index=False)
    questions = "\n".join(f"- **Q{i}** [open] {q}" for i, q in enumerate(QUESTIONS, 1))
    ledger = (REPO_ROOT / "scientist" / "templates" / "ledger.md").read_text(encoding="utf-8")
    (workspace / "research" / "ledger.md").write_text(ledger.replace("{questions}", questions), encoding="utf-8")
    _make_venv(workspace, say=say)


def _make_venv(workspace: Path, *, say=print) -> None:
    requirements = REPO_ROOT / "scientist" / "requirements-research.txt"
    venv = workspace / ".venv"
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", str(venv), "--python", "3.11", "-q"], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python"), "-r", str(requirements)], check=True)
    else:
        subprocess.run([shutil.which("python3") or "python3", "-m", "venv", str(venv)], check=True)
        subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    say(f"  venv ready: {venv}")


def sealed_dir(demo_id: str) -> Path:
    return SEALED_ROOT / demo_id


def load_key(demo_id: str) -> dict[str, Any]:
    return json.loads((sealed_dir(demo_id) / "key.json").read_text(encoding="utf-8"))


def workspaces(demo_id: str) -> dict[str, Path]:
    return {label: DEMO_ROOT / demo_id / label for label in ("dataset-1", "dataset-2")}


def init(seed: int | None = None, *, say=print) -> str:
    demo_id = f"demo-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:4]}"
    key = generator.make_answer_key(seed)
    key.update({"demo_id": demo_id, "created_at": time.time()})
    sealed = sealed_dir(demo_id)
    sealed.mkdir(parents=True, mode=0o700)
    key_path = sealed / "key.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    key_path.chmod(0o600)
    for label, ws in workspaces(demo_id).items():
        say(f"creating workspace {ws}")
        make_workspace(key, label, ws, say=say)
    results = RESULTS_ROOT / demo_id
    results.mkdir(parents=True, exist_ok=True)
    (results / "manifest.json").write_text(json.dumps({
        "demo_id": demo_id,
        "workspaces": {label: str(ws) for label, ws in workspaces(demo_id).items()},
        "sealed_key": str(key_path),
        "pass_criteria_declared_before_run": generator.CRITERIA,
        "created_at": key["created_at"],
    }, indent=2), encoding="utf-8")
    say(f"demo {demo_id} initialised; answer key sealed at {key_path} (not shown)")
    return demo_id


def _labels(cases: str) -> list[str]:
    return ["dataset-1", "dataset-2"] if cases == "all" else [c.strip() for c in cases.split(",") if c.strip()]


def run(demo_id: str, *, rounds: int, cases: str = "all", parallel: bool = True, max_hours: float = 4.0, resume: bool = False) -> dict[str, str]:
    statuses: dict[str, str] = {}
    models = load_role_models()

    def one(label: str) -> None:
        ws = workspaces(demo_id)[label]
        run_id = f"{demo_id}-{label}"
        say = lambda msg: print(f"[{label}] {msg}", flush=True)  # noqa: E731
        try:
            if resume:
                statuses[label] = resume_run(workspace=ws, run_id=run_id, rounds=rounds, max_hours=max_hours, say=say)
            else:
                statuses[label] = start_run(workspace=ws, task=demo_task(rounds), rounds=rounds, run_id=run_id,
                                            models=models, max_hours=max_hours, say=say)
            persist_run_records(ws, run_id)
        except Exception as exc:  # report and keep the other case running
            statuses[label] = f"error: {exc!r}"
            say(f"run failed: {exc!r}")

    labels = _labels(cases)
    if parallel and len(labels) > 1:
        threads = [threading.Thread(target=one, args=(label,)) for label in labels]
        for thread in threads:
            thread.start()
            time.sleep(5)
        for thread in threads:
            thread.join()
    else:
        for label in labels:
            one(label)
    return statuses


def evaluate_demo(demo_id: str, *, cases: str = "all") -> dict[str, Any]:
    key = load_key(demo_id)
    results = RESULTS_ROOT / demo_id
    results.mkdir(parents=True, exist_ok=True)
    sensitive = [str(SEALED_ROOT), str(REPO_ROOT), "scientist/demo", "key.json", "answer_key"]
    summary: dict[str, Any] = {"demo_id": demo_id, "cases": {}}
    for label in _labels(cases):
        ws = workspaces(demo_id)[label]
        run_id = f"{demo_id}-{label}"
        print(f"[{label}] evaluating on fresh data …", flush=True)
        outcome = evaluate.evaluate_case(key, label, ws, sensitive=sensitive)
        print(f"[{label}] protocol check:", flush=True)
        outcome["protocol_check_pass"] = run_check(ws, run_id)
        summary["cases"][label] = outcome
        case_dir = results / label
        case_dir.mkdir(parents=True, exist_ok=True)
        for rel in ("research/ledger.md", "research/conclusions.json"):
            if (ws / rel).exists():
                shutil.copy2(ws / rel, case_dir / Path(rel).name)
        for rel in ("research/predictor", "research/tools", f"research/runs/{run_id}", f"research/audits/{run_id}"):
            if (ws / rel).exists():
                _copy_tree(ws / rel, case_dir / Path(rel).parts[1])  # predictor/, tools/, runs/, audits/
    # Unblinding happens only now, after every case has been scored.
    shutil.copy2(sealed_dir(demo_id) / "key.json", results / "answer_key.json")
    (results / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (results / "evaluation.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    return summary


def render(summary: dict[str, Any]) -> str:
    lines = [f"# Demo evaluation — `{summary['demo_id']}`", "",
             "Scored on fresh assets generated after the run (never seen by the agent).", "",
             "| Dataset | Hidden truth | Agent claim | MSE ratio vs pooled (95% CI) | Captured gain | Fresh-data verdict | Protocol check | Leak |",
             "|---|---|---|---|---|---|---|---|"]
    for label, case in summary["cases"].items():
        scores = case.get("scores") or {}
        ratio = f"{scores['mse_ratio_vs_pooled']:.4f} ({scores['mse_ratio_ci95'][0]:.4f}–{scores['mse_ratio_ci95'][1]:.4f})" if scores else "n/a"
        gain = scores.get("captured_gain")
        truth = case["true_config"]
        truth_text = (f"split: phi={[round(p, 3) for p in truth['phi']]}, proxy={truth['proxy']}" if truth["kind"] == "split"
                      else f"single process: phi={round(truth['phi'][0], 3)}")
        lines.append(
            f"| {label} | {truth_text} | {case.get('claim')} | {ratio} | {'—' if gain is None else f'{gain:.2f}'} "
            f"| {'PASS' if case['verdict']['pass'] else 'FAIL'} | {'PASS' if case.get('protocol_check_pass') else 'FAIL'} "
            f"| {'YES' if case['leakage']['leak_detected'] else 'no'} |"
        )
    lines += ["", "Verdict reasons:"]
    for label, case in summary["cases"].items():
        lines.append(f"- {label}: {'; '.join(case['verdict']['reasons']) or 'all criteria met'}")
        claim = case.get("claim_record") or {}
        if claim.get("characteristics_associated") is not None:
            lines.append(f"  - characteristics the agent associated with groups: {claim.get('characteristics_associated')}")
        if case["leakage"]["hook_denials"]:
            lines.append(f"  - hook denials during the run: {len(case['leakage']['hook_denials'])}")
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> int:
    demo_id = args.demo_id
    if args.action in {"init", "all"}:
        demo_id = init(args.seed)
    if not demo_id:
        raise SystemExit("--demo-id is required for this action")
    if args.action in {"run", "all"}:
        print(run(demo_id, rounds=args.rounds, cases=args.cases, parallel=not args.sequential, max_hours=args.max_hours))
    if args.action == "resume":
        print(run(demo_id, rounds=args.rounds, cases=args.cases, parallel=not args.sequential, max_hours=args.max_hours, resume=True))
    if args.action in {"evaluate", "all", "resume"}:
        summary = evaluate_demo(demo_id, cases=args.cases)
        return 0 if all(c["verdict"]["pass"] for c in summary["cases"].values()) else 1
    return 0
