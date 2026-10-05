"""Drive LongHorizon-Harness runs through its documented web control API.

Why a thin wrapper instead of plain `lh-harness run`:
- LongHorizon's CLI cannot resume. Resuming ("continue the same round ledger")
  exists only in its supervisor, which is exposed through `lh-harness web`
  (`POST /api/runs`, `POST /api/runs/{id}/resume`). Creating runs through the same
  supervisor makes every run resumable.
- Supervised runs raise a human approval gate at the end of the run (completed /
  round limit reached) and whenever the Manager asks a question. Unattended
  runs need a deterministic policy for those gates. This wrapper never grants
  extra rounds, so the configured round limit holds.
- Role processes must not inherit the identity of an enclosing Claude Code session.

Everything else (role prompts, isolation, auditing, round accounting) is
LongHorizon's own behaviour.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import RoleModels, child_environment

TERMINAL = {"completed", "failed", "cancelled", "blocked", "incomplete", "stopped", "aborted"}

NO_OPERATOR_ANSWER = (
    "No human operator is available in this unattended scientist run. Decide using "
    "SCIENTIST.md, record the decision as an explicit assumption in research/ledger.md, and proceed."
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def lh_harness_binary() -> str:
    found = shutil.which("lh-harness")
    if not found:
        raise SystemExit(
            "lh-harness is not on PATH. Install it with `uv tool install lh-harness` "
            "(see scientist/README.md)."
        )
    return found


class WebSupervisor:
    """A private `lh-harness web` process bound to localhost for one launcher call."""

    def __init__(self, workspace: Path, runs_root: Path, env: dict[str, str], log_path: Path):
        self.workspace = workspace
        self.runs_root = runs_root
        self.env = env
        self.log_path = log_path
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 60.0) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self.process = subprocess.Popen(
            [
                lh_harness_binary(), "web",
                "--workspace-root", str(self.workspace),
                "--runs-root", str(self.runs_root),
                "--host", "127.0.0.1", "--port", str(self.port), "--no-open",
            ],
            cwd=str(self.workspace), env=self.env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        log.close()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"lh-harness web exited early; see {self.log_path}")
            try:
                self.request("GET", "/api/meta")
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        raise RuntimeError(f"lh-harness web did not become ready; see {self.log_path}")

    def request(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict[str, Any]:
        data = json.dumps(body or {}).encode("utf-8") if method != "GET" else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@dataclass
class RunPaths:
    workspace: Path
    run_id: str

    @property
    def runs_root(self) -> Path:
        return self.workspace / ".lh-harness" / "runs"

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def state_dir(self) -> Path:
        return self.workspace / ".lh-harness" / "scientist"

    @property
    def rounds_jsonl(self) -> Path:
        return self.run_dir / "lh_harness" / "role_orchestration" / "rounds.jsonl"

    @property
    def report_json(self) -> Path:
        return self.run_dir / "lh_harness" / "report.json"

    @property
    def launcher_record(self) -> Path:
        return self.state_dir / "runs" / f"{self.run_id}.json"


def _record(paths: RunPaths, update: dict[str, Any]) -> dict[str, Any]:
    paths.launcher_record.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(paths.launcher_record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {"run_id": paths.run_id, "workspace": str(paths.workspace), "budgets": [], "gates": []}
    for key, value in update.items():
        if key in {"budgets", "gates"}:
            current.setdefault(key, []).extend(value)
        else:
            current[key] = value
    paths.launcher_record.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def _round_lines(paths: RunPaths, seen: int, say: Callable[[str], None]) -> int:
    try:
        lines = paths.rounds_jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        return seen
    for line in lines[seen:]:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        header = " / ".join(
            part.split(":", 1)[-1].strip()
            for part in (record.get("auditor_report") or "").splitlines()[:3]
        )
        say(f"  round {record.get('round_index')}: route={record.get('next_step')} audit=[{header or 'none'}]")
    return len(lines)


def _handle_gates(sup: WebSupervisor, paths: RunPaths, snapshot: dict[str, Any], answered: list[int],
                  say: Callable[[str], None]) -> None:
    for approval in snapshot.get("approvals") or []:
        if approval.get("status") != "pending":
            continue
        trigger = (approval.get("context") or {}).get("trigger", "")
        if trigger == "needs_input" and not answered:
            action, note, user_input = "continue", "unattended: no operator; manager told to decide", NO_OPERATOR_ANSWER
            answered.append(1)
        else:
            # completed, max_rounds, needs_human, repeated_failure, or a second question:
            # end the run. Never grant extra rounds from an unattended gate.
            action, note, user_input = "stop", f"unattended policy for gate '{trigger}'", ""
        sup.request(
            "POST",
            f"/api/runs/{paths.run_id}/approvals/{approval['approval_id']}/resolve",
            {"action": action, "reason": note, "user_input": user_input},
        )
        _record(paths, {"gates": [{"ts": time.time(), "trigger": trigger, "action": action, "note": note}]})
        say(f"  gate '{trigger}' -> {action}")


def _wait(sup: WebSupervisor, paths: RunPaths, *, max_hours: float, say: Callable[[str], None]) -> str:
    deadline = time.time() + max_hours * 3600
    seen = 0
    answered: list[int] = []
    status = "unknown"
    stop_sent = False
    while True:
        try:
            snapshot = sup.request("GET", f"/api/runs/{paths.run_id}/snapshot")
            _handle_gates(sup, paths, snapshot, answered, say)
            info = sup.request("GET", f"/api/runs/{paths.run_id}/status")
        except RuntimeError as exc:
            say(f"  (status poll failed: {exc})")
            time.sleep(5)
            continue
        status = str(info.get("status") or "unknown")
        seen = _round_lines(paths, seen, say)
        if status in TERMINAL and not info.get("alive"):
            return status
        if time.time() > deadline and not stop_sent:
            say(f"  wall-clock limit of {max_hours} h reached; stopping the run")
            sup.request("POST", f"/api/runs/{paths.run_id}/stop")
            stop_sent = True
        time.sleep(5)


def start_run(*, workspace: Path, task: str, rounds: int, run_id: str, models: RoleModels,
              max_hours: float = 4.0, say: Callable[[str], None] = print) -> str:
    paths = RunPaths(workspace, run_id)
    env = child_environment(workspace, run_id=run_id, runs_root=paths.runs_root)
    sup = WebSupervisor(workspace, paths.runs_root, env, paths.state_dir / "web" / f"{run_id}.log")
    _record(paths, {"budgets": [rounds], "created_at": time.time(), "models": models.api_roles()})
    sup.start()
    try:
        body = {
            "task": task,
            "agent": models.agent,
            "model": models.model,
            "roles": models.api_roles(),
            "workspace": str(workspace),
            "run_id": run_id,
            "max_rounds": rounds,
            "prompt_language": "en",
        }
        if models.reasoning_effort:
            body["reasoning_effort"] = models.reasoning_effort
        sup.request("POST", "/api/runs", body)
        say(f"started run {run_id} ({rounds} rounds) in {workspace}")
        status = _wait(sup, paths, max_hours=max_hours, say=say)
    finally:
        sup.stop()
    _record(paths, {"final_status": status, "finished_at": time.time()})
    return status


def resume_run(*, workspace: Path, run_id: str, rounds: int, max_hours: float = 4.0,
               say: Callable[[str], None] = print) -> str:
    paths = RunPaths(workspace, run_id)
    if not paths.run_dir.is_dir():
        raise SystemExit(f"no such run: {paths.run_dir}")
    env = child_environment(workspace, run_id=run_id, runs_root=paths.runs_root)
    sup = WebSupervisor(workspace, paths.runs_root, env, paths.state_dir / "web" / f"{run_id}.log")
    _record(paths, {"budgets": [rounds], "resumed_at": time.time()})
    sup.start()
    try:
        sup.request("POST", f"/api/runs/{run_id}/resume", {"mode": "continue", "extra_rounds": rounds})
        say(f"resumed run {run_id} with {rounds} more round(s)")
        status = _wait(sup, paths, max_hours=max_hours, say=say)
    finally:
        sup.stop()
    _record(paths, {"final_status": status, "finished_at": time.time()})
    return status
