"""Launcher + LongHorizon integration with a stub `claude` (no model calls).

Verifies, against the real lh-harness supervisor:
- a run created through the web API stops exactly at its round limit,
- the unattended gate policy ends the run instead of granting rounds,
- `resume` continues the same run's round ledger for exactly the extra rounds.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from scientist.config import RoleModels
from scientist.launcher import RunPaths, resume_run, start_run

STUB_BIN = Path(__file__).resolve().parent / "stub_bin"

pytestmark = pytest.mark.skipif(shutil.which("lh-harness") is None, reason="lh-harness not installed")


@pytest.fixture()
def stub_path(monkeypatch):
    monkeypatch.setenv("PATH", f"{STUB_BIN}{os.pathsep}{os.environ['PATH']}")


def test_round_limit_gate_policy_and_resume(tmp_path, stub_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    models = RoleModels(agent="claude_code", model="stub-model",
                        roles={"auditor": {"agent": "claude_code", "model": "stub-auditor"}})
    messages: list[str] = []
    status = start_run(workspace=workspace, task="Stub task.", rounds=2, run_id="stub-run",
                       models=models, max_hours=0.2, say=messages.append)
    paths = RunPaths(workspace, "stub-run")
    rounds = [json.loads(line) for line in paths.rounds_jsonl.read_text().splitlines()]
    assert [r["round_index"] for r in rounds] == [1, 2]
    report = json.loads(paths.report_json.read_text())
    assert report["abort_reason"] == "max_rounds_exhausted"
    assert status in {"incomplete", "stopped", "completed", "failed", "cancelled", "blocked"}
    record = json.loads(paths.launcher_record.read_text())
    assert [g["trigger"] for g in record["gates"]] == ["max_rounds"]
    assert all(g["action"] == "stop" for g in record["gates"])
    assert (workspace / "stub_progress.txt").read_text().count("executor ran") == 2

    resume_run(workspace=workspace, run_id="stub-run", rounds=1, max_hours=0.2, say=messages.append)
    rounds = [json.loads(line) for line in paths.rounds_jsonl.read_text().splitlines()]
    assert [r["round_index"] for r in rounds] == [1, 2, 3]
    assert (workspace / "stub_progress.txt").read_text().count("executor ran") == 3
    record = json.loads(paths.launcher_record.read_text())
    assert record["budgets"] == [2, 1]
    # The auditor really ran with its own model (role bindings reached LongHorizon).
    meta = (paths.run_dir / "lh_harness" / "role_orchestration" / "rounds" / "round_001" / "auditor_raw_trajectory.jsonl").read_text()
    assert "stub-auditor" in meta
