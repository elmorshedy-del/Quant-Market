"""Resolve agent/model settings from the project's LongHorizon config."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import REPO_ROOT

LH_CONFIG = REPO_ROOT / ".lh-harness" / "config.toml"

# Variables that identify an *enclosing* Claude Code session (e.g. when this
# launcher itself runs inside Claude Code). Role processes must not inherit them,
# or they would attach to the parent's session and messaging channel. They do
# not exist in an ordinary terminal, so removing them is harmless there.
PARENT_SESSION_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_REMOTE_SESSION_ID",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_PID",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_AFTER_LAST_COMPACT",
)


@dataclass
class RoleModels:
    agent: str = "claude_code"
    model: str | None = None
    reasoning_effort: str | None = None
    roles: dict[str, dict[str, str]] = field(default_factory=dict)
    timeouts: dict[str, int] = field(default_factory=dict)
    guard_exclude_paths: list[str] = field(default_factory=list)

    def api_roles(self) -> dict[str, dict[str, str]]:
        """Role bindings in the shape LongHorizon's web API accepts."""
        out: dict[str, dict[str, str]] = {}
        for role in ("manager", "executor", "auditor"):
            spec = self.roles.get(role, {})
            entry = {"agent": spec.get("agent") or self.agent}
            model = spec.get("model") or self.model
            if model:
                entry["model"] = model
            effort = spec.get("reasoning_effort") or self.reasoning_effort
            if effort:
                entry["reasoning_effort"] = effort
            out[role] = entry
        return out


def load_role_models(path: Path = LH_CONFIG) -> RoleModels:
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    run = data.get("run", {})
    roles = {
        name: {k: str(v) for k, v in spec.items() if k in {"agent", "model", "reasoning_effort"}}
        for name, spec in (run.get("roles") or {}).items()
        if isinstance(spec, dict) and name in {"manager", "executor", "auditor"}
    }
    return RoleModels(
        agent=str(run.get("agent") or "claude_code"),
        model=run.get("model"),
        reasoning_effort=run.get("reasoning_effort"),
        roles=roles,
        timeouts={k: int(v) for k, v in (run.get("timeouts") or {}).items()},
        guard_exclude_paths=list(run.get("guard_exclude_paths") or []),
    )


def child_environment(workspace: Path, *, run_id: str, runs_root: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in PARENT_SESSION_VARS}
    venv = workspace / ".venv"
    if (venv / "bin" / "python").exists():
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(venv)
    env.update({
        "SCIENTIST_RUN_ID": run_id,
        "SCIENTIST_RUNS_ROOT": str(runs_root),
        # The auditor's read-only guard snapshots the workspace; bytecode caches
        # written by an auditor's `python` call would look like a mutation.
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    })
    return env
