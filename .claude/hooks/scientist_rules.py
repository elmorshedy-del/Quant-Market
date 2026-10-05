"""Shared, stdlib-only rules for the scientist setup.

Used at run time by the Claude Code hook (`scientist_gate.py`, same directory) and
after a run by `scientist check`. Everything here checks *form and presence*
(fields exist, are non-empty, tools were actually used). It cannot judge scientific
quality; that is the Auditor's job and the reader's.

Keep this module free of third-party imports: hooks run under the system python3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- ledger

LEDGER_RELPATH = "research/ledger.md"

LEDGER_SECTIONS = (
    "Questions",
    "Observations and sources",
    "Measurement and selection assumptions",
    "Competing hypotheses",
    "Predictions",
    "Experiment results",
    "Rejected explanations",
    "Unresolved questions",
    "Next useful actions",
    "Audit log",
)

ENTRY_FIELDS = (
    "Question",
    "Reasoning move",
    "Justification",
    "Prediction (registered before running)",
    "Procedure",
    "Actual result",
    "Verification status",
    "Change in belief",
    "Artifacts",
)

# The per-investigation fields the user-facing spec requires.
REQUIRED_ENTRY_FIELDS = (
    "Reasoning move",
    "Justification",
    "Prediction (registered before running)",
    "Actual result",
    "Verification status",
    "Change in belief",
)

VERIFICATION_STATUSES = (
    "pending audit",
    "audited: confirmed",
    "audited: disputed",
    "audited: partially confirmed",
)

_PLACEHOLDER_VALUES = {"", "tbd", "todo", "...", "…", "-", "none", "n/a", "pending"}

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_FIELD_LINE_RE = re.compile(r"^\s*[-*]\s*\*\*([^*]+?)\*\*\s*:?\s*(.*)$")
_ENTRY_ID_RE = re.compile(r"^\s*(R[0-9][\w.-]*)")
_HYP_ID_RE = re.compile(r"\*\*(H\d+[a-z]?)\*\*")


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().rstrip(":").strip()).lower()


def strip_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text or "")


def split_sections(text: str) -> dict[str, str]:
    """Map `## Heading` -> body (HTML comments removed)."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in strip_comments(text).splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(lines)
            current = line[3:].strip()
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines)
    return sections


def find_section(sections: dict[str, str], name: str) -> str | None:
    target = _norm(name)
    for key, body in sections.items():
        if _norm(key) == target or _norm(key).startswith(target):
            return body
    return None


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current: str | None = None
    for line in body.splitlines():
        match = _FIELD_LINE_RE.match(line)
        if match:
            current = match.group(1).strip().rstrip(":").strip()
            fields[current] = match.group(2).strip()
        elif current is not None and line.strip():
            fields[current] = (fields[current] + "\n" + line.strip()).strip()
    return fields


def field_value(fields: dict[str, str], name: str) -> str:
    """Look a field up by exact name, then by its leading word(s)."""
    target = _norm(name)
    for key, value in fields.items():
        if _norm(key) == target:
            return value
    head = target.split(" (")[0]
    for key, value in fields.items():
        if _norm(key).split(" (")[0] == head:
            return value
    return ""


def is_placeholder(value: str) -> bool:
    cleaned = value.strip().strip("_*` ").lower()
    if cleaned in _PLACEHOLDER_VALUES:
        return True
    return bool(re.fullmatch(r"<[^>]*>", cleaned))


def parse_entries(ledger_text: str) -> list[dict[str, Any]]:
    sections = split_sections(ledger_text)
    body = find_section(sections, "Experiment results") or ""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    buffer: list[str] = []
    for line in body.splitlines():
        if line.startswith("### "):
            if current is not None:
                current["fields"] = parse_fields("\n".join(buffer))
                current["body"] = "\n".join(buffer)
                entries.append(current)
            heading = line[4:].strip()
            id_match = _ENTRY_ID_RE.match(heading)
            current = {"id": id_match.group(1) if id_match else heading, "heading": heading}
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        current["fields"] = parse_fields("\n".join(buffer))
        current["body"] = "\n".join(buffer)
        entries.append(current)
    return entries


def entry_problems(entry: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    fields = entry.get("fields", {})
    for name in REQUIRED_ENTRY_FIELDS:
        value = field_value(fields, name)
        if name == "Verification status":
            if not value or not value.strip().lower().startswith(VERIFICATION_STATUSES):
                problems.append(
                    f"{entry['id']}: 'Verification status' must start with one of: "
                    + ", ".join(VERIFICATION_STATUSES)
                )
            continue
        if not value or is_placeholder(value):
            problems.append(f"{entry['id']}: field '{name}' is missing or empty")
    return problems


def ledger_problems(ledger_text: str) -> list[str]:
    sections = split_sections(ledger_text)
    problems = [f"missing section '## {name}'" for name in LEDGER_SECTIONS if find_section(sections, name) is None]
    for entry in parse_entries(ledger_text):
        problems.extend(entry_problems(entry))
    return problems


def hypothesis_ids(section_body: str | None) -> set[str]:
    return set(_HYP_ID_RE.findall(section_body or ""))


def rejected_ids(ledger_text: str) -> set[str]:
    return hypothesis_ids(find_section(split_sections(ledger_text), "Rejected explanations"))


def preservation_problems(old_text: str, new_text: str) -> list[str]:
    """Rejected hypotheses and investigation entries may never disappear.

    A rejected hypothesis may leave *Rejected explanations* only if it reappears
    under *Competing hypotheses* marked `reopened`.
    """
    problems: list[str] = []
    new_sections = split_sections(new_text)
    new_rejected = hypothesis_ids(find_section(new_sections, "Rejected explanations"))
    competing = find_section(new_sections, "Competing hypotheses") or ""
    for hyp in sorted(rejected_ids(old_text) - new_rejected):
        reopened = re.search(rf"\*\*{re.escape(hyp)}\*\*[^\n]*reopen", competing, re.I)
        if not reopened:
            problems.append(f"rejected hypothesis {hyp} was removed without a recorded re-opening")
    old_ids = {entry["id"] for entry in parse_entries(old_text)}
    new_ids = {entry["id"] for entry in parse_entries(new_text)}
    for entry_id in sorted(old_ids - new_ids):
        problems.append(f"investigation entry {entry_id} was deleted")
    return problems


def changed_entries(old_text: str, new_text: str) -> list[dict[str, Any]]:
    old = {entry["id"]: entry.get("body", "") for entry in parse_entries(old_text)}
    return [entry for entry in parse_entries(new_text) if old.get(entry["id"]) != entry.get("body", "")]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- manager output

MANAGER_FIELDS = (
    "Reading of last result",
    "Blocker",
    "Options considered",
    "Selected move",
    "Why this move",
    "Belief-changing observation",
    "Executor assignment",
    "Budget and stopping rule",
)

CONCLUSION_STATUSES = ("supported", "rejected", "insufficient evidence", "open")

_ROUTE_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*next\s*:\s*\**\s*(gui|cli|ask|done|complete|blocked)\b"
)
_DECISION_HEADER_RE = re.compile(r"(?i)scientific\s+decision\s+record\s*:?\**")
_CONCLUSION_HEADER_RE = re.compile(r"(?i)conclusion\s+record\s*:?\**")
_ROUND_RE = re.compile(r"Current management round:\s*(\d+)")
_SECTION_STOP_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*(?:acceptance\s+criteria|related\s+audit\s+reports|related\s+audited\s+state|"
    r"boundaries|next\s*:|current\s+task\s+state|task\s+contract|dependency\s+assessment)\b"
)


def manager_route(text: str) -> str:
    matches = _ROUTE_RE.findall(text or "")
    if not matches:
        return "invalid"
    route = matches[-1].lower()
    return "done" if route == "complete" else route


def _label_pattern(label: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(part) for part in label.split())
    return re.compile(rf"(?im)^\s*(?:[-*•]|\d+[.)])?\s*\**\s*{words}\s*\**\s*:\s*\**")


def decision_record(text: str) -> dict[str, str]:
    """Extract the Scientific decision record fields (last record wins)."""
    text = text or ""
    headers = list(_DECISION_HEADER_RE.finditer(text))
    if not headers:
        return {}
    block = text[headers[-1].end():]
    stop = _SECTION_STOP_RE.search(block)
    if stop:
        block = block[: stop.start()]
    positions: list[tuple[int, int, str]] = []
    for label in MANAGER_FIELDS:
        match = _label_pattern(label).search(block)
        if match:
            positions.append((match.start(), match.end(), label))
    positions.sort()
    record: dict[str, str] = {}
    for index, (_, end, label) in enumerate(positions):
        stop_at = positions[index + 1][0] if index + 1 < len(positions) else len(block)
        record[label] = block[end:stop_at].strip()
    return record


def count_options(value: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•]|\([a-z0-9]\))\s+\S", value or ""))


def known_skills(workspace: str | os.PathLike[str]) -> set[str]:
    root = Path(workspace) / ".claude" / "skills"
    if not root.is_dir():
        return set()
    return {path.parent.name for path in root.glob("*/SKILL.md")}


def manager_problems(text: str, *, round_index: int | None, skills: Iterable[str]) -> list[str]:
    route = manager_route(text)
    problems: list[str] = []
    if route in {"gui", "cli"}:
        record = decision_record(text)
        if not record:
            return [
                "the `Task:` section has no 'Scientific decision record:' block "
                "(required before every executor assignment; see SCIENTIST.md §5)"
            ]
        for label in MANAGER_FIELDS:
            value = record.get(label, "")
            if label == "Reading of last result":
                if (round_index or 1) >= 2 and (not value or is_placeholder(value) or "none yet" in value.lower()):
                    problems.append(
                        "'Reading of last result' must interpret the last audited result (this is not round 1)"
                    )
                continue
            if not value or is_placeholder(value):
                problems.append(f"decision record field '{label}' is missing or empty")
        if count_options(record.get("Options considered", "")) < 2:
            problems.append("'Options considered' must list at least two materially different numbered options")
        selected = record.get("Selected move", "").lower()
        skill_set = {name.lower() for name in skills}
        if skill_set and not any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", selected) for name in skill_set):
            problems.append(
                "'Selected move' must name one of the installed skills: " + ", ".join(sorted(skill_set))
            )
    elif route in {"done", "blocked"}:
        headers = list(_CONCLUSION_HEADER_RE.finditer(text or ""))
        tail = (text or "")[headers[-1].end():].lower() if headers else ""
        if not headers or not any(status in tail for status in CONCLUSION_STATUSES):
            problems.append(
                "ending a run requires a 'Conclusion record:' giving each research question a status "
                "(supported / rejected / insufficient evidence / open)"
            )
    return problems


def round_index_from_prompt(prompt_text: str) -> int | None:
    match = _ROUND_RE.search(prompt_text or "")
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------- auditor output

AUDITOR_FIELDS = (
    "Evidence check",
    "Calculation check",
    "Interpretation check",
    "Ledger check",
    "Verdict on claims",
)

_HEADER_PATTERNS = (
    re.compile(r"(?i)^\**\s*status\s*:\s*\**\s*(complete|incomplete|blocked)\b"),
    re.compile(r"(?i)^\**\s*integrity\s*:\s*\**\s*(clean|suspect|violation)\b"),
    re.compile(r"(?i)^\**\s*contract\s+audit\s*:\s*\**\s*(aligned|unknown|needs_revision|invalid)\b"),
)
_AUDIT_HEADER_RE = re.compile(r"(?i)scientific\s+audit\s*:?\**")


def auditor_header(text: str) -> dict[str, str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()][:3]
    keys = ("status", "integrity", "contract_audit")
    header: dict[str, str] = {}
    for key, pattern, line in zip(keys, _HEADER_PATTERNS, lines):
        match = pattern.match(line)
        if match:
            header[key] = match.group(1).lower()
    return header


def audit_record(text: str) -> dict[str, str]:
    headers = list(_AUDIT_HEADER_RE.finditer(text or ""))
    if not headers:
        return {}
    block = (text or "")[headers[-1].end():]
    positions: list[tuple[int, int, str]] = []
    for label in AUDITOR_FIELDS:
        match = _label_pattern(label).search(block)
        if match:
            positions.append((match.start(), match.end(), label))
    positions.sort()
    record: dict[str, str] = {}
    for index, (_, end, label) in enumerate(positions):
        stop_at = positions[index + 1][0] if index + 1 < len(positions) else len(block)
        value = block[end:stop_at]
        # A trailing harness section (e.g. "State update for manager:") ends the last field.
        value = re.split(r"(?im)^\s*\**\s*(?:acceptance-constraint backcheck|state update for manager)\b", value)[0]
        record[label] = value.strip()
    return record


def auditor_problems(text: str) -> list[str]:
    problems: list[str] = []
    if len(auditor_header(text)) < 3:
        problems.append(
            "the first three non-empty lines must remain the harness control header "
            "(Status / Integrity / Contract audit)"
        )
    record = audit_record(text)
    if not record:
        problems.append("the report has no 'Scientific audit:' block (SCIENTIST.md §5, Auditor)")
        return problems
    for label in AUDITOR_FIELDS:
        value = record.get(label, "")
        if not value or is_placeholder(value):
            problems.append(f"scientific audit field '{label}' is missing or empty")
    return problems


def calculation_not_applicable(text: str) -> bool:
    value = audit_record(text).get("Calculation check", "").lower()
    return value.startswith("not applicable")


# ------------------------------------------------------------------- transcripts

def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        pass
    return records


def tool_uses(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls from a Claude Code transcript or stream-json log."""
    uses: list[dict[str, Any]] = []
    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses.append({"name": block.get("name", ""), "input": block.get("input") or {}})
    return uses


def first_user_text(records: Iterable[dict[str, Any]]) -> str:
    for record in records:
        if record.get("type") != "user":
            continue
        message = record.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            if texts:
                return "\n".join(texts)
    return ""


def last_assistant_text(records: Iterable[dict[str, Any]]) -> str:
    text = ""
    for record in records:
        if record.get("type") == "result" and isinstance(record.get("result"), str):
            text = record["result"]
            continue
        if record.get("type") != "assistant":
            continue
        message = record.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            if any(part.strip() for part in parts):
                text = "\n".join(parts)
    return text


# --------------------------------------------------------------- tool boundaries

# Tools a harness role may use. Anything else (messaging, scheduling, artifacts,
# remote sessions, worktrees, MCP servers) is outward-facing or out of scope.
HARNESS_ALLOWED_TOOLS = {
    "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "NotebookRead",
    "Bash", "BashOutput", "KillShell", "KillBash", "Glob", "Grep", "LS",
    "Skill", "TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "TaskOutput", "TaskStop", "ToolSearch", "WebSearch", "WebFetch",
}
WEB_TOOLS = {"WebSearch", "WebFetch"}

SYSTEM_PATH_PREFIXES = (
    "/usr/", "/bin/", "/sbin/", "/lib", "/etc/", "/dev/", "/proc/self", "/proc/cpuinfo",
    "/proc/meminfo", "/sys/", "/tmp/", "/var/tmp/", "/opt/",
)

_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_ABS_PATH_RE = re.compile(r"(?:^|(?<=[\s'\"=(,:\[{]))(/[A-Za-z._~][^\s'\"`;|&<>(){}\[\],]*)")
_HOME_RE = re.compile(r"(?:^|(?<=[\s'\"=(,:]))~(?=/|\s|$|['\"])|\$HOME\b|\$\{HOME\}")
_ROOT_SEARCH_RE = re.compile(r"(?:^|\s)/(?:\s|$|\*)")
_DOTDOT_RE = re.compile(r"(?:^|(?<=[\s'\"=(,:]))((?:[\w.-]+/)*\.\.(?:/[\w.-]*)*)")


def _within(path: str, root: str) -> bool:
    path = os.path.normpath(path)
    root = os.path.normpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def path_violations(text: str, *, workspace: str, cwd: str | None = None, extra_allowed: Iterable[str] = ()) -> list[str]:
    """Absolute or escaping paths in `text` that leave the workspace.

    A heuristic for keeping a blinded investigation inside its workspace. Paths
    built at run time (string concatenation in a script) are invisible to it; the
    post-run leakage audit is the backstop.
    """
    text = _URL_RE.sub(" ", text or "")
    cwd = cwd or workspace
    allowed = [os.path.normpath(p) for p in extra_allowed if p]
    violations: list[str] = []
    if _HOME_RE.search(text):
        violations.append("home-directory reference (~ or $HOME)")
    if _ROOT_SEARCH_RE.search(text):
        violations.append("filesystem-root reference ('/')")
    for match in _ABS_PATH_RE.finditer(text):
        candidate = match.group(1)
        normalized = os.path.normpath(candidate)
        if _within(normalized, workspace) or any(_within(normalized, a) for a in allowed):
            continue
        if any((normalized + "/").startswith(prefix) or normalized.startswith(prefix) for prefix in SYSTEM_PATH_PREFIXES):
            continue
        violations.append(candidate)
    for match in _DOTDOT_RE.finditer(text):
        candidate = match.group(1)
        if not _within(os.path.normpath(os.path.join(cwd, candidate)), workspace):
            violations.append(candidate)
    return sorted(set(violations))


def tool_input_text(tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        keys = ("command", "file_path", "path", "pattern", "notebook_path", "url", "glob")
        return "\n".join(str(tool_input[key]) for key in keys if tool_input.get(key))
    return str(tool_input or "")
