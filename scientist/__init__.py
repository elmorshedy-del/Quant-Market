"""Scientist agent setup on top of LongHorizon-Harness (see scientist/README.md)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"

# The enforcement rules live next to the hook so that a workspace copy of
# `.claude/` is self-contained; the CLI reuses the same module.
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))
