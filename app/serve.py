from __future__ import annotations

import os

DEFAULT_PORT = 8000


def _resolve_port() -> int:
    raw = os.getenv("PORT", str(DEFAULT_PORT)).strip()
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PORT
    return port if port > 0 else DEFAULT_PORT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=_resolve_port())
