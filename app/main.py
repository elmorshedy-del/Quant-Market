from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .engine import list_strategy_catalog, run_tournament
from .models import TournamentRequest
from .rate_limit import InMemorySlidingWindowRateLimiter


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="GLM-5 paper-aligned quant strategy tournament backtester",
)


def _resolve_cors_origins() -> list[str]:
    if settings.allowed_origins:
        return settings.allowed_origins
    if settings.app_env.strip().lower() == "production":
        return []
    return ["*"]


_cors_origins = _resolve_cors_origins()
_allow_credentials = _cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

_limiter = InMemorySlidingWindowRateLimiter(
    window_seconds=settings.rate_limit_window_seconds,
    max_requests=settings.rate_limit_max_requests,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "app": settings.app_name,
        "env": settings.app_env,
        "data_source": settings.data_source,
        "data_source_allow_fallback": settings.data_source_allow_fallback,
        "data_source_fallback_order": settings.data_source_fallback_order,
        "polygon_configured": bool(settings.polygon_api_key),
        "cors_origins": _cors_origins,
        "rate_limit": {
            "window_seconds": settings.rate_limit_window_seconds,
            "max_requests": settings.rate_limit_max_requests,
        },
    }


@app.get("/api/strategies")
def strategies() -> dict:
    return {
        "strategies": list_strategy_catalog(),
    }


@app.post("/api/tournament/run")
def tournament_run(payload: TournamentRequest, request: Request, response: Response):
    if payload.start_date >= payload.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")
    rate_key = f"run:{client_ip}"
    decision = _limiter.check(rate_key)
    if not decision.allowed:
        response.headers["Retry-After"] = str(decision.retry_after_seconds)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {decision.retry_after_seconds}s.",
        )

    try:
        result = run_tournament(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return result.model_dump()
