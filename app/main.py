from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .engine import list_strategy_catalog, run_tournament
from .models import TournamentRequest


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="GLM-5 paper-aligned quant strategy tournament backtester",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    }


@app.get("/api/strategies")
def strategies() -> dict:
    return {
        "strategies": list_strategy_catalog(),
    }


@app.post("/api/tournament/run")
def tournament_run(payload: TournamentRequest):
    if payload.start_date >= payload.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        result = run_tournament(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return result.model_dump()
