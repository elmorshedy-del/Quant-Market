from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import MAX_TICKERS, MIN_TICKERS


class TournamentRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=MIN_TICKERS, max_length=MAX_TICKERS)
    start_date: date
    end_date: date
    initial_capital: float = Field(default=100000, gt=0)
    trading_cost_bps: float = Field(default=10, ge=0, le=500)
    borrow_cost_bps_annual: float = Field(default=50, ge=0, le=2000)
    risk_free_rate_annual: float = Field(default=0.02, ge=-0.05, le=0.2)
    strategy_ids: list[str] | None = None

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, raw: list[str]) -> list[str]:
        cleaned = []
        for ticker in raw:
            symbol = "".join(ch for ch in ticker.upper().strip() if ch.isalnum() or ch in {".", "-"})
            if symbol:
                cleaned.append(symbol)
        unique = list(dict.fromkeys(cleaned))
        if len(unique) < MIN_TICKERS:
            raise ValueError("At least two valid tickers are required.")
        return unique


class StrategyMeta(BaseModel):
    strategy_id: str
    name: str
    family: str
    implemented: bool
    complexity: str
    data_requirements: str
    notes: str | None = None


class StrategyResult(BaseModel):
    strategy_id: str
    name: str
    family: str
    implemented: bool
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    turnover_annual: float
    hit_rate_pct: float
    cvar_5_pct: float
    final_equity: float
    expected_pnl: float
    probability_of_skill: float
    repeatability_score: float
    complexity: str
    skipped_reason: str | None = None


class TournamentResponse(BaseModel):
    run_id: str
    tickers: list[str]
    start_date: date
    end_date: date
    benchmark: dict[str, Any]
    ranking: list[StrategyResult]
    skipped: list[StrategyResult]
    metadata: dict[str, Any]
