from __future__ import annotations

import os
from dataclasses import dataclass


TRADING_DAYS_PER_YEAR = 252
MIN_TICKERS = 2
MAX_TICKERS = 30
MIN_HISTORY_DAYS = 90
DEFAULT_LOOKBACK_SHORT_DAYS = 20
DEFAULT_LOOKBACK_MEDIUM_DAYS = 63
DEFAULT_LOOKBACK_LONG_DAYS = 126
DEFAULT_BOOTSTRAP_SAMPLES = 500
MAX_ABSOLUTE_LEVERAGE = 1.0


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Quant Strategy Tournament (GLM-5 Aligned)")
    app_env: str = os.getenv("APP_ENV", "development")
    data_cache_dir: str = os.getenv("DATA_CACHE_DIR", "data/cache")
    data_source: str = os.getenv("DATA_SOURCE", "yfinance")

    default_initial_capital: float = float(os.getenv("DEFAULT_INITIAL_CAPITAL", "100000"))
    default_trading_cost_bps: float = float(os.getenv("DEFAULT_TRADING_COST_BPS", "10"))
    default_borrow_cost_bps_annual: float = float(os.getenv("DEFAULT_BORROW_COST_BPS_ANNUAL", "50"))
    default_risk_free_rate_annual: float = float(os.getenv("DEFAULT_RISK_FREE_RATE_ANNUAL", "0.02"))
    default_target_vol_annual: float = float(os.getenv("DEFAULT_TARGET_VOL_ANNUAL", "0.15"))


settings = Settings()
