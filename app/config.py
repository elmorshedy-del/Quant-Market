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
DEFAULT_PAIR_FORMATION_DAYS = 252
DEFAULT_PCA_FORMATION_DAYS = 252
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_ML_MIN_TRAIN_DAYS = 252
DEFAULT_ML_RETRAIN_FREQUENCY_DAYS = 21
DEFAULT_ML_EMBARGO_DAYS = 5
MAX_ABSOLUTE_LEVERAGE = 1.0
SUPPORTED_DATA_SOURCES = {"yfinance", "polygon", "auto"}


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_allowed_origins(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",")]
    return [value for value in values if value]


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",")]
    cleaned = [value for value in values if value]
    return cleaned if cleaned else list(default)


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Quant Strategy Tournament (GLM-5 Aligned)")
    app_env: str = os.getenv("APP_ENV", "development")
    data_cache_dir: str = os.getenv("DATA_CACHE_DIR", "data/cache")
    data_source: str = os.getenv("DATA_SOURCE", "yfinance").strip().lower()
    data_source_allow_fallback: bool = _parse_bool(os.getenv("DATA_SOURCE_ALLOW_FALLBACK"), True)
    data_source_fallback_order: list[str] = None  # type: ignore[assignment]
    data_request_timeout_seconds: int = int(os.getenv("DATA_REQUEST_TIMEOUT_SECONDS", "30"))
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "").strip()
    polygon_base_url: str = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").strip()
    polygon_adjusted_bars: bool = _parse_bool(os.getenv("POLYGON_ADJUSTED_BARS"), True)
    polygon_request_limit: int = int(os.getenv("POLYGON_REQUEST_LIMIT", "50000"))
    polygon_retry_attempts: int = int(os.getenv("POLYGON_RETRY_ATTEMPTS", "3"))
    polygon_retry_backoff_seconds: float = float(os.getenv("POLYGON_RETRY_BACKOFF_SECONDS", "2"))
    polygon_retry_max_sleep_seconds: float = float(os.getenv("POLYGON_RETRY_MAX_SLEEP_SECONDS", "30"))
    allowed_origins: list[str] = None  # type: ignore[assignment]
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    rate_limit_max_requests: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
    show_survivorship_warning: bool = _parse_bool(os.getenv("SHOW_SURVIVORSHIP_WARNING"), True)

    default_initial_capital: float = float(os.getenv("DEFAULT_INITIAL_CAPITAL", "100000"))
    default_trading_cost_bps: float = float(os.getenv("DEFAULT_TRADING_COST_BPS", "10"))
    default_borrow_cost_bps_annual: float = float(os.getenv("DEFAULT_BORROW_COST_BPS_ANNUAL", "50"))
    default_risk_free_rate_annual: float = float(os.getenv("DEFAULT_RISK_FREE_RATE_ANNUAL", "0.02"))
    default_target_vol_annual: float = float(os.getenv("DEFAULT_TARGET_VOL_ANNUAL", "0.15"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_origins", _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS")))
        fallback_order = _parse_csv_values(
            os.getenv("DATA_SOURCE_FALLBACK_ORDER"),
            default=["polygon", "yfinance"],
        )
        normalized_fallback = [source.strip().lower() for source in fallback_order]
        object.__setattr__(self, "data_source_fallback_order", normalized_fallback)


settings = Settings()
