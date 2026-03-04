from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from .config import MIN_HISTORY_DAYS, SUPPORTED_DATA_SOURCES, settings


class DataLoadError(RuntimeError):
    pass


def _normalize_download_frame(frame: Any, ticker: str) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataLoadError(f"No data returned for ticker '{ticker}'.")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    required_columns = {"Close", "Volume"}
    missing = required_columns - set(normalized.columns)
    if missing:
        raise DataLoadError(f"Ticker '{ticker}' missing columns: {', '.join(sorted(missing))}")

    normalized = normalized[["Close", "Volume"]].sort_index()
    if normalized["Close"].isna().all():
        raise DataLoadError(f"Ticker '{ticker}' has no valid Close values.")
    if normalized["Volume"].isna().all():
        raise DataLoadError(f"Ticker '{ticker}' has no valid Volume values.")

    return normalized


def _resolve_provider_chain() -> list[str]:
    primary = settings.data_source.strip().lower()
    if primary not in SUPPORTED_DATA_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_DATA_SOURCES))
        raise DataLoadError(f"Unsupported DATA_SOURCE '{primary}'. Supported values: {supported}.")

    if primary == "auto":
        auto_chain = [
            source
            for source in settings.data_source_fallback_order
            if source in SUPPORTED_DATA_SOURCES and source != "auto"
        ]
        chain = auto_chain if auto_chain else ["yfinance"]
    else:
        chain = [primary]
        if settings.data_source_allow_fallback:
            for source in settings.data_source_fallback_order:
                if source in SUPPORTED_DATA_SOURCES and source not in chain and source != "auto":
                    chain.append(source)

    usable_chain: list[str] = []
    for source in chain:
        if source == "polygon" and not settings.polygon_api_key:
            continue
        usable_chain.append(source)

    if not usable_chain:
        raise DataLoadError(
            "No usable data providers configured. "
            "Set DATA_SOURCE=yfinance or configure POLYGON_API_KEY for polygon."
        )

    return usable_chain


def _cache_key(tickers: list[str], start_date: str, end_date: str, provider_chain: list[str]) -> str:
    provider_part = ",".join(provider_chain)
    raw = f"{','.join(sorted(tickers))}|{start_date}|{end_date}|1d|providers:{provider_part}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_paths(key: str) -> tuple[Path, Path, Path]:
    cache_dir = Path(settings.data_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"close_{key}.csv",
        cache_dir / f"volume_{key}.csv",
        cache_dir / f"providers_{key}.json",
    )


def _truncate_error_detail(raw: str, max_chars: int = 220) -> str:
    clean = " ".join(raw.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _retry_sleep_seconds(attempt: int, retry_after_header: str | None) -> float:
    base = max(0.5, float(settings.polygon_retry_backoff_seconds))
    max_sleep = max(base, float(settings.polygon_retry_max_sleep_seconds))

    if retry_after_header:
        try:
            parsed = float(retry_after_header.strip())
            if parsed > 0:
                return min(max_sleep, parsed)
        except ValueError:
            pass

    return min(max_sleep, base * (2**attempt))


def _download_single_ticker_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        frame = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=settings.data_request_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to download ticker '{ticker}': {exc}") from exc

    return _normalize_download_frame(frame, ticker)


def _download_single_ticker_polygon(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not settings.polygon_api_key:
        raise DataLoadError("POLYGON_API_KEY is required when DATA_SOURCE=polygon.")

    base_url = settings.polygon_base_url.rstrip("/")
    url = f"{base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": str(settings.polygon_adjusted_bars).lower(),
        "sort": "asc",
        "limit": max(1, settings.polygon_request_limit),
        "apiKey": settings.polygon_api_key,
    }

    max_attempts = max(0, int(settings.polygon_retry_attempts))
    response: requests.Response | None = None

    for attempt in range(max_attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=settings.data_request_timeout_seconds,
            )
        except requests.RequestException as exc:
            if attempt < max_attempts:
                sleep_seconds = _retry_sleep_seconds(attempt, None)
                time.sleep(sleep_seconds)
                continue
            raise DataLoadError(f"Polygon request failed for '{ticker}': {exc}") from exc

        if response.status_code == 429:
            if attempt < max_attempts:
                sleep_seconds = _retry_sleep_seconds(attempt, response.headers.get("Retry-After"))
                time.sleep(sleep_seconds)
                continue
            raise DataLoadError(
                f"Polygon rate limit hit for '{ticker}' (HTTP 429). "
                "Consider fewer tickers/time range or enable yfinance fallback."
            )

        if response.status_code >= 500:
            if attempt < max_attempts:
                sleep_seconds = _retry_sleep_seconds(attempt, response.headers.get("Retry-After"))
                time.sleep(sleep_seconds)
                continue
            raise DataLoadError(
                f"Polygon upstream error for '{ticker}' (HTTP {response.status_code}): "
                f"{_truncate_error_detail(response.text)}"
            )

        if response.status_code >= 400:
            raise DataLoadError(
                f"Polygon error for '{ticker}' (HTTP {response.status_code}): "
                f"{_truncate_error_detail(response.text)}"
            )

        break

    if response is None:
        raise DataLoadError(f"Polygon returned no response for ticker '{ticker}'.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise DataLoadError(f"Polygon returned non-JSON response for '{ticker}'.") from exc

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        status = payload.get("status", "unknown")
        message = payload.get("error") or payload.get("message") or "no bars returned"
        raise DataLoadError(f"Polygon returned no data for '{ticker}' (status={status}, message={message}).")

    frame = pd.DataFrame(results)

    def pick_column(candidates: tuple[str, ...]) -> str | None:
        for name in candidates:
            if name in frame.columns:
                return name
        return None

    time_col = pick_column(("t", "timestamp"))
    close_col = pick_column(("c", "close", "Close"))
    volume_col = pick_column(("v", "volume", "Volume"))
    if time_col is None or close_col is None or volume_col is None:
        cols = ", ".join(map(str, frame.columns.tolist()))
        raise DataLoadError(
            f"Polygon payload missing expected bar columns for '{ticker}'. "
            f"Found columns: [{cols}]"
        )

    index = pd.to_datetime(frame[time_col], unit="ms", utc=True, errors="coerce").dt.tz_localize(None)
    # Use positional arrays to avoid pandas label-alignment mismatch between
    # frame RangeIndex and the DatetimeIndex we assign here.
    normalized = pd.DataFrame(
        {
            "Close": pd.to_numeric(frame[close_col], errors="coerce").to_numpy(),
            "Volume": pd.to_numeric(frame[volume_col], errors="coerce").to_numpy(),
        },
        index=index,
    )
    normalized = normalized[normalized.index.notna()]
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    try:
        return _normalize_download_frame(normalized, ticker)
    except DataLoadError as exc:
        sample_keys = ", ".join(map(str, list(results[0].keys()))) if results else "none"
        status = payload.get("status", "unknown")
        raise DataLoadError(
            f"{exc} Polygon status={status}; first-row keys=[{sample_keys}]"
        ) from exc


def _download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    provider: str,
) -> pd.DataFrame:
    if provider == "yfinance":
        return _download_single_ticker_yfinance(ticker, start_date, end_date)
    if provider == "polygon":
        return _download_single_ticker_polygon(ticker, start_date, end_date)
    raise DataLoadError(f"Unsupported provider '{provider}' for ticker '{ticker}'.")


def load_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    provider_chain = _resolve_provider_chain()
    key = _cache_key(tickers, start_date, end_date, provider_chain)
    close_path, volume_path, provider_path = _cache_paths(key)

    if close_path.exists() and volume_path.exists() and provider_path.exists():
        close = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        provider_map = json.loads(provider_path.read_text(encoding="utf-8"))
        return close.sort_index(), volume.sort_index(), provider_map

    close_map: dict[str, pd.Series] = {}
    volume_map: dict[str, pd.Series] = {}
    provider_map: dict[str, str] = {}

    for ticker in tickers:
        frame: pd.DataFrame | None = None
        errors: list[str] = []
        selected_provider = ""

        for provider in provider_chain:
            try:
                frame = _download_single_ticker(ticker, start_date, end_date, provider)
                selected_provider = provider
                break
            except DataLoadError as exc:
                errors.append(f"{provider}: {exc}")

        if frame is None:
            provider_text = ", ".join(provider_chain)
            detail = "; ".join(errors) if errors else "No provider-level error details available."
            raise DataLoadError(
                f"Failed to load '{ticker}' from providers [{provider_text}]. Details: {detail}"
            )

        close_map[ticker] = frame["Close"]
        volume_map[ticker] = frame["Volume"]
        provider_map[ticker] = selected_provider

    close = pd.DataFrame(close_map).sort_index().ffill().dropna(how="all")
    volume = pd.DataFrame(volume_map).sort_index().ffill().dropna(how="all")

    if close.empty or len(close) < MIN_HISTORY_DAYS:
        raise DataLoadError(
            f"Not enough history. Need at least {MIN_HISTORY_DAYS} bars, got {len(close)}."
        )

    close.to_csv(close_path)
    volume.to_csv(volume_path)
    provider_path.write_text(json.dumps(provider_map, indent=2, sort_keys=True), encoding="utf-8")
    return close, volume, provider_map
