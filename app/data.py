from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import MIN_HISTORY_DAYS, settings


class DataLoadError(RuntimeError):
    pass


def _cache_key(tickers: list[str], start_date: str, end_date: str) -> str:
    raw = f"{','.join(sorted(tickers))}|{start_date}|{end_date}|1d"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_paths(key: str) -> tuple[Path, Path]:
    cache_dir = Path(settings.data_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"close_{key}.csv", cache_dir / f"volume_{key}.csv"


def _download_single_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise DataLoadError(f"No data returned for ticker '{ticker}'.")
    required_columns = {"Close", "Volume"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise DataLoadError(f"Ticker '{ticker}' missing columns: {', '.join(sorted(missing))}")
    return frame


def load_price_data(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = _cache_key(tickers, start_date, end_date)
    close_path, volume_path = _cache_paths(key)

    if close_path.exists() and volume_path.exists():
        close = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        return close.sort_index(), volume.sort_index()

    close_map: dict[str, pd.Series] = {}
    volume_map: dict[str, pd.Series] = {}

    for ticker in tickers:
        frame = _download_single_ticker(ticker, start_date, end_date)
        close_map[ticker] = frame["Close"]
        volume_map[ticker] = frame["Volume"]

    close = pd.DataFrame(close_map).sort_index().ffill().dropna(how="all")
    volume = pd.DataFrame(volume_map).sort_index().ffill().dropna(how="all")

    if close.empty or len(close) < MIN_HISTORY_DAYS:
        raise DataLoadError(
            f"Not enough history. Need at least {MIN_HISTORY_DAYS} bars, got {len(close)}."
        )

    close.to_csv(close_path)
    volume.to_csv(volume_path)
    return close, volume
