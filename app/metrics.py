from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DEFAULT_BOOTSTRAP_SAMPLES, TRADING_DAYS_PER_YEAR


def _safe_float(value: float | int | np.floating) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    compounded = (1.0 + returns).prod()
    years = max(len(returns) / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
    return _safe_float(compounded ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    return _safe_float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float) -> float:
    if returns.empty:
        return 0.0
    daily_rf = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    vol = annualized_volatility(excess)
    if vol <= 0:
        return 0.0
    return _safe_float((excess.mean() * TRADING_DAYS_PER_YEAR) / vol)


def sortino_ratio(returns: pd.Series, risk_free_rate_annual: float) -> float:
    if returns.empty:
        return 0.0
    daily_rf = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_vol <= 0:
        return 0.0
    return _safe_float((excess.mean() * TRADING_DAYS_PER_YEAR) / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return _safe_float(drawdown.min())


def calmar_ratio(cagr: float, max_dd: float) -> float:
    if max_dd >= 0:
        return 0.0
    return _safe_float(cagr / abs(max_dd))


def turnover_annualized(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    return _safe_float(turnover.mean() * TRADING_DAYS_PER_YEAR)


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return _safe_float((returns > 0).mean())


def cvar_5pct(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    var_5 = returns.quantile(0.05)
    tail = returns[returns <= var_5]
    if tail.empty:
        return 0.0
    return _safe_float(tail.mean())


def bootstrap_probability_of_skill(returns: pd.Series, risk_free_rate_annual: float, n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES) -> float:
    if returns.empty or len(returns) < 30:
        return 0.0

    rng = np.random.default_rng(42)
    arr = returns.to_numpy()
    if arr.size == 0:
        return 0.0

    positive = 0
    for _ in range(n_samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        s = sharpe_ratio(pd.Series(sample), risk_free_rate_annual)
        if s > 0:
            positive += 1

    return _safe_float(positive / n_samples)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def deflated_sharpe_confidence(sharpe: float, returns: pd.Series) -> float:
    n = len(returns)
    if n < 30:
        return 0.0

    skew = _safe_float(returns.skew())
    kurt = _safe_float(returns.kurt())
    sr = float(sharpe)

    adjustment = 1.0 - (skew * sr) / 6.0 - ((kurt + 3.0) * (sr**2)) / 24.0
    adjusted_sr = sr * max(0.05, adjustment)
    z = adjusted_sr * math.sqrt(n / TRADING_DAYS_PER_YEAR)
    return _safe_float(normal_cdf(z))
