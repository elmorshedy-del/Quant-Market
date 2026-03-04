from __future__ import annotations

import math
from statistics import NormalDist

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


def _expected_max_sharpe_under_null(num_trials: int) -> float:
    # Approximation from Bailey & Lopez de Prado for the expected maximum
    # Sharpe across N independent trials under the null.
    n = max(1, int(num_trials))
    if n <= 1:
        return 0.0

    standard_normal = NormalDist()
    euler_gamma = 0.5772156649015329
    p1 = min(0.999999, max(0.000001, 1.0 - 1.0 / n))
    p2 = min(0.999999, max(0.000001, 1.0 - 1.0 / (n * math.e)))
    z1 = standard_normal.inv_cdf(p1)
    z2 = standard_normal.inv_cdf(p2)
    return (1.0 - euler_gamma) * z1 + euler_gamma * z2


def deflated_sharpe_confidence(sharpe: float, returns: pd.Series, num_trials: int = 1) -> float:
    n = len(returns)
    if n < 30:
        return 0.0

    skew = _safe_float(returns.skew())
    # pandas kurt() returns excess kurtosis.
    excess_kurtosis = _safe_float(returns.kurt())
    sr = float(sharpe)
    baseline_sr = _expected_max_sharpe_under_null(num_trials)

    variance_term = 1.0 - (skew * sr) + ((excess_kurtosis) / 4.0) * (sr**2)
    if variance_term <= 0:
        return 0.0

    sigma_sr = math.sqrt(variance_term / max(1.0, n - 1.0))
    if sigma_sr <= 0:
        return 0.0

    z = (sr - baseline_sr) / sigma_sr
    return _safe_float(normal_cdf(z))


def _block_bootstrap_indices(length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    if block_size <= 1:
        return rng.integers(0, length, size=length)

    starts = rng.integers(0, length, size=max(1, length // block_size + 2))
    indices: list[int] = []
    for start in starts:
        for offset in range(block_size):
            idx = (int(start) + offset) % length
            indices.append(idx)
            if len(indices) >= length:
                return np.array(indices[:length], dtype=int)
    return np.array(indices[:length], dtype=int)


def white_reality_check_pvalue(
    strategy_return_frame: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 500,
    block_size: int = 5,
) -> float:
    if strategy_return_frame.empty or benchmark_returns.empty:
        return 1.0

    aligned_benchmark = benchmark_returns.reindex(strategy_return_frame.index).fillna(0.0)
    relative = strategy_return_frame.sub(aligned_benchmark, axis=0).dropna(how="all")
    if relative.empty:
        return 1.0

    observed_best = float((relative.mean() * TRADING_DAYS_PER_YEAR).max())
    centered = relative.sub(relative.mean(), axis=1).fillna(0.0)
    if centered.empty:
        return 1.0

    rng = np.random.default_rng(123)
    exceedances = 0
    arr = centered.to_numpy(dtype=float)

    for _ in range(max(1, n_bootstrap)):
        idx = _block_bootstrap_indices(len(centered), block_size, rng)
        sampled = arr[idx]
        sampled_best = float((sampled.mean(axis=0) * TRADING_DAYS_PER_YEAR).max())
        if sampled_best >= observed_best:
            exceedances += 1

    return _safe_float((exceedances + 1) / (n_bootstrap + 1))


def probability_of_backtest_overfitting(strategy_return_frame: pd.DataFrame, n_slices: int = 8) -> float:
    if strategy_return_frame.empty or strategy_return_frame.shape[1] < 2:
        return 1.0

    returns = strategy_return_frame.dropna(how="all").fillna(0.0)
    total = len(returns)
    if total < n_slices * 10:
        return 1.0

    slice_size = total // n_slices
    if slice_size < 10:
        return 1.0

    slices = []
    for i in range(n_slices):
        start = i * slice_size
        end = (i + 1) * slice_size if i < n_slices - 1 else total
        if end - start >= 10:
            slices.append(returns.iloc[start:end])

    if len(slices) < 4:
        return 1.0

    rng = np.random.default_rng(7)
    trials = min(40, 2 ** len(slices))
    overfit = 0
    valid = 0
    strategy_names = list(returns.columns)

    for _ in range(trials):
        mask = rng.random(len(slices)) > 0.5
        if mask.sum() == 0 or mask.sum() == len(slices):
            continue

        train = pd.concat([slice_frame for slice_frame, selected in zip(slices, mask) if selected], axis=0)
        test = pd.concat([slice_frame for slice_frame, selected in zip(slices, mask) if not selected], axis=0)
        if train.empty or test.empty:
            continue

        train_sharpes = {name: sharpe_ratio(train[name], 0.0) for name in strategy_names}
        best_train = max(train_sharpes, key=train_sharpes.get)

        test_sharpes = np.array([sharpe_ratio(test[name], 0.0) for name in strategy_names], dtype=float)
        if test_sharpes.size == 0:
            continue
        best_test_value = sharpe_ratio(test[best_train], 0.0)
        percentile = float((test_sharpes <= best_test_value).mean())
        valid += 1
        if percentile < 0.5:
            overfit += 1

    if valid == 0:
        return 1.0

    return _safe_float(overfit / valid)
