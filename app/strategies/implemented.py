from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

from ..config import (
    DEFAULT_LOOKBACK_LONG_DAYS,
    DEFAULT_LOOKBACK_MEDIUM_DAYS,
    DEFAULT_LOOKBACK_SHORT_DAYS,
    DEFAULT_ML_EMBARGO_DAYS,
    DEFAULT_ML_MIN_TRAIN_DAYS,
    DEFAULT_ML_RETRAIN_FREQUENCY_DAYS,
    DEFAULT_PAIR_FORMATION_DAYS,
    DEFAULT_PCA_FORMATION_DAYS,
    TRADING_DAYS_PER_YEAR,
    settings,
)
from .base import (
    BaseStrategy,
    StrategyContext,
    StrategyMetaInfo,
    long_only_from_score,
    long_short_from_score,
    normalize_weight_frame,
)


EPSILON = 1e-9


def _zero_weights(context: StrategyContext) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=context.prices.index, columns=context.prices.columns)


def _rolling_pair_for_window(window_returns: pd.DataFrame) -> tuple[str, str] | None:
    if window_returns.empty or window_returns.shape[1] < 2:
        return None

    corr = window_returns.corr().replace([np.inf, -np.inf], np.nan)
    if corr.empty:
        return None

    np.fill_diagonal(corr.values, np.nan)
    if corr.isna().all().all():
        return None

    best = corr.stack().idxmax()
    if not isinstance(best, tuple) or len(best) != 2:
        return None
    return str(best[0]), str(best[1])


def _build_walk_forward_predictions(
    dataset: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    feature_columns: list[str],
    model_builder,
    min_train_days: int = DEFAULT_ML_MIN_TRAIN_DAYS,
    retrain_frequency_days: int = DEFAULT_ML_RETRAIN_FREQUENCY_DAYS,
    embargo_days: int = DEFAULT_ML_EMBARGO_DAYS,
) -> pd.DataFrame:
    prediction_frame = pd.DataFrame(index=all_dates)
    if dataset.empty:
        return prediction_frame

    unique_dates = sorted(dataset["date"].unique())
    if len(unique_dates) < min_train_days + embargo_days + 1:
        return prediction_frame

    last_fit_idx = -1
    model = None
    predictions: list[pd.DataFrame] = []

    for idx in range(min_train_days + embargo_days, len(unique_dates)):
        prediction_date = unique_dates[idx]
        train_end_idx = idx - embargo_days
        if train_end_idx <= 0:
            continue

        if model is None or (idx - last_fit_idx) >= retrain_frequency_days:
            train_dates = unique_dates[:train_end_idx]
            train = dataset[dataset["date"].isin(train_dates)]
            if train.empty:
                continue
            model = model_builder()
            model.fit(train[feature_columns], train["target"])
            last_fit_idx = idx

        prediction_rows = dataset[dataset["date"] == prediction_date]
        if prediction_rows.empty or model is None:
            continue

        predicted = prediction_rows[["date", "ticker"]].copy()
        predicted["prediction"] = model.predict(prediction_rows[feature_columns])
        predictions.append(predicted)

    if not predictions:
        return prediction_frame

    merged = pd.concat(predictions, ignore_index=True)
    pivot = merged.pivot(index="date", columns="ticker", values="prediction")
    return pivot.reindex(all_dates).fillna(0.0)


class BuyAndHoldStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="buy_and_hold",
        name="Buy and Hold Equal Weight",
        family="Baseline",
        implemented=True,
        complexity="low",
        data_requirements="daily_ohlcv",
        notes="Baseline benchmark for tournament sanity checks.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        n_assets = len(context.prices.columns)
        if n_assets == 0:
            return _zero_weights(context)
        weights = pd.DataFrame(1.0 / n_assets, index=context.prices.index, columns=context.prices.columns)
        return weights


class TimeSeriesMomentumStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="time_series_momentum",
        name="Time-Series Momentum (TSM)",
        family="Time-Series Momentum / Trend",
        implemented=True,
        complexity="low",
        data_requirements="daily_ohlcv",
        notes="Sign of long-horizon return per asset; volatility-normalized by cross-asset gross leverage cap.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        lookback = DEFAULT_LOOKBACK_LONG_DAYS
        momentum = context.prices.pct_change(lookback).shift(1)
        signal = np.sign(momentum).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return normalize_weight_frame(signal)


class CrossSectionalMomentumStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="cross_sectional_momentum",
        name="Cross-Sectional Momentum",
        family="Cross-Sectional Momentum",
        implemented=True,
        complexity="low",
        data_requirements="daily_ohlcv",
        notes="Long top quantile winners and short bottom quantile losers based on lagged returns.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        lookback = DEFAULT_LOOKBACK_LONG_DAYS
        scores = context.prices.pct_change(lookback).shift(1)
        return long_short_from_score(scores, quantile=0.30)


class SingleAssetMeanReversionStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="single_asset_mean_reversion",
        name="Single-Asset Mean Reversion",
        family="Mean Reversion",
        implemented=True,
        complexity="low",
        data_requirements="daily_ohlcv",
        notes="Trade z-score reversion of each asset versus rolling mean.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        window = DEFAULT_LOOKBACK_SHORT_DAYS
        rolling_mean = context.prices.rolling(window).mean()
        rolling_std = context.prices.rolling(window).std().replace(0, np.nan)
        z_score = ((context.prices - rolling_mean) / rolling_std).shift(1)
        signal = -z_score.clip(lower=-2.0, upper=2.0)
        return normalize_weight_frame(signal)


class CrossSectionalMeanReversionStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="cross_sectional_mean_reversion",
        name="Cross-Sectional Mean Reversion",
        family="Mean Reversion",
        implemented=True,
        complexity="low",
        data_requirements="daily_ohlcv",
        notes="Long lagged underperformers and short lagged outperformers across the universe.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        lagged_returns = context.returns.shift(1)
        cross_section_residual = lagged_returns.sub(lagged_returns.mean(axis=1), axis=0)
        scores = -cross_section_residual
        return long_short_from_score(scores, quantile=0.30)


class PairsSpreadReversionStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="pairs_spread_reversion",
        name="Pairs Spread Reversion",
        family="Statistical Arbitrage / Pairs",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Chooses rolling-window pair and rolling hedge ratio using only historical data.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        if context.prices.shape[1] < 2:
            return _zero_weights(context)

        weights = _zero_weights(context)
        lagged_prices = context.prices.shift(1)
        lagged_returns = lagged_prices.pct_change().replace([np.inf, -np.inf], np.nan)
        min_required = max(DEFAULT_PAIR_FORMATION_DAYS, DEFAULT_LOOKBACK_MEDIUM_DAYS)

        for row_idx in range(min_required, len(lagged_prices)):
            returns_window = lagged_returns.iloc[row_idx - DEFAULT_PAIR_FORMATION_DAYS : row_idx]
            pair = _rolling_pair_for_window(returns_window)
            if pair is None:
                continue

            ticker_a, ticker_b = pair
            pair_prices = lagged_prices[[ticker_a, ticker_b]].iloc[row_idx - DEFAULT_PAIR_FORMATION_DAYS : row_idx]
            if pair_prices.isna().any().any():
                continue

            log_a = np.log(pair_prices[ticker_a].replace(0, np.nan))
            log_b = np.log(pair_prices[ticker_b].replace(0, np.nan))
            valid = log_a.notna() & log_b.notna()
            if valid.sum() < DEFAULT_LOOKBACK_MEDIUM_DAYS:
                continue

            beta = np.polyfit(log_b[valid], log_a[valid], deg=1)[0]
            spread_window = (log_a - beta * log_b).dropna()
            spread_std = spread_window.std(ddof=1)
            if not np.isfinite(spread_std) or spread_std <= EPSILON:
                continue

            z_score = (spread_window.iloc[-1] - spread_window.mean()) / spread_std
            z_clamped = float(np.clip(z_score, -2.0, 2.0))
            current_date = lagged_prices.index[row_idx]
            weights.at[current_date, ticker_a] = -z_clamped
            weights.at[current_date, ticker_b] = z_clamped

        return normalize_weight_frame(weights)


class PCAResidualReversionStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="pca_residual_reversion",
        name="PCA Residual Reversion",
        family="Statistical Arbitrage / PCA",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Uses rolling PCA decomposition from historical windows only.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        lagged_returns = context.returns.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if lagged_returns.shape[1] < 2:
            return _zero_weights(context)

        weights = _zero_weights(context)
        min_required = max(DEFAULT_PCA_FORMATION_DAYS, lagged_returns.shape[1] + 1)

        for row_idx in range(min_required, len(lagged_returns)):
            window = lagged_returns.iloc[row_idx - DEFAULT_PCA_FORMATION_DAYS : row_idx]
            if window.empty:
                continue

            covariance = np.cov(window.to_numpy(dtype=float).T)
            if not np.isfinite(covariance).all():
                continue

            eigvals, eigvecs = np.linalg.eigh(covariance)
            if eigvals.size == 0:
                continue

            principal = eigvecs[:, np.argmax(eigvals)]
            current_return = lagged_returns.iloc[row_idx].to_numpy(dtype=float)
            factor_projection = float(np.dot(current_return, principal))
            common_component = principal * factor_projection
            residual = current_return - common_component

            current_date = lagged_returns.index[row_idx]
            weights.loc[current_date] = -residual

        return normalize_weight_frame(weights)


class VolatilityTargetTrendStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="volatility_target_trend",
        name="Volatility Targeted Trend Overlay",
        family="Volatility Targeting / Overlay",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Applies trend direction with target volatility scaling to equal-weight basket.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        market_price = context.prices.mean(axis=1)
        market_return = market_price.pct_change().fillna(0.0)
        trend_signal = np.sign(market_price.pct_change(DEFAULT_LOOKBACK_LONG_DAYS).shift(1)).fillna(0.0)

        realized_vol = market_return.rolling(DEFAULT_LOOKBACK_SHORT_DAYS).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        target_vol = float(settings.default_target_vol_annual)
        leverage = (target_vol / (realized_vol + EPSILON)).clip(lower=0.0, upper=1.5).fillna(0.0)

        n_assets = len(context.prices.columns)
        if n_assets == 0:
            return _zero_weights(context)

        base = pd.DataFrame(1.0 / n_assets, index=context.prices.index, columns=context.prices.columns)
        signed_scale = (trend_signal * leverage).rename("scale")
        weights = base.mul(signed_scale, axis=0)
        return normalize_weight_frame(weights)


class RiskParityInverseVolStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="risk_parity_inverse_vol",
        name="Risk Parity (Inverse Vol Approx)",
        family="Risk Parity",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Inverse-vol weighting as a robust risk parity approximation.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        rolling_vol = context.returns.rolling(DEFAULT_LOOKBACK_MEDIUM_DAYS).std().shift(1)
        inv_vol = 1.0 / (rolling_vol + EPSILON)
        inv_vol = inv_vol.clip(lower=0.0)
        return normalize_weight_frame(inv_vol)


class ValueQualityProxyStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="value_quality_proxy",
        name="Value + Quality Proxy",
        family="Fundamental Factors (Proxy)",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Price-only proxy: long-term underperformance (value proxy) + low volatility (quality proxy).",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        value_proxy = -context.prices.pct_change(252).shift(1)
        quality_proxy = -context.returns.rolling(DEFAULT_LOOKBACK_MEDIUM_DAYS).std().shift(1)

        value_rank = value_proxy.rank(axis=1, pct=True)
        quality_rank = quality_proxy.rank(axis=1, pct=True)
        composite = value_rank + quality_rank
        return long_only_from_score(composite, top_quantile=0.40)


class ProfitabilityProxyStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="profitability_proxy",
        name="Profitability Proxy",
        family="Fundamental Factors (Proxy)",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Price-only profitability proxy: return-to-volatility efficiency score.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        return_6m = context.prices.pct_change(DEFAULT_LOOKBACK_LONG_DAYS).shift(1)
        vol_3m = context.returns.rolling(DEFAULT_LOOKBACK_MEDIUM_DAYS).std().shift(1)
        efficiency = return_6m / (vol_3m + EPSILON)
        return long_short_from_score(efficiency, quantile=0.30)


class RegimeFilteredMomentumStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="regime_filtered_momentum",
        name="Regime-Filtered Momentum",
        family="Regime Switching",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Momentum exposure scaled down in high-volatility regimes.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        lookback = DEFAULT_LOOKBACK_LONG_DAYS
        trend_signal = np.sign(context.prices.pct_change(lookback).shift(1)).fillna(0.0)

        market_return = context.returns.mean(axis=1)
        market_vol = market_return.rolling(DEFAULT_LOOKBACK_SHORT_DAYS).std().shift(1)
        vol_threshold = market_vol.expanding(min_periods=DEFAULT_LOOKBACK_MEDIUM_DAYS).quantile(0.75)
        regime_scale = pd.Series(1.0, index=market_vol.index)
        regime_scale[market_vol > vol_threshold] = 0.35

        scaled = trend_signal.mul(regime_scale, axis=0)
        return normalize_weight_frame(scaled)


class ElasticNetForecastStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="elastic_net_forecast",
        name="Elastic Net Forecast",
        family="ML (Linear)",
        implemented=True,
        complexity="high",
        data_requirements="daily_ohlcv",
        notes="Cross-sectional return forecast with lagged technical features.",
    )

    def _build_dataset(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        dataset_rows: list[pd.DataFrame] = []

        for ticker in prices.columns:
            frame = pd.DataFrame(index=prices.index)
            frame["ret_1d"] = returns[ticker].shift(1)
            frame["ret_5d"] = prices[ticker].pct_change(5).shift(1)
            frame["ret_20d"] = prices[ticker].pct_change(20).shift(1)
            frame["vol_20d"] = returns[ticker].rolling(20).std().shift(1)
            frame["mom_63d"] = prices[ticker].pct_change(63).shift(1)
            frame["target"] = returns[ticker].shift(-1)
            frame["ticker"] = ticker
            dataset_rows.append(frame.reset_index(names="date"))

        dataset = pd.concat(dataset_rows, ignore_index=True)
        dataset = dataset.dropna(subset=["ret_1d", "ret_5d", "ret_20d", "vol_20d", "mom_63d", "target"]) 
        return dataset

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        dataset = self._build_dataset(context.prices)
        if dataset.empty:
            return _zero_weights(context)

        features = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "mom_63d"]
        prediction_pivot = _build_walk_forward_predictions(
            dataset=dataset,
            all_dates=context.prices.index,
            feature_columns=features,
            model_builder=lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42),
            min_train_days=DEFAULT_ML_MIN_TRAIN_DAYS,
            retrain_frequency_days=DEFAULT_ML_RETRAIN_FREQUENCY_DAYS,
            embargo_days=DEFAULT_ML_EMBARGO_DAYS,
        )

        return long_short_from_score(prediction_pivot, quantile=0.30)


class GradientBoostingForecastStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="gradient_boosting_forecast",
        name="Gradient Boosting Forecast",
        family="ML (Tree-Based)",
        implemented=True,
        complexity="high",
        data_requirements="daily_ohlcv",
        notes="Non-linear predictor using lagged return and volatility features.",
    )

    def _build_dataset(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        dataset_rows: list[pd.DataFrame] = []

        for ticker in prices.columns:
            frame = pd.DataFrame(index=prices.index)
            frame["ret_1d"] = returns[ticker].shift(1)
            frame["ret_5d"] = prices[ticker].pct_change(5).shift(1)
            frame["ret_20d"] = prices[ticker].pct_change(20).shift(1)
            frame["vol_20d"] = returns[ticker].rolling(20).std().shift(1)
            frame["vol_63d"] = returns[ticker].rolling(63).std().shift(1)
            frame["mom_63d"] = prices[ticker].pct_change(63).shift(1)
            frame["target"] = returns[ticker].shift(-1)
            frame["ticker"] = ticker
            dataset_rows.append(frame.reset_index(names="date"))

        dataset = pd.concat(dataset_rows, ignore_index=True)
        dataset = dataset.dropna(subset=["ret_1d", "ret_5d", "ret_20d", "vol_20d", "vol_63d", "mom_63d", "target"]) 
        return dataset

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        dataset = self._build_dataset(context.prices)
        if dataset.empty:
            return _zero_weights(context)

        features = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "vol_63d", "mom_63d"]
        prediction_pivot = _build_walk_forward_predictions(
            dataset=dataset,
            all_dates=context.prices.index,
            feature_columns=features,
            model_builder=lambda: GradientBoostingRegressor(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            ),
            min_train_days=DEFAULT_ML_MIN_TRAIN_DAYS,
            retrain_frequency_days=DEFAULT_ML_RETRAIN_FREQUENCY_DAYS,
            embargo_days=DEFAULT_ML_EMBARGO_DAYS,
        )

        return long_short_from_score(prediction_pivot, quantile=0.30)


IMPLEMENTED_STRATEGY_CLASSES: dict[str, type[BaseStrategy]] = {
    BuyAndHoldStrategy.meta.strategy_id: BuyAndHoldStrategy,
    TimeSeriesMomentumStrategy.meta.strategy_id: TimeSeriesMomentumStrategy,
    CrossSectionalMomentumStrategy.meta.strategy_id: CrossSectionalMomentumStrategy,
    SingleAssetMeanReversionStrategy.meta.strategy_id: SingleAssetMeanReversionStrategy,
    CrossSectionalMeanReversionStrategy.meta.strategy_id: CrossSectionalMeanReversionStrategy,
    PairsSpreadReversionStrategy.meta.strategy_id: PairsSpreadReversionStrategy,
    PCAResidualReversionStrategy.meta.strategy_id: PCAResidualReversionStrategy,
    VolatilityTargetTrendStrategy.meta.strategy_id: VolatilityTargetTrendStrategy,
    RiskParityInverseVolStrategy.meta.strategy_id: RiskParityInverseVolStrategy,
    ValueQualityProxyStrategy.meta.strategy_id: ValueQualityProxyStrategy,
    ProfitabilityProxyStrategy.meta.strategy_id: ProfitabilityProxyStrategy,
    RegimeFilteredMomentumStrategy.meta.strategy_id: RegimeFilteredMomentumStrategy,
    ElasticNetForecastStrategy.meta.strategy_id: ElasticNetForecastStrategy,
    GradientBoostingForecastStrategy.meta.strategy_id: GradientBoostingForecastStrategy,
}
