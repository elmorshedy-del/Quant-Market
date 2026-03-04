from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

from ..config import (
    DEFAULT_LOOKBACK_LONG_DAYS,
    DEFAULT_LOOKBACK_MEDIUM_DAYS,
    DEFAULT_LOOKBACK_SHORT_DAYS,
    TRADING_DAYS_PER_YEAR,
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
        notes="Chooses most-correlated pair in-sample and trades rolling spread z-score reversion.",
    )

    def _pick_pair(self, prices: pd.DataFrame) -> tuple[str, str] | None:
        returns = prices.pct_change().dropna(how="all")
        corr = returns.corr().replace([np.inf, -np.inf], np.nan)
        np.fill_diagonal(corr.values, np.nan)
        if corr.isna().all().all():
            return None
        best = corr.stack().idxmax()
        if not isinstance(best, tuple) or len(best) != 2:
            return None
        return str(best[0]), str(best[1])

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        pair = self._pick_pair(context.prices)
        if pair is None:
            return _zero_weights(context)

        ticker_a, ticker_b = pair
        if ticker_a not in context.prices.columns or ticker_b not in context.prices.columns:
            return _zero_weights(context)

        log_a = np.log(context.prices[ticker_a].replace(0, np.nan)).ffill()
        log_b = np.log(context.prices[ticker_b].replace(0, np.nan)).ffill()

        valid = log_a.notna() & log_b.notna()
        if valid.sum() < DEFAULT_LOOKBACK_LONG_DAYS:
            return _zero_weights(context)

        beta = np.polyfit(log_b[valid], log_a[valid], deg=1)[0]
        spread = log_a - beta * log_b
        spread_mean = spread.rolling(DEFAULT_LOOKBACK_MEDIUM_DAYS).mean()
        spread_std = spread.rolling(DEFAULT_LOOKBACK_MEDIUM_DAYS).std().replace(0, np.nan)
        z = ((spread - spread_mean) / spread_std).shift(1).clip(-2.0, 2.0)

        weights = _zero_weights(context)
        weights[ticker_a] = -z
        weights[ticker_b] = z
        return normalize_weight_frame(weights)


class PCAResidualReversionStrategy(BaseStrategy):
    meta = StrategyMetaInfo(
        strategy_id="pca_residual_reversion",
        name="PCA Residual Reversion",
        family="Statistical Arbitrage / PCA",
        implemented=True,
        complexity="medium",
        data_requirements="daily_ohlcv",
        notes="Removes first principal component then mean-reverts idiosyncratic residuals.",
    )

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        returns = context.returns.copy().fillna(0.0)
        if returns.shape[1] < 2:
            return _zero_weights(context)

        matrix = returns.to_numpy(dtype=float)
        covariance = np.cov(matrix.T)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        principal = eigvecs[:, np.argmax(eigvals)]

        common_component = np.outer(matrix @ principal, principal)
        residual = matrix - common_component
        residual_df = pd.DataFrame(residual, index=returns.index, columns=returns.columns)
        signal = -residual_df.shift(1)
        return normalize_weight_frame(signal)


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
        target_vol = 0.15
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

        unique_dates = sorted(dataset["date"].unique())
        split_idx = int(len(unique_dates) * 0.70)
        if split_idx < 60:
            return _zero_weights(context)
        split_date = unique_dates[split_idx]

        features = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "mom_63d"]
        train = dataset[dataset["date"] < split_date]
        if train.empty:
            return _zero_weights(context)

        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42)
        model.fit(train[features], train["target"])

        dataset["prediction"] = model.predict(dataset[features])
        prediction_pivot = dataset.pivot(index="date", columns="ticker", values="prediction")
        prediction_pivot = prediction_pivot.reindex(context.prices.index).fillna(0.0)

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

        unique_dates = sorted(dataset["date"].unique())
        split_idx = int(len(unique_dates) * 0.70)
        if split_idx < 60:
            return _zero_weights(context)
        split_date = unique_dates[split_idx]

        features = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "vol_63d", "mom_63d"]
        train = dataset[dataset["date"] < split_date]
        if train.empty:
            return _zero_weights(context)

        model = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(train[features], train["target"])

        dataset["prediction"] = model.predict(dataset[features])
        prediction_pivot = dataset.pivot(index="date", columns="ticker", values="prediction")
        prediction_pivot = prediction_pivot.reindex(context.prices.index).fillna(0.0)

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
