from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR, settings
from .data import DataLoadError, load_price_data
from .metrics import (
    annualized_return,
    probability_of_backtest_overfitting,
    calmar_ratio,
    bootstrap_probability_of_skill,
    cvar_5pct,
    deflated_sharpe_confidence,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover_annualized,
    white_reality_check_pvalue,
)
from .models import StrategyResult, TournamentRequest, TournamentResponse
from .strategies.base import StrategyContext
from .strategies.registry import get_implemented_strategy_instances, list_strategy_meta


COMPLEXITY_PENALTY = {
    "low": 1.00,
    "medium": 0.85,
    "high": 0.70,
    "very_high": 0.55,
}


class TournamentEngine:
    def __init__(self) -> None:
        self._strategies = get_implemented_strategy_instances()

    def _evaluate_strategy(
        self,
        strategy_id: str,
        strategy,
        context: StrategyContext,
        request: TournamentRequest,
        num_trials: int,
    ) -> tuple[StrategyResult, pd.Series]:
        raw_weights = strategy.generate_weights(context)
        if raw_weights is None or raw_weights.empty:
            raise ValueError("Strategy returned empty weights.")

        weights = raw_weights.reindex(context.prices.index).fillna(0.0)
        returns = context.returns.reindex(context.prices.index).fillna(0.0)

        executed_weights = weights.shift(1).fillna(0.0)

        gross = (executed_weights * returns).sum(axis=1)

        turnover_daily = executed_weights.diff().abs().sum(axis=1).fillna(0.0)
        trading_cost = turnover_daily * (request.trading_cost_bps / 10_000.0)

        borrow_daily_rate = (request.borrow_cost_bps_annual / 10_000.0) / TRADING_DAYS_PER_YEAR
        short_exposure = executed_weights.clip(upper=0).abs().sum(axis=1)
        borrow_cost = short_exposure * borrow_daily_rate

        net = gross - trading_cost - borrow_cost

        equity_curve = (1.0 + net).cumprod() * request.initial_capital
        cagr = annualized_return(net)
        sharpe = sharpe_ratio(net, request.risk_free_rate_annual)
        sortino = sortino_ratio(net, request.risk_free_rate_annual)
        max_dd = max_drawdown(net)
        calmar = calmar_ratio(cagr, max_dd)
        turn_annual = turnover_annualized(executed_weights)
        hit = hit_rate(net)
        cvar = cvar_5pct(net)
        prob_skill = bootstrap_probability_of_skill(net, request.risk_free_rate_annual)
        dsr_conf = deflated_sharpe_confidence(sharpe, net, num_trials=num_trials)

        rolling_sharpe = net.rolling(63).apply(
            lambda x: sharpe_ratio(pd.Series(x), request.risk_free_rate_annual), raw=False
        )
        rolling_std = float(rolling_sharpe.std(skipna=True) or 0.0)
        regime_stability = float(1.0 / (1.0 + max(0.0, rolling_std)))

        complexity_penalty = COMPLEXITY_PENALTY.get(strategy.meta.complexity, 0.75)
        turnover_penalty = max(0.1, 1.0 - min(turn_annual / 8.0, 0.9))

        repeatability_score = (
            0.45 * prob_skill
            + 0.25 * dsr_conf
            + 0.20 * regime_stability
            + 0.10 * turnover_penalty
        ) * complexity_penalty

        final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else request.initial_capital
        expected_pnl = final_equity - request.initial_capital

        return StrategyResult(
            strategy_id=strategy_id,
            name=strategy.meta.name,
            family=strategy.meta.family,
            implemented=True,
            total_return_pct=float((final_equity / request.initial_capital - 1.0) * 100.0),
            cagr_pct=float(cagr * 100.0),
            sharpe=float(sharpe),
            sortino=float(sortino),
            max_drawdown_pct=float(max_dd * 100.0),
            calmar=float(calmar),
            turnover_annual=float(turn_annual),
            hit_rate_pct=float(hit * 100.0),
            cvar_5_pct=float(cvar * 100.0),
            final_equity=float(final_equity),
            expected_pnl=float(expected_pnl),
            probability_of_skill=float(prob_skill),
            repeatability_score=float(repeatability_score),
            complexity=strategy.meta.complexity,
            skipped_reason=None,
        ), net

    def _benchmark(self, prices: pd.DataFrame, initial_capital: float) -> dict[str, float]:
        benchmark_price = prices.mean(axis=1)
        benchmark_returns = benchmark_price.pct_change().fillna(0.0)
        equity = (1.0 + benchmark_returns).cumprod() * initial_capital
        return {
            "name": "Equal-Weight Buy & Hold Basket",
            "total_return_pct": float((equity.iloc[-1] / initial_capital - 1.0) * 100.0),
            "cagr_pct": float(annualized_return(benchmark_returns) * 100.0),
            "max_drawdown_pct": float(max_drawdown(benchmark_returns) * 100.0),
            "final_equity": float(equity.iloc[-1]),
        }

    def run(self, request: TournamentRequest) -> TournamentResponse:
        run_id = str(uuid.uuid4())
        started = time.time()

        close, volume, provider_map = load_price_data(
            tickers=request.tickers,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
        )
        context = StrategyContext(prices=close, volumes=volume)

        selected_ids = (
            [sid for sid in request.strategy_ids if sid]
            if request.strategy_ids
            else list(self._strategies.keys())
        )
        selected_implemented_count = max(1, sum(1 for sid in selected_ids if sid in self._strategies))

        ranking: list[StrategyResult] = []
        skipped: list[StrategyResult] = []
        strategy_net_series: dict[str, pd.Series] = {}

        strategy_meta_lookup = {m.strategy_id: m for m in list_strategy_meta()}

        for strategy_id in selected_ids:
            meta = strategy_meta_lookup.get(strategy_id)
            strategy = self._strategies.get(strategy_id)

            if strategy is None:
                reason = "Strategy is not implemented yet in this build."
                if meta is None:
                    reason = "Unknown strategy id."
                skipped.append(
                    StrategyResult(
                        strategy_id=strategy_id,
                        name=meta.name if meta else strategy_id,
                        family=meta.family if meta else "Unknown",
                        implemented=False,
                        total_return_pct=0.0,
                        cagr_pct=0.0,
                        sharpe=0.0,
                        sortino=0.0,
                        max_drawdown_pct=0.0,
                        calmar=0.0,
                        turnover_annual=0.0,
                        hit_rate_pct=0.0,
                        cvar_5_pct=0.0,
                        final_equity=request.initial_capital,
                        expected_pnl=0.0,
                        probability_of_skill=0.0,
                        repeatability_score=0.0,
                        complexity=meta.complexity if meta else "unknown",
                        skipped_reason=reason,
                    )
                )
                continue

            try:
                result, net_series = self._evaluate_strategy(
                    strategy_id,
                    strategy,
                    context,
                    request,
                    num_trials=selected_implemented_count,
                )
                ranking.append(result)
                strategy_net_series[strategy_id] = net_series
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    StrategyResult(
                        strategy_id=strategy_id,
                        name=strategy.meta.name,
                        family=strategy.meta.family,
                        implemented=True,
                        total_return_pct=0.0,
                        cagr_pct=0.0,
                        sharpe=0.0,
                        sortino=0.0,
                        max_drawdown_pct=0.0,
                        calmar=0.0,
                        turnover_annual=0.0,
                        hit_rate_pct=0.0,
                        cvar_5_pct=0.0,
                        final_equity=request.initial_capital,
                        expected_pnl=0.0,
                        probability_of_skill=0.0,
                        repeatability_score=0.0,
                        complexity=strategy.meta.complexity,
                        skipped_reason=f"Failed during execution: {exc}",
                    )
                )

        ranking = sorted(
            ranking,
            key=lambda r: (r.expected_pnl, r.repeatability_score, r.sharpe),
            reverse=True,
        )

        benchmark_returns = context.prices.mean(axis=1).pct_change().fillna(0.0)
        strategy_return_frame = pd.DataFrame(strategy_net_series).reindex(context.prices.index).fillna(0.0)
        wrc_pvalue = white_reality_check_pvalue(
            strategy_return_frame=strategy_return_frame,
            benchmark_returns=benchmark_returns,
        )
        pbo_estimate = probability_of_backtest_overfitting(strategy_return_frame)

        elapsed = time.time() - started
        warnings: list[str] = []
        providers_used = sorted({provider for provider in provider_map.values() if provider})

        if settings.show_survivorship_warning:
            if "yfinance" in providers_used:
                warnings.append(
                    "yfinance data is free/lagged and may be survivorship-biased (current tickers only). "
                    "Use survivorship-free point-in-time universes before production decisions."
                )
            else:
                warnings.append(
                    "Provider bars alone do not remove survivorship bias. "
                    "Use point-in-time universes that include delisted names before production decisions."
                )

        if len(set(provider_map.values())) > 1:
            warnings.append(
                "Mixed provider run: some tickers used fallback providers due to upstream errors/rate-limits."
            )
        if settings.data_source != "auto" and settings.data_source not in providers_used:
            warnings.append(
                f"Configured DATA_SOURCE '{settings.data_source}' was unavailable. Fallback providers were used."
            )

        metadata = {
            "run_seconds": round(elapsed, 2),
            "bar_count": int(len(close)),
            "data_source_config": settings.data_source,
            "data_providers_used": providers_used,
            "data_provider_by_ticker": provider_map,
            "strategies_requested": len(selected_ids),
            "strategies_completed": len(ranking),
            "strategies_skipped": len(skipped),
            "framework": "GLM-5 paper aligned tournament with lagged execution, costs, and repeatability ranking",
            "cost_model": {
                "trading_cost_bps": request.trading_cost_bps,
                "borrow_cost_bps_annual": request.borrow_cost_bps_annual,
                "execution": "signal at t, execution at t+1"
            },
            "statistical_method": {
                "bootstrap_samples": "default",
                "dsr_trials": selected_implemented_count,
                "white_reality_check_pvalue": round(float(wrc_pvalue), 6),
                "pbo_estimate": round(float(pbo_estimate), 6),
            },
            "warnings": warnings,
        }

        benchmark = self._benchmark(close, request.initial_capital)

        return TournamentResponse(
            run_id=run_id,
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            benchmark=benchmark,
            ranking=ranking,
            skipped=skipped,
            metadata=metadata,
        )


def list_strategy_catalog() -> list[dict]:
    return [asdict(meta) for meta in list_strategy_meta()]


def run_tournament(request: TournamentRequest) -> TournamentResponse:
    engine = TournamentEngine()
    try:
        return engine.run(request)
    except DataLoadError as exc:
        raise ValueError(str(exc)) from exc
