from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import MAX_ABSOLUTE_LEVERAGE


@dataclass(frozen=True)
class StrategyMetaInfo:
    strategy_id: str
    name: str
    family: str
    implemented: bool
    complexity: str
    data_requirements: str
    notes: str = ""


@dataclass(frozen=True)
class StrategyContext:
    prices: pd.DataFrame
    volumes: pd.DataFrame

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change().fillna(0)


class BaseStrategy:
    meta: StrategyMetaInfo

    def generate_weights(self, context: StrategyContext) -> pd.DataFrame:
        raise NotImplementedError


def _normalize_abs(row: pd.Series, max_leverage: float = MAX_ABSOLUTE_LEVERAGE) -> pd.Series:
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = row.abs().sum()
    if gross <= 0:
        return row * 0.0
    scaled = row / gross
    return scaled * max_leverage


def normalize_weight_frame(raw: pd.DataFrame, max_leverage: float = MAX_ABSOLUTE_LEVERAGE) -> pd.DataFrame:
    if raw.empty:
        return raw
    return raw.apply(lambda row: _normalize_abs(row, max_leverage=max_leverage), axis=1)


def long_only_from_score(scores: pd.DataFrame, top_quantile: float = 0.5) -> pd.DataFrame:
    def transform(row: pd.Series) -> pd.Series:
        threshold = row.quantile(1 - top_quantile)
        long_mask = row >= threshold
        weights = pd.Series(0.0, index=row.index)
        if long_mask.sum() > 0:
            weights.loc[long_mask] = 1.0
        return _normalize_abs(weights, max_leverage=1.0)

    return scores.apply(transform, axis=1)


def long_short_from_score(scores: pd.DataFrame, quantile: float = 0.3) -> pd.DataFrame:
    def transform(row: pd.Series) -> pd.Series:
        hi = row.quantile(1 - quantile)
        lo = row.quantile(quantile)
        w = pd.Series(0.0, index=row.index)
        w.loc[row >= hi] = 1.0
        w.loc[row <= lo] = -1.0
        return _normalize_abs(w, max_leverage=1.0)

    return scores.apply(transform, axis=1)
