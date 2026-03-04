from __future__ import annotations

from .base import StrategyMetaInfo
from .implemented import IMPLEMENTED_STRATEGY_CLASSES


PLANNED_ONLY = [
    StrategyMetaInfo(
        strategy_id="intraday_order_flow_imbalance",
        name="Intraday Order Flow Imbalance (OFI)",
        family="Intraday Microstructure",
        implemented=False,
        complexity="very_high",
        data_requirements="intraday_l2_orderbook",
        notes="Planned. Requires level-2 order book data and intraday execution simulator.",
    ),
    StrategyMetaInfo(
        strategy_id="sequence_model_lstm",
        name="Sequence Model (LSTM)",
        family="ML (Sequence)",
        implemented=False,
        complexity="very_high",
        data_requirements="daily_ohlcv_plus_feature_store",
        notes="Planned. Requires deep-learning stack and strict walk-forward validation pipeline.",
    ),
    StrategyMetaInfo(
        strategy_id="hierarchical_risk_parity",
        name="Hierarchical Risk Parity (HRP)",
        family="Portfolio Optimization",
        implemented=False,
        complexity="high",
        data_requirements="daily_ohlcv_plus_covariance_clustering",
        notes="Planned. Requires robust hierarchical clustering and cluster variance recursion module.",
    ),
]


def get_implemented_strategy_instances():
    return {strategy_id: cls() for strategy_id, cls in IMPLEMENTED_STRATEGY_CLASSES.items()}


def list_strategy_meta() -> list[StrategyMetaInfo]:
    implemented = [instance.meta for instance in get_implemented_strategy_instances().values()]
    all_meta = implemented + PLANNED_ONLY
    return sorted(all_meta, key=lambda item: (not item.implemented, item.family, item.name))
