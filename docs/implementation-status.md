# GLM-5 Paper Alignment Status

This repository follows the document in `docs/glm5-quant-strategy-paper.md`.

## Implemented in this build
- Buy and Hold Equal Weight
- Time-Series Momentum
- Cross-Sectional Momentum
- Single-Asset Mean Reversion
- Cross-Sectional Mean Reversion
- Pairs Spread Reversion
- PCA Residual Reversion
- Volatility Target Trend Overlay
- Risk Parity (Inverse Vol approximation)
- Value + Quality (price-based proxy)
- Profitability (price-based proxy)
- Regime Filtered Momentum
- Elastic Net Forecast
- Gradient Boosting Forecast

## Planned / explicitly not implemented yet
- Intraday Order Flow Imbalance (needs L2 intraday data)
- Sequence Model (LSTM) (needs deep-learning training stack)
- Hierarchical Risk Parity (full HRP clustering implementation)

## Why some are marked planned
The current default data mode uses free lagged daily bars. Certain paper strategies require richer data and heavier infra (intraday order book, GPU DL training, advanced clustering modules).

## Data and validation status
- Provider abstraction implemented: `yfinance`, `polygon`, and fallback/auto mode.
- UI/API now expose White Reality Check p-value and PBO estimate.
- Survivorship-bias warning is explicitly surfaced in API metadata and UI.
- Still required for production-grade inference: point-in-time universes including delisted symbols.
