# Quant Strategy Tournament (GLM-5 Aligned)

A deployable web app that runs a **quant strategy tournament** on configurable market data providers and ranks strategies by both P&L and repeatability.

This repo is aligned to the GLM-5 strategy paper in:
- `docs/glm5-quant-strategy-paper.md`

## What it does
- Pulls daily market data from configurable providers (`yfinance` or paid `polygon`)
- Runs multiple strategy families in one framework
- Applies lagged execution (`signal t`, `fill t+1`)
- Applies trading + borrow cost model
- Computes performance and risk metrics
- Computes repeatability score (`skill probability + deflated sharpe confidence + stability + turnover penalty`)
- Computes White Reality Check p-value + PBO estimate for tournament-level robustness
- Returns ranked leaderboard via API and browser UI

## Data provider modes
- `DATA_SOURCE=yfinance`: free, lagged, easy to start.
- `DATA_SOURCE=polygon`: paid API key (`POLYGON_API_KEY`) for more reliable bars.
- `DATA_SOURCE=auto`: follows `DATA_SOURCE_FALLBACK_ORDER` (for example `polygon,yfinance`).

Important:
- Better bars do not automatically remove survivorship bias.
- Production-grade research still needs point-in-time universes including delisted symbols.

## Current implementation scope
See `docs/implementation-status.md`.

## Quick start (local)

Use Python 3.12 (recommended). This repo is prepared for Railway's Python 3.12 runtime.

```bash
cd quant-strategy-tournament-glm5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8010
```

Open:
- http://127.0.0.1:8010

## API

### Health
`GET /api/health`

### Strategy catalog
`GET /api/strategies`

### Run tournament
`POST /api/tournament/run`

Example body:

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "SPY", "QQQ"],
  "start_date": "2024-01-01",
  "end_date": "2026-01-01",
  "initial_capital": 100000,
  "trading_cost_bps": 10,
  "borrow_cost_bps_annual": 50,
  "risk_free_rate_annual": 0.02
}
```

## Deployment

### Docker

```bash
docker build -t quant-tournament .
docker run -p 8000:8000 quant-tournament
```

### Railway/Render/Fly
- `Dockerfile` and `Procfile` included.
- `railway.toml`, `nixpacks.toml`, and `runtime.txt` are included for Railway.
- Start command:

```bash
python -m app.serve
```

Recommended Railway env vars:

```bash
APP_ENV=production
DATA_SOURCE=polygon
POLYGON_API_KEY=your_key_here
DATA_SOURCE_ALLOW_FALLBACK=true
DATA_SOURCE_FALLBACK_ORDER=polygon,yfinance
DATA_REQUEST_TIMEOUT_SECONDS=30
POLYGON_RETRY_ATTEMPTS=3
POLYGON_RETRY_BACKOFF_SECONDS=2
POLYGON_RETRY_MAX_SLEEP_SECONDS=30
DATA_CACHE_DIR=data/cache
ALLOWED_ORIGINS=https://your-frontend-domain.com
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_MAX_REQUESTS=10
SHOW_SURVIVORSHIP_WARNING=true
DEFAULT_TRADING_COST_BPS=10
DEFAULT_BORROW_COST_BPS_ANNUAL=50
DEFAULT_RISK_FREE_RATE_ANNUAL=0.02
```

## Notes
- This is a research backtesting environment, not production execution software.
- Fundamental and intraday strategies require richer data feeds for full-fidelity implementation.
- Paid bars improve reliability, but survivorship-free constituent history is still required for institutional-quality validation.
- Free Polygon tiers can hit HTTP 429 quickly on multi-ticker scans; use retries/fallback or a paid tier.
- Always perform paper trading before live deployment.
