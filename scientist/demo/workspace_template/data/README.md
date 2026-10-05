# Data

A synthetic daily-return panel. Each asset has static characteristics and one return series.

| File | Columns | Notes |
|---|---|---|
| `explore/assets.csv` | `asset_id, sector, size, liquidity, value, age` | `sector` is categorical (A–F); the other four are standardized characteristics |
| `explore/returns.csv` | `asset_id, t, ret` | simple daily returns, `t = 1..40`, no missing values |
| `confirmation/assets.csv`, `confirmation/returns.csv` | same as above | different assets from the same source; **locked** (see below) |

`confirmation/` is held out for confirming findings. A hook keeps it locked until at least one
prediction is registered in the ledger's *Predictions* section as `- **P<n>** ...`. Use it to
test registered predictions, not to explore.

After the run, a frozen copy of your predictor (`research/predictor/`) will be scored on
fresh assets from the same source, which do not exist during the run.
