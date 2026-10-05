# Predictor interface

The deliverable `research/predictor/predict.py` must run as

    .venv/bin/python research/predictor/predict.py --assets ASSETS.csv --history HISTORY.csv --out OUT.csv

from the workspace root, where

- `ASSETS.csv` has the columns of `data/explore/assets.csv`, for assets **never seen before**;
- `HISTORY.csv` has `asset_id, t, ret` for `t = 1..h` (h ≥ 40) for those assets;
- `OUT.csv` must contain `asset_id, prediction`: each asset's predicted return at `t = h + 1`.

It is called once per target period with a growing history. It may read files under `research/`
(for example, parameters you fitted). It must finish within 60 seconds for 1,000 assets, and it
must not modify files.

Scoring after the run is mean squared error of one-step-ahead predictions on fresh assets.
