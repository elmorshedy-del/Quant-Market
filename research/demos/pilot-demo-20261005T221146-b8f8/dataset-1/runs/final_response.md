# Final reply — run demo-20261005T221146-b8f8-dataset-1

# Results: return-process investigation (synthetic daily-return panel)

**Short answer:** The returns do not come from one process. The investigation found two kinds of assets, each following its own AR(1) process, meaning next-period return depends linearly on the asset's own previous return. Asset age is the characteristic linked to which kind an asset is. An independent audit checked this conclusion and the forecaster built on it, and found no problems. One caveat: the ledger has not yet been updated to record that audit, so on paper the work is still "pending audit."

## Where things stand on each question
- **Q1 (what produces each asset's next return):** marked **supported**. The answer is an AR(1) process: next return is a linear function of the asset's last return, with different settings for each of the two asset types. Age sets the starting estimate of which type an asset is. The asset's full return history then updates that estimate.
- **Q2 (one process or several):** marked **supported** for multiple processes, with two groups.

Both statuses cite ledger entries R1–R6. The parameter estimates themselves (each type's persistence and noise level, and how age separates the types) are recorded in `research/conclusions.json` and `research/predictor/params.json`. I have not restated the numbers here because this summary did not re-verify them.

## What was delivered
- **`research/ledger.md`**: six experiment entries (R1–R6), seven predictions (P1–P7) registered before the confirmation data was opened, and statuses for both questions.
- **Confirmation test:** run on the held-out confirmation assets after the predictions were registered. The output is in `research/artifacts/`, including `confirm_log.txt`.
- **`research/predictor/predict.py`** (with `params.json`): the auditor read it and confirmed it is a real forecaster that mixes the two AR(1) types, not a placeholder.
  - It reads only its parameter file and its inputs, and writes only to the requested output path.
  - It handles missing age, assets with no history, assets with a single observation, and missing return values.
- **`research/conclusions.json`**: lists the type of variation found, the number of processes, the parameter description, the associated characteristics and the supporting ledger entries.
- **`research/tools/`**: nine reusable analysis scripts.

## What is still unfinished
- **Ledger statuses:** all six entries and both question statuses still say "pending audit." The audit came after them and confirmed the work, but there was no further round to write that back. Until someone updates them, the first completion criterion (statuses reflecting the audits) is not formally met. The run's "incomplete" label comes from hitting the one-round limit, not from a failed check.
- **Risk for the frozen predictor:** its accuracy depends on age continuing to separate the two types in the new assets it will be scored on. If those assets have a different age mix, or age is less informative for them, accuracy will drop.
- **Prediction is not proof of mechanism:** the two-type model forecasts well and survived the held-out confirmation. That shows it predicts well. It does not show that age itself causes the difference between the types.
