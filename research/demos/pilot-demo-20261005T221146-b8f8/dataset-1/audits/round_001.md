# Audit — run demo-20261005T221146-b8f8-dataset-1, round 1

_Copied by software from the LongHorizon auditor report; do not edit._

Status: complete
Integrity: clean
Contract audit: aligned

**Audit facts**
- The deliverables exist. They are `research/ledger.md`, `research/conclusions.json`, `research/predictor/predict.py` with `params.json`, nine files in `research/tools/`, and logs in `research/artifacts/`.
- `grep -n "^### R"` on the ledger lists R1–R6. The text "pending audit" appears 8 times: six R entries plus Q1 and Q2.
- P1–P7 are in the Predictions section, and Q1 and Q2 carry statuses ("supported", with entry ids).
- I read `predict.py` directly. It is a genuine two-type AR(1) mixture forecaster, not a placeholder:
  - It uses an age-gated prior and the likelihood of the asset's full history.
  - It reads only `params.json` and the input files, and writes only `--out`.
  - It handles missing age, assets with no history, a single observation, and NaN returns.
- I read `research/artifacts/confirm_log.txt`. Its values match the Executor's table:
  - P1: r = 0.759, CI (0.682, 0.820).
  - P2: ΔBIC 1021, φ = −0.430 and +0.446.
  - P3: age coefficient 1.41, p = 1.9e-7, AUC 0.749.
  - P4: size coefficient +0.23, one-sided p = 0.86, failed.
  - P5: Δll = 330 per 100 assets, within-type r = −0.214 and +0.102.
  - P6: p = 0.92.
  - P7: MSE ratio 0.833 against pooled and 0.831 against zero, paired p = 2.3e-14.
  - Holm-adjusted values are in the log.
- `conclusions.json` has exactly the five required keys. `heterogeneity` is `multiple_processes` and `n_processes` is 2.
- I did not touch `.lh-harness`.
- The Executor's stop-hook complaint concerns the harness, not the work. The artifacts and logs show the scripts were run.

**Scientific audit:**
- **Evidence check:**
  - I read `predict.py` and `confirm_log.txt`, and ran grep checks on the ledger.
  - I recomputed the explore-side claims with my own code: split-half persistence, the spread of per-asset φ, the lag-2 mixture artefact, and the age and size relationships to φ. All matched.
  - I ran the predictor on explore history t ≤ 39.
- **Calculation check:**
  - The first-half versus second-half correlation of per-asset AR(1) φ is 0.7437, recomputed again in this turn. The Executor claimed 0.744.
  - The sd of per-asset φ is 0.403, which matches.
  - The pooled lag-2 correlation is 0.179, against the ledger's regression coefficient of 0.176. The two measures differ slightly and are consistent.
  - The correlation of age with per-asset φ is +0.45. The correlation of size with φ is −0.17, weaker, which is consistent with size failing confirmation.
  - `predict.py` took 1.19 s on 300 assets and wrote 300 finite rows in `asset_id, prediction` format. I did not rerun the 1,000-asset timing of 1.21 s that the Executor reported.
  - I did not rerun the confirmation analyses, because that would reopen the confirmation data. I read the logged results.
- **Interpretation check:**
  - Predictive and mechanistic claims are kept separate, and "why age relates to type" is stated as not established.
  - 33 tests were run against a stated cap of about 30. This is disclosed, and Holm is applied over all 33.
  - The size handle (explore Holm p = 0.049) failed confirmation and was rejected, which is the correct handling.
  - The bootstrap-resolution limit on the dispersion and mixture p-values is disclosed.
  - The `multiple_processes` conclusion rests on persistence, the age handle and out-of-sample MSE. It is not overclaimed.
  - The final predictor is refit on explore + confirmation and so has no clean holdout. The ledger says so, and 0.833 remains the honest out-of-sample figure.
  - The P5 within-type r of −0.214 sits near its 0.25 bound. It is not a failure and is flagged as open.
- **Ledger check:**
  - The R entries carry the §6 fields with `pending audit`.
  - The rejected size handle is moved with a revisit condition.
  - I did not verify the Audit log or `research/audits/` byte for byte, but nothing indicates hand edits.
  - The ledger text states that P1–P7 were written before the confirmation data was opened. File timestamps are consistent with this but cannot prove it by themselves.
- **Verdict on claims:**
  - Two-type AR(1) structure: confirmed on explore data by my recomputation; confirmation results are logged and credible but not rerun.
  - Age handle: confirmed from the explore recomputation and the confirmation log.
  - Size handle rejected: confirmed.
  - Predictor validity: confirmed.
  - Mechanism: appropriately limited to the statistical structure.

**Gaps and next step:**
- Minor only: one unused line in `run_explore.py`, and the near-bound within-type correlation noted above.
- This was the final round, so the next step is the harness's audit-log update.

**Acceptance-constraint backcheck:**
- **Contract conclusion:** aligned.
- **Original constraint inventory:**
  - Final consumer and state carriers: the ledger, `conclusions.json`, and the frozen predictor scored on fresh assets.
  - Authoritative inputs: `data/explore` and `data/confirmation`. Confirmation is locked until a P entry exists.
  - Files: R entries in §6 format, `predict.py`, `conclusions.json` with the required keys and allowed values, and tools with docstrings.
  - Predictor interface: run from the workspace root, handle h ≥ 40, finish in under 60 s for 1,000 assets, modify no files.
  - Forbidden: edits to the Audit log, inflated or invented results, a placeholder predictor, and paths outside the workspace.
- **Contract coverage check:** No omissions or distortions found. The "no refit on confirmation before it is recorded" rule is consistent with the original request, and the Executor followed it.
- **Per-constraint backcheck:**
  1. Full §6 entries with `pending audit`: verified by grep.
  2. Predictions registered before confirmation: verified from ledger text and the lock design. It is not independently provable from timestamps alone, which is non-blocking.
  3. Confirmation used only for testing, with the final refit after R5 and disclosed: verified from ledger text.
  4. Predictive and mechanistic claims separated: verified in `conclusions.json` and the ledger.
  5. Test count and correction: verified. 33 tests with Holm, and the overshoot of the cap is disclosed.
  6. Predictor interface: verified by running it on 300 assets and by reading the source. The 1,000-asset timing is the Executor's claim and is plausible given the 1.19 s run.
  7. Work stays in the workspace: no evidence of violation.
  8. Audit log and `research/audits/` not edited: no evidence of edits.
- **Blocking constraints:** none.
- **Possible scoring risks:** The final refit has no held-out estimate. The predictor depends on the age handle holding in fresh assets, and a shift in the age distribution would degrade accuracy.
- **Over-narrow or incorrect interpretation:** none.
- **Recommended contract revision:** none.

**State update for manager:**
- Trusted: `research/ledger.md` R1–R6, P1–P7 and H entries; `research/conclusions.json` (multiple_processes, n=2, age); `research/predictor/predict.py` and `params.json`; the `research/tools/` scripts.
- Residual caveats: the final predictor has no clean holdout, and the 33 tests exceeded the manager's cap of about 30, though they are fully corrected.
- Untrusted artifacts: none.
- The research budget is exhausted, so the run can finish with Q1 and Q2 marked "supported", pending the harness audit-log update.
