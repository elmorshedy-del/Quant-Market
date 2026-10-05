# Research Ledger

Durable scientific state for this investigation. Format and rules: `SCIENTIST.md` §6.
Read this before starting work, and append to it rather than rewriting it. Rejected
hypotheses are never deleted. The *Audit log* is written by software.

Statement labels: **[Obs]** observation · **[Asm]** assumption · **[Int]** interpretation · **[Pred]** prediction.

## Questions

- **Q1** [supported: two-type AR(1) in own last return, type predicted by age — R2, R3, R5, R6; pending audit] What process generates each asset's next-period return in this dataset, and how does it depend on the information available (the asset's own past returns and its characteristics)?
- **Q2** [supported: multiple processes (2) — R2, R3, R5; pending audit] Do all assets share one such process, or are there distinct groups of assets whose next-period returns are generated differently? Conclude: single process, multiple processes, or insufficient evidence.

## Observations and sources

- [Obs] `data/explore/assets.csv`: 300 assets with columns asset_id, sector (A–F: C 61, F 60, E 50, D 46, A 45, B 38), and size, liquidity, value, age (standardized; mean about 0, sd about 1; pairwise |corr| ≤ 0.05). Source: inline pandas summary, R1.
- [Obs] `data/explore/returns.csv`: 12,000 rows = 300 assets × t=1..40, no missing values. Pooled mean ret 0.00064, sd 0.0152. Source: R1.
- [Obs] Pooled clustered OLS of r_t on r_{t-1..t-5}: lag1 0.016 (p=0.51), lag2 0.176 (p=5e-48), lag3 0.025 (p=0.03), lags 4–5 ≈ 0. Source: `research/tools/run_explore.py`, R2.
- [Obs] Per-asset AR(1) phi is bimodal (histogram over 10 bins on [-1,1]: 0 7 79 43 19 24 68 55 5 0). Its sd is 0.403, against a 95% single-process noise floor of 0.166. Source: R2.
- [Obs] The confirmation data was opened only after P1–P7 were written (R5). `data/confirmation`: 150 assets × t=1..40 (sectors B 32, E 29, C 25, D 23, F 21, A 20). Source: `research/tools/run_confirm.py`.
- [Obs] Confirmation (frozen explore model): P1, P2, P3, P5, P6 and P7 pass; P4 fails. Source: `research/artifacts/confirm_results.json`, R5.

## Measurement and selection assumptions

- [Asm] The data are synthetic. There is no survivorship or selection: every generated asset is present with a full t=1..40 history (README: "no missing values").
- [Asm] `ret` is a simple return per period. The process is taken to be stationary within t=1..40, with no common time factor assumed (not tested beyond pooled regressions).
- [Asm] The confirmation assets and the future scoring assets are draws from the same generator as the explore assets (README: "same source").
- [Asm] Each asset has one fixed latent type for its whole history. This is an asset-level mixture, not regime switching over time. Half-split persistence (R2) is the evidence that the type is stable.
- [Asm] Bootstrap noise floors use Gaussian innovations from the fitted model. Non-Gaussian tails are not modelled.
- [Asm] Open decision, recorded because no human is available: the final predictor is refit on explore + confirmation after the confirmation test is recorded (R6). The confirmation claims in R5 use only the frozen explore fit.

## Competing hypotheses

- **H1** [rejected, see Rejected explanations] Returns are i.i.d. around an asset-common mean. Next return is unpredictable from the past. — Predicts: per-asset AC within the noise floor, zero-forecast MSE ≈ best. — R2.
- **H2** [rejected, see Rejected explanations] One pooled AR(p) process shared by all assets. — Predicts: per-asset phi dispersion within the floor, K=1 preferred. — R2.
- **H3** [rejected, see Rejected explanations] One process whose AR(1) coefficient loads continuously on characteristics (phi_i = b0 + b·chars). — Predicts: unimodal phi given chars, phi correlated with age within any grouping, continuous model ≥ discrete mixture. — R2, R5.
- **H6** [rejected, see Rejected explanations] Mean return depends on characteristics or sector. — Predicts: residual per-asset means correlate with chars or differ by sector. — R2, R3, R5.
- **H4** [active, supported in R2–R5] Two latent AR(1) types: a reversal type (phi ≈ −0.41) and a momentum type (phi ≈ +0.40), with a common small intercept (≈ 0.0005) and similar innovation sd (≈ 0.0135 vs 0.0142). The type is a permanent property of the asset. — Predicts: bimodal per-asset phi, high half-split persistence, a K=2 mixture far ahead of K=1, and a pooled lag-2 coefficient ≈ E[phi²] with a pooled lag-1 ≈ 0. — R2, R3, R5.
- **H5** [active, age part supported in R5; size part rejected in R5 (P4 failed)] The probability of the momentum type rises with `age` (logit slope ≈ +1.4 per sd). The weak exploratory `size` term (≈ −0.47 per sd) did not replicate. Liquidity, value and sector do not predict type. — Predicts: in new assets, age is positively associated with the history-inferred type. — R2, R5.
- **H7** [active, supported in R3/R5] Per-asset innovation volatility varies continuously across assets beyond the noise floor and is persistent. It is not explained by characteristics. It is a nuisance for typing, not a third dynamic type. — R3, R5.

## Predictions

- **P1** (from H4, registered in R4, 2026-10-05) In `data/confirmation`, the correlation between per-asset AR(1) phi estimated on t=1..20 and on t=21..40 is > 0.5, with the lower 95% CI bound > 0.3. — data/confirmation.
- **P2** (from H4, registered in R4, 2026-10-05) In confirmation, a K=2 asset-level AR(1) mixture beats K=1 by ΔBIC > 10. The refit component phis fall in [−0.55, −0.28] and [+0.27, +0.53]. On per-asset standardized returns, K=3 does not beat K=2 by ΔBIC > 10. — data/confirmation.
- **P3** (from H5, registered in R4, 2026-10-05) In confirmation, types are labelled from each asset's history alone, using the frozen explore components with no gate (posterior > 0.5 = momentum). A logistic regression of that label on age then has a positive coefficient with p < 0.01. The frozen explore gate probability (0.130 − 0.468·size + 1.492·age) predicts the label with AUC > 0.70. — data/confirmation.
- **P4** (from H5, low confidence, registered in R4, 2026-10-05) In the same logistic regression with age and size, the size coefficient is < 0 with one-sided p < 0.05. — data/confirmation.
- **P5** (from H4 versus H3, registered in R4, 2026-10-05) Discrete rather than continuous loading. In confirmation, the explore-frozen gated K=2 mixture has a higher log-likelihood than the explore-frozen continuous-loading single process (phi = b0 + b1·age + b2·size) by > 5 per 100 assets. Within each history-labelled type, |corr(per-asset phi, age)| < 0.25. — data/confirmation.
- **P6** (from H6-null, registered in R4, 2026-10-05) Sector does not shift mean returns. An ANOVA of per-asset residual means (after the frozen mixture forecast) across sectors gives p > 0.05. — data/confirmation.
- **P7** (predictive, from H4+H5, registered in R4, 2026-10-05) The frozen explore `mix_gated` forecaster is applied to confirmation targets t=21..40, each using history 1..t-1. Its MSE is < 0.90 × pooled AR(1) MSE, < 0.90 × zero-forecast MSE, and < 0.97 × per-asset OLS AR(1) MSE. The paired per-asset MSE difference versus pooled is one-sided p < 0.01. — data/confirmation.
- Correction plan registered with P1–P7: Holm across the p-values of P1, P2 (bootstrap-free ΔBIC criterion is reported separately), P3, P4, P5-within-type correlations, P6 and P7, at α = 0.05.

## Experiment results

### R1 — Data schema and planted-truth validation of the mixture pipeline
- **Question:** Can the asset-level AR(1) mixture pipeline recover a known two-type structure and avoid inventing one in a homogeneous panel of the same shape (n=300, T=40)?
- **Reasoning move:** planted-truth (protocol), in support of hidden-states-and-trajectories
- **Justification:** A mixture-of-regressions EM can find spurious components. Before it is trusted on real data it must recover a planted answer and pass a negative control.
- **Prediction (registered before running):** On the planted two-type panel (phi = −0.5 and +0.4, logistic gate slope 1.5), phis are recovered within 0.05, accuracy > 0.9, ΔBIC(1→2) > 10, and the gate slope is positive. On the homogeneous panel, ΔBIC(1→2) < 10.
- **Procedure:** `.venv/bin/python research/tools/run_planted_truth.py` (simulator and fitter in `research/tools/panel_tools.py`). Data schema read with inline pandas (`data/explore/*.csv`).
- **Actual result:** [Obs] Planted panel: recovered phi = (−0.519, +0.421), sig = (0.0136, 0.0140), accuracy 1.000, BIC K1/K2/K3 = −64050.1 / −66551.9 / −66532.0 (ΔBIC12 = 2501.7, K=3 not preferred), gated slope 1.78. Homogeneous panel: K=2 phis (−0.032, 0.077), ΔBIC12 = −18.7, so no split. Printed PASS in 2.6 s. Schema facts are in *Observations*.
- **Verification status:** pending audit
- **Change in belief:** The pipeline can be trusted to tell two AR types from one at this n and T.
- **Artifacts:** `research/tools/panel_tools.py`, `research/tools/run_planted_truth.py`

### R2 — Pooled dynamics, per-asset noise floor, persistence, mixture and handle search (explore only)
- **Question:** Do the pooled dynamics hide distinct, persistent per-asset types, and does any characteristic predict the type? Are mean returns linked to characteristics? (Q1, Q2)
- **Reasoning move:** hidden-states-and-trajectories
- **Justification:** The pooled lag-1 coefficient is about 0 but lag 2 is strong. That pattern is what a mixture of ±phi AR(1) types produces (lag-2 ≈ E[phi²] > 0 while E[phi] ≈ 0). Only per-unit analysis plus a persistence test can tell such a mixture from noise.
- **Prediction (registered before running):** (Manager's belief-changing observation.) "Multiple processes" is favored if the dispersion of per-asset AR(1) exceeds the simulated homogeneous noise floor at p < 0.01, the half-split estimates correlate positively with a 95% CI excluding 0, a ≥2-component mixture beats one component by ΔBIC > 10, and the confirmation results reproduce this under Holm. "Single process" is favored if dispersion is within the floor, persistence ≈ 0, and pooled ≥ grouped on confirmation.
- **Procedure:** `.venv/bin/python research/tools/run_explore.py` (log `research/artifacts/explore_log.txt`). This produced 28 counted tests with Holm over all of them (`research/artifacts/explore_tests.csv`). Noise floors come from 500 parametric bootstraps of the K=1 fit. Mixture LRTs were bootstrapped with B=40. Gate tests are LR versus the ungated K=2 model (χ²).
- **Actual result:** [Obs] Pooled lag2 = 0.176 (p_Holm 1.4e-46). Lag1 = 0.016, lag3 = 0.025, lags 4–5 ≈ 0 (none survive Holm). Per-asset phi sd = 0.403 against a floor mean of 0.156 and 95th percentile of 0.166 (bootstrap p = 0.002, the resolution limit). Half-split persistence of phi: r = 0.744, 95% CI (0.688, 0.790), p_Holm 1.3e-52. Assets with a negative first-half phi have a median second-half phi of −0.397; those with a positive first-half phi, +0.255. Persistence of the per-asset mean: r = 0.007, CI (−0.106, 0.120). Persistence of the per-asset sd: r = 0.563. Mixtures: K=1 phi 0.025, BIC −64731.5. K=2 phi (−0.412, +0.402), c (0.00047, 0.00051), sig (0.0135, 0.0142), BIC −66452.5 (ΔBIC12 = 1720.9). K=3 BIC −66680.2 (splits the positive type by sig: 0.0110 vs 0.0162). Component intercepts do not differ (LR 0.02, p 0.88); sigmas differ slightly (p_Holm 0.014). Gate tests (raw p → Holm): age slope +1.43, LR 79.5, p 4.8e-19 → 1.2e-17. Size −0.38, p 0.0018 → 0.040. Liquidity p 0.97, value p 0.58, sector (5 df) p 0.13, all → 1. Within-type corr(per-asset phi, age): −0.10 and +0.10 (p ≈ 0.2), so no continuous loading inside a type. A continuous-loading single process (phi = 0.021 + 0.212·age − 0.078·size) has BIC −65259.7, worse than ungated K=2 at −66452.5 by 1193. Mean-return effects of size, liquidity, value and age: all |r| < 0.07, p > 0.25. Sector ANOVA on residual means: p 0.0026 (p_Holm 0.046 within these 28). Final explore gated model (age + size): phi (−0.411, +0.403), c (0.00046, 0.00052), sig (0.0135, 0.0142), logit P(momentum) = 0.130 − 0.468·size + 1.492·age, BIC −66532.1 (`research/artifacts/explore_fit.json`).
- **Verification status:** pending audit
- **Change in belief:** [Int] Strongly toward H4, two persistent AR(1) types of opposite sign, and H5, an age handle with a weaker size handle. Against H1 (i.i.d.), H2 (one pooled AR) and H3 (continuous loading). The pooled lag-2 effect is explained as a mixture artefact (0.5·0.41² + 0.5·0.40² ≈ 0.165 versus the observed 0.176), not as a separate lag-2 mechanism. The extra K=3 component and the per-asset mean dispersion were followed up in R3.
- **Artifacts:** `research/tools/run_explore.py`, `research/artifacts/explore_tests.csv`, `research/artifacts/explore_fit.json`, `research/artifacts/explore_log.txt`

### R3 — Follow-ups: correct noise floors, volatility artefact, sector means (explore only)
- **Question:** Is the K=3 preference a third dynamic type or per-asset volatility heterogeneity? Is the excess per-asset mean dispersion in R2 real or an artefact of using a K=1 floor? Do the mixture LR p-values survive a higher-resolution bootstrap?
- **Reasoning move:** revise-assumptions (the R2 noise floor assumed K=1, and a raw-scale mixture conflates volatility with type)
- **Justification:** These are the failure checks in the hidden-states skill: "dispersion from scale, not type", and a noise floor taken under the wrong null.
- **Prediction (registered before running):** If K=3 reflects volatility only, then on per-asset standardized returns K=3 will not beat K=2 by ΔBIC > 10. If the per-asset mean dispersion comes from the AR mixture, it will fall within the K=2 floor.
- **Procedure:** `.venv/bin/python research/tools/run_explore2.py` (log `research/artifacts/explore2_log.txt`). This added 5 tests (cumulative 33) and re-ran Holm over all 33 (`research/artifacts/explore_tests_all.csv`).
- **Actual result:** [Obs] Per-asset mean dispersion versus the K=2 floor: 0.00293 against a 95th percentile of 0.00314, p 0.41, so it is within the floor. Per-asset sigma dispersion versus the K=2 floor: 0.00314 against 0.00175, p 0.002 (p_Holm 0.052, resolution-limited). K=2 versus K=1 LR = 1743.7; the maximum null LR over 200 bootstraps was 20.5 (p ≤ 0.005, the resolution limit; p_Holm 0.10 only because of that resolution). Standardized returns: K=2 phi (−0.41, +0.34), BIC 31839.2; K=3 BIC 31848.3 (ΔBIC = −9.1); K=4 31865.8. The K=3 vs K=2 LR is 13.7 against a null 95th percentile of 19.8 (p 0.15). Log per-asset sigma on characteristics + sector: F p 0.10. With all 33 tests under Holm, the sector mean effect has p_Holm 0.057 (not significant), the size gate 0.049, the age gate 1.4e-17, and phi persistence 1.5e-52.
- **Verification status:** pending audit
- **Change in belief:** [Int] The third raw component is a volatility artefact, so H7 is a nuisance and not a third type. Two dynamic types remain. There is no evidence of stable per-asset mean differences. The sector mean effect is not established after correction, so its null is registered as P6. The tests that are limited by bootstrap resolution (dispersion, mixture LR) cannot reach Holm significance with 33 tests at B ≤ 500. The type evidence that survives correction rests on half-split persistence (p_Holm 1.5e-52) and the age gate (p_Holm 1.4e-17). The ΔBIC of 1721 is reported separately as a non-p-value criterion. The test count (33) exceeded the Manager's soft cap of about 30 by 3, and all 33 are included in the correction.
- **Artifacts:** `research/tools/run_explore2.py`, `research/artifacts/explore_tests_all.csv`, `research/artifacts/explore2_log.txt`

### R4 — Forecaster selection by asset-level cross-validation (explore only) and registration of P1–P7
- **Question:** Which forecaster minimizes one-step MSE on assets unseen by the fit, and how much does typing help? (predictive part of Q1)
- **Reasoning move:** deriving-predictions
- **Justification:** The scoring uses unseen assets. The forecaster must be chosen out of sample before the confirmation data is opened, and the predictions must be fixed in writing.
- **Prediction (registered before running):** If H4 holds, the mixture forecasters beat pooled AR(1) by more than 5% in MSE on held-out assets, and they beat per-asset OLS, which is hurt by noisy T ≤ 40 estimates.
- **Procedure:** `.venv/bin/python research/tools/run_cv.py` (log `research/artifacts/cv_log.txt`, `research/artifacts/cv_explore.json`). This is 2-fold asset-level CV × 5 repeats: fit on half the assets, then forecast the other half at t=21..40 from history 1..t-1. A planted-truth sanity check runs on a simulated panel. After this, P1–P7 were written into *Predictions* before any access to `data/confirmation`.
- **Actual result:** [Obs] CV MSE (ratio versus pooled): zero 2.282e-4 (0.998), pooled 2.287e-4 (1.000), per-asset OLS 2.061e-4 (0.901), ungated mixture 1.927e-4 (0.842), gated mixture (age, size) 1.921e-4 (0.840), standardized-likelihood gated 1.922e-4 (0.841), gate only (characteristics, no history) 2.185e-4 (0.955). Planted check: mix_gated/pooled = 0.794, PASS. Chosen forecaster: `mix_gated`, with the type posterior from the age+size gate prior × the history likelihood under each AR(1) component, and forecast Σ_k p_k (c_k + phi_k r_T).
- **Verification status:** pending audit
- **Change in belief:** [Int] Typing from history carries most of the predictive gain (0.842). Characteristics alone give a smaller gain (0.955), and they add only marginally once 20+ periods of history are available (0.840). P1–P7 registered.
- **Artifacts:** `research/tools/forecasters.py`, `research/tools/run_cv.py`, `research/artifacts/cv_explore.json`, `research/artifacts/cv_log.txt`

### R5 — One-shot confirmation of P1–P7 on data/confirmation
- **Question:** Do the registered structure, handle and predictive predictions hold on 150 assets that played no part in exploration? (Q1, Q2)
- **Reasoning move:** deriving-predictions
- **Justification:** The exploratory findings came from a counted search of 33 tests. Under SCIENTIST §4 a type and handle claim stays provisional until it is confirmed on held-out data against predictions written down first.
- **Prediction (registered before running):** P1–P7 exactly as written in *Predictions* (registered in R4 before confirmation access). These include the Manager's belief-changing observation: "multiple processes" requires persistence, ΔBIC > 10, a lower confirmation MSE for the grouped model than for pooled after Holm, and a characteristic handle reproducing the predicted sign with p < 0.01.
- **Procedure:** `.venv/bin/python research/tools/run_confirm.py`, run once (log `research/artifacts/confirm_log.txt`, results `research/artifacts/confirm_results.json`). It uses only parameters frozen from explore (`research/artifacts/explore_fit.json`, plus the pooled and continuous models refit on explore inside the script). The refit on confirmation is limited to what P2 asks (K=1/2 and standardized K=2/3 BIC). Holm is applied across the 7 confirmation p-values.
- **Actual result:** [Obs]
  - **P1 PASS:** half-split phi r = 0.759, 95% CI (0.682, 0.820), p = 2.0e-29 (Holm 1.4e-28).
  - **P2 PASS:** ΔBIC(1→2) = 1021.2. Refit phis are −0.430 and +0.446, c = (0.00058, 0.00031), sig = (0.0147, 0.0141), momentum share 0.54. Standardized ΔBIC(2→3) = −12.0.
  - **P3 PASS:** logit of the history label on age gives a coefficient of +1.41, p = 1.9e-7 (Holm 9.7e-7). The coefficient is 1.37 controlling for size. The frozen gate has AUC 0.749 and the label share is 0.533.
  - **P4 FAIL:** size coefficient +0.23 (wrong sign), one-sided p = 0.86.
  - **P5 PASS:** the frozen mixture log-likelihood is 16397.9 versus 15902.9 for the continuous-loading model, a margin of +330 per 100 assets. Within-type corr(phi, age) = −0.214 (p 0.075) and +0.102 (p 0.37), both below 0.25 in absolute value (Holm 0.30 and 1).
  - **P6 PASS (null held):** sector ANOVA F = 0.29, p = 0.92.
  - **P7 PASS:** MSE zero 2.516e-4, pooled 2.509e-4, per-asset OLS 2.241e-4, mix_gated 2.090e-4. Ratios: 0.833 versus pooled, 0.831 versus zero, 0.933 versus OLS. Paired t = 8.34, one-sided p = 2.3e-14 (Holm 1.4e-13).
- **Verification status:** pending audit
- **Change in belief:** [Int] The two-type AR(1) structure (H4) and the age handle (H5-age) replicate on fresh assets with the predicted signs and magnitudes. The size handle does not replicate, so it is rejected; it was a marginal exploratory survivor (Holm 0.049), consistent with a false positive. Continuous loading (H3) and sector mean effects (H6) are disfavored. The type-0 within-type correlation of −0.21 is close to the 0.25 bound and is noted as an open point, not a failure. Q2 → multiple processes (2). The predictive claim is confirmed. The mechanistic claim covers only the statistical generating structure (discrete types versus continuous versus pooled, discriminated by P1, P2 and P5). The cause of the age association is not established.
- **Artifacts:** `research/tools/run_confirm.py`, `research/artifacts/confirm_results.json`, `research/artifacts/confirm_log.txt`

### R6 — Final predictor: refit on explore + confirmation (after R5), implementation and acceptance tests
- **Question:** Build the frozen predictor for fresh assets using the structure confirmed in R5.
- **Reasoning move:** reusable-instruments (with the acceptance-gate and timing-discipline protocols)
- **Justification:** The scoring is one-step MSE on unseen assets with h ≥ 40. R4 and R5 showed that the gated two-type mixture beats the alternatives out of sample. The refit uses all 450 assets for tighter parameters. It happens after R5 was recorded, so it cannot leak into the confirmation claims. Size is dropped because the registered P4 failed. That decision follows a pre-registered test and is not a new search.
- **Prediction (registered before running):** The refit parameters stay within the P2 bands (phi in [−0.55, −0.28] and [+0.27, +0.53]). The CLI runs in < 60 s on 1,000 assets, produces `asset_id, prediction`, and modifies no files. On data simulated from the fitted model, its MSE is within 1% of the oracle that knows each asset's true type. predict() with the frozen explore parameters reproduces the R5 MSE exactly.
- **Procedure:** `.venv/bin/python research/tools/fit_final.py` writes `research/predictor/params.json` (log `research/artifacts/fit_final_log.txt`). `.venv/bin/python research/tools/test_predictor.py` runs the tests (log `research/artifacts/test_predictor_log.txt`, test files in `research/artifacts/predictor_test/`). An edge-case CLI run covers missing age, an asset with no history, and a NaN return (`research/artifacts/predictor_test/edge_*.csv`).
- **Actual result:** [Obs]
  - Final parameters: phi = (−0.4181, +0.4185), c = (0.000498, 0.000443), sig = (0.01392, 0.01418). The gate logit P(momentum) = 0.223 + 1.419·age. BIC −99346.9, against −99345.3 with size added and −96483.5 for K=1.
  - Test 1: explore t=40 from t ≤ 39 gives 300 finite rows with columns `asset_id, prediction` in 1.05 s. MSE 1.65e-4 versus zero 1.91e-4; this is in-sample, because the params include explore.
  - Test 2: frozen explore params on confirmation reproduce the R5 MSE, 2.090258e-4 = 2.090258e-4.
  - Test 3: 1,000 synthetic unseen assets with h = 60 run in 1.21 s. MSE 1.904e-4 versus oracle 1.899e-4 (+0.3%) and zero 2.382e-4; corr(pred, oracle) = 0.992.
  - Test 4: no files under `research/` or `data/` changed, apart from the `tee` log being written by the test harness itself.
  - Edge cases: all produce finite predictions.
- **Verification status:** pending audit
- **Change in belief:** None about the science. The predictor is a faithful, fast implementation of the confirmed model. Its parameters are fitted on explore + confirmation, so no held-out estimate of the final refit's MSE exists. The R5 ratio of about 0.83 versus pooled is the best available out-of-sample estimate, made with the explore-only parameters.
- **Artifacts:** `research/tools/fit_final.py`, `research/tools/test_predictor.py`, `research/predictor/predict.py`, `research/predictor/params.json`, `research/conclusions.json`, `research/artifacts/test_predictor_log.txt`

## Rejected explanations

- **H1** rejected in R2/R5: returns are predictable from their own past. Per-asset phi dispersion is 2.6× the noise floor, half-split persistence r = 0.74 (explore) and 0.76 (confirmation), and the zero forecast loses to the mixture by 17% MSE on confirmation. Evidence: R2, R5. Revisit only if: fresh assets show mix/zero MSE ratio ≥ 0.98.
- **H2** rejected in R2/R5: K=2 beats K=1 by ΔBIC 1721 (explore) and 1021 (confirmation). The pooled lag-2 effect is explained as a mixture artefact. Evidence: R2, R3, R5. Revisit only if: on fresh assets the per-asset phi is unimodal or the half-split persistence is ≈ 0.
- **H3** rejected in R2/R5: phi is bimodal, within-type corr(phi, age) ≈ 0, and the discrete mixture beats continuous loading (ΔBIC 1193 explore; +330 log-lik per 100 assets on confirmation, frozen). Evidence: R2, R5. Revisit only if: a continuous model with nonlinear age loading matches the mixture's out-of-sample likelihood, or the within-type age correlation (−0.21 in confirmation type 0) becomes significant on more data.
- **H5-size** (size as a type handle) rejected in R5: the registered P4 failed (coefficient +0.23, wrong sign, one-sided p = 0.86). The explore support was marginal (Holm 0.049 of 33). Evidence: R2, R3, R5. Revisit only if: a larger fresh sample gives a size coefficient < 0 with p < 0.01.
- **H6** rejected (not established) in R2/R3/R5: no characteristic mean effect (all |r| < 0.07). The sector effect did not survive Holm in explore (0.057) and was null on confirmation (p 0.92). Per-asset mean dispersion is within the K=2 noise floor. Evidence: R2, R3, R5. Revisit only if: fresh data show sector or characteristic mean differences with p < 0.01, or per-asset mean persistence > 0.2.

## Unresolved questions

- Why does age predict type? The association is predictive only; no causal mechanism is claimed or testable in this synthetic panel.
- Is the type a deterministic function of age plus unobserved noise (a latent threshold), or a probabilistic draw? Both give the logistic gate seen here.
- Per-asset volatility varies persistently (half-split sd r = 0.56) and is unexplained by characteristics (F p = 0.10). Its generating law is unknown.
- In confirmation type 0, corr(phi, age) = −0.21 (p 0.075). Whether there is weak residual continuous loading inside a type remains undecided at this power.
- Bootstrap-resolution-limited p-values (dispersion, mixture LR) do not reach Holm significance with B ≤ 500 and 33 tests. Larger B would settle this at a cost of minutes.

## Next useful actions

- Score the frozen predictor on the fresh assets. The expected MSE ratio is about 0.83 versus pooled AR(1) (R5).
- With more assets: test a latent-threshold model (type = 1[a + b·age + u > 0]) against the logistic gate, and test nonlinear age effects.
- Model per-asset volatility (for example a lognormal random scale) and check whether it improves type posteriors for short histories (R4 showed no gain at h ≥ 20).
- Re-run the R3 bootstraps with B = 5,000 to remove the resolution limit on Holm-corrected p-values (a few minutes).

## Audit log

_Software-maintained from auditor reports; do not edit by hand._

| Run | Round | Status | Integrity | Contract audit | Report |
|---|---|---|---|---|---|
| demo-20261005T221146-b8f8-dataset-1 | 1 | complete | clean | aligned | `research/audits/demo-20261005T221146-b8f8-dataset-1/round_001.md` |
