# Manager plan — run demo-20261005T221146-b8f8-dataset-1, round 1 (route: cli)

_Copied by software from LongHorizon's round record._

Current task state:
- Completed: nothing yet. No auditor reports exist.
- Incomplete: all deliverables are still to do. That means ledger entries R1 and up, registered predictions P1 and up, a confirmation test, a status for Q1 and Q2, `research/predictor/predict.py`, and `research/conclusions.json`.
- Blockers/Risks: there is only one round, so exploration, registration, confirmation, the predictor and the conclusions must all happen in this single CLI round. The confirmation data stays locked until a `- **P<n>**` line exists in the ledger's Predictions section. Each series is short (t = 1..40), so per-asset estimates will be noisy. That raises the risk of regression to the mean and of searching too many subgroups.
- Untrusted/Do not reuse: none so far. Manager read-only observation (not audited): the ledger is an empty template, and `data/README.md` and `research/predictor/README.md` describe the schema and interface quoted below.

Task contract:
- Interpretation: answer Q1, which asks what process generates each asset's next return given its own past returns and characteristics. Answer Q2, which asks whether all assets share one process, there are several, or the evidence is insufficient. Deliver a frozen predictor and a conclusions file.
- Authoritative inputs:
  - `data/explore/assets.csv` with columns `asset_id, sector, size, liquidity, value, age`
  - `data/explore/returns.csv` with columns `asset_id, t, ret`, t = 1..40
  - `data/confirmation/*` in the same format, used only to test registered predictions
  - `SCIENTIST.md` and `research/predictor/README.md`
- Final-state carriers:
  - `research/ledger.md`: R entries with all §6 fields, P entries in Predictions, H entries in Competing hypotheses or Rejected explanations, and Q1 and Q2 statuses updated in Questions
  - `research/predictor/predict.py`, plus any fitted parameter files under `research/`
  - `research/conclusions.json` with keys `heterogeneity` (one of `single_process`, `multiple_processes`, `insufficient_evidence`), `n_processes` (integer or null), `description` (with parameter estimates), `characteristics_associated` (a list), and `evidence_entries` (ledger ids)
  - reusable code in `research/tools/`
- Acceptance constraints:
  1. Every substantial investigation has a full §6 entry with status `pending audit`.
  2. Each prediction is registered before the confirmation data is opened.
  3. Confirmation data is used only to test predictions. Nothing may be refit or reselected on it in a way that leaks into the claims.
  4. Predictive claims and mechanistic claims are kept separate.
  5. Every split or covariate tried is counted and corrected for.
  6. `predict.py` runs as `.venv/bin/python research/predictor/predict.py --assets A --history H --out O` from the workspace root. It handles h ≥ 40 and unseen assets, finishes in under 60 s for 1,000 assets, writes `asset_id, prediction`, and modifies no files.
  7. Work stays inside the workspace and never touches `.lh-harness`.
  8. The ledger's Audit log and `research/audits/` are never edited by hand.
- Unacceptable shortcuts:
  - inventing results
  - inflating a weak split into "multiple processes"
  - fitting the predictor on confirmation data before the confirmation test is recorded
  - declaring a mechanism from predictive fit alone
  - a placeholder predictor
- Evidence: the auditor's independent recomputation of key numbers and inspection of the files.

Dependency assessment:
- Target state: all four completion criteria above, satisfied in the workspace files.
- State creator: CLI.
- Satisfied prerequisites: none audited.
- Unsatisfied prerequisites: the exploratory analysis, prediction registration, the confirmation test, the predictor and the conclusions.
- Routing rationale: this is the last round, so it cannot be spent on a prerequisite alone. I am routing one coherent CLI investigation that delivers everything. Its single dominant state change is "the research record and its artifacts are produced", and the steps run in a strict order inside it.

Next: cli

Task:
Scientific decision record:
- Reading of last result: none yet (this is round 1, and the ledger is empty).
- Blocker: we do not know whether returns depend on their own past returns (autoregressive or momentum/reversal terms) or on characteristics. We also do not know whether that dependence is homogeneous across assets or a mixture of types. The pooled estimate could hide opposite-signed groups.
- Options considered:
  1. hidden-states-and-trajectories: estimate a pooled baseline (mean, AR(1), characteristic interactions), then per-asset dynamics compared against a simulated noise floor, a persistence test across half-samples, a mixture-versus-single comparison by BIC, and a search for a characteristic or sector handle. This also yields a predictor. Expected information is high, because it answers Q1 and Q2 together. Cost is moderate.
  2. revealing-comparisons: compare sectors and characteristic bins on mean return, holding exposure fixed. This is cheaper. It speaks to cross-sectional mean differences but misses time-series dynamics that differ by type.
  3. deriving-predictions alone: fit a pooled model and register out-of-sample MSE predictions. This gives a fast predictor but cannot settle Q2.
- Selected move: hidden-states-and-trajectories
- Why this move: Q2 is explicitly a mixture question. A near-zero pooled autocorrelation could hide groups with momentum and groups with reversal. Only per-unit analysis plus a persistence test can tell that apart from noise. It also produces the predictor directly. deriving-predictions will be used for the confirmation step, with the planted-truth and acceptance-gate protocols applied.
- Belief-changing observation: "multiple processes" is favored if all of the following hold:
  - the dispersion of per-asset AR(1) coefficients (or the chosen statistic) exceeds the simulated homogeneous noise floor at p < 0.01;
  - the first-half and second-half estimates are positively correlated, with a 95% CI excluding 0;
  - a mixture with 2 or more components beats a single component by ΔBIC > 10;
  - on confirmation data, after Holm correction across the full count of splits tried, the registered group-specific model lowers one-step MSE relative to the pooled model by a pre-stated margin, and any characteristic handle (for example sector or a characteristic sign) reproduces the predicted group coefficients with the same signs and CIs excluding 0.

  "Single process" is favored if dispersion stays within the noise floor, persistence is about 0, and the pooled model matches or beats the grouped model on confirmation data. If the signals conflict, or the test lacks power, the answer is "insufficient evidence".
- Executor assignment: carry out steps (a) to (h) in order.
  - (a) Read the data, record schema observations and assumptions (synthetic, no survivorship, t = 1..40), and register competing hypotheses H1, H2, … covering: i.i.d. mean only; pooled AR(p); a mean or AR loading on characteristics; a mixture of latent types, with or without a characteristic handle.
  - (b) Planted-truth check. Simulate a panel of the same shape with a known 2-type AR structure and confirm the pipeline recovers it. Also simulate a homogeneous panel and confirm the pipeline finds no split.
  - (c) Exploration on `data/explore` only:
    - pooled regressions of ret_{t+1} on ret_t (plus lags 2–5 and rolling means if warranted) and on characteristics and their interactions with ret_t;
    - per-asset statistics against the noise floor;
    - the half-split persistence test;
    - a Gaussian mixture or EM over per-asset regressions, compared by BIC;
    - the covariate handle search.

    Keep a running count of every test and split, and save the code as documented tools in `research/tools/`.
  - (d) Before touching `data/confirmation`, write predictions P1..Pn in the ledger Predictions section with numeric thresholds. These cover the group structure and handle, the coefficient signs and magnitudes, and the MSE of the chosen model against the pooled and zero baselines.
  - (e) Run the confirmation test once and record the results honestly, including any failures.
  - (f) Fit the final predictor. Fitting on explore plus confirmation is allowed only after the confirmation results are recorded, and the ledger must say so. Implement `research/predictor/predict.py` to the README interface. For unseen assets, assign type from characteristics if a confirmed handle exists; otherwise use shrunken per-asset or pooled estimates from the history. Test it on a held-out split of t ≤ 40, and time it on 1,000 synthetic assets.
  - (g) Write `research/conclusions.json` with the exact keys, plus ledger entries R1..Rn with all §6 fields and status `pending audit`. Mark Q1 and Q2 as supported, rejected, insufficient evidence, or open, with entry ids. Move rejected hypotheses to Rejected explanations with `Revisit only if:` clauses. Fill in Unresolved questions and Next useful actions.
  - (h) Report the exact commands, paths and key numbers, including the test count and the corrected p-values.
- Budget and stopping rule: this is a single round, so keep the total compute to minutes and time a small run first. Stop the exploratory search after a bounded set of at most about 30 counted tests. If no split survives correction plus persistence, stop searching and conclude single process or insufficient evidence. Do not search further after seeing the confirmation data.

Acceptance criteria:
- The ledger contains R entries with all fields, P entries registered before confirmation access, H entries in their correct sections, and statuses for Q1 and Q2.
- `research/predictor/predict.py` runs on a sample (for example, explore assets with history t ≤ 39, or a synthetic set of 1,000 assets) and produces a valid `asset_id, prediction` CSV within the time limit, without modifying files.
- `research/conclusions.json` is valid JSON with exactly the required keys and allowed values.
- Tools in `research/tools/` have docstrings.
- The report lists the exact commands and numbers so the auditor can recompute them.

Related audit reports: none. This is round 1.

Related audited state: none. The ledger is an unaudited empty template (read by the manager).

Boundaries:
- Work only inside the workspace, and never access `.lh-harness`.
- Use confirmation data only after a P entry is registered, and only for confirming.
- Do not hand-edit the Audit log or `research/audits/`.
- Do not claim a mechanism beyond what the evidence discriminates.
- Do not ask the human anything. Record open decisions as explicit assumptions in the ledger.