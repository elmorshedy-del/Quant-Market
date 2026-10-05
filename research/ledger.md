# Research Ledger — Quant-Market

Durable scientific state for this project. Format and rules: `SCIENTIST.md` §6.
Read this before starting work, and append to it rather than rewriting it. Rejected
hypotheses are never deleted. The *Audit log* is written by software.

Statement labels: **[Obs]** observation · **[Asm]** assumption · **[Int]** interpretation · **[Pred]** prediction.

## Questions

- **Q1** [open] Do the tournament's top-ranked strategies keep their rank on data not
  used for ranking, or does the ranking mostly reflect selection across many strategies?
  (Proposed from `README.md` and `docs/implementation-status.md`; not yet investigated.)

## Observations and sources

- **O1** [Obs, documentation claim, unverified here] The app ranks strategies by P&L and
  a repeatability score, and reports White's Reality Check p-value and a PBO estimate.
  Source: `README.md`, "What it does".
- **O2** [Obs, documentation claim, unverified here] Execution is lagged: signal at
  *t*, fill at *t+1*. Source: `README.md`.

## Measurement and selection assumptions

- **A1** [Asm] Ticker universes come from current lists, so delisted names are missing:
  survivorship bias. Source: `README.md` ("Better bars do not automatically remove
  survivorship bias"). Affects any cross-sectional or strategy-ranking claim.
- **A2** [Asm] yfinance bars are free and may be revised or lagged. Polygon bars are
  adjusted when `POLYGON_ADJUSTED_BARS=true`. Results may depend on the provider.
  Source: `README.md`, `.env.example`.

## Competing hypotheses

_None registered yet. Use `- **H<n>** [active] statement — what it predicts — entry ids`._

## Predictions

_None registered yet. Use `- **P<n>** (from H<n>, registered in R<id>, <date>) statement — threshold — data`._

## Experiment results

_No investigations yet. Each substantial investigation gets an entry in this format:_

<!--
### R<id> — <short title>
- **Question:** 
- **Reasoning move:** 
- **Justification:** 
- **Prediction (registered before running):** 
- **Procedure:** 
- **Actual result:** 
- **Verification status:** pending audit
- **Change in belief:** 
- **Artifacts:** 
-->

## Rejected explanations

_None yet. Use `- **H<n>** rejected in R<id>: reason. Evidence: … Revisit only if: …`._

## Unresolved questions

_None yet._

## Next useful actions

- Run a 3-round investigation of Q1 with `scientist/bin/scientist run --question @research/questions/q1.md`
  (requires network access for market data).

## Audit log

_Software-maintained from auditor reports; do not edit by hand._
