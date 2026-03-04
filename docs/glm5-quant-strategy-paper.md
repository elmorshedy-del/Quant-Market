# Quant Strategy Tournament: Math, Implementation, and Ranking

**Executive Summary**
This document outlines a rigorous framework for a "Strategy Tournament" to identify robust alpha sources in equities. We define 16 implementation-ready strategies spanning momentum, reversion, fundamental factors, and machine learning. A unified backtest framework is established to ensure fair comparison, incorporating transaction costs, survivorship bias, and statistical significance testing. The final ranking relies on a "Likelihood to Repeat" score, prioritizing strategies with high statistical confidence and economic rationale over those with merely high backtested returns.

---

## 1. Strategy Definitions: Math and Implementation

### Family 1: Time-Series Momentum (Trend Following)

**Strategy 1: Time-Series Momentum (TSM)**
*   **Intuition:** Assets with positive (negative) past returns will continue to perform well (poorly) in the near future. Exploits investor herding and slow information diffusion.
*   **Math:**
    Let $r_{t-k:t}$ be the cumulative return from $t-k$ to $t$.
    Signal $S_{i,t} = \text{sign}(r_{t-k:t})$.
    Volatility Targeting Weight: $w_{i,t} = S_{i,t} \times \frac{\sigma_{target}}{\sigma_{i,t}^{EWM}}$.
    Where $\sigma_{i,t}^{EWM}$ is the exponentially weighted volatility ($\lambda=0.94$).
*   **Signal Construction:** Lookback $k \in \{12, 24, 36\}$ months. Use overlapping holding periods (e.g., 1-month hold rebalanced daily/weekly).
*   **Position Sizing:** Volatility targeting ($\sigma_{target} = 15\%$ annualized) to normalize risk across assets.
*   **Turnover/TC:** Low turnover (annualized ~50-100%). TC model: $Cost = \frac{1}{2} Spread + Impact$.
*   **Failure Modes:** "Fire sale" risk during market reversals. Whipsaw in sideways markets. Diagnostics: Check rolling Sharpe vs. VIX levels.

### Family 2: Cross-Sectional Momentum

**Strategy 2: Cross-Sectional Momentum (CSM)**
*   **Intuition:** Winners relative to the market continue to win; losers relative to the market continue to lose.
*   **Math:**
    Rank assets $i=1 \dots N$ based on past return $r_{t-k:t}$.
    $S_{i,t} = \text{Rank}(r_{i, t-k:t})$.
    Weight $w_{i,t} = S_{i,t} - \bar{S}$ (Zero-cost long/short portfolio).
*   **Signal Construction:** Lookback 12 months, skip most recent 1 month (to avoid reversal microstructure noise). Hold 1 month.
*   **Position Sizing:** Decile portfolios. Top decile long, bottom decile short. Dollar-neutral.
*   **Failure Modes:** Momentum crashes (e.g., 2009). High tail risk. Mitigate with volatility scaling or dynamic position limits.

### Family 3: Mean Reversion

**Strategy 3: Single-Asset Mean Reversion (MR-S)**
*   **Intuition:** Prices fluctuate around a fair value; extreme deviations correct.
*   **Math:**
    $Z_{i,t} = \frac{P_{i,t} - \mu_{i,t}(n)}{\sigma_{i,t}(n)}$.
    Signal $S_{i,t} = -Z_{i,t}$ (Short if price > mean).
    Where $\mu$ is moving average, $\sigma$ is rolling std dev.
*   **Signal Construction:** Lookback $n=20$ days (short-term reversion).
*   **Position Sizing:** Inverse volatility.
*   **Failure Modes:** Catching a falling knife (trending markets). Must filter with trend filter (e.g., only revert if long-term trend is flat).

**Strategy 4: Cross-Sectional Mean Reversion (MR-CS) / Industry Neutral**
*   **Intuition:** Stocks that outperformed their sector peers yesterday will underperform tomorrow (short-term reversal).
*   **Math:**
    $r_{i,t}^{resid} = r_{i,t} - r_{sector,t}$.
    Signal $S_{i,t+1} = -r_{i,t}^{resid}$.
*   **Position Sizing:** Equal weight within sector buckets.
*   **Turnover:** Very high (daily rebalance). Requires low slippage.

### Family 4: Statistical Arbitrage

**Strategy 5: Pairs Trading (Cointegration)**
*   **Intuition:** Two assets historically move together; divergence presents an arbitrage opportunity.
*   **Math:**
    Engle-Granger Test: Regress $P_A$ on $P_B$. $Spread_t = P_{A,t} - \beta P_{B,t}$.
    If $Spread_t$ is stationary (ADF test p-value < 0.05), trade.
    Signal $S_t = -Z(Spread_t)$.
*   **Signal Construction:** Formation period 1 year. Trade if $|Z| > 2$. Exit if $|Z| < 0.5$.
*   **Failure Modes:** Structural break (relationship decouples). Stop-loss if spread exceeds $4\sigma$.

**Strategy 6: PCA Residual Reversion**
*   **Intuition:** Isolate the idiosyncratic component of returns after removing market/sector factors.
*   **Math:**
    Decompose returns $R = F \Lambda + \epsilon$.
    Trade $\epsilon$ reversion: $S_{i,t} = -\epsilon_{i,t-1}$.
*   **Implementation:** Use top 5 PCA components as factors.

### Family 5: Volatility / Risk Overlays

**Strategy 7: Volatility Targeting (VT)**
*   **Intuition:** Constant volatility exposure improves risk-adjusted returns by de-leveraging in turbulent times.
*   **Math:**
    Leverage $L_t = \frac{\sigma_{target}}{\sigma_{t}^{forecast}}$.
    Forecast $\sigma_{t}^{forecast} = GARCH(1,1)$ or EWMA.
*   **Position Sizing:** Apply $L_t$ to a passive benchmark (e.g., SPY).

**Strategy 8: Risk Parity (RP)**
*   **Intuition:** Equalize risk contribution from all assets.
*   **Math:**
    Find weights $w$ such that $RC_i = w_i \frac{\partial \sigma_p}{\partial w_i} = RC_j$ for all $i,j$.
    Solved via optimization: $\min_w \sum (\sigma_i w_i - \sigma_j w_j)^2$.

### Family 6: Fundamental Factors

**Strategy 9: Value + Quality (Composite)**
*   **Intuition:** Buy cheap companies with solid financials.
*   **Math:**
    Value Score $V = \text{Rank}(\frac{B}{P}) + \text{Rank}(\frac{CF}{P})$.
    Quality Score $Q = \text{Rank}(ROE) + \text{Rank}(Accruals) + \text{Rank}(Leverage)$.
    Signal $S = V + Q$.
*   **Signal Construction:** Update quarterly using point-in-time data (avoid look-ahead bias).
*   **Failure Modes:** Value traps. Quality usually protects against this.

**Strategy 10: Profitability (Novy-Marx)**
*   **Intuition:** Gross profitability predicts returns.
*   **Math:**
    $GP = \frac{Revenue - COGS}{Assets}$.
    Signal $S = \text{Rank}(GP)$.

### Family 7: Intraday Microstructure

**Strategy 11: Order Flow Imbalance (OFI)**
*   **Intuition:** Aggressive buying pressure predicts short-term price moves.
*   **Math:**
    $OFI_t = \Delta V_{bid} \mathbb{1}_{P_t=P_{bid}} - \Delta V_{ask} \mathbb{1}_{P_t=P_{ask}}$.
    Signal $S = \beta \times OFI_t$.
*   **Data:** Requires Level 2 order book data.
*   **Capacity:** Very limited capacity. High turnover. Slippage is the primary cost.

### Family 8: Regime Switching

**Strategy 12: Hidden Markov Model (HMM) Regime Filter**
*   **Intuition:** Market has "Bull" and "Bear" states with different parameters.
*   **Math:**
    Model returns $r_t \sim \mathcal{N}(\mu_{S_t}, \sigma^2_{S_t})$ where $S_t \in \{1, 2\}$.
    Use Baum-Welch algorithm to estimate parameters. Viterbi algorithm to decode current state.
*   **Application:** Turn off trend-following strategies if $P(Bear) > 0.7$.

### Family 9: Machine Learning

**Strategy 13: Linear Model (Elastic Net)**
*   **Intuition:** Linear combination of features with regularization to prevent overfitting.
*   **Math:**
    $\min_{\beta} ||y - X\beta||^2 + \lambda_1 ||\beta||_1 + \lambda_2 ||\beta||_2^2$.
    Target $y$: Next day return. Features: Momentum, Reversion, Volume changes.

**Strategy 14: Tree-Based (Gradient Boosting - XGBoost)**
*   **Intuition:** Capture non-linear interactions between features.
*   **Math:**
    $\hat{y} = \sum_{k=1}^K f_k(x)$, where $f_k$ are regression trees.
    Loss function: $L(\phi) = \sum l(y_i, \hat{y}_i) + \sum \Omega(f_k)$.
*   **Implementation:** Purged K-Fold CV is mandatory. Feature importance analysis required.

**Strategy 15: Sequence Model (LSTM)**
*   **Intuition:** Learn temporal dependencies in sequences of returns/features.
*   **Math:**
    $h_t = \tanh(W_h [h_{t-1}, x_t] + b_h)$ (Simplified).
    Output $y_t = W_y h_t$.
*   **Failure Modes:** Extreme overfitting. Requires massive regularization (Dropout, Early Stopping).

### Family 10: Portfolio Optimization

**Strategy 16: Hierarchical Risk Parity (HRP)**
*   **Intuition:** Improve stability of risk parity by clustering assets.
*   **Math:**
    1. Calculate distance matrix $D$ from correlation matrix.
    2. Hierarchical Clustering (Ward's method).
    3. Recursive bisection to assign weights based on cluster variance.
*   **Advantage:** Inversion-free (robust to ill-conditioned covariance matrices).

---

## 2. Backtest Tournament Framework

### Data Schema & Universe
*   **Universe:** S&P 500 or Russell 1000 (point-in-time).
*   **Survivorship Bias Control:** Must include delisted stocks. Use "Point-in-Time" (PIT) data files.
*   **Corporate Actions:** Use Total Return series (Dividends reinvested). Adjust for splits.
*   **Data Structure:**
    *   `Date`, `AssetID`, `Open`, `High`, `Low`, `Close`, `Volume`, `VWAP`.
    *   Fundamental Data: `Date`, `AssetID`, `BookValue`, `EPS`, `Revenue`, `ReportDate` (PIT).

### Train/Validation/Test Splits
*   **Method:** Walk-Forward Optimization (anchored or rolling).
    *   Train: Expanding window (e.g., 2000-2010).
    *   Validation: 1 year (e.g., 2011).
    *   Test: Out-of-sample (OOS) period (e.g., 2012-2023).
*   **Purged Cross-Validation:** For ML models, purge observations immediately following training data to prevent leakage (embargo period).

### Execution Model
*   **Convention:** Next Open execution.
    *   Signal generated at $t$ (Close). Trade executed at $t+1$ (Open).
*   **Slippage:**
    *   Assumption: Trade size $< 5\%$ daily volume.
    *   Slippage Model: $Slippage = 10 \text{ bps}$ (conservative for large caps).
*   **Transaction Costs:**
    *   Commission: $0.001$ per share.
    *   Borrow Cost: For shorts, apply `FeeRate` (e.g., 50 bps annualized) on short position value.

### Metrics
1.  **P&L:** Cumulative returns series.
2.  **Sharpe Ratio:** $SR = \frac{E[R_p - R_f]}{\sigma_p}$ (Annualized).
3
3.  **Sortino Ratio:** $Sortino = \frac{E[R_p - R_f]}{\sigma_{downside}}$, where $\sigma_{downside}$ is the standard deviation of negative returns only.
4.  **Max Drawdown (MDD):** $MDD = \min_{t} \left( \frac{P_t - \max_{k \le t} P_k}{\max_{k \le t} P_k} \right)$.
5.  **Calmar Ratio:** $CAGR / |MDD|$.
6.  **Turnover:** $\frac{1}{T} \sum_{t} \sum_{i} |w_{i,t} - w_{i,t-1}|$ (annualized).
7.  **Hit Rate:** Percentage of periods with positive returns.
8.  **Tail Risk:** 5% CVaR (Conditional Value at Risk) of daily returns.

### Statistical Significance Tests
To avoid data snooping, we apply:
1.  **Bootstrap:** Resample the strategy returns series 1,000 times to construct a confidence interval around the Sharpe Ratio.
2.  **Deflated Sharpe Ratio (DSR):** Adjusts the Sharpe for non-normality (skew/kurtosis) and sample length. $DSR \approx \hat{SR} \sqrt{1 - \frac{Skew^2}{n}}$.
3.  **White's Reality Check (WRC):** A bootstrap method to test if the best strategy from a set of $N$ strategies genuinely outperforms a benchmark, accounting for the multiple testing penalty.

### Ranking Methodology
Strategies are ranked by a composite score:
$$ Score = w_1 \cdot \text{Net Sharpe} + w_2 \cdot \text{DSR Confidence} - w_3 \cdot \text{Turnover} $$
Where Net Sharpe deducts transaction costs.

---

## 3. Likelihood to Repeat: Bayesian Repeatability Score

This section estimates the probability that a strategy's backtested performance is a false positive.

### Bayesian Framework
We treat the True Sharpe Ratio ($SR_{true}$) as a random variable.
Prior: Assume a skeptical prior $SR_{true} \sim \mathcal{N}(0, 0.5)$ (reflecting market efficiency).
Likelihood: Observed Sharpe $\hat{SR} \sim \mathcal{N}(SR_{true}, \frac{1}{\sqrt{T}})$.

Posterior Estimate:
$$ P(SR_{true} > 0 | \text{Data}) = \text{Probability of Skill} $$

### Repeatability Score Calculation
We define the **Repeatability Score ($R$)** as:
$$ R_i = P(SR_{true} > 0) \times \text{Regime Stability} \times \text{Implementation Quality} $$

*   **Regime Stability:** Measured by the standard deviation of rolling 12-month Sharpe ratios. Lower variance = higher stability.
*   **Implementation Quality:** A heuristic score based on turnover (lower is better) and capacity (higher is better).

### Final Ranking Table (Template)

| Rank | Strategy | Exp. Net Sharpe | Prob(Skill) | Turnover | Complexity | Repeatability Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Value + Quality | 0.85 | 0.92 | Low | Med | **0.88** |
| 2 | Risk Parity (HRP) | 0.65 | 0.98 | Low | High | **0.85** |
| 3 | TSM (Trend) | 0.70 | 0.75 | Med | Low | **0.72** |
| 4 | XGBoost (ML) | 1.20 | 0.55 | High | V.High | **0.45** |
| ... | ... | ... | ... | ... | ... | ... |

*Note: ML strategies often show high backtested Sharpe but lower Repeatability Scores due to overfitting risk.*

---

## 4. Can an LLM be used as a helper?

**Answer: Yes, but strictly in an advisory and generative capacity.**

### Safe/Useful Roles
1.  **Feature Ideation:** "Suggest 5 novel features based on volume and volatility interactions." (LLMs excel at pattern matching literature).
2.  **Code Generation:** Writing boilerplate pandas/numpy code for standard indicators (e.g., "Write a function to calculate RSI").
3.  **Anomaly Triage:** Summarizing logs. "Scan this error log and identify the top 3 data integrity issues."
4.  **Report Generation:** Drafting the monthly strategy performance commentary based on provided metrics.

### Unsafe Roles
1.  **Direct Signal Generation:** "Should I buy AAPL today?" (LLMs hallucinate real-time data and lack mathematical precision).
2.  **Auto-Trading:** Executing orders based on LLM output without deterministic guardrails.
3.  **Math Verification:** LLMs are notoriously bad at complex arithmetic. Always verify equations externally.

### Recommended Architecture
**The "LLM-in-the-Loop" Design:**
1.  **Researcher** prompts LLM for code/ideas.
2.  **LLM** generates Python script.
3.  **Deterministic Engine** (Human-verified code) runs the backtest.
4.  **LLM** analyzes the *output text/logs* to summarize results.
5.  **Human** makes the final deployment decision.

---

## 5. Programming Roadmap: Phased Build Plan

A step-by-step guide to building the tournament system.

### Phase 1: Infrastructure & Data (Weeks 1-2)
*   **Goal:** Clean, survivorship-bias-free data pipeline.
*   **Tasks:**
    *   Ingest pricing data (OHLCV) and point-in-time fundamentals.
    *   Build a universal data handler class that adjusts for splits/dividends automatically.
    *   Implement the `TransactionCostModel` class (slippage + fees).
*   **Fastest Signal-to-Noise:** Verify data integrity first. Check for missing delistings.

### Phase 2: The "Dumb" Baselines (Weeks 3-4)
*   **Goal:** Establish the hurdle rate.
*   **Tasks:**
    *   Implement Strategy 1 (TSM) and Strategy 9 (Value).
    *   Run backtests on the "Test" period only.
    *   If these don't work, the data or cost model is likely broken.
*   **Output:** Baseline Sharpe ratios.

### Phase 3: Advanced Alpha & Optimization (Weeks 5-8)
*   **Goal:** Run the tournament contenders.
*   **Tasks:**
    *   Implement ML strategies (XGBoost, Elastic Net) with Purged CV.
    *   Implement Optimization overlays (HRP, Risk Parity).
    *   Run the "Tournament Loop": Iterate through all 16 strategies with identical cost assumptions.

### Phase 4: Significance Testing & Ranking (Week 9)
*   **Goal:** Filter false positives.
*   **Tasks:**
    *   Code the Bootstrap and Deflated Sharpe Ratio functions.
    *   Calculate the "Likelihood to Repeat" score for all strategies.
    *   Generate the final ranking table.

### Phase 5: A/B Testing & Paper Trading (Ongoing)
*   **Goal:** Real-time validation.
*   **Tasks:**
    *   Deploy the top 3 strategies to a paper trading environment.
    *   Monitor live Sharpe vs. Expected Sharpe.
    *   Trigger a "kill switch" if live performance deviates > 2 sigma from backtest.

---

## 6. Backtests To Be Done (Explicit Checklist)

Run these tests for every strategy before ranking final winners:

1. **Baseline gross backtest (no costs):** Verify raw signal edge and implementation correctness.
2. **Net backtest with realistic costs:** Include commissions, spread, slippage, borrow fees, and financing.
3. **Execution-lag variants:** Compare (signal at close, fill next open) vs (fill next close) to quantify execution sensitivity.
4. **Walk-forward stability test:** Expanding and rolling windows; log performance drift between train/validation/test slices.
5. **Parameter robustness grid:** Sweep core hyperparameters (lookbacks, thresholds, holding periods) and map performance surfaces.
6. **Subperiod/regime test:** Evaluate separately in bull, bear, high-volatility, and low-volatility periods.
7. **Universe perturbation test:** Re-run on different universes (e.g., S&P 500, Russell 1000, sector subsets) to test transferability.
8. **Cost-stress test:** Increase transaction costs/slippage by +25%, +50%, +100% and measure Sharpe/P&L degradation.
9. **Capacity curve test:** Scale notional and participation rate to estimate market-impact breakpoints.
10. **Bootstrap + White Reality Check + Deflated Sharpe:** Control for multiple-testing/data-snooping bias.
11. **Ablation test (for composite/ML strategies):** Remove feature groups one-by-one to identify true signal contributors.
12. **Paper-trading shadow test:** Run top candidates live in paper mode for at least 8-12 weeks before production capital.
