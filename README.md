# GB Day-Ahead Electricity Price Forecasting for Battery Trading

## Background: What Does a Battery Trading Algorithm Need?

Grid-scale batteries participate in the GB day-ahead wholesale electricity market by buying cheap and selling expensive across the 48 half-hour settlement periods of a day. The value of a battery's strategy depends critically on the **shape and spread of the daily price profile** — not just individual half-hour prices. A few key questions the trading algorithm must answer every morning:

- At which settlement periods should the battery charge, and at which should it discharge?
- What is the expected profit from a given charging schedule, and how uncertain is it?
- What is the risk of extreme outcomes (negative spreads, forced curtailment)?

A point forecast of each half-hour price answers the first question poorly and cannot address the second or third at all. To make good trading decisions, the algorithm needs:

1. **Per-period uncertainty estimates** — how wide is the price distribution at each settlement period?
2. **Joint scenarios** — if the morning is cheap, how does that constrain the afternoon? Battery dispatch decisions depend on the shape of the *whole day*, not each period independently.

These requirements motivated the two-component probabilistic forecasting system described here.

---

## System Design

The forecasting system is deliberately split into two components with complementary responsibilities:

- **LightGBM quantile regression** handles the *marginal* price distribution at each settlement period independently.
- **A t-copula** handles the *dependence structure* across settlement periods, converting marginal distributions into coherent full-day scenarios.

This separation keeps the modelling tractable: a single LightGBM model trained on all periods learns shared price dynamics efficiently, while the copula focuses purely on cross-period correlation without having to model price levels.

---

## Exploratory Data Analysis

Before building any models, EDA was conducted to understand the data generating process and feature relationships. See [`notebooks/prob_distribution_eda.ipynb`](notebooks/prob_distribution_eda.ipynb) for full details.

Key observations:
- GB day-ahead prices are highly non-Gaussian: right-skewed with occasional negative prices and extreme spikes.
- Strong intra-day seasonality — distinct off-peak, morning ramp, midday, and evening peak regimes — driven by settlement period.
- Clear annual seasonality via month and weekday effects.
- Lagged prices (lag-1d and lag-7d at the same settlement period) carry strong predictive signal, reflecting day-over-day and week-over-week mean reversion.
- Net load (demand minus wind and embedded renewable generation) is the primary fundamental driver of price level.
- Scarcity features (De-rated Margin and Loss of Load Probability) are important for capturing tail risk and spike events.

---

## Feature Engineering

Features were shortlisted primarily based on their marginal correlation with the target price and their availability strictly before the day-ahead decision window (no leakage). Day-ahead bids for day D are submitted before 10:00 on D-1; at that point only D-2 has a complete full-day record of physical actuals such as DRM and interconnector flows.

**Calendar features:**
- `settlement_period` — within-day position (1–48); captures intra-day shape.
- `month` — seasonal price level and shape changes.
- `weekday` / `is_weekend` — demand and price level differ markedly on weekends.

**Lagged price features:**
- `price_lag1d` — same-period price from yesterday; strong autoregressive signal.
- `price_lag7d` — same-period price from a week ago; captures weekly seasonality.
- `price_roll7d_mean` — rolling 7-day mean price; smoothed level indicator.

**Fundamental features:**
- `net_load` — demand minus wind forecast minus embedded wind forecast minus embedded solar forecast. The single most important driver of price level.
- `drm_eve_mean`, `drm_min` — De-rated Margin statistics from D-2 (last complete day of actuals at bid time); low DRM indicates tightening supply.
- `lolp_max` — maximum Loss of Load Probability from D-2; captures scarcity/tail spike risk.

---

## Regime Analysis (Discarded)

Clustering was applied to normalised daily price shapes and summary statistics (price level, net load, spread) to explore whether distinct market regimes exist that could be used as a conditioning feature in the LightGBM models.

The resulting clusters did not separate into clearly distinct day types. The main source of variation in daily price shapes is continuous seasonal change rather than discrete regime switches. Adding a cluster label as a feature to LightGBM was therefore not justified and was dropped from the modelling scope.

A possible future direction would be to redesign regimes around price level, spread, and scarcity/tail behaviour rather than shape.

---

## LightGBM Marginal Quantile Model

See [`notebooks/lightgbm_quantile.ipynb`](notebooks/lightgbm_quantile.ipynb) for full methodology and evaluation.

### Methodology

One **global** LightGBM quantile regression model is trained per quantile level (Q01, Q05, Q10, Q25, Q50, Q75, Q90, Q95, Q99). "Global" means all 48 settlement periods share a single model, with `settlement_period` as a categorical feature — this avoids 48 separately trained models and lets the model share information across periods.

Training follows an **expanding-window rolling-origin backtest** over 24 monthly folds (February 2024 – February 2026), with a 12-month burn-in period, 1-month validation window (for early stopping), and 1-month test window. The naive benchmark is a shifted lag-1d distribution: the lag-1d price plus the empirical training residual quantile.

Minor quantile crossings (concentrated at the Q01/Q05 and Q95/Q99 pairs) are corrected by row-wise sorting before saving predictions.

### Results

**Point forecast (P50 vs lag-1d naive):**

| Metric | LightGBM P50 | Naive lag-1d |
|--------|-------------|--------------|
| MAE    | 15.42 £/MWh | 20.59 £/MWh  |
| RMSE   | 27.06 £/MWh | 38.43 £/MWh  |

The model beats the naive benchmark consistently across most months and settlement periods. The exception is February/March, where low day-over-day volatility makes the raw lag-1d hard to beat.

**Probabilistic calibration:**
The 90% and 98% prediction intervals achieve close to nominal coverage while being considerably narrower than the naive residual baseline — roughly half the width at the 90%/98% level. The 98% interval (Q01–Q99) is particularly important as it directly anchors the inverse CDF tail used when generating copula scenarios.

The reliability diagram shows an upward level bias concentrated in the Q25–Q90 range (empirical coverage exceeds nominal by up to +4%). Tail quantiles (Q01, Q95, Q99) are well-calibrated with near-zero bias. This is a level bias, not a width problem.

**Shape diagnostics:**
The P50 correctly identifies the peak-3 settlement periods on ~46% of days and the cheapest-3 on ~28% of days. The daily spread (high minus low) is systematically compressed in point forecasts: the model under-predicts daily highs and over-predicts daily lows on average (mean spread error −19.9 £/MWh). This is expected for a regression-to-the-mean estimator and is precisely why probabilistic scenarios (via the copula) are the right tool for spread-sensitive trading decisions rather than the P50 alone.

---

## t-Copula Dependence Model

See [`notebooks/copula_fitting.ipynb`](notebooks/copula_fitting.ipynb) for full methodology and evaluation.

### Methodology

After obtaining marginal quantile forecasts from LightGBM, the copula converts them into coherent daily price scenarios. The approach has three steps:

1. **PIT transformation** — actual prices are transformed to uniform probability integral transform (PIT) values `U ∈ (0,1)` using the LightGBM quantile forecasts as the marginal CDFs. This decouples the dependence structure from the marginal distributions.

2. **Copula fitting** — three copula models are compared on rolling out-of-sample folds:
   - *Independence*: each period is drawn independently.
   - *Gaussian copula*: Pearson correlation of Gaussian latents; captures linear dependence.
   - *t-copula*: Spearman rank correlation converted to Pearson, with a single degree-of-freedom parameter ν estimated by grid search + scalar optimisation on a held-out validation slice.

3. **Scenario generation** — 500 correlated uniform vectors per day are drawn from the fitted copula and mapped back to prices via the LightGBM inverse CDF (piecewise-linear interpolation across the 9 quantile levels, with linear extrapolation beyond Q01/Q99).

The copula is re-fitted each month using an expanding training window, mirroring the LightGBM backtest structure (12 folds, 365-day burn-in, 30-day test blocks).

### Results

**Log score (higher is better):**

| Model        | Mean log-score per day | Std   |
|--------------|----------------------|-------|
| Independence | 0.000                | —     |
| Gaussian     | 34.7                 | 33.7  |
| t-copula     | 42.8                 | 26.2  |

The t-copula significantly outperforms the Gaussian (paired t-test: t=6.53, p<0.0001) with lower variance across days — it is both better on average and more consistent. The fitted degrees-of-freedom ν ≈ 24–33 across folds, indicating that the t-copula's heavier tails versus the Gaussian are empirically warranted.

**Dependence preservation:**
Both copulas reproduce adjacent-period Spearman correlations accurately (MAE ≈ 0.02–0.03 vs 0.86 for independence). At longer lags (13+ periods apart), both copulas overestimate dependence — the stationary global correlation matrix inflates long-range correlations due to days with regime-wide price shifts in the training data. This is most visible for overnight-to-evening period pairs, where empirical correlations are near zero but both copulas predict noticeably positive values.

**Spread CRPS (continuous ranked probability score for pairwise spreads):**

| Pair                  | Independence | Gaussian | t-copula |
|-----------------------|-------------|----------|----------|
| Adjacent (SP 1,2)     | 6.0         | 2.7      | 2.5      |
| Adjacent (SP 17,18)   | 6.6         | 2.8      | 2.7      |
| 4-apart (SP 1,5)      | 7.0         | 4.9      | 4.8      |
| 4-apart (SP 33,37)    | 10.0        | 9.2      | 9.1      |
| Overnight→Evening     | 12.5        | 12.3     | 12.2     |
| Midday→Evening        | 13.4        | 13.3     | 13.2     |

For adjacent spreads the copulas achieve a ~59% reduction in CRPS versus independence. For distant pairs the gain shrinks to near zero, consistent with both copulas overestimating long-range dependence.

---

## How This System Supports Battery Trading

The combined LightGBM + copula system outputs 500 full-day price scenarios each morning, each a vector of 48 correlated prices that respects both the marginal distributions and the within-day dependence structure.

These scenarios support battery trading in several concrete ways:

**Optimal dispatch planning:** The 500 scenarios can be fed directly into a stochastic optimisation problem (e.g., stochastic MPC or scenario-tree DP) that maximises expected revenue subject to battery state-of-charge constraints. This is strictly better than planning against a single point forecast because it accounts for spread uncertainty and avoids false precision.

**Spread distribution forecasting:** A key battery profitability driver is the difference between peak and off-peak prices within a day. The copula directly captures this joint structure. The scenario distribution of (price_peak − price_off_peak) is a well-calibrated forecast of tomorrow's arbitrage opportunity.

**Charging period selection:** The cheapest charging periods and most profitable discharge periods can be identified probabilistically across the 500 scenarios, with an explicit confidence level rather than a binary point-forecast decision.

---

## Limitations

**Spread compression in the marginal model:** The LightGBM P50 compresses the daily high-low spread relative to reality (mean spread error −19.9 £/MWh). This flows through to scenario generation: the median scenario will systematically understate the arbitrage opportunity. The copula partially compensates by sampling from the full quantile range, but systematic P50 bias is an upstream problem.

**Overestimated long-range dependence:** Both copulas use a single stationary, unstructured 48×48 correlation matrix. This overestimates dependence between distant settlement periods (e.g., overnight vs evening peak). For a battery that charges overnight and discharges in the evening, this leads to inflated estimates of the joint probability that both periods are simultaneously extreme, slightly overstating the benefit of the full-day arbitrage strategy.

**No structural correlation model:** The correlation matrix has no imposed structure (e.g., decay with lag, block-diagonal). Periods far apart in the day that have genuinely near-zero empirical correlation are assigned small but positive model correlations. A lag-decay or block-structured parametrisation could improve this.

**Single global ν:** The t-copula uses a single degrees-of-freedom parameter across all 48 periods. In reality, tail dependence likely varies by time of day — peak periods may exhibit stronger joint tail behaviour. A vine copula or block t-copula would allow heterogeneous tail dependence but at considerable added complexity.

**Marginal calibration level bias:** The mild upward level bias in LightGBM quantile predictions (especially in winter months) means scenario prices are on average slightly higher than reality. This could cause the algorithm to overestimate the profitability of selling and underestimate the cost of buying in those months.

**Out-of-distribution events:** The model is trained on GB market data from 2023 onward. Structural market changes (new interconnectors, large new storage capacity, policy interventions) can shift the dependence structure in ways the copula cannot anticipate. Model performance should be monitored and the copula re-calibrated regularly.
