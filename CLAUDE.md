This project is about building a **day-ahead GB wholesale electricity price forecasting system** for battery trading.

The system's goal is to forecast the **48 half-hour prices for tomorrow** in a way that is useful for trading, not just statistically accurate. That means it should produce:

* **probabilistic half-hour price forecasts** such as multiple quantiles
* **coherent full-day scenarios** that preserve realistic cross-hour structure

The implementation is intentionally **modular**. The current scope focuses on three parts:

1. **Feature store**
   Build an "as-of" half-hourly dataset using only information known before day-ahead bidding.
   Main inputs include:

   * historical market index price
   * demand forecast
   * day-ahead wind forecast
   * embedded solar/wind forecast
   * LoLP / De-rated Margin
   * interconnector flows aggregated from 5-minute to half-hour features

2. **Marginal probabilistic price model**
   Train **global LightGBM quantile models** to predict half-hour price quantiles for tomorrow.
   Inputs are the engineered core features.

3. **Scenario model**
   Fit a **t-copula** to historical dependence across the 48 half-hours.
   This converts marginal quantile forecasts into realistic daily price scenarios that preserve within-day structure.

The key design principle is:

* **LightGBM handles half-hour marginal price distributions**
* **t-copula handles cross-hour dependence**

So the final system should answer:

* what each half-hour price distribution looks like tomorrow
* what realistic full-day price paths could occur

A critical implementation requirement is **no leakage**: all features and training logic must mimic what would have been known before the actual day-ahead decision time.

## Regime model: deferred

Clustering analysis on z-scored daily price shapes showed that the main source of variation is seasonal shape change, not discrete market-state clustering. Historical days do not separate into clearly distinct day types with the current features and normalization. This does not justify a regime-conditioning layer for now.

Possible future direction: redesign regimes around price level, spread, and scarcity/tail behavior rather than shape.
