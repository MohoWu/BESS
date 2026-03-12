"""Feature engineering pipeline for the clustering analysis.

Steps
-----
1. build_half_hourly_panel   — align all series to (delivery_date, SP) MultiIndex
2. filter_eligible_days      — drop incomplete / DST-anomaly days
3. normalize_shape_within_day — per-day z-score for price, net_load, drm
4. construct_daily_features  — derive shape vectors + summary stats
5. scale_features            — RobustScaler across days
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.clustering.config import (
    FEATURE_COLS,
    MIN_COMPLETENESS,
    STANDARD_SPS_PER_DAY,
)

logger = logging.getLogger(__name__)

# Settlement periods that correspond to "midday" (SP 25–36 ≈ 12:00–18:00)
_MIDDAY_SPS = list(range(25, 37))
# Settlement periods that correspond to "evening peak" (SP 33–42 ≈ 16:00–21:00)
_EVENING_SPS = list(range(33, 43))
# Morning ramp (SP 13–20 ≈ 06:00–10:00)
_MORNING_SPS = list(range(13, 21))


# ---------------------------------------------------------------------------
# Step 1 — build panel
# ---------------------------------------------------------------------------


def build_half_hourly_panel(
    mip: pd.DataFrame,
    demand: pd.DataFrame,
    wind: pd.DataFrame | None,
    embedded: pd.DataFrame | None,
) -> pd.DataFrame:
    """Align all series to a (settlement_date, settlement_period) MultiIndex.

    Parameters
    ----------
    mip, demand:
        DataFrames with columns ``settlement_date``, ``settlement_period``
        plus their value columns (``price``, ``demand``).
    wind:
        DataFrame with ``wind_da`` or None if not available.
    embedded:
        DataFrame with ``emb_solar``, ``emb_wind`` or None if not available.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(settlement_date, settlement_period)`` with columns:
        ``price``, ``demand``,
        optionally ``wind_da``, ``emb_solar``, ``emb_wind``, ``net_load``.
    """
    # Merge on date + SP
    panel = mip.set_index(["settlement_date", "settlement_period"])

    panel = panel.join(
        demand.set_index(["settlement_date", "settlement_period"])[["demand"]],
        how="outer",
    )

    if wind is not None:
        panel = panel.join(
            wind.set_index(["settlement_date", "settlement_period"])[["wind_da"]],
            how="left",
        )
    else:
        panel["wind_da"] = 0.0

    if embedded is not None:
        panel = panel.join(
            embedded.set_index(["settlement_date", "settlement_period"])[
                ["emb_solar", "emb_wind"]
            ],
            how="left",
        )
    else:
        panel["emb_solar"] = 0.0
        panel["emb_wind"] = 0.0

    # Compute net load: demand minus all renewable generation forecasts
    panel["net_load"] = (
        panel["demand"]
        - panel["wind_da"].fillna(0)
        - panel["emb_wind"].fillna(0)
        - panel["emb_solar"].fillna(0)
    )

    panel = panel.sort_index()
    logger.info("Panel shape after join: %s", panel.shape)
    return panel


# ---------------------------------------------------------------------------
# Step 2 — filter eligible days
# ---------------------------------------------------------------------------


def filter_eligible_days(
    panel: pd.DataFrame,
    min_completeness: float = MIN_COMPLETENESS,
) -> tuple[pd.DataFrame, list]:
    """Drop days that fail completeness or DST checks.

    Criteria
    --------
    * Days where settlement period count ≠ 48 (DST transitions: 46 or 50 SPs).
    * Days where any required series has < ``min_completeness`` fraction of
      non-null values across the 48 settlement periods.

    Returns
    -------
    (filtered_panel, dropped_dates)
    """
    required_cols = ["price", "demand", "net_load"]
    present_cols = [c for c in required_cols if c in panel.columns]

    dates = panel.index.get_level_values("settlement_date").unique()
    dropped: list = []
    keep: list = []

    for d in dates:
        day = panel.loc[d]
        n_sps = len(day)

        # DST check
        if n_sps != STANDARD_SPS_PER_DAY:
            logger.debug("Dropping %s — %d SPs (DST anomaly)", d, n_sps)
            dropped.append((d, f"DST anomaly ({n_sps} SPs)"))
            continue

        # Completeness check
        fail = False
        for col in present_cols:
            if col not in day.columns:
                continue
            frac = day[col].notna().mean()
            if frac < min_completeness:
                logger.debug("Dropping %s — %s completeness %.2f", d, col, frac)
                dropped.append((d, f"{col} completeness {frac:.2f}"))
                fail = True
                break
        if not fail:
            keep.append(d)

    filtered = panel.loc[keep]
    logger.info(
        "Days kept: %d / %d  (dropped: %d)", len(keep), len(dates), len(dropped)
    )
    return filtered, dropped


# ---------------------------------------------------------------------------
# Step 3 — within-day shape normalisation
# ---------------------------------------------------------------------------


def normalize_shape_within_day(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply per-day z-score normalisation to price, net_load, and drm.

    Assumption: if a day's std ≈ 0 (constant series — e.g., data error or
    grid outage), shape values are set to 0 (flat shape). This affects very
    few days and avoids division-by-zero. The original raw columns are kept.
    """
    cols_to_norm = ["price", "net_load"]
    norm_cols = {c: f"{c}_norm" for c in cols_to_norm if c in panel.columns}

    panel = panel.copy()
    for raw_col, norm_col in norm_cols.items():
        panel[norm_col] = panel.groupby(level="settlement_date")[raw_col].transform(
            _zscore_with_flat_guard
        )

    return panel


def _zscore_with_flat_guard(series: pd.Series) -> pd.Series:
    """Z-score a series; return zeros if std ≈ 0 (flat-guard)."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma < 1e-6:
        return pd.Series(0.0, index=series.index)
    return ((series - mu) / sigma).fillna(0.0)


# ---------------------------------------------------------------------------
# Step 4 — daily feature construction
# ---------------------------------------------------------------------------


def construct_daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a daily-grain feature DataFrame from the half-hourly panel.

    Assumes ``normalize_shape_within_day`` has already been applied so that
    ``price_norm``, ``net_load_norm``, and ``drm_norm`` columns exist.

    Returns
    -------
    pd.DataFrame
        Indexed by ``settlement_date`` with shape vectors + summary stats.
    """
    records = []
    dates = panel.index.get_level_values("settlement_date").unique()

    for d in dates:
        day = panel.loc[d].sort_index()  # sort by settlement_period
        row: dict = {}

        # ---- shape vectors (one value per SP) ----
        for sp in range(1, STANDARD_SPS_PER_DAY + 1):
            sp_row = day.loc[sp] if sp in day.index else None

            def _get(col, default=np.nan):
                if sp_row is None:
                    return default
                v = (
                    sp_row.get(col, default)
                    if hasattr(sp_row, "get")
                    else getattr(sp_row, col, default)
                )
                return v if pd.notna(v) else default

            row[f"price_shape_{sp}"] = _get("price_norm", 0.0)
            row[f"netload_shape_{sp}"] = _get("net_load_norm", 0.0)

        prices = (
            day["price"].dropna() if "price" in day.columns else pd.Series(dtype=float)
        )
        netloads = (
            day["net_load"].dropna()
            if "net_load" in day.columns
            else pd.Series(dtype=float)
        )
        # ---- price summary stats ----
        row["price_mean"] = prices.mean()
        row["price_min"] = prices.min()
        row["price_max"] = prices.max()
        row["price_range"] = prices.max() - prices.min()
        row["price_std"] = prices.std()

        midday_prices = _sp_values(day, "price", _MIDDAY_SPS)
        evening_prices = _sp_values(day, "price", _EVENING_SPS)
        row["midday_price_mean"] = (
            midday_prices.mean() if len(midday_prices) else np.nan
        )
        row["evening_price_mean"] = (
            evening_prices.mean() if len(evening_prices) else np.nan
        )
        row["evening_peak_minus_midday_trough"] = (
            evening_prices.max() - midday_prices.min()
            if len(evening_prices) and len(midday_prices)
            else np.nan
        )

        # ---- net load summary stats ----
        row["netload_mean"] = netloads.mean()
        row["netload_max"] = netloads.max()
        row["netload_min"] = netloads.min()
        row["netload_range"] = netloads.max() - netloads.min()
        row["netload_std"] = netloads.std()

        morning_nl = _sp_values(day, "net_load", _MORNING_SPS)
        evening_nl = _sp_values(day, "net_load", _EVENING_SPS)
        # Ramp = difference between the mean of the later half and earlier half of the window
        row["netload_morning_ramp"] = _ramp(morning_nl)
        row["netload_evening_ramp"] = _ramp(evening_nl)

        # ---- metadata ----
        d_ts = pd.Timestamp(d)
        row["month"] = d_ts.month
        row["weekday"] = d_ts.weekday()  # 0=Mon, 6=Sun
        row["is_weekend"] = int(d_ts.weekday() >= 5)

        row["settlement_date"] = d
        records.append(row)

    daily_df = pd.DataFrame(records).set_index("settlement_date")
    logger.info("Daily feature matrix shape: %s", daily_df.shape)
    return daily_df


def _sp_values(day: pd.DataFrame, col: str, sps: list[int]) -> pd.Series:
    """Extract values for specific settlement periods from a day slice."""
    if col not in day.columns:
        return pd.Series(dtype=float)
    idx = [sp for sp in sps if sp in day.index]
    return day.loc[idx, col].dropna() if idx else pd.Series(dtype=float)


def _ramp(series: pd.Series) -> float:
    """Simple ramp measure: mean of second half minus mean of first half."""
    if len(series) < 2:
        return np.nan
    mid = len(series) // 2
    return float(series.iloc[mid:].mean() - series.iloc[:mid].mean())


# ---------------------------------------------------------------------------
# Step 5 — across-day scaling
# ---------------------------------------------------------------------------


def scale_features(
    daily_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, RobustScaler, list[str]]:
    """Fit a RobustScaler on feature columns and return scaled array.

    Parameters
    ----------
    daily_df:
        Daily-grain DataFrame produced by ``construct_daily_features``.
    feature_cols:
        Columns to use for clustering. Defaults to ``FEATURE_COLS`` from config.

    Returns
    -------
    (X_scaled, scaler, feature_cols)
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in daily_df.columns]

    X = daily_df[feature_cols].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Scaled feature matrix: %s", X_scaled.shape)
    return X_scaled, scaler, feature_cols
