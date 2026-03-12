"""Configuration constants for the clustering pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE = "2026-03-01"

# ---------------------------------------------------------------------------
# Settlement period conventions
# ---------------------------------------------------------------------------
STANDARD_SPS_PER_DAY = 48  # half-hours in a standard day
# Days with SP count != 48 are dropped (DST transitions give 46 or 50 SPs)

# ---------------------------------------------------------------------------
# Price thresholds
# ---------------------------------------------------------------------------
NEGATIVE_PRICE_THRESHOLD = 0  # £/MWh — below this is a "negative price" SP
SPIKE_PRICE_THRESHOLD = 200  # £/MWh — above this is a "price spike" SP

# ---------------------------------------------------------------------------
# LoLP thresholds
# ---------------------------------------------------------------------------
LOLP_HIGH_THRESHOLD = 0.5
LOLP_MID_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
MIN_COMPLETENESS = 0.95  # minimum fraction of SPs required per series per day

# ---------------------------------------------------------------------------
# Elexon API
# ---------------------------------------------------------------------------
ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_BATCH_DAYS = 30  # max days per API request

# ---------------------------------------------------------------------------
# NESO API
# ---------------------------------------------------------------------------
NESO_API_URL = "https://api.neso.energy/api/3/action/datastore_search_sql"
NESO_WIND_DATASET_ID = "7524ec65-f782-4258-aaf8-5b926c17b966"
NESO_EMBEDDED_DATASET_ID = "db6c038f-98af-4570-ab60-24d71ebd0ae5"  # live/current day only
# Year-specific historical archives
NESO_EMBEDDED_DATASET_BY_YEAR: dict[int, str] = {
    2023: "26c9ef64-ce43-4e22-b984-ef013636aacb",
    2024: "06abd00a-ef6b-488b-9b6d-5e08fdc0c890",
    2025: "fc13df13-2dad-4a1c-b9e3-4569efba4955",
    2026: "d6375700-69c2-4c25-8bde-883a205d742e",
}
NESO_BATCH_DAYS = 30  # days per NESO API request
NESO_PAGE_SIZE = 10000  # rows per paginated request

# Dataset identifiers
ELEXON_MIP_DATASET = "balancing/pricing/market-index"
ELEXON_DEMAND_DATASET = "forecast/demand/day-ahead/latest"
ELEXON_LOLP_DATASET = "forecast/system/loss-of-load"

# ---------------------------------------------------------------------------
# Column name mappings per data source
# ---------------------------------------------------------------------------

# MIP (Market Index Data) — returned by Elexon MID endpoint
MIP_COLS = {
    "settlement_date": "settlementDate",
    "settlement_period": "settlementPeriod",
    "price": "price",  # may also appear as "marketIndexPrice"
    "price_alt": "marketIndexPrice",
    "volume": "volume",
}

# Demand forecast (NDF) — returned by Elexon NDF endpoint
DEMAND_COLS = {
    "settlement_date": "settlementDate",
    "settlement_period": "settlementPeriod",
    "demand": "demand",  # may appear as "nationalDemand" or "transmissionSystemDemand"
    "demand_alt": "nationalDemand",
    "demand_alt2": "transmissionSystemDemand",
}

# LoLP / De-rated Margin (LOLPDRM) — returned by Elexon LOLPDRM endpoint
LOLP_COLS = {
    "settlement_date": "settlementDate",
    "settlement_period": "settlementPeriod",
    "lolp": "lossOfLoadProbability",
    "drm": "deratedMargin",
    "drm_alt": "drmForecast",
}

# NESO day-ahead wind forecast (manual CSV download)
# Expected columns after renaming — adjust WIND_COL_MAP if actual CSV differs
WIND_COL_MAP = {
    # raw_col_name: canonical_name
    "Settlement Date": "settlement_date",
    "Settlement_period": "settlement_period",
    "Incentive_forecast": "wind_da",
    # Fallbacks for alternative column names in NESO exports:
    "Date": "settlement_date",
    "SETTLEMENT_PERIOD": "settlement_period",
    "WIND_FORECAST": "wind_da",
    "Wind Forecast (MW)": "wind_da",
}

# NESO embedded solar/wind forecast (manual CSV download)
EMBEDDED_COL_MAP = {
    "Settlement Date": "settlement_date",
    "Settlement Period": "settlement_period",
    "Embedded Solar Forecast": "emb_solar",
    "Embedded Wind Forecast": "emb_wind",
    # Fallbacks:
    "SETTLEMENT_DATE": "settlement_date",
    "SETTLEMENT_PERIOD": "settlement_period",
    "EMBEDDED_SOLAR_FORECAST": "emb_solar",
    "EMBEDDED_WIND_FORECAST": "emb_wind",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MIP_CACHE_DIR = RAW_DATA_DIR / "mip"
DEMAND_CACHE_DIR = RAW_DATA_DIR / "demand"
LOLP_CACHE_DIR = RAW_DATA_DIR / "lolp_drm"
WIND_DATA_DIR = RAW_DATA_DIR / "wind"
EMBEDDED_DATA_DIR = RAW_DATA_DIR / "embedded"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLUSTERING_DAILY_PARQUET = PROCESSED_DATA_DIR / "clustering_daily.parquet"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "clustering"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
KMEANS_PATH_TEMPLATE = ARTIFACTS_DIR / "kmeans_k{k}.pkl"
GMM_PATH_TEMPLATE = ARTIFACTS_DIR / "gmm_k{k}.pkl"
REGIME_LABELS_PATH = ARTIFACTS_DIR / "regime_labels.json"

# ---------------------------------------------------------------------------
# Feature column groups (used by pipeline and evaluation)
# ---------------------------------------------------------------------------
SHAPE_COLS = [f"price_shape_{sp}" for sp in range(1, 49)] + [
    f"netload_shape_{sp}" for sp in range(1, 49)
]

SUMMARY_COLS = [
    "price_mean",
    "price_min",
    "price_max",
    "price_range",
    "price_std",
    "midday_price_mean",
    "evening_price_mean",
    "evening_peak_minus_midday_trough",
    "netload_mean",
    "netload_max",
    "netload_min",
    "netload_range",
    "netload_std",
    "netload_evening_ramp",
    "netload_morning_ramp",
]

METADATA_COLS = ["month", "weekday", "is_weekend"]

FEATURE_COLS = SHAPE_COLS + SUMMARY_COLS
