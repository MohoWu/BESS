"""Data loading functions for the clustering pipeline.

Each loader:
1. Checks for a cached file in data/raw/ and loads it if present.
2. Otherwise fetches from the relevant API (Elexon or NESO).
3. Saves the raw result to cache before returning.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from urllib import parse

from src.clustering.config import (
    DEMAND_CACHE_DIR,
    DEMAND_COLS,
    ELEXON_BASE_URL,
    ELEXON_BATCH_DAYS,
    ELEXON_DEMAND_DATASET,
    ELEXON_LOLP_DATASET,
    ELEXON_MIP_DATASET,
    EMBEDDED_COL_MAP,
    EMBEDDED_DATA_DIR,
    LOLP_CACHE_DIR,
    LOLP_COLS,
    MIP_CACHE_DIR,
    MIP_COLS,
    NESO_API_URL,
    NESO_BATCH_DAYS,
    NESO_EMBEDDED_DATASET_BY_YEAR,
    NESO_PAGE_SIZE,
    NESO_WIND_DATASET_ID,
    WIND_COL_MAP,
    WIND_DATA_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic Elexon fetcher
# ---------------------------------------------------------------------------


def fetch_elexon(
    dataset: str,
    start: str,
    end: str,
    batch_days: int = ELEXON_BATCH_DAYS,
    retry_wait: float = 2.0,
    max_retries: int = 3,
    data_providers: str | None = None,
) -> pd.DataFrame:
    """Fetch a dataset from the Elexon BMRS API in batched date windows.

    Parameters
    ----------
    dataset:
        Elexon dataset code, e.g. ``"MID"``, ``"NDF"``, ``"LOLPDRM"``.
    start, end:
        Inclusive date strings ``"YYYY-MM-DD"``.
    batch_days:
        Number of calendar days per API request (default 30).

    Returns
    -------
    pd.DataFrame
        Concatenated results across all batches.
    """
    url = f"{ELEXON_BASE_URL}/{dataset}"
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    frames: list[pd.DataFrame] = []
    cursor = start_dt

    while cursor <= end_dt:
        batch_end = min(cursor + timedelta(days=batch_days - 1), end_dt)
        params = {
            "from": f"{cursor.isoformat()}T00:00Z",
            "to": f"{batch_end.isoformat()}T23:59Z",
            "format": "json",
        }
        if data_providers is not None:
            params["dataProviders"] = data_providers

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                data = (
                    payload.get("data", payload)
                    if isinstance(payload, dict)
                    else payload
                )
                if data:
                    frames.append(pd.json_normalize(data))
                logger.debug(
                    "Fetched %s %s–%s: %d rows",
                    dataset,
                    cursor,
                    batch_end,
                    len(data) if data else 0,
                )
                break
            except requests.RequestException as exc:
                logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(retry_wait * attempt)
                else:
                    raise

        cursor = batch_end + timedelta(days=1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# MIP loader
# ---------------------------------------------------------------------------


def load_mip(
    start_date: str,
    end_date: str,
    cache_dir: Path = MIP_CACHE_DIR,
) -> pd.DataFrame:
    """Load Market Index Price data, fetching from Elexon if not cached.

    Returns a DataFrame with columns: settlement_date, settlement_period, price.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"mip_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.info("Loading MIP from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching MIP from Elexon API (%s to %s)…", start_date, end_date)
    raw = fetch_elexon(
        ELEXON_MIP_DATASET, start_date, end_date, batch_days=7, data_providers="APXMIDP"
    )

    if raw.empty:
        logger.warning("MIP fetch returned no data.")
        return pd.DataFrame(columns=["settlement_date", "settlement_period", "price", "volume"])  # type: ignore[call-overload]

    # Normalise column names
    df = _normalise_mip(raw)
    df.to_parquet(cache_file, index=False)
    logger.info("MIP cached to %s (%d rows)", cache_file, len(df))
    return df


def _normalise_mip(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Elexon MIP columns to canonical names."""
    col_candidates = {
        "settlement_date": [MIP_COLS["settlement_date"]],
        "settlement_period": [MIP_COLS["settlement_period"]],
        "price": [MIP_COLS["price"], MIP_COLS["price_alt"]],
        "volume": [MIP_COLS["volume"]],
    }
    df = _pick_columns(raw, col_candidates)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    df["settlement_period"] = df["settlement_period"].astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    # If multiple rows per (date, SP) — e.g. multiple data providers — take mean
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False)[["price", "volume"]].mean()  # type: ignore[assignment]
    return df


# ---------------------------------------------------------------------------
# Demand forecast loader
# ---------------------------------------------------------------------------


def load_demand_forecast(
    start_date: str,
    end_date: str,
    cache_dir: Path = DEMAND_CACHE_DIR,
) -> pd.DataFrame:
    """Load National Demand Forecast, fetching from Elexon if not cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"demand_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.info("Loading demand from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info(
        "Fetching demand forecast from Elexon API (%s to %s)…", start_date, end_date
    )
    raw = fetch_elexon(ELEXON_DEMAND_DATASET, start_date, end_date, batch_days=7)

    if raw.empty:
        logger.warning("Demand fetch returned no data.")
        return pd.DataFrame(columns=["settlement_date", "settlement_period", "demand"])  # type: ignore[call-overload]

    df = _normalise_demand(raw)
    df.to_parquet(cache_file, index=False)
    logger.info("Demand cached to %s (%d rows)", cache_file, len(df))
    return df


def _normalise_demand(raw: pd.DataFrame) -> pd.DataFrame:
    col_candidates = {
        "settlement_date": [DEMAND_COLS["settlement_date"]],
        "settlement_period": [DEMAND_COLS["settlement_period"]],
        "demand": [
            DEMAND_COLS["demand"],
            DEMAND_COLS["demand_alt"],
            DEMAND_COLS["demand_alt2"],
        ],
    }
    df = _pick_columns(raw, col_candidates)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    df["settlement_period"] = df["settlement_period"].astype(int)
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False)[["demand"]].mean()  # type: ignore[assignment]
    return df


# ---------------------------------------------------------------------------
# LoLP / De-rated Margin loader
# ---------------------------------------------------------------------------


def load_lolp_drm(
    start_date: str,
    end_date: str,
    cache_dir: Path = LOLP_CACHE_DIR,
) -> pd.DataFrame:
    """Load LoLP and De-rated Margin data, fetching from Elexon if not cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"lolp_drm_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.info("Loading LoLP/DRM from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching LoLP/DRM from Elexon API (%s to %s)…", start_date, end_date)
    raw = fetch_elexon(ELEXON_LOLP_DATASET, start_date, end_date, batch_days=7)

    if raw.empty:
        logger.warning("LoLP/DRM fetch returned no data.")
        return pd.DataFrame(columns=["settlement_date", "settlement_period", "lolp", "drm"])  # type: ignore[call-overload]

    df = _normalise_lolp(raw)
    df.to_parquet(cache_file, index=False)
    logger.info("LoLP/DRM cached to %s (%d rows)", cache_file, len(df))
    return df


def _normalise_lolp(raw: pd.DataFrame) -> pd.DataFrame:
    col_candidates = {
        "settlement_date": [LOLP_COLS["settlement_date"]],
        "settlement_period": [LOLP_COLS["settlement_period"]],
        "lolp": [LOLP_COLS["lolp"]],
        "drm": [LOLP_COLS["drm"], LOLP_COLS["drm_alt"]],
    }
    df = _pick_columns(raw, col_candidates)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    df["settlement_period"] = df["settlement_period"].astype(int)
    df["lolp"] = pd.to_numeric(df["lolp"], errors="coerce")
    df["drm"] = pd.to_numeric(df["drm"], errors="coerce")
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False)[["lolp", "drm"]].mean()  # type: ignore[assignment]
    return df


# ---------------------------------------------------------------------------
# NESO API fetcher
# ---------------------------------------------------------------------------


def fetch_neso(
    dataset_id: str,
    start: str,
    end: str,
    datetime_col: str,
    batch_days: int = NESO_BATCH_DAYS,
    page_size: int = NESO_PAGE_SIZE,
    retry_wait: float = 2.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch a dataset from the NESO CKAN datastore API in batched date windows.

    Parameters
    ----------
    dataset_id:
        NESO datastore resource UUID.
    start, end:
        Inclusive date strings ``"YYYY-MM-DD"``.
    datetime_col:
        The column name in the dataset to filter on (e.g. ``"Datetime_GMT"``).
    batch_days:
        Calendar days per batch request.
    page_size:
        Rows per paginated request within each batch.

    Returns
    -------
    pd.DataFrame
        Concatenated results across all batches and pages.
    """
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    frames: list[pd.DataFrame] = []
    cursor = start_dt

    while cursor <= end_dt:
        batch_end = min(cursor + timedelta(days=batch_days - 1), end_dt)
        start_iso = f"{cursor.isoformat()}T00:00:00.000Z"
        end_iso = f"{batch_end.isoformat()}T23:59:59.999Z"

        offset = 0
        total = 0
        while True:
            sql = (
                f'SELECT COUNT(*) OVER () AS _count, * FROM "{dataset_id}" '
                f'WHERE "{datetime_col}" >= \'{start_iso}\' '
                f'AND "{datetime_col}" <= \'{end_iso}\' '
                f'ORDER BY "_id" ASC '
                f"LIMIT {page_size} OFFSET {offset}"
            )
            params = {"sql": sql}

            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(
                        NESO_API_URL,
                        params=parse.urlencode(params),
                        timeout=60,
                    )
                    resp.raise_for_status()
                    result = resp.json().get("result", {})
                    records = result.get("records", [])
                    total = int(records[0]["_count"]) if records else 0
                    if records:
                        frames.append(pd.DataFrame(records))
                    logger.debug(
                        "NESO %s %s–%s offset=%d: %d/%d rows",
                        dataset_id[:8],
                        cursor,
                        batch_end,
                        offset,
                        len(records),
                        total,
                    )
                    break
                except requests.RequestException as exc:
                    logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, exc)
                    if attempt < max_retries:
                        time.sleep(retry_wait * attempt)
                    else:
                        raise

            offset += page_size
            if offset >= total:
                break

        cursor = batch_end + timedelta(days=1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# NESO wind and embedded loaders
# ---------------------------------------------------------------------------


def load_wind_forecast(
    start_date: str,
    end_date: str,
    cache_dir: Path = WIND_DATA_DIR,
) -> pd.DataFrame:
    """Load NESO day-ahead wind forecast from the API, caching locally.

    Returns DataFrame with: settlement_date, settlement_period, wind_da.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"wind_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.info("Loading wind forecast from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching wind forecast from NESO API (%s to %s)…", start_date, end_date)
    raw = fetch_neso(
        NESO_WIND_DATASET_ID,
        start_date,
        end_date,
        datetime_col="Datetime_GMT",
    )

    if raw.empty:
        logger.warning("Wind forecast fetch returned no data.")
        return pd.DataFrame(columns=["settlement_date", "settlement_period", "wind_da"])  # type: ignore[call-overload]

    df = _apply_col_map(raw, WIND_COL_MAP)
    _require_cols(df, ["settlement_date", "settlement_period", "wind_da"], "NESO wind API")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    df["settlement_period"] = df["settlement_period"].astype(int)
    df["wind_da"] = pd.to_numeric(df["wind_da"], errors="coerce")
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False)[["wind_da"]].mean()  # type: ignore[assignment]

    df.to_parquet(cache_file, index=False)
    logger.info("Wind forecast cached to %s (%d rows)", cache_file, len(df))
    return df  # type: ignore[return-value]


def _fetch_day_embedded_da(
    dataset_id: str,
    forecast_day: date,
    settlement_date_col: str,
    page_size: int,
    retry_wait: float,
    max_retries: int,
) -> list[pd.DataFrame]:
    """Fetch embedded DA forecast for a single forecast day (blocking, with pagination)."""
    settlement_dt = forecast_day + timedelta(days=1)
    forecast_start = f"{forecast_day.isoformat()}T08:00:00.000"
    forecast_end = f"{forecast_day.isoformat()}T08:59:59.999"
    settlement_str = settlement_dt.isoformat()
    next_day_str = (settlement_dt + timedelta(days=1)).isoformat()

    frames: list[pd.DataFrame] = []
    offset = 0
    total = 0

    while True:
        sql = (
            f'SELECT COUNT(*) OVER () AS _count, * FROM "{dataset_id}" '
            f'WHERE "Forecast_Datetime" >= \'{forecast_start}\' '
            f'AND "Forecast_Datetime" <= \'{forecast_end}\' '
            f'AND "{settlement_date_col}" >= \'{settlement_str}T00:00:00\' '
            f'AND "{settlement_date_col}" < \'{next_day_str}T00:00:00\' '
            f'ORDER BY "_id" ASC '
            f"LIMIT {page_size} OFFSET {offset}"
        )
        params = {"sql": sql}

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(
                    NESO_API_URL,
                    params=parse.urlencode(params),
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json().get("result", {})
                records = result.get("records", [])
                total = int(records[0]["_count"]) if records else 0
                if records:
                    frames.append(pd.DataFrame(records))
                logger.debug(
                    "NESO embedded DA %s → settlement %s: %d rows",
                    forecast_day,
                    settlement_dt,
                    len(records),
                )
                break
            except requests.RequestException as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s", attempt, max_retries, forecast_day, exc
                )
                if attempt < max_retries:
                    time.sleep(retry_wait * attempt)
                else:
                    raise

        offset += page_size
        if offset >= total:
            break

    return frames


def fetch_neso_embedded_da(
    dataset_id: str,
    start: str,
    end: str,
    settlement_date_col: str,
    page_size: int = NESO_PAGE_SIZE,
    retry_wait: float = 2.0,
    max_retries: int = 3,
    concurrency: int = 10,
) -> pd.DataFrame:
    """Fetch NESO embedded forecast day-by-day using concurrent threads.

    For each forecast day D, issues one query with:
      - Forecast_Datetime in [D 08:00:00, D 08:59:59]
      - Settlement date column == D + 1

    Up to ``concurrency`` days are fetched simultaneously via a thread pool.

    Parameters
    ----------
    dataset_id:
        NESO datastore resource UUID (year-specific archive).
    start, end:
        Inclusive date strings ``"YYYY-MM-DD"`` for the *forecast* day D.
    settlement_date_col:
        Raw column name for the settlement date in this dataset.
    concurrency:
        Maximum number of simultaneous in-flight requests (default 10).
    """
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    days: list[date] = []
    cursor = start_dt
    while cursor <= end_dt:
        days.append(cursor)
        cursor += timedelta(days=1)

    months_in_range = sorted({(d.year, d.month) for d in days})
    for year, month in months_in_range:
        logger.info("Queuing embedded DA forecast for %d-%02d…", year, month)

    # Submit all days to the thread pool; collect results keyed by day
    day_results: dict[date, list[pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _fetch_day_embedded_da,
                dataset_id, day, settlement_date_col,
                page_size, retry_wait, max_retries,
            ): day
            for day in days
        }
        for future in as_completed(futures):
            day = futures[future]
            day_results[day] = future.result()

    # Aggregate in chronological order and log monthly summaries
    month_rows: dict[tuple[int, int], int] = {}
    all_frames: list[pd.DataFrame] = []
    for day in days:
        day_frames = day_results[day]
        key = (day.year, day.month)
        month_rows[key] = month_rows.get(key, 0) + sum(len(f) for f in day_frames)
        all_frames.extend(day_frames)

    for (year, month), count in sorted(month_rows.items()):
        logger.info("  Completed %d-%02d: %d rows fetched", year, month, count)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def load_embedded_forecast(
    start_date: str,
    end_date: str,
    cache_dir: Path = EMBEDDED_DATA_DIR,
) -> pd.DataFrame:
    """Load NESO embedded solar/wind day-ahead forecast from the API, caching locally.

    For each forecast day D, fetches only the forecast published 08:00–09:00 and
    with settlement_date == D + 1. Queries are issued day-by-day at the API level
    to avoid downloading all forecast horizons.

    Returns DataFrame with: settlement_date, settlement_period, emb_solar, emb_wind.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"embedded_da_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.info("Loading embedded forecast from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info(
        "Fetching embedded forecast from NESO API (%s to %s)…", start_date, end_date
    )

    # Detect which raw column name is used for settlement date in this dataset
    # (checked against the first non-empty response; assumed consistent across years)
    sd_raw_col_candidates = [
        k for k, v in EMBEDDED_COL_MAP.items() if v == "settlement_date"
    ]

    start_year = date.fromisoformat(start_date).year
    end_year = date.fromisoformat(end_date).year
    year_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        dataset_id = NESO_EMBEDDED_DATASET_BY_YEAR.get(year)
        if dataset_id is None:
            logger.warning(
                "No embedded forecast dataset ID configured for year %d — skipping.", year
            )
            continue
        year_start = max(start_date, f"{year}-01-01")
        # Forecast day D must be within this year's archive; settlement is D+1
        year_end = min(end_date, f"{year}-12-31")

        # Determine which settlement_date column name this dataset uses by probing
        # the first candidate that appears in the dataset
        sd_raw_col: str | None = None
        for candidate in sd_raw_col_candidates:
            probe_sql = (
                f'SELECT * FROM "{dataset_id}" LIMIT 1'
            )
            try:
                resp = requests.get(
                    NESO_API_URL,
                    params=parse.urlencode({"sql": probe_sql}),
                    timeout=30,
                )
                resp.raise_for_status()
                records = resp.json().get("result", {}).get("records", [])
                if records and candidate in records[0]:
                    sd_raw_col = candidate
                    break
            except requests.RequestException:
                pass

        if sd_raw_col is None:
            raise ValueError(
                f"Cannot detect settlement_date column in NESO embedded dataset {dataset_id}. "
                f"Tried: {sd_raw_col_candidates}"
            )

        logger.info("Year %d: settlement_date column = '%s'", year, sd_raw_col)
        chunk = fetch_neso_embedded_da(
            dataset_id, year_start, year_end, settlement_date_col=sd_raw_col
        )
        if not chunk.empty:
            year_frames.append(chunk)

    raw = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()

    if raw.empty:
        logger.warning("Embedded forecast fetch returned no data.")
        return pd.DataFrame(columns=["settlement_date", "settlement_period", "emb_solar", "emb_wind"])  # type: ignore[call-overload, reportArgumentType]

    df = _apply_col_map(raw, EMBEDDED_COL_MAP)
    _require_cols(
        df,
        ["settlement_date", "settlement_period", "emb_solar", "emb_wind"],
        "NESO embedded API",
    )
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    df["settlement_period"] = df["settlement_period"].astype(int)
    for col in ["emb_solar", "emb_wind"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False)[
        ["emb_solar", "emb_wind"]
    ].mean()  # type: ignore[assignment]

    df.to_parquet(cache_file, index=False)
    logger.info("Embedded forecast cached to %s (%d rows)", cache_file, len(df))
    return df  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_columns(
    raw: pd.DataFrame, col_candidates: dict[str, list[str]]
) -> pd.DataFrame:
    """Select one column per canonical name from a list of candidate raw names."""
    result = {}
    for canonical, candidates in col_candidates.items():
        matched = next((c for c in candidates if c in raw.columns), None)
        if matched is None:
            # Try case-insensitive match
            lower_map = {c.lower(): c for c in raw.columns}
            matched = next(
                (lower_map[c.lower()] for c in candidates if c.lower() in lower_map),
                None,
            )
        if matched is None:
            raise KeyError(
                f"Cannot find column '{canonical}' in DataFrame. "
                f"Tried: {candidates}. Available: {list(raw.columns)}"
            )
        result[canonical] = raw[matched]
    return pd.DataFrame(result)


def _apply_col_map(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns in df according to col_map (raw -> canonical)."""
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=rename)


def _require_cols(df: pd.DataFrame, required: list[str], source: object) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} after applying column map for {source}. "
            f"Available: {list(df.columns)}. Update WIND_COL_MAP / EMBEDDED_COL_MAP in config.py."
        )
