"""Fetch and persist EMA (energy_market_analysis) national forecasts.

EMA produces 168h-ahead generation/load forecasts for the DE-LU bidding zone
from weather data, available by ~08:00 UTC. Column names match SMARD convention.

Usage:
    from src.data.ema import fetch_ema_forecasts, load_ema_history, save_ema_snapshot

    df = fetch_ema_forecasts()          # fetch latest forecast
    save_ema_snapshot(df)               # persist as daily parquet snapshot
    history = load_ema_history()        # load all accumulated snapshots
"""

from datetime import datetime, timezone
from pathlib import Path
import time

from loguru import logger
import pandas as pd

from src.config import DATA_DIR, EMA_DATA_DIR, EMA_HISTORICAL_FORECASTS_DIR

# URL for when the CSV is deployed to GitHub Pages.
# Falls back to local EMA repo path if the URL is unavailable.
EMA_CSV_URL = (
    "https://smnfrse.github.io/energy_market_analysis/data/DE/downloads/national_forecasts.csv"
)

# Local path (both repos on the same machine)
EMA_LOCAL_PATH = (
    Path.home()
    / "projects"
    / "energy_market_analysis"
    / "deploy"
    / "data"
    / "DE"
    / "downloads"
    / "national_forecasts.csv"
)

# Columns that overlap with EP's merged dataset (the ones we care about)
EMA_OVERLAP_COLUMNS = [
    "prognostizierte_erzeugung_wind_und_photovoltaik",
    "prognostizierte_erzeugung_sonstige",
    "prognostizierte_erzeugung_gesamt",
    "prognostizierter_verbrauch_gesamt",
    "prognostizierter_verbrauch_residuallast",
]


def fetch_ema_forecasts(
    url: str = EMA_CSV_URL,
    local_path: Path = EMA_LOCAL_PATH,
) -> pd.DataFrame | None:
    """Download latest EMA national forecasts.

    Tries URL first, falls back to local file path.

    Returns:
        UTC-indexed DataFrame with SMARD-compatible column names, or None on failure.
    """
    df = _try_url(url)
    if df is None:
        df = _try_local(local_path)
    if df is None:
        logger.warning("Could not load EMA forecasts from URL or local path")
        return None

    # Ensure UTC DatetimeIndex
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "date_utc"
    logger.info(
        f"Loaded EMA forecasts: {len(df)} rows, "
        f"{df.index.min()} to {df.index.max()}, "
        f"columns: {list(df.columns)}"
    )
    return df


def _try_url(url: str) -> pd.DataFrame | None:
    """Attempt to fetch CSV from URL."""
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        logger.info(f"Fetched EMA forecasts from URL ({len(df)} rows)")
        return df
    except Exception as e:
        logger.debug(f"URL fetch failed: {e}")
        return None


def _try_local(local_path: Path) -> pd.DataFrame | None:
    """Attempt to load CSV from local filesystem."""
    if not local_path.exists():
        logger.debug(f"Local EMA file not found: {local_path}")
        return None
    try:
        df = pd.read_csv(local_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded EMA forecasts from local path ({len(df)} rows)")
        return df
    except Exception as e:
        logger.debug(f"Local file read failed: {e}")
        return None


def save_ema_snapshot(df: pd.DataFrame, snapshot_dir: Path = EMA_DATA_DIR) -> Path:
    """Persist EMA forecast as a dated parquet snapshot.

    Each snapshot contains the full 168h forecast as published that day.

    Args:
        df: EMA forecast DataFrame (UTC-indexed).
        snapshot_dir: Directory for snapshots (default: data/ema/).

    Returns:
        Path to the written parquet file.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = snapshot_dir / f"{today}.parquet"
    df.to_parquet(path)
    logger.info(f"Saved EMA snapshot: {path} ({len(df)} rows)")
    return path


def load_ema_history(snapshot_dir: Path = EMA_DATA_DIR) -> pd.DataFrame:
    """Load accumulated EMA forecast history from daily snapshots.

    Concatenates all parquet files in snapshot_dir. For overlapping timestamps
    across snapshots, keeps the most recent snapshot's values.

    Returns:
        Combined UTC-indexed DataFrame, or empty DataFrame if no snapshots.
    """
    if not snapshot_dir.exists():
        logger.warning(f"EMA snapshot directory does not exist: {snapshot_dir}")
        return pd.DataFrame()

    files = sorted(snapshot_dir.glob("*.parquet"))
    if not files:
        logger.warning(f"No EMA snapshots found in {snapshot_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["_snapshot_date"] = f.stem  # e.g., "2026-03-15"
        dfs.append(df)

    combined = pd.concat(dfs)
    # Keep last snapshot's values for duplicate timestamps
    combined = combined.sort_values("_snapshot_date").groupby(level=0).last()
    combined = combined.drop(columns=["_snapshot_date"])

    logger.info(f"Loaded EMA history: {len(files)} snapshots, {len(combined)} unique hours")
    return combined


def build_ema_training_data(
    merged_df: pd.DataFrame,
    ema_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Replace prognostizierte_* columns with EMA equivalents where available.

    For timestamps where EMA data exists (typically 2022+), replaces the 5 overlap
    columns with EMA forecast values. For timestamps without EMA data (pre-2022),
    keeps existing values (SMARD forecasts / backfilled actuals).

    Adds a `forecast_source` regime indicator:
        0 = original SMARD/actuals (no EMA data available)
        1 = EMA backtest forecasts

    Args:
        merged_df: EP's merged dataset with prognostizierte_* columns.
        ema_forecasts: Output of load_ema_historical_forecasts() — must contain
            EMA_OVERLAP_COLUMNS, indexed by UTC timestamp.

    Returns:
        Copy of merged_df with EMA columns swapped in and forecast_source added.
    """
    df = merged_df.copy()

    # Determine which timestamps have EMA coverage
    ema_idx = df.index.intersection(ema_forecasts.index)
    logger.info(
        f"EMA coverage: {len(ema_idx)} of {len(df)} hours ({len(ema_idx) / len(df) * 100:.1f}%)"
    )

    if len(ema_idx) == 0:
        logger.warning("No overlapping timestamps between merged_df and EMA forecasts")
        df["forecast_source"] = 0
        return df

    # Replace overlap columns where EMA data exists
    replaced = []
    for col in EMA_OVERLAP_COLUMNS:
        if col in df.columns and col in ema_forecasts.columns:
            df.loc[ema_idx, col] = ema_forecasts.loc[ema_idx, col]
            replaced.append(col)
        else:
            logger.warning(
                f"Column {col} missing from {'merged_df' if col not in df.columns else 'ema_forecasts'}"
            )

    # Add regime indicators
    df["forecast_source"] = 0
    df.loc[ema_idx, "forecast_source"] = 1

    # Pass through is_true_forecast from EMA data.
    # Default=1 for non-EMA rows: SMARD forecasts use actual forecast weather.
    # Only hindcast rows (actual weather, not forecast) get 0.
    df["is_true_forecast"] = 1
    if "is_true_forecast" in ema_forecasts.columns:
        df.loc[ema_idx, "is_true_forecast"] = ema_forecasts.loc[ema_idx, "is_true_forecast"]

    n_smard = (df["forecast_source"] == 0).sum()
    n_ema = (df["forecast_source"] == 1).sum()
    n_hindcast = ((df["is_true_forecast"] == 0) & (df["forecast_source"] == 1)).sum()
    logger.info(
        f"Replaced {len(replaced)} columns: {replaced}. "
        f"forecast_source: {n_smard} SMARD/actuals, {n_ema} EMA "
        f"({n_hindcast} hindcast)"
    )
    return df


def load_ema_historical_forecasts(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load EMA hindcast and backtest parquet files from the EMA repo output.

    Globs for hindcast_*.parquet and backtest_*.parquet (skipping per_tso_* files).
    Validates that each file contains all EMA_OVERLAP_COLUMNS before including it.
    Clips negative generation/load values to zero. For overlapping timestamps,
    prefers backtest over hindcast (backtest has realistic forecast errors).

    Args:
        path: Directory containing parquet files. Defaults to EMA_HISTORICAL_FORECASTS_DIR.

    Returns:
        UTC-indexed DataFrame with EMA_OVERLAP_COLUMNS + `source` + `is_true_forecast`.
        is_true_forecast: 1 for backtest (forecast weather), 0 for hindcast (actual weather).
    """
    if path is None:
        # Try in-repo fallback first (works in CI), then cross-repo path (local dev)
        fallback_path = DATA_DIR / "ema" / "historical"
        if fallback_path.exists() and any(fallback_path.glob("*.parquet")):
            path = fallback_path
        else:
            path = EMA_HISTORICAL_FORECASTS_DIR

    path = Path(path)
    if not path.exists():
        logger.warning(
            f"EMA historical forecasts directory not found: {path}. "
            "Checked both data/ema/historical/ and EMA repo path."
        )
        return pd.DataFrame()

    # Discover hindcast and backtest files (skip per_tso_*)
    candidates = sorted(path.glob("hindcast_*.parquet")) + sorted(path.glob("backtest_*.parquet"))
    candidates = [f for f in candidates if not f.name.startswith("per_tso_")]

    if not candidates:
        logger.warning(f"No hindcast_*.parquet or backtest_*.parquet files found in {path}")
        return pd.DataFrame()

    required = set(EMA_OVERLAP_COLUMNS)
    parts = []
    for pq in candidates:
        df = pd.read_parquet(pq)
        missing = required - set(df.columns)
        if missing:
            logger.warning(f"Skipping {pq.name}: missing columns {missing}")
            continue

        # Assign is_true_forecast based on filename
        is_backtest = pq.name.startswith("backtest_")
        df["is_true_forecast"] = 1 if is_backtest else 0
        if "source" not in df.columns:
            df["source"] = "backtest" if is_backtest else "hindcast"

        logger.info(
            f"Loaded {pq.name}: {len(df)} rows, "
            f"{df.index.min()} to {df.index.max()}, "
            f"source={df['source'].iloc[0]}"
        )
        parts.append(df)

    if not parts:
        logger.warning(
            f"No valid files with required columns in {path}. Required: {EMA_OVERLAP_COLUMNS}"
        )
        return pd.DataFrame()

    combined = pd.concat(parts)

    # Ensure UTC index
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    combined.index.name = "date_utc"

    combined = combined.sort_index()

    # Dedup: prefer backtest over hindcast for overlapping timestamps
    if combined.index.duplicated().any():
        # Sort so backtest (is_true_forecast=1) comes last, then keep last
        combined = combined.sort_values(["is_true_forecast"], kind="mergesort")
        combined = combined.sort_index(kind="mergesort")
        n_dups = combined.index.duplicated(keep="last").sum()
        logger.info(f"Removing {n_dups} duplicate timestamps (preferring backtest)")
        combined = combined[~combined.index.duplicated(keep="last")]

    # Clip negative values to zero on overlap columns
    combined[EMA_OVERLAP_COLUMNS] = combined[EMA_OVERLAP_COLUMNS].clip(lower=0)

    # Filter to overlap columns + metadata
    meta_cols = ["source", "is_true_forecast"]
    combined = combined[EMA_OVERLAP_COLUMNS + meta_cols]

    n_hindcast = (combined["is_true_forecast"] == 0).sum()
    n_backtest = (combined["is_true_forecast"] == 1).sum()
    logger.info(
        f"EMA historical forecasts: {len(combined)} hours "
        f"({n_hindcast} hindcast, {n_backtest} backtest), "
        f"{combined.index.min()} to {combined.index.max()}"
    )
    return combined


# =============================================================================
# Combined EMA data (all sources)
# =============================================================================


def _fetch_live_with_retry(max_retries: int = 3, delay_min: int = 15) -> pd.DataFrame | None:
    """Fetch live EMA forecast, retrying if it doesn't cover tomorrow.

    After fetching, checks whether the forecast's max timestamp extends beyond
    12 hours from now (i.e., covers tomorrow). If stale, waits and retries.

    Returns:
        EMA forecast DataFrame, or None if all attempts fail.
    """
    df = None
    for attempt in range(max_retries + 1):
        df = fetch_ema_forecasts()
        if df is not None and df.index.max() > pd.Timestamp.now("UTC") + pd.Timedelta(hours=12):
            return df  # covers tomorrow
        if attempt < max_retries:
            logger.warning(
                f"EMA forecast stale (max={df.index.max() if df is not None else 'N/A'}), "
                f"retrying in {delay_min}min ({attempt + 1}/{max_retries})"
            )
            time.sleep(delay_min * 60)
    logger.warning("EMA forecast still stale after retries, proceeding anyway")
    return df


def get_combined_ema_data(include_live: bool = False) -> pd.DataFrame:
    """Combine all EMA data sources into a single DataFrame.

    Priority (later overwrites earlier for duplicate timestamps):
    1. Historical hindcast/backtest (2022 to ~2026-03-06)
    2. Daily snapshots from data/ema/ (accumulated from previous runs)
    3. Live 168h forecast from EMA GitHub Pages (if include_live=True)

    Args:
        include_live: If True, fetch today's live forecast and save as snapshot.

    Returns:
        Combined UTC-indexed DataFrame with EMA_OVERLAP_COLUMNS, or empty
        DataFrame if no data is available.
    """
    parts: list[pd.DataFrame] = []

    # 1. Historical hindcast/backtest
    historical = load_ema_historical_forecasts()
    if not historical.empty:
        # Keep only overlap columns (drop source/is_true_forecast metadata for combining)
        hist_cols = [c for c in EMA_OVERLAP_COLUMNS if c in historical.columns]
        meta_cols = [c for c in ["source", "is_true_forecast"] if c in historical.columns]
        parts.append(historical[hist_cols + meta_cols])
        logger.info(f"Historical: {len(historical)} hours")

    # 2. Daily snapshots
    snapshots = load_ema_history()
    if not snapshots.empty:
        snap_cols = [c for c in EMA_OVERLAP_COLUMNS if c in snapshots.columns]
        if snap_cols:
            snap_df = snapshots[snap_cols].copy()
            snap_df["source"] = "snapshot"
            snap_df["is_true_forecast"] = 1  # snapshots use forecast weather
            parts.append(snap_df)
            logger.info(f"Snapshots: {len(snap_df)} hours")

    # 3. Live forecast
    if include_live:
        live = _fetch_live_with_retry()
        if live is not None:
            save_ema_snapshot(live)
            live_cols = [c for c in EMA_OVERLAP_COLUMNS if c in live.columns]
            if live_cols:
                live_df = live[live_cols].copy()
                live_df["source"] = "live"
                live_df["is_true_forecast"] = 1
                parts.append(live_df)
                logger.info(f"Live: {len(live_df)} hours")

    if not parts:
        logger.warning("No EMA data available from any source")
        return pd.DataFrame()

    combined = pd.concat(parts)

    # Ensure UTC index
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    combined.index.name = "date_utc"

    # Dedup: later sources (snapshot, live) overwrite earlier (historical)
    # Since parts are appended in priority order, keep last
    if combined.index.duplicated().any():
        n_dups = combined.index.duplicated(keep="last").sum()
        combined = combined[~combined.index.duplicated(keep="last")]
        logger.info(f"Deduped {n_dups} overlapping timestamps (live > snapshot > historical)")

    # Clip negative values
    overlap_present = [c for c in EMA_OVERLAP_COLUMNS if c in combined.columns]
    combined[overlap_present] = combined[overlap_present].clip(lower=0)

    logger.info(
        f"Combined EMA data: {len(combined)} hours, "
        f"{combined.index.min()} to {combined.index.max()}"
    )
    return combined
