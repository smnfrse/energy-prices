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

from loguru import logger
import pandas as pd

from src.config import EMA_DATA_DIR

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
