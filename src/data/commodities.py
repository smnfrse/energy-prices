"""
Commodity price data download and update functions.

Downloads commodity price data from:
- ICAP Carbon Action API (EU carbon allowances)
- Yahoo Finance (TTF gas futures, Brent oil futures)
- FRED (Federal Reserve Economic Data - gas prices for TTF gap reconstruction)

Update logic has been moved to src/sources.py (IcapSource, YahooSource, FredSource).
This module retains the download functions (API-specific parsing) and the CLI.
"""

import io
import os
from pathlib import Path
from typing import Optional

from fredapi import Fred
from loguru import logger
import pandas as pd
import requests
import typer
import yfinance as yf

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.config.commodities import (
    COLUMN_NAMES,
    FRED_SERIES,
    ICAP_PHASE_SPLIT_TIMESTAMP,
    ICAP_START_TIMESTAMP,
    ICAP_SYSTEMS,
    PRICE_RANGES,
    TICKERS,
)

app = typer.Typer()


def download_icap_carbon(
    system_id: int, start_date: int, end_date: int, output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Download carbon prices from ICAP Carbon Action API.

    Args:
        system_id: ICAP system ID (33 for 2014-2018, 35 for 2019+)
        start_date: Unix timestamp in milliseconds
        end_date: Unix timestamp in milliseconds
        output_path: Optional path to save CSV (if None, doesn't save)

    Returns:
        DataFrame with Date index and carbon price columns
    """
    url = "https://allowancepriceexplorer.icapcarbonaction.com/systems/reports/price/download"
    params = {"systemIds": system_id, "startDate": start_date, "endDate": end_date}

    logger.info(f"Downloading ICAP carbon data (systemId={system_id})...")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse CSV from response text
        df = pd.read_csv(io.StringIO(response.text), skiprows=1)

        # Clean up: remove unnamed columns and trailing spaces
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()

        # Parse date column
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # Select relevant columns
        columns_to_keep = {}
        if "Primary Market" in df.columns:
            columns_to_keep["Primary Market"] = "carbon_primary"
        if "Secondary Market" in df.columns:
            columns_to_keep["Secondary Market"] = "carbon_secondary"
        if "Exchange rate EUR/USD" in df.columns:
            columns_to_keep["Exchange rate EUR/USD"] = "eur_usd_rate"

        df = df[list(columns_to_keep.keys())].copy()
        df.columns = list(columns_to_keep.values())

        # Convert to numeric (handles empty strings as NaN)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            f"  Downloaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}"
        )

        if output_path:
            df.to_csv(output_path)
            logger.info(f"  Saved to {output_path}")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"  Error downloading ICAP data: {e}")
        raise
    except Exception as e:
        logger.error(f"  Error processing ICAP data: {e}")
        raise


def download_icap_full_history(output_dir: Path) -> pd.DataFrame:
    """
    Download complete carbon price history using both ICAP systemIds.

    Args:
        output_dir: Directory to save the combined CSV

    Returns:
        Combined DataFrame with complete history
    """
    output_path = output_dir / "carbon_icap_historical.csv"

    # Download Phase 3 (2014-2018)
    phase3 = download_icap_carbon(
        system_id=ICAP_SYSTEMS["eu_ets_phase3"],
        start_date=ICAP_START_TIMESTAMP,
        end_date=ICAP_PHASE_SPLIT_TIMESTAMP,
        output_path=None,
    )

    # Download Phase 4 (2019-present)
    end_timestamp = int(pd.Timestamp.now().timestamp() * 1000)
    phase4 = download_icap_carbon(
        system_id=ICAP_SYSTEMS["eu_ets_phase4"],
        start_date=ICAP_PHASE_SPLIT_TIMESTAMP,
        end_date=end_timestamp,
        output_path=None,
    )

    # Combine both phases
    combined = pd.concat([phase3, phase4]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Save combined dataset
    combined.to_csv(output_path)
    logger.info(f"Combined carbon data: {len(combined)} records")
    logger.info(f"  Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
    logger.info(f"  Saved to {output_path}")

    return combined


def download_yahoo_ticker(
    ticker: str, start_date: str, end_date: str, interval: str = "1d"
) -> pd.DataFrame:
    """
    Download commodity futures data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., 'TTF=F', 'BZ=F')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d' for daily, '1h' for hourly)

    Returns:
        DataFrame with date index and OHLCV columns
    """
    logger.info(f"Downloading {ticker} from Yahoo Finance...")

    try:
        data = yf.download(
            ticker, start=start_date, end=end_date, interval=interval, progress=False
        )

        if data.empty:
            logger.warning(f"  No data returned for {ticker}")
            return pd.DataFrame()

        # If multi-level columns (when multiple tickers), flatten
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Rename index to Date
        data.index.name = "Date"

        # Keep only Close price and Volume
        df = data[["Close", "Volume"]].copy()
        df.columns = ["price", "volume"]

        logger.info(
            f"  Downloaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}"
        )

        return df

    except Exception as e:
        logger.error(f"  Error downloading {ticker}: {e}")
        raise


def download_fred_series(
    series_id: str, start_date: str, end_date: str, api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Download economic data from FRED (Federal Reserve Economic Data).

    Args:
        series_id: FRED series identifier (e.g., 'PNGASEUUSDM', 'DHHNGSP')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_key: FRED API key (if None, reads from FRED_API_KEY env var)

    Returns:
        DataFrame with date index and 'value' column
    """
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY not found in environment variables")

    logger.info(f"Downloading FRED series {series_id}...")

    try:
        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

        df = pd.DataFrame({"value": series})
        df.index.name = "Date"

        logger.info(
            f"  Downloaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}"
        )

        return df

    except Exception as e:
        logger.error(f"  Error downloading FRED series {series_id}: {e}")
        raise


def download_fred_gas_data(output_dir: Path, start_date: str = "2014-01-01") -> None:
    """
    Download natural gas price data from FRED.

    Args:
        output_dir: Directory to save raw CSV files
        start_date: Start date in 'YYYY-MM-DD' format
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Download EU monthly gas prices
    eu_gas = download_fred_series(
        FRED_SERIES["eu_gas_monthly"], start_date=start_date, end_date=end_date
    )
    eu_gas_path = output_dir / "fred_eu_gas_monthly.csv"
    eu_gas.to_csv(eu_gas_path)
    logger.info(f"  Saved to {eu_gas_path}")

    # Download US Henry Hub daily prices
    us_gas = download_fred_series(
        FRED_SERIES["us_henryhub_daily"], start_date=start_date, end_date=end_date
    )
    us_gas_path = output_dir / "fred_us_henryhub_daily.csv"
    us_gas.to_csv(us_gas_path)
    logger.info(f"  Saved to {us_gas_path}")


def download_all_commodities(output_dir: Path, start_date: str = "2014-01-01") -> None:
    """
    Download all commodity price data from scratch.

    Args:
        output_dir: Directory to save raw CSV files
        start_date: Start date for Yahoo Finance data (YYYY-MM-DD)
    """
    from src.data.sources import FredSource, IcapSource

    output_dir.mkdir(parents=True, exist_ok=True)
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("DOWNLOADING COMMODITY PRICE DATA")
    logger.info("=" * 60)

    # 1. Carbon from ICAP (both phases)
    logger.info("\n[1/6] Carbon prices - ICAP historical...")
    IcapSource().download(output_dir / "carbon_icap_historical.csv")

    # 2. Carbon Real-time from Yahoo Finance (CO2.L)
    logger.info("\n[2/6] Carbon prices - Real-time (CO2.L ETC)...")
    carbon_rt = download_yahoo_ticker(
        TICKERS["carbon_realtime"], start_date=start_date, end_date=end_date
    )
    if not carbon_rt.empty:
        carbon_rt_path = output_dir / "carbon_realtime_daily.csv"
        carbon_rt.to_csv(carbon_rt_path)
        logger.info(f"  Saved to {carbon_rt_path}")

    # 3. TTF Gas from Yahoo Finance
    logger.info("\n[3/6] TTF gas futures (Yahoo Finance)...")
    ttf = download_yahoo_ticker(TICKERS["ttf"], start_date=start_date, end_date=end_date)
    if not ttf.empty:
        ttf_path = output_dir / "ttf_daily.csv"
        ttf.to_csv(ttf_path)
        logger.info(f"  Saved to {ttf_path}")

    # 4. Brent Oil from Yahoo Finance
    logger.info("\n[4/6] Brent crude oil futures (Yahoo Finance)...")
    brent = download_yahoo_ticker(TICKERS["brent"], start_date=start_date, end_date=end_date)
    if not brent.empty:
        brent_path = output_dir / "brent_daily.csv"
        brent.to_csv(brent_path)
        logger.info(f"  Saved to {brent_path}")

    # 5. FRED Gas Data (for TTF gap reconstruction)
    logger.info("\n[5/6] Natural gas prices - FRED (EU monthly + US daily)...")
    try:
        FredSource().download(output_dir, start_date)
    except Exception as e:
        logger.warning(f"  FRED download failed: {e}")
        logger.warning("  TTF gap reconstruction will not be available")

    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)


def update_all_commodities(data_dir: Path, redundancy_days: int = 14) -> None:
    """
    Update all commodity data sources with recent data.

    Uses DataSource classes for sources that have them, falls back to
    direct functions for others.

    Args:
        data_dir: Directory containing existing CSV files
        redundancy_days: Number of days to re-fetch (default 14)
    """
    from src.data.sources import FredSource, IcapSource, YahooSource

    logger.info("=" * 60)
    logger.info("UPDATING COMMODITY PRICE DATA")
    logger.info("=" * 60)

    # Update carbon (ICAP)
    logger.info("\n[1/6] Updating carbon prices (ICAP)...")
    carbon_path = data_dir / "carbon_icap_historical.csv"
    if carbon_path.exists():
        IcapSource().update(carbon_path, redundancy_days)
    else:
        logger.warning(f"  {carbon_path} not found, skipping")

    # Update carbon real-time (CO2.L)
    logger.info("\n[2/6] Updating carbon prices (CO2.L real-time)...")
    carbon_rt_path = data_dir / "carbon_realtime_daily.csv"
    if carbon_rt_path.exists():
        YahooSource(TICKERS["carbon_realtime"], "CO2.L").update(carbon_rt_path, redundancy_days)
    else:
        logger.warning(f"  {carbon_rt_path} not found, skipping")

    # Update TTF
    logger.info("\n[3/6] Updating TTF gas futures...")
    ttf_path = data_dir / "ttf_daily.csv"
    if ttf_path.exists():
        YahooSource(TICKERS["ttf"], "TTF").update(ttf_path, redundancy_days)
    else:
        logger.warning(f"  {ttf_path} not found, skipping")

    # Update Brent
    logger.info("\n[4/6] Updating Brent crude oil futures...")
    brent_path = data_dir / "brent_daily.csv"
    if brent_path.exists():
        YahooSource(TICKERS["brent"], "Brent").update(brent_path, redundancy_days)
    else:
        logger.warning(f"  {brent_path} not found, skipping")

    # Update FRED gas data
    logger.info("\n[5/6] Updating FRED gas data...")
    try:
        FredSource().update(data_dir, redundancy_days)
    except Exception as e:
        logger.warning(f"  FRED update failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("UPDATE COMPLETE")
    logger.info("=" * 60)


# ============================================================================
# CLI COMMANDS
# ============================================================================


@app.command()
def main(
    start_date: str = "2014-01-01",
    output_dir: Path = typer.Option(
        RAW_DATA_DIR / "prices", help="Output directory for raw CSV files"
    ),
):
    """Download all commodity price data from scratch."""
    download_all_commodities(output_dir, start_date)


@app.command()
def update(
    data_dir: Path = typer.Option(
        RAW_DATA_DIR / "prices", help="Directory with existing CSV files"
    ),
    redundancy_days: int = typer.Option(14, help="Number of days to re-fetch"),
):
    """Update existing commodity data with recent prices."""
    update_all_commodities(data_dir, redundancy_days)


def load_raw_commodities(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all raw commodity CSV files.

    Args:
        data_dir: Directory containing raw CSV files

    Returns:
        Dictionary with keys 'carbon', 'carbon_realtime', 'ttf', 'brent' mapping to DataFrames
    """
    data_dict = {}

    # Load carbon (ICAP)
    carbon_path = data_dir / "carbon_icap_historical.csv"
    if carbon_path.exists():
        logger.info(f"Loading carbon data from {carbon_path}")
        carbon = pd.read_csv(carbon_path, index_col="Date", parse_dates=True)
        carbon.index = pd.to_datetime(carbon.index, utc=True)
        carbon = carbon[["carbon_primary"]].copy()
        carbon.columns = [COLUMN_NAMES["carbon_price"]]
        data_dict["carbon"] = carbon
        logger.info(
            f"  Carbon (ICAP): {len(carbon)} records, {carbon.index[0].date()} to {carbon.index[-1].date()}"
        )
    else:
        logger.warning(f"  {carbon_path} not found")

    # Load carbon real-time (CO2.L)
    carbon_rt_path = data_dir / "carbon_realtime_daily.csv"
    if carbon_rt_path.exists():
        logger.info(f"Loading carbon real-time data from {carbon_rt_path}")
        carbon_rt = pd.read_csv(carbon_rt_path, index_col="Date", parse_dates=True)
        carbon_rt.index = pd.to_datetime(carbon_rt.index, utc=True)
        carbon_rt = carbon_rt[["price"]].copy()
        carbon_rt.columns = [COLUMN_NAMES["carbon_realtime_price"]]
        data_dict["carbon_realtime"] = carbon_rt
        logger.info(
            f"  Carbon (CO2.L): {len(carbon_rt)} records, {carbon_rt.index[0].date()} to {carbon_rt.index[-1].date()}"
        )
    else:
        logger.warning(f"  {carbon_rt_path} not found")

    # Load TTF (Yahoo Finance)
    ttf_path = data_dir / "ttf_daily.csv"
    if ttf_path.exists():
        logger.info(f"Loading TTF data from {ttf_path}")
        ttf = pd.read_csv(ttf_path, index_col="Date", parse_dates=True)
        ttf.index = pd.to_datetime(ttf.index, utc=True)
        ttf = ttf[["price"]].copy()
        ttf.columns = [COLUMN_NAMES["ttf_price"]]
        data_dict["ttf"] = ttf
        logger.info(f"  TTF: {len(ttf)} records, {ttf.index[0].date()} to {ttf.index[-1].date()}")
    else:
        logger.warning(f"  {ttf_path} not found")

    # Load Brent (Yahoo Finance)
    brent_path = data_dir / "brent_daily.csv"
    if brent_path.exists():
        logger.info(f"Loading Brent data from {brent_path}")
        brent = pd.read_csv(brent_path, index_col="Date", parse_dates=True)
        brent.index = pd.to_datetime(brent.index, utc=True)
        brent = brent[["price"]].copy()
        brent.columns = [COLUMN_NAMES["brent_price"]]
        data_dict["brent"] = brent
        logger.info(
            f"  Brent: {len(brent)} records, {brent.index[0].date()} to {brent.index[-1].date()}"
        )
    else:
        logger.warning(f"  {brent_path} not found")

    # Load FRED EU gas (monthly)
    fred_eu_path = data_dir / "fred_eu_gas_monthly.csv"
    if fred_eu_path.exists():
        logger.info(f"Loading FRED EU gas data from {fred_eu_path}")
        fred_eu = pd.read_csv(fred_eu_path, index_col="Date", parse_dates=True)
        fred_eu.index = pd.to_datetime(fred_eu.index, utc=True)
        data_dict["fred_eu_gas"] = fred_eu
        logger.info(
            f"  FRED EU gas: {len(fred_eu)} records, {fred_eu.index[0].date()} to {fred_eu.index[-1].date()}"
        )
    else:
        logger.info(f"  {fred_eu_path} not found (TTF gap reconstruction unavailable)")

    # Load FRED US Henry Hub (daily)
    fred_us_path = data_dir / "fred_us_henryhub_daily.csv"
    if fred_us_path.exists():
        logger.info(f"Loading FRED US Henry Hub data from {fred_us_path}")
        fred_us = pd.read_csv(fred_us_path, index_col="Date", parse_dates=True)
        fred_us.index = pd.to_datetime(fred_us.index, utc=True)
        data_dict["fred_us_gas"] = fred_us
        logger.info(
            f"  FRED US gas: {len(fred_us)} records, {fred_us.index[0].date()} to {fred_us.index[-1].date()}"
        )
    else:
        logger.info(f"  {fred_us_path} not found (TTF gap reconstruction unavailable)")

    # Load ICAP EUR/USD exchange rate (from carbon data)
    if carbon_path.exists():
        carbon_full = pd.read_csv(carbon_path, index_col="Date", parse_dates=True)
        carbon_full.index = pd.to_datetime(carbon_full.index, utc=True)
        if "eur_usd_rate" in carbon_full.columns:
            eur_usd = carbon_full[["eur_usd_rate"]].copy()
            data_dict["eur_usd_rate"] = eur_usd
            logger.info(f"  EUR/USD rate: {len(eur_usd)} records from carbon data")

    return data_dict


def reconstruct_ttf_gap(
    ttf_yahoo: pd.DataFrame,
    fred_eu_monthly: pd.DataFrame,
    fred_us_daily: pd.DataFrame,
    eur_usd_rate: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct TTF natural gas prices for the Dec 2014 - Oct 2017 gap period.

    Uses a three-step methodology:
    1. Adjust EU monthly prices using bias correction from overlap period
    2. Center US daily prices (subtract monthly mean)
    3. Combine: adjusted_EU_monthly + centered_US_daily

    All conversions: USD/MMBtu -> EUR/MMBtu -> EUR/MWh

    Args:
        ttf_yahoo: Yahoo Finance TTF data (Oct 2017+)
        fred_eu_monthly: FRED EU gas import price (monthly, USD/MMBtu)
        fred_us_daily: FRED US Henry Hub price (daily, USD/MMBtu)
        eur_usd_rate: EUR/USD exchange rate from ICAP carbon data

    Returns:
        DataFrame with date index and 'ttf_eur_per_mwh' column (gap filled)
    """
    from src.config.commodities import TTF_GAP_OVERLAP_START, UNIT_CONVERSION_MMBTU_TO_MWH

    logger.info("Reconstructing TTF gap (Dec 2014 - Oct 2017)...")

    # Prepare Yahoo TTF (has column name from COLUMN_NAMES)
    ttf_col = COLUMN_NAMES["ttf_price"]
    if ttf_col not in ttf_yahoo.columns:
        logger.error("TTF Yahoo data missing expected column")
        return ttf_yahoo

    yahoo_ttf = ttf_yahoo[ttf_col].copy()

    # Step 0: Convert FRED data to EUR/MWh
    logger.info("  [1/4] Converting FRED data to EUR/MWh...")

    # Prepare EUR/USD exchange rate (forward-fill to daily)
    eur_usd_daily = eur_usd_rate["eur_usd_rate"].copy()

    # Convert EU monthly: USD/MMBtu -> EUR/MWh
    eu_gas_usd_monthly = fred_eu_monthly["value"].copy()
    eu_gas_usd_daily = eu_gas_usd_monthly.resample("D").ffill()
    eu_eur_usd = eur_usd_daily.reindex(eu_gas_usd_daily.index, method="ffill")
    eu_gas_eur_mwh = (eu_gas_usd_daily / eu_eur_usd) * UNIT_CONVERSION_MMBTU_TO_MWH

    # Convert US daily: USD/MMBtu -> EUR/MWh
    us_gas_usd = fred_us_daily["value"].copy()
    us_eur_usd = eur_usd_daily.reindex(us_gas_usd.index, method="ffill")
    us_gas_eur_mwh = (us_gas_usd / us_eur_usd) * UNIT_CONVERSION_MMBTU_TO_MWH

    logger.info(f"  EU gas: {len(eu_gas_eur_mwh)} values (monthly -> daily forward-filled)")
    logger.info(f"  US gas: {len(us_gas_eur_mwh)} values (daily)")

    # Step 1: Calculate bias correction from overlap period
    logger.info("  [2/4] Calculating bias correction from overlap...")

    overlap_start = pd.Timestamp(TTF_GAP_OVERLAP_START, tz="UTC")

    eu_overlap = eu_gas_eur_mwh[eu_gas_eur_mwh.index >= overlap_start]
    yahoo_overlap = yahoo_ttf[yahoo_ttf.index >= overlap_start]

    common_dates = eu_overlap.index.intersection(yahoo_overlap.index)
    if len(common_dates) == 0:
        logger.warning("No overlap dates found, using EU gas without correction")
        bias_correction = 0.0
    else:
        eu_overlap_aligned = eu_overlap.loc[common_dates]
        yahoo_overlap_aligned = yahoo_overlap.loc[common_dates]

        bias_correction = (yahoo_overlap_aligned - eu_overlap_aligned).mean()
        correlation = yahoo_overlap_aligned.corr(eu_overlap_aligned)

        logger.info(f"  Overlap: {len(common_dates)} daily observations")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  Bias correction: {bias_correction:.2f} EUR/MWh")

    eu_gas_adjusted = eu_gas_eur_mwh + bias_correction

    # Step 2: Center US daily prices (extract variation patterns)
    logger.info("  [3/4] Centering US daily prices...")

    us_monthly_mean = us_gas_eur_mwh.resample("MS").transform("mean")
    us_gas_centered = us_gas_eur_mwh - us_monthly_mean

    logger.info(
        f"  US centered: mean={us_gas_centered.mean():.4f}, std={us_gas_centered.std():.2f}"
    )

    # Step 3: Combine sources
    logger.info("  [4/4] Combining EU baseline + US variation...")

    eu_gas_adjusted_daily = eu_gas_adjusted.resample("D").ffill()

    combined_index = eu_gas_adjusted_daily.index.union(us_gas_centered.index)
    eu_baseline = eu_gas_adjusted_daily.reindex(combined_index, method="ffill")
    us_variation = us_gas_centered.reindex(combined_index, fill_value=0)

    reconstructed = eu_baseline + us_variation

    # Step 4: Merge reconstructed with Yahoo TTF
    ttf_unified = yahoo_ttf.copy()

    gap_start = reconstructed.index.min()
    gap_end = yahoo_ttf.index.min() - pd.Timedelta(days=1)

    gap_mask = (reconstructed.index >= gap_start) & (reconstructed.index <= gap_end)
    gap_data = reconstructed[gap_mask]

    ttf_unified = pd.concat([gap_data, ttf_unified]).sort_index()
    ttf_unified = ttf_unified[~ttf_unified.index.duplicated(keep="last")]

    logger.info(f"  Gap filled: {len(gap_data)} days ({gap_start.date()} to {gap_end.date()})")
    logger.info(
        f"  Reconstructed range: {reconstructed.min():.2f} - {reconstructed.max():.2f} EUR/MWh"
    )
    logger.info(f"  Total TTF data: {ttf_unified.notna().sum()} days")

    # Validate reconstructed prices
    ttf_min, ttf_max = PRICE_RANGES["ttf"]
    outside_range = reconstructed[(reconstructed < ttf_min) | (reconstructed > ttf_max)]
    if len(outside_range) > 0:
        logger.warning(
            f"  {len(outside_range)} reconstructed values outside expected range "
            f"EUR {ttf_min}-{ttf_max}"
        )

    result = pd.DataFrame({ttf_col: ttf_unified})
    return result


def combine_commodities_daily(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all commodity sources into a single daily dataframe.

    Uses outer join to preserve all dates from all sources.
    Combines ICAP carbon (historical) with CO2.L (real-time) to create unified carbon column.
    Forward-fills weekend/holiday gaps within valid data range for each commodity.

    Args:
        data_dict: Dictionary of DataFrames from load_raw_commodities()

    Returns:
        Combined daily DataFrame with columns: carbon_eur_per_ton, ttf_eur_per_mwh, brent_usd_per_barrel
    """
    logger.info("Combining commodity sources...")

    # Check if TTF gap reconstruction is available
    fred_available = (
        "fred_eu_gas" in data_dict
        and "fred_us_gas" in data_dict
        and "eur_usd_rate" in data_dict
        and "ttf" in data_dict
    )

    if fred_available:
        logger.info("  FRED data available - reconstructing TTF gap...")
        reconstructed_ttf = reconstruct_ttf_gap(
            ttf_yahoo=data_dict["ttf"],
            fred_eu_monthly=data_dict["fred_eu_gas"],
            fred_us_daily=data_dict["fred_us_gas"],
            eur_usd_rate=data_dict["eur_usd_rate"],
        )
        data_dict["ttf"] = reconstructed_ttf

    # Start with empty DataFrame
    combined = pd.DataFrame()

    # Join all sources on date index (outer join)
    skip_keys = {"fred_eu_gas", "fred_us_gas", "eur_usd_rate"}
    for source_name, df in data_dict.items():
        if source_name in skip_keys:
            continue
        if combined.empty:
            combined = df
        else:
            combined = combined.join(df, how="outer")

    combined = combined.sort_index()

    # Create unified carbon column: Use ICAP where available, CO2.L otherwise
    carbon_col = COLUMN_NAMES["carbon_price"]
    carbon_rt_col = COLUMN_NAMES["carbon_realtime_price"]

    if carbon_col in combined.columns and carbon_rt_col in combined.columns:
        logger.info("Creating unified carbon column (ICAP + CO2.L)...")

        overlap_mask = combined[carbon_col].notna() & combined[carbon_rt_col].notna()
        overlap_data = combined[overlap_mask]

        if len(overlap_data) > 0:
            bias_correction = (overlap_data[carbon_col] - overlap_data[carbon_rt_col]).mean()
            correlation = overlap_data[carbon_col].corr(overlap_data[carbon_rt_col])

            logger.info(f"  Overlap period: {len(overlap_data)} days")
            logger.info(f"  Correlation: {correlation:.4f}")
            logger.info(f"  Bias correction (ICAP - CO2.L): {bias_correction:.2f} EUR/ton")
        else:
            bias_correction = 0.0
            logger.warning("  No overlap between ICAP and CO2.L - using CO2.L without correction")

        unified_carbon = combined[carbon_col].copy()

        missing_mask = unified_carbon.isna()
        co2l_corrected = combined.loc[missing_mask, carbon_rt_col] + bias_correction
        unified_carbon[missing_mask] = co2l_corrected

        icap_count = combined[carbon_col].notna().sum()
        co2l_count = missing_mask.sum() - (
            unified_carbon.isna().sum() - combined[carbon_col].isna().sum()
        )
        logger.info(f"  ICAP data points: {icap_count}")
        logger.info(f"  CO2.L data points used (bias-corrected): {co2l_count}")
        logger.info(f"  Total unified: {unified_carbon.notna().sum()}")

        combined[carbon_col] = unified_carbon
        combined = combined.drop(columns=[carbon_rt_col])

    # Ensure all expected columns exist
    expected_cols = [
        COLUMN_NAMES["carbon_price"],
        COLUMN_NAMES["ttf_price"],
        COLUMN_NAMES["brent_price"],
    ]

    for col in expected_cols:
        if col not in combined.columns:
            logger.warning(f"  {col} not found in data, adding as NaN column")
            combined[col] = pd.NA

    combined = combined[expected_cols]

    logger.info(
        f"Combined: {len(combined)} records, {combined.index[0].date()} to {combined.index[-1].date()}"
    )
    logger.info(f"  Columns: {list(combined.columns)}")

    # Report missing value counts BEFORE forward-fill
    logger.info("Missing values (before forward-fill):")
    for col in combined.columns:
        missing_count = combined[col].isna().sum()
        missing_pct = 100 * missing_count / len(combined)
        logger.info(f"  {col}: {missing_count} ({missing_pct:.1f}%)")

    # Forward-fill weekend/holiday gaps (within valid data range only)
    logger.info("Forward-filling weekend/holiday gaps...")
    for col in combined.columns:
        first_valid_idx = combined[col].first_valid_index()
        last_valid_idx = combined[col].last_valid_index()

        if first_valid_idx is None or last_valid_idx is None:
            logger.info(f"  {col}: No valid data to forward-fill")
            continue

        mask = (combined.index >= first_valid_idx) & (combined.index <= last_valid_idx)
        combined.loc[mask, col] = combined.loc[mask, col].ffill()

        filled_count = mask.sum() - combined.loc[mask, col].isna().sum()
        logger.info(
            f"  {col}: Filled {filled_count} values within {first_valid_idx.date()} to {last_valid_idx.date()}"
        )

    # Report missing value counts AFTER forward-fill
    logger.info("Missing values (after forward-fill):")
    for col in combined.columns:
        missing_count = combined[col].isna().sum()
        missing_pct = 100 * missing_count / len(combined)
        logger.info(f"  {col}: {missing_count} ({missing_pct:.1f}%)")

    return combined


def validate_price_ranges(df: pd.DataFrame) -> None:
    """
    Validate that prices fall within expected ranges.

    Args:
        df: DataFrame with commodity prices
    """
    from src.config.commodities import PRICE_RANGES

    logger.info("Validating price ranges...")

    if COLUMN_NAMES["carbon_price"] in df.columns:
        carbon_col = COLUMN_NAMES["carbon_price"]
        carbon_min, carbon_max = PRICE_RANGES["carbon"]
        outside_range = df[(df[carbon_col] < carbon_min) | (df[carbon_col] > carbon_max)][
            carbon_col
        ].dropna()

        if len(outside_range) > 0:
            logger.warning(
                f"  {len(outside_range)} carbon prices outside expected range "
                f"EUR {carbon_min}-{carbon_max}"
            )
            logger.warning(
                f"  Range found: EUR {outside_range.min():.2f} - {outside_range.max():.2f}"
            )
        else:
            logger.info(f"  Carbon prices: OK (EUR {carbon_min}-{carbon_max})")

    if COLUMN_NAMES["ttf_price"] in df.columns:
        ttf_col = COLUMN_NAMES["ttf_price"]
        ttf_min, ttf_max = PRICE_RANGES["ttf"]
        outside_range = df[(df[ttf_col] < ttf_min) | (df[ttf_col] > ttf_max)][ttf_col].dropna()

        if len(outside_range) > 0:
            logger.warning(
                f"  {len(outside_range)} TTF prices outside expected range EUR {ttf_min}-{ttf_max}"
            )
            logger.warning(
                f"  Range found: EUR {outside_range.min():.2f} - {outside_range.max():.2f}"
            )
        else:
            logger.info(f"  TTF prices: OK (EUR {ttf_min}-{ttf_max})")

    if COLUMN_NAMES["brent_price"] in df.columns:
        brent_col = COLUMN_NAMES["brent_price"]
        brent_min, brent_max = PRICE_RANGES["brent"]
        outside_range = df[(df[brent_col] < brent_min) | (df[brent_col] > brent_max)][
            brent_col
        ].dropna()

        if len(outside_range) > 0:
            logger.warning(
                f"  {len(outside_range)} Brent prices outside expected range "
                f"USD {brent_min}-{brent_max}"
            )
            logger.warning(
                f"  Range found: USD {outside_range.min():.2f} - {outside_range.max():.2f}"
            )
        else:
            logger.info(f"  Brent prices: OK (USD {brent_min}-{brent_max})")


def forward_fill_to_hourly(
    daily_df: pd.DataFrame, smard_timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Forward-fill daily commodity prices to hourly frequency.

    Only forward-fills up to yesterday to avoid data leakage - we shouldn't
    assume today's commodity prices equal yesterday's close.

    Args:
        daily_df: DataFrame with daily commodity prices (date index)
        smard_timestamps: Hourly timestamps from SMARD dataset

    Returns:
        DataFrame with hourly frequency, matching SMARD timestamps exactly
    """
    logger.info("Forward-filling to hourly frequency...")
    logger.info(f"  Input: {len(daily_df)} daily observations")
    logger.info(f"  Target: {len(smard_timestamps)} hourly observations")

    # Only forward-fill up to yesterday to avoid data leakage
    now = pd.Timestamp.now(tz="UTC").normalize()
    valid_timestamps = smard_timestamps[smard_timestamps.normalize() < now]

    logger.info(
        f"  Filtering to timestamps before today: {len(valid_timestamps)} / {len(smard_timestamps)}"
    )

    smard_dates = valid_timestamps.normalize()
    hourly_df = daily_df.reindex(smard_dates, method="ffill")
    hourly_df.index = valid_timestamps

    logger.info(f"  Output: {len(hourly_df)} hourly observations")
    logger.info(f"  Timestamp range: {hourly_df.index[0]} to {hourly_df.index[-1]}")

    logger.info("Missing values (hourly):")
    for col in hourly_df.columns:
        missing_count = hourly_df[col].isna().sum()
        missing_pct = 100 * missing_count / len(hourly_df)
        logger.info(f"  {col}: {missing_count} ({missing_pct:.1f}%)")

    return hourly_df


def process_commodities(raw_dir: Path, output_dir: Path, smard_path: Path) -> None:
    """
    Full commodity processing pipeline.

    Args:
        raw_dir: Directory with raw CSV files
        output_dir: Directory to save processed parquet files
        smard_path: Path to SMARD hourly dataset (for timestamps)
    """
    logger.info("=" * 60)
    logger.info("PROCESSING COMMODITY PRICE DATA")
    logger.info("=" * 60)

    # 1. Load raw data
    logger.info("[1/6] Loading raw data...")
    data_dict = load_raw_commodities(raw_dir)

    if not data_dict:
        logger.error("No commodity data found")
        raise typer.Exit(code=1)

    # 2. Combine to daily
    logger.info("[2/6] Combining sources...")
    daily_df = combine_commodities_daily(data_dict)

    # 3. Validate ranges
    logger.info("[3/6] Validating data...")
    validate_price_ranges(daily_df)

    # 4. Save daily dataframe
    logger.info("[4/6] Saving daily data...")
    daily_path = output_dir / "commodity_prices_daily.parquet"
    daily_df.to_parquet(daily_path)
    logger.info(f"  Saved: {daily_path}")
    logger.info(f"  Shape: {daily_df.shape}")

    # 5. Load SMARD timestamps
    logger.info("[5/6] Loading SMARD hourly timestamps...")
    if not smard_path.exists():
        logger.error(f"SMARD hourly dataset not found at {smard_path}")
        raise typer.Exit(code=1)

    smard = pd.read_parquet(smard_path)
    smard_timestamps = smard.index
    logger.info(f"  SMARD shape: {smard.shape}")
    logger.info(f"  SMARD timestamps: {smard_timestamps[0]} to {smard_timestamps[-1]}")

    # 6. Forward-fill to hourly
    logger.info("[6/6] Forward-filling to hourly...")
    hourly_df = forward_fill_to_hourly(daily_df, smard_timestamps)

    # 7. Save hourly dataframe
    hourly_path = output_dir / "commodity_prices_hourly.parquet"
    hourly_df.to_parquet(hourly_path)
    logger.info(f"Saved: {hourly_path}")
    logger.info(f"Shape: {hourly_df.shape}")

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Daily data:  {daily_path}")
    logger.info(f"Hourly data: {hourly_path}")
    logger.info("Hourly data ready to merge with SMARD dataset during feature engineering.")


@app.command()
def process(
    raw_dir: Path = typer.Option(RAW_DATA_DIR / "prices", help="Raw data directory"),
    output_dir: Path = typer.Option(INTERIM_DATA_DIR, help="Output directory for processed data"),
    smard_path: Path = typer.Option(
        INTERIM_DATA_DIR / "merged_dataset_hourly.parquet", help="SMARD hourly dataset path"
    ),
):
    """Process raw commodity data and forward-fill to hourly frequency."""
    process_commodities(raw_dir, output_dir, smard_path)


if __name__ == "__main__":
    app()
