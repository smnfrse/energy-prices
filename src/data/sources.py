"""Data source abstraction with shared update logic.

Deduplicates the update-with-redundancy pattern that appears across SMARD, ICAP,
Yahoo Finance, and FRED data sources.

Each concrete DataSource knows how to download and incrementally update its data.
The base class provides the generic update flow: load existing → calc overlap →
fetch new → trim → concat → save.
"""

from abc import ABC, abstractmethod
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
import pandas as pd

from src.config import RAW_DATA_DIR
from src.config.smard import (
    get_filter_dict_for_region,
    resolution_periods,
)
from src.data.smard import (
    DataNotAvailableError,
    _get_filename_for_key,
    get_all_data,
    get_timestamps,
)


class DataSource(ABC):
    """Base class for all data sources with shared update logic."""

    index_col: str = "Date"

    @abstractmethod
    def download(self, output_path: Path, **kwargs) -> None:
        """Download full history and save to output_path."""

    def update(self, csv_path: Path, redundancy_days: int = 14) -> pd.DataFrame:
        """Generic update: load existing -> calc overlap -> fetch new -> trim -> concat -> save.

        Args:
            csv_path: Path to existing CSV file.
            redundancy_days: Number of days to re-fetch for corrections.

        Returns:
            Updated DataFrame.
        """
        existing = pd.read_csv(csv_path, index_col=self.index_col, parse_dates=True)
        last_date = existing.index.max()
        logger.info(f"Current data ends: {last_date.date()}")

        redundancy_start = last_date - pd.Timedelta(days=redundancy_days)
        new_data = self._fetch_update(redundancy_start)

        if new_data.empty:
            logger.info("No new data available")
            return existing

        # Find overlap and trim
        overlap_mask = existing.index >= redundancy_start
        if overlap_mask.any():
            overlap_idx = existing.index.get_loc(existing.index[overlap_mask][0])
            trimmed = existing.iloc[:overlap_idx]
        else:
            trimmed = existing

        combined = pd.concat([trimmed, new_data]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_csv(csv_path)

        new_records = len(combined) - len(existing)
        logger.info(f"Added {new_records} new records, ends {combined.index.max().date()}")
        return combined

    @abstractmethod
    def _fetch_update(self, start_date) -> pd.DataFrame:
        """Source-specific fetch for update window. Returns DataFrame with DatetimeIndex."""


# =============================================================================
# SMARD API Source
# =============================================================================


class SmardSource(DataSource):
    """Data source for SMARD API (German energy market data).

    Handles downloading and updating multiple measure keys from the SMARD API.
    Overrides the base update() since SMARD uses bisect-based timestamp logic
    rather than date-based overlap.
    """

    def __init__(self, region: str = "DE-LU", resolution: str = "quarterhour"):
        self.region = region
        self.resolution = resolution

    def download(
        self,
        output_dir: Path,
        keys: list[int] | None = None,
        start_date: pd.Timestamp | None = None,
        max_workers: int = 10,
        **kwargs,
    ) -> None:
        """Download full history (or partial from start_date) for all keys.

        Args:
            output_dir: Directory to save CSV files.
            keys: List of filter keys. If None, uses all keys for the region.
            start_date: If provided, only download data from this date onwards.
                Used for bootstrapping missing CSVs with recent data.
            max_workers: Number of parallel download threads.
        """
        if keys is None:
            source_dict = get_filter_dict_for_region(self.region)
            keys = list(source_dict.keys())

        output_dir.mkdir(parents=True, exist_ok=True)

        def _download_key(key):
            filename = _get_filename_for_key(key, self.region)
            output_path = Path(output_dir / f"{filename}.csv")

            if output_path.exists():
                logger.info(f"{filename} has already been created")
                return

            try:
                if start_date is not None:
                    all_stamps = get_timestamps(key, self.region, self.resolution)
                    cutoff_ms = int(start_date.timestamp() * 1000)
                    filtered = [t for t in all_stamps.timestamps.to_list() if t >= cutoff_ms]
                    if not filtered:
                        logger.info(f"No recent data for {filename}, skipping")
                        return
                    logger.info(
                        f"Bootstrapping {filename} from {start_date.date()}"
                        f" ({len(filtered)} timestamps)"
                    )
                    df = get_all_data(
                        key,
                        region=self.region,
                        resolution=self.resolution,
                        timestamp_list=filtered,
                    )
                else:
                    logger.info(f"Attempting to download data for {filename}")
                    df = get_all_data(key, region=self.region, resolution=self.resolution)
                if df.empty:
                    logger.info(f"No data returned for {filename}, skipping")
                    return
                df.to_csv(output_path, index=False)
                logger.success(f"{filename}.csv successfully saved to {output_dir}")
            except DataNotAvailableError:
                logger.warning(f"{filename} not available for region {self.region}, skipping")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_key, key): key for key in keys}
            for future in as_completed(futures):
                future.result()  # propagate exceptions

    def update(
        self, output_dir: Path, keys: list[int] | None = None, redundancy_days: int = 14
    ) -> None:
        """Update existing data for all keys.

        Args:
            output_dir: Directory containing CSV files.
            keys: List of filter keys. If None, uses all keys for the region.
            redundancy_days: Days to re-download for data correction.
        """
        if keys is None:
            source_dict = get_filter_dict_for_region(self.region)
            keys = list(source_dict.keys())

        # Separate missing keys (need bootstrap) from existing keys (incremental update)
        bootstrap_keys = []
        update_keys = []
        for key in keys:
            filename = _get_filename_for_key(key, self.region)
            output_path = Path(output_dir / f"{filename}.csv")
            if output_path.exists():
                update_keys.append(key)
            else:
                bootstrap_keys.append(key)

        # Bootstrap missing keys in one parallel batch
        if bootstrap_keys:
            bootstrap_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45)
            logger.info(
                f"Bootstrapping {len(bootstrap_keys)} missing keys (last 45 days, parallel)"
            )
            self.download(output_dir, keys=bootstrap_keys, start_date=bootstrap_start)

        # Incremental update for existing keys
        for key in update_keys:
            filename = _get_filename_for_key(key, self.region)
            output_path = Path(output_dir / f"{filename}.csv")

            try:
                update_stamps = self._get_update_timestamps(
                    key, redundancy_days=redundancy_days, output_dir=output_dir
                )
            except IndexError:
                logger.debug(f"No updates found for {filename}.csv")
                continue
            except DataNotAvailableError:
                logger.warning(f"{filename} not available for region {self.region}, skipping")
                continue

            update_df = get_all_data(
                key, region=self.region, resolution=self.resolution, timestamp_list=update_stamps
            )
            existing_df = pd.read_csv(output_path)

            logger.info(f"Updating {filename}.csv with {len(update_df)} new timestamps.")

            first_update_ts = update_df["timestamp"].min()
            existing_df_trimmed = existing_df[existing_df["timestamp"] < first_update_ts]
            result_df = pd.concat([existing_df_trimmed, update_df], ignore_index=True)

            result_df.to_csv(output_path, index=False)
            logger.success(f"{filename}.csv successfully updated.")

    def _get_update_timestamps(self, filter_key, redundancy_days=14, output_dir=None):
        """Determine which timestamps need updating with redundancy window.

        Only considers timestamps with valid data to avoid issues with future
        NaN-filled timestamps from forecasted data.
        """
        if output_dir is None:
            output_dir = RAW_DATA_DIR / "smard_api"

        filename = _get_filename_for_key(filter_key, self.region)
        output_path = Path(output_dir / f"{filename}.csv")
        check_df = pd.read_csv(output_path)

        if self.resolution not in resolution_periods:
            raise ValueError(
                f"Invalid resolution: {self.resolution}. Must be one of {list(resolution_periods.keys())}"
            )

        periods_per_day = resolution_periods[self.resolution]
        redundancy_periods = int(redundancy_days * periods_per_day)

        # Filter to only rows with valid data (avoid future NaN-filled timestamps)
        check_df_valid = check_df[check_df["value"].notna()]

        if len(check_df_valid) >= redundancy_periods:
            last_stamp = check_df_valid.timestamp.iloc[-redundancy_periods]
        else:
            last_stamp = check_df_valid.timestamp.iloc[0] if len(check_df_valid) > 0 else None

        available_stamps = get_timestamps(filter_key, self.region, self.resolution)
        stamps_list = available_stamps.timestamps.to_list()

        idx = bisect.bisect_left(stamps_list, last_stamp)
        return stamps_list[idx - 1 :]

    def _fetch_update(self, start_date) -> pd.DataFrame:
        """Not used for SmardSource (overrides update() directly)."""
        raise NotImplementedError("SmardSource uses custom update() logic")


# =============================================================================
# ICAP Carbon Source
# =============================================================================


class IcapSource(DataSource):
    """Data source for ICAP Carbon Action API (EU carbon allowances)."""

    def __init__(self):
        from src.config.commodities import ICAP_SYSTEMS

        self.system_id = ICAP_SYSTEMS["eu_ets_phase4"]

    def download(self, output_path: Path, **kwargs) -> None:
        """Download full history (delegates to commodities module)."""
        from src.data.commodities import download_icap_full_history

        download_icap_full_history(output_path.parent)

    def _fetch_update(self, start_date) -> pd.DataFrame:
        """Fetch recent carbon data from ICAP Phase 4."""
        from src.data.commodities import download_icap_carbon

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(pd.Timestamp.now().timestamp() * 1000)

        return download_icap_carbon(
            system_id=self.system_id,
            start_date=start_ms,
            end_date=end_ms,
            output_path=None,
        )


# =============================================================================
# Yahoo Finance Source
# =============================================================================


class YahooSource(DataSource):
    """Data source for Yahoo Finance commodity futures."""

    def __init__(self, ticker: str, name: str):
        self.ticker = ticker
        self.name = name

    def download(self, output_path: Path, start_date: str = "2014-01-01", **kwargs) -> None:
        """Download full history from Yahoo Finance."""
        from src.data.commodities import download_yahoo_ticker

        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = download_yahoo_ticker(self.ticker, start_date=start_date, end_date=end_date)
        if not df.empty:
            df.to_csv(output_path)
            logger.info(f"Saved {self.name} to {output_path}")

    def _fetch_update(self, start_date) -> pd.DataFrame:
        """Fetch recent data from Yahoo Finance."""
        from src.data.commodities import download_yahoo_ticker

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        return download_yahoo_ticker(self.ticker, start_date=start_str, end_date=end_str)


# =============================================================================
# FRED Source
# =============================================================================


class FredSource(DataSource):
    """Data source for FRED (Federal Reserve Economic Data).

    Overrides update() for the two-series download pattern (EU monthly + US daily).
    """

    def download(self, output_dir: Path, start_date: str = "2014-01-01", **kwargs) -> None:
        """Download FRED gas data (EU monthly + US daily)."""
        from src.data.commodities import download_fred_gas_data

        download_fred_gas_data(output_dir, start_date)

    def update(self, data_dir: Path, redundancy_days: int = 14) -> pd.DataFrame:
        """Update FRED gas data with recent observations.

        Args:
            data_dir: Directory containing existing CSV files.
            redundancy_days: Number of days to re-fetch.

        Returns:
            Updated DataFrame (last updated series).
        """
        from src.config.commodities import FRED_SERIES
        from src.data.commodities import download_fred_series

        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        last_combined = pd.DataFrame()

        # Update EU monthly gas prices
        logger.info("Updating FRED EU gas prices (monthly)...")
        eu_gas_path = data_dir / "fred_eu_gas_monthly.csv"
        if eu_gas_path.exists():
            existing = pd.read_csv(eu_gas_path, index_col="Date", parse_dates=True)
            last_date = existing.index.max()
            logger.info(f"  Current data ends: {last_date.date()}")

            # For monthly data, go back 2 months to catch revisions
            redundancy_start = last_date - pd.Timedelta(days=60)
            start_date = redundancy_start.strftime("%Y-%m-%d")

            new_data = download_fred_series(FRED_SERIES["eu_gas_monthly"], start_date, end_date)

            overlap_dates = existing.index >= redundancy_start
            if overlap_dates.any():
                overlap_idx = existing.index.get_loc(existing.index[overlap_dates][0])
                trimmed = existing.iloc[:overlap_idx]
            else:
                trimmed = existing

            combined = pd.concat([trimmed, new_data]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_csv(eu_gas_path)

            new_records = len(combined) - len(existing)
            logger.info(f"  Added {new_records} new records, ends: {combined.index.max().date()}")
        else:
            logger.warning(f"  {eu_gas_path} not found, skipping")

        # Update US Henry Hub daily prices
        logger.info("Updating FRED US Henry Hub prices (daily)...")
        us_gas_path = data_dir / "fred_us_henryhub_daily.csv"
        if us_gas_path.exists():
            existing = pd.read_csv(us_gas_path, index_col="Date", parse_dates=True)
            last_date = existing.index.max()
            logger.info(f"  Current data ends: {last_date.date()}")

            redundancy_start = last_date - pd.Timedelta(days=redundancy_days)
            start_date = redundancy_start.strftime("%Y-%m-%d")

            new_data = download_fred_series(FRED_SERIES["us_henryhub_daily"], start_date, end_date)

            overlap_dates = existing.index >= redundancy_start
            if overlap_dates.any():
                overlap_idx = existing.index.get_loc(existing.index[overlap_dates][0])
                trimmed = existing.iloc[:overlap_idx]
            else:
                trimmed = existing

            combined = pd.concat([trimmed, new_data]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_csv(us_gas_path)

            new_records = len(combined) - len(existing)
            logger.info(f"  Added {new_records} new records, ends: {combined.index.max().date()}")
            last_combined = combined
        else:
            logger.warning(f"  {us_gas_path} not found, skipping")

        return last_combined

    def _fetch_update(self, start_date) -> pd.DataFrame:
        """Not used for FredSource (overrides update() directly)."""
        raise NotImplementedError("FredSource uses custom update() logic")


# =============================================================================
# Energy Charts Source
# =============================================================================


class EnergyChartsSource(DataSource):
    """Data source for energy-charts.info REST API."""

    index_col = "time"

    def __init__(self, series_name: str = "da_price_de_lu"):
        from src.config.energy_charts import SERIES

        self.series_name = series_name
        self.config = SERIES[series_name]

    def download(self, output_path: Path, start_date: str = "2015-01-01") -> None:
        """Download full history and save to output_path."""
        from src.data.energy_charts import fetch_price

        bzn = self.config["params"]["bzn"]
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp.now(tz="UTC")

        logger.info(
            f"Downloading Energy Charts {self.series_name} from {start.date()} to {end.date()}"
        )
        df = fetch_price(bzn, start, end)
        df.to_csv(output_path)
        logger.info(f"Saved {len(df)} rows to {output_path}")

    def _fetch_update(self, start_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch recent data for the redundancy window."""
        from src.data.energy_charts import fetch_price

        bzn = self.config["params"]["bzn"]
        end = pd.Timestamp.now(tz="UTC")
        return fetch_price(bzn, start_date, end)
