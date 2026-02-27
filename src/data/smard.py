"""SMARD API functions and CLI for data downloads.

API functions:
- get_timestamps(): Fetch available timestamps for a measure
- get_data(): Fetch time series data for a specific timestamp
- get_all_data(): Orchestrate fetching all data for a measure
- _get_filename_for_key(): Map filter keys to filenames
- DataNotAvailableError: Custom exception

CLI commands (Typer):
- main: Download data from scratch
- update: Incrementally update existing data
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import requests
import typer

from src.config import RAW_DATA_DIR
from src.config.smard import (
    get_camel_dict_for_region,
    get_filter_dict_for_region,
)

app = typer.Typer()


# =============================================================================
# API functions
# =============================================================================


class DataNotAvailableError(Exception):
    """Raised when data is not available for a given filter/region combination."""

    pass


def get_timestamps(filter, region="DE-LU", resolution="quarterhour"):
    """Fetch a list of available timestamps for a particular measure, region and resolution.

    Raises:
        DataNotAvailableError: If the API returns 404 (data not available for this combination)
    """
    url = f"https://smard.api.proxy.bund.dev/app/chart_data/{filter}/{region}/index_{resolution}.json"
    response = requests.get(url)

    if response.status_code == 404:
        raise DataNotAvailableError(
            f"Data not available for filter={filter}, region={region}, resolution={resolution}"
        )

    data = response.json()
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df.timestamps, unit="ms", utc=True)

    return df


def get_data(filter: int, timestamp, region="DE-LU", resolution="quarterhour"):
    """Get data from SMARD for a particular filter, region, resolution and timestamp.

    NB: valid timestamps for this step are only on a roughly weekly basis.
    """
    url = (
        f"https://smard.api.proxy.bund.dev/app/chart_data/{filter}/{region}/"
        f"{filter}_{region}_{resolution}_{timestamp}.json"
    )
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data["series"], columns=["timestamp", "value"])
    df["time"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
    return df


def get_all_data(filter, region="DE-LU", resolution="quarterhour", timestamp_list="all"):
    """Fetch all available data for a particular filter, region and resolution.

    Args:
        filter: the filter code for the series you wish to access
        region: the region for the data request
        resolution: the frequency of the data
        timestamp_list: the default value 'all' fetches all available timestamps,
                        otherwise you should pass a list of valid timestamps
    """
    full_df = pd.DataFrame()

    if timestamp_list == "all":
        timestamps_df = get_timestamps(filter, region, resolution)
        timestamps = timestamps_df.timestamps
    else:
        timestamps = timestamp_list

    for ts in timestamps:
        df = get_data(filter, ts, region, resolution)
        full_df = pd.concat([full_df, df])

    full_df["region"] = region
    full_df["measure"] = filter

    region_filter_dict = get_filter_dict_for_region(region)
    full_df["measure_desc"] = region_filter_dict.get(filter, "No description")
    return full_df


def _get_filename_for_key(key, region="DE-LU"):
    """Get the filename for a key using the region-specific camel dict.

    Args:
        key: The filter key
        region: 'DE-LU' (default) or 'DE-AT-LU'

    Returns:
        The filename (without extension) for the key
    """
    camel_dict = get_camel_dict_for_region(region)
    if key in camel_dict:
        return camel_dict[key]
    else:
        raise KeyError(f"Key {key} not found in camel_dict for region {region}")


# =============================================================================
# CLI
# =============================================================================


def _parse_keys(keys: str = None, include_capacity: bool = False, region: str = "DE-LU"):
    """Parse keys for both CLI commands."""
    if keys is None:
        source_dict = get_filter_dict_for_region(region, include_capacity=include_capacity)
        keys_list = list(source_dict.keys())
    else:
        if "," in keys:
            keys_list = [int(k.strip()) for k in keys.split(",")]
        else:
            keys_list = [int(keys.strip())]
    return keys_list


def _setup_output_dir(dir_name: str, dir_parent: Path):
    """Set up output directory."""
    output_dir = dir_parent / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@app.command()
def main(
    keys: str = None,
    dir_name: str = "smard_api",
    dir_parent: Path = RAW_DATA_DIR,
    region: str = "DE-LU",
    resolution: str = "quarterhour",
    include_capacity: bool = typer.Option(
        False, "--include-capacity", help="Include installed capacity keys"
    ),
):
    """Generate data from scratch."""
    from src.data.sources import SmardSource

    keys_list = _parse_keys(keys, include_capacity, region)
    output_dir = _setup_output_dir(dir_name, dir_parent)

    logger.info(f"Generating dataset from scratch for {region} with {resolution} resolution...")
    logger.info(f"Using {len(keys_list)} keys for region {region}")

    source = SmardSource(region=region, resolution=resolution)
    source.download(output_dir, keys=keys_list)
    logger.success("Dataset generation complete.")


@app.command()
def update(
    keys: str = None,
    dir_name: str = "smard_api",
    dir_parent: Path = RAW_DATA_DIR,
    redundancy_days: int = 14,
    region: str = "DE-LU",
    resolution: str = "quarterhour",
    include_capacity: bool = typer.Option(
        False, "--include-capacity", help="Include installed capacity keys"
    ),
):
    """Update existing data."""
    from src.data.sources import SmardSource

    keys_list = _parse_keys(keys, include_capacity, region)
    output_dir = _setup_output_dir(dir_name, dir_parent)

    logger.info(f"Updating dataset for {region} with {resolution} resolution...")
    logger.info(f"Using {len(keys_list)} keys for region {region}")

    source = SmardSource(region=region, resolution=resolution)
    source.update(output_dir, keys=keys_list, redundancy_days=redundancy_days)
    logger.success("Dataset update complete.")


if __name__ == "__main__":
    app()
