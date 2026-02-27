"""Data processing: combine raw CSVs and merge datasets with regime indicators.

The merge pipeline combines, merges, adds regimes, cleans missing values,
and (for hourly) normalizes DST transitions before saving to processed/.
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from src.config import ENERGY_CHARTS_DIR, RAW_DIRS, get_path
from src.config.processing import (
    BIDDING_AREA_SPLIT,
    QUARTER_HOURLY_START,
)
from src.config.smard import camel_dict_all_regions

app = typer.Typer()


# =============================================================================
# CSV Combination Functions
# =============================================================================


def load_and_concat(input_dir: Path) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate into a single DataFrame.

    Args:
        input_dir: Directory containing CSV files to load.

    Returns:
        Concatenated DataFrame in long format with parsed datetime column.
    """
    dfs = []
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        # Parse time column with ISO8601 format to handle mixed timezone formats
        # Some timestamps have "+00:00" suffix, some don't - utc=True handles both
        df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
        logger.debug(f"Loaded {file_path.name}: {len(df)} rows")
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Concatenated DataFrame has {len(full_df)} rows")

    return full_df


def _build_column_mapping(measure_codes: list) -> dict:
    """Build a mapping from measure codes to unique column names.

    Uses camel_dict_all_regions which includes cross-border flows for both
    DE-LU and DE-AT-LU regions. This ensures that cross-border flows from
    different regions with the same description get the same column name,
    allowing them to be properly merged.

    Handles duplicate names by appending the measure code.

    Args:
        measure_codes: List of measure codes present in the data.

    Returns:
        Dict mapping measure code -> unique column name.
    """
    # First pass: identify which names would be duplicates
    name_counts = {}
    for code in measure_codes:
        name = camel_dict_all_regions.get(code, str(code))
        name_counts[name] = name_counts.get(name, 0) + 1

    # Second pass: build mapping, adding code suffix for duplicates
    mapping = {}
    for code in measure_codes:
        name = camel_dict_all_regions.get(code, str(code))
        if name_counts[name] > 1:
            mapping[code] = f"{name}_{int(code)}"
        else:
            mapping[code] = name

    return mapping


def to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format DataFrame to wide format with one column per measure.

    Args:
        df: Long-format DataFrame with columns: time, measure, value, etc.

    Returns:
        Wide-format DataFrame with time as index and measures as columns.
    """
    # Drop columns not needed for the pivot
    wide_df = df.drop(columns=["timestamp", "measure_desc", "region"], errors="ignore")

    # Pivot to wide format
    wide_df = wide_df.pivot_table(index="time", columns="measure", values="value")

    # Build column mapping that handles duplicates
    column_mapping = _build_column_mapping(wide_df.columns.tolist())
    wide_df = wide_df.rename(columns=column_mapping)

    logger.info(f"Wide DataFrame shape: {wide_df.shape}")

    return wide_df


def combine_data(input_dir: Path, output_path: Path) -> None:
    """Orchestrate loading, transforming, and saving data as Parquet.

    Args:
        input_dir: Directory containing CSV files.
        output_path: Path for output Parquet file.
    """
    logger.info(f"Processing data from {input_dir}")

    # Load and concatenate
    full_df = load_and_concat(input_dir)

    # Transform to wide format
    wide_df = to_wide_format(full_df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Parquet
    wide_df.to_parquet(output_path)
    logger.info(f"Saved Parquet to {output_path}")


def combine_data_incremental(
    input_dir: Path, output_path: Path, redundancy_days: int = 14
) -> None:
    """Incrementally update parquet with new CSV data.

    Loads existing parquet, finds the last timestamp with valid data, then processes
    new rows from CSVs newer than (last_timestamp - redundancy_days). The redundancy
    window ensures that recently updated data (from SMARD update with redundancy) is
    re-processed and replaces any NaN values in the existing parquet.

    Args:
        input_dir: Directory containing CSV files.
        output_path: Path for output Parquet file.
        redundancy_days: Days to re-process for data corrections (default: 14).
    """
    if not output_path.exists():
        logger.info("No existing parquet found, running full processing")
        return combine_data(input_dir, output_path)

    # Load existing data and find last timestamp with actual data
    existing_df = pd.read_parquet(output_path)

    # Find the latest timestamp where we have valid REALIZED data (not forecasts)
    # Use consumption data as the reference since it's always realized (never forecasted)
    # and is critical for the dataset
    consumption_col = "stromverbrauch_gesamt_(netzlast)"

    if consumption_col in existing_df.columns:
        last_timestamp = existing_df[consumption_col].last_valid_index()
        logger.info("Using consumption column for incremental update reference")
    else:
        # Fallback: use median of all last_valid timestamps to avoid being fooled by forecasts
        last_valid_timestamps = [
            existing_df[col].last_valid_index()
            for col in existing_df.columns
            if existing_df[col].last_valid_index() is not None
        ]

        if last_valid_timestamps:
            # Use median timestamp - more conservative than max, avoids forecast timestamps
            import statistics

            last_timestamp = statistics.median(last_valid_timestamps)
            logger.info("Using median of last_valid timestamps as reference")
        else:
            last_timestamp = existing_df.index.min()

    # Apply redundancy window to re-process recently updated data
    redundancy_start = last_timestamp - pd.Timedelta(days=redundancy_days)

    max_timestamp = existing_df.index.max()
    logger.info(
        f"Existing data: {len(existing_df)} rows, "
        f"last valid data at {last_timestamp}, "
        f"max timestamp {max_timestamp}, "
        f"redundancy window starts at {redundancy_start}"
    )

    # Load CSVs and filter to rows >= redundancy_start
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    new_dfs = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        # Parse time column with ISO8601 format to handle mixed timezone formats
        df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
        new_rows = df[df["time"] >= redundancy_start]
        if len(new_rows) > 0:
            logger.debug(f"{file_path.name}: {len(new_rows)} rows in update window")
            new_dfs.append(new_rows)

    if not new_dfs:
        logger.info("No new data to process")
        return

    # Concatenate and pivot new data
    new_long = pd.concat(new_dfs, ignore_index=True)
    logger.info(f"Found {len(new_long)} rows in update window across {len(new_dfs)} files")

    new_wide = to_wide_format(new_long)
    logger.info(f"Pivoted to {len(new_wide)} unique timestamps")

    # Remove redundancy window from existing data, then append updated data
    # This ensures updated values replace any NaN values in the redundancy window
    existing_trimmed = existing_df[existing_df.index < redundancy_start]
    logger.info(
        f"Trimmed existing data from {len(existing_df)} to {len(existing_trimmed)} rows "
        f"(removed redundancy window)"
    )

    # Combine trimmed existing + updated data
    combined = pd.concat([existing_trimmed, new_wide])
    combined = combined.sort_index()

    # Remove any duplicate timestamps (keep latest values)
    combined = combined[~combined.index.duplicated(keep="last")]

    # Save
    combined.to_parquet(output_path)
    new_added = len(combined) - len(existing_df)
    logger.info(f"Update complete: {len(combined)} total rows ({new_added:+d} net change)")


# =============================================================================
# Dataset Merging Functions
# =============================================================================


def load_interim_data(resolution: str = "quarterhour") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both interim parquet files.

    Args:
        resolution: Data resolution - "quarterhour" or "hour".

    Returns:
        Tuple of (df_de_lu, df_de_at_lu) DataFrames.
    """
    de_lu_path = get_path("combined_de_lu", resolution)
    de_at_lu_path = get_path("combined_de_at_lu", resolution)

    logger.info(f"Loading DE-LU data from {de_lu_path}")
    df_de_lu = pd.read_parquet(de_lu_path)
    logger.info(
        f"DE-LU shape: {df_de_lu.shape}, range: {df_de_lu.index.min()} to {df_de_lu.index.max()}"
    )

    logger.info(f"Loading DE-AT-LU data from {de_at_lu_path}")
    df_de_at_lu = pd.read_parquet(de_at_lu_path)
    logger.info(
        f"DE-AT-LU shape: {df_de_at_lu.shape}, range: {df_de_at_lu.index.min()} to {df_de_at_lu.index.max()}"
    )

    return df_de_lu, df_de_at_lu


def merge_datasets(
    df_de_lu: pd.DataFrame, df_de_at_lu: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Merge DE-AT-LU and DE-LU datasets at the cutoff point.

    Takes DE-AT-LU data before cutoff and DE-LU data from cutoff onwards.
    Handles column alignment by taking the union of columns.

    Args:
        df_de_lu: DE-LU dataset (post-split primary).
        df_de_at_lu: DE-AT-LU dataset (pre-split primary).
        cutoff: Timestamp where DE-LU data starts being used.

    Returns:
        Merged DataFrame sorted by time index.
    """
    # Split each dataset at cutoff
    df_pre = df_de_at_lu[df_de_at_lu.index < cutoff].copy()
    df_post = df_de_lu[df_de_lu.index >= cutoff].copy()

    logger.info(f"Pre-cutoff rows (DE-AT-LU): {len(df_pre)}")
    logger.info(f"Post-cutoff rows (DE-LU): {len(df_post)}")

    # Concatenate with outer join to preserve all columns
    merged = pd.concat([df_pre, df_post], axis=0)
    merged = merged.sort_index()

    logger.info(f"Merged shape: {merged.shape}")

    return merged


def _load_energy_charts_fallback() -> "pd.Series | None":
    """Load Energy Charts DA price as fallback for target_price gaps.

    Returns None gracefully if the file is absent (e.g. first-time setup).
    """
    ec_path = ENERGY_CHARTS_DIR / "da_price_de_lu.csv"
    if not ec_path.exists():
        return None
    ec_df = pd.read_csv(ec_path, index_col="time", parse_dates=True)
    ec_df.index = pd.to_datetime(ec_df.index, utc=True)
    return ec_df["value"]


def create_unified_target(
    df: pd.DataFrame, ec_fallback: "pd.Series | None" = None
) -> pd.DataFrame:
    """Create a unified target_price column from the two price series.

    Pre-split: uses marktpres_deutschland_austria_luxembourg
    Post-split: uses marktpreis_deutschland_luxemburg

    Args:
        df: DataFrame with both price columns.

    Returns:
        DataFrame with added target_price column.
    """
    df = df.copy()

    pre_col = "marktpres_deutschland_austria_luxembourg"
    post_col = "marktpreis_deutschland_luxemburg"

    if pre_col not in df.columns or post_col not in df.columns:
        raise ValueError(f"Missing required columns: {pre_col} and/or {post_col}")

    # Combine: use post-split price first (has priority), then fall back to pre-split
    df["target_price"] = df[post_col].combine_first(df[pre_col])

    if ec_fallback is not None:
        n_gaps = df["target_price"].isna().sum()
        if n_gaps > 0:
            df["target_price"] = df["target_price"].fillna(ec_fallback.reindex(df.index))
            n_filled = n_gaps - df["target_price"].isna().sum()
            logger.info(f"Filled {n_filled} target_price gaps from Energy Charts")
        else:
            logger.info("No target_price gaps — Energy Charts fallback not needed")

    non_null_count = df["target_price"].notna().sum()
    logger.info(f"Created target_price with {non_null_count} non-null values")

    return df


def add_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime indicator columns based on key dates.

    Adds:
    - regime_de_at_lu: 1 if time < BIDDING_AREA_SPLIT, else 0
    - regime_quarter_hourly: 1 if time >= QUARTER_HOURLY_START, else 0

    Args:
        df: DataFrame with datetime index.

    Returns:
        DataFrame with regime dummy columns added.
    """
    df = df.copy()

    # Regime for bidding area (1 = old DE-AT-LU regime)
    df["regime_de_at_lu"] = (df.index < BIDDING_AREA_SPLIT).astype(int)

    # Regime for quarter-hourly pricing (1 = new quarter-hourly regime)
    df["regime_quarter_hourly"] = (df.index >= QUARTER_HOURLY_START).astype(int)

    logger.info(f"regime_de_at_lu=1 count: {df['regime_de_at_lu'].sum()}")
    logger.info(f"regime_quarter_hourly=1 count: {df['regime_quarter_hourly'].sum()}")

    return df


def merge_commodities(
    df: pd.DataFrame,
    commodity_path: Path | None = None,
    resolution: str = "quarterhour",
) -> pd.DataFrame:
    """Merge commodity prices onto the main dataset.

    Loads the daily commodity parquet, reindexes it to the target timestamps via
    forward-fill, then joins. Using daily data directly means any timestamps —
    including inference rows newer than the last update-data run — always get
    actual prices rather than stale values.

    Args:
        df: Main dataset with datetime index (UTC).
        commodity_path: Path to daily commodity parquet. If None, uses default daily path.
        resolution: Data resolution — unused, kept for API compatibility.

    Returns:
        DataFrame with commodity columns merged.
    """
    if commodity_path is None:
        commodity_path = get_path("commodity_daily")

    if not commodity_path.exists():
        logger.warning(f"Commodity file not found at {commodity_path}, skipping commodity merge")
        return df

    logger.info(f"Loading commodity data from {commodity_path}")
    commodity_df = pd.read_parquet(commodity_path)
    logger.info(f"Loaded {len(commodity_df)} rows, {len(commodity_df.columns)} commodity columns")

    # Reindex daily prices to the exact target timestamps.
    # Normalising to UTC midnight aligns each hour with its calendar day's closing price.
    target_dates = df.index.normalize()
    commodity_aligned = commodity_df.reindex(target_dates, method="ffill")
    commodity_aligned.index = df.index

    df = df.join(commodity_aligned, how="left")

    # Log missing value statistics per commodity column
    for col in commodity_df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = 100 * missing_count / len(df)
        logger.info(f"  {col}: {missing_count} missing ({missing_pct:.1f}%)")

    logger.info(f"Merged dataset shape after commodities: {df.shape}")
    return df


def run_merge_pipeline(
    output_path: Path | None = None, resolution: str = "quarterhour"
) -> pd.DataFrame:
    """Execute the full merge pipeline.

    Steps:
    1. Load interim parquet files
    2. Merge at cutoff point
    3. Create unified target
    4. Add regime dummies
    5. Merge commodities
    6. Handle missing values
    7. Normalize DST transitions (hourly only)
    8. Save to processed directory

    Args:
        output_path: Optional custom output path. Defaults based on resolution.
        resolution: Data resolution - "quarterhour" or "hour".

    Returns:
        Merged DataFrame.
    """
    from src.features.transforms import handle_missing_values

    if output_path is None:
        output_path = get_path("merged", resolution)

    # Step 1: Load data
    df_de_lu, df_de_at_lu = load_interim_data(resolution)

    # Step 2: Merge datasets
    df = merge_datasets(df_de_lu, df_de_at_lu, cutoff=BIDDING_AREA_SPLIT)

    # Step 3: Create unified target (with Energy Charts fallback for SMARD gaps)
    ec_fallback = _load_energy_charts_fallback()
    df = create_unified_target(df, ec_fallback=ec_fallback)

    # Step 4: Add regime dummies
    df = add_regime_dummies(df)

    # Step 5: Merge commodities
    df = merge_commodities(df, resolution=resolution)

    # Step 6: Handle missing values
    logger.info("Applying missing value handling")
    df = handle_missing_values(df)

    # Step 7: Normalize DST (hourly only)
    if resolution == "hour":
        from src.features.ts_transforms import normalize_dst

        logger.info("Normalizing DST transitions")
        df = normalize_dst(df)

    # Step 8: Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Saved merged dataset to {output_path}")

    return df


def run_merge_pipeline_incremental(
    output_path: Path | None = None,
    resolution: str = "quarterhour",
    redundancy_days: int = 14,
    source_path: Path | None = None,
) -> pd.DataFrame:
    """Incrementally update the merged dataset with new data.

    Only processes new rows from DE-LU interim parquet (DE-AT-LU is historical only).
    Applies transformations, cleans missing values, normalizes DST (hourly), and
    appends to existing merged dataset.

    Args:
        output_path: Optional custom output path. Defaults based on resolution.
        resolution: Data resolution - "quarterhour" or "hour".
        redundancy_days: Number of days to reprocess for data corrections (default 14).
        source_path: If set, read base timestamps from this dataset and write only
            delta rows to output_path (add-data mode). If None, reads from and
            writes to output_path (standard update-data mode).

    Returns:
        Updated merged DataFrame.
    """
    if output_path is None:
        output_path = get_path("merged", resolution)

    _source = source_path or output_path

    if not _source.exists():
        if source_path is None:
            logger.info("No existing merged dataset found, running full merge pipeline")
            return run_merge_pipeline(output_path, resolution)
        raise FileNotFoundError(f"Source dataset not found: {_source}")

    # Load the dataset that provides base timestamps (source) and the one we'll update (output)
    existing_df = pd.read_parquet(_source)

    # Find the latest timestamp where we have valid REALIZED data (not forecasts)
    # Use consumption data as the reference
    consumption_col = "stromverbrauch_gesamt_(netzlast)"

    if consumption_col in existing_df.columns:
        last_timestamp = existing_df[consumption_col].last_valid_index()
        logger.info("Using consumption column for incremental update reference")
    else:
        # Fallback: use median of all last_valid timestamps
        last_valid_timestamps = [
            existing_df[col].last_valid_index()
            for col in existing_df.columns
            if existing_df[col].last_valid_index() is not None
        ]

        if last_valid_timestamps:
            import statistics

            last_timestamp = statistics.median(last_valid_timestamps)
            logger.info("Using median of last_valid timestamps as reference")
        else:
            last_timestamp = existing_df.index.min()

    max_timestamp = existing_df.index.max()

    # Calculate redundancy window start
    redundancy_start = last_timestamp - pd.Timedelta(days=redundancy_days)

    logger.info(
        f"Existing merged data: {len(existing_df)} rows, "
        f"last valid data at {last_timestamp}, "
        f"max timestamp {max_timestamp}, "
        f"redundancy window from {redundancy_start}"
    )

    # Load DE-LU interim data (DE-AT-LU is historical, no updates)
    de_lu_path = get_path("combined_de_lu", resolution)
    df_de_lu = pd.read_parquet(de_lu_path)

    # Filter to rows >= redundancy_start (not just > last_timestamp)
    # This ensures we reprocess the redundancy window for data corrections
    new_rows = df_de_lu[df_de_lu.index >= redundancy_start].copy()

    if len(new_rows) == 0:
        logger.info("No new data to merge")
        return existing_df

    logger.info(f"Found {len(new_rows)} rows in redundancy window to reprocess")

    # Create target_price for new rows (post-split, so use DE-LU price)
    post_col = "marktpreis_deutschland_luxemburg"
    if post_col in new_rows.columns:
        new_rows["target_price"] = new_rows[post_col]
        # Fill any SMARD gaps with Energy Charts fallback
        ec_fallback = _load_energy_charts_fallback()
        if ec_fallback is not None:
            n_gaps = new_rows["target_price"].isna().sum()
            if n_gaps > 0:
                new_rows["target_price"] = new_rows["target_price"].fillna(
                    ec_fallback.reindex(new_rows.index)
                )
                n_filled = n_gaps - new_rows["target_price"].isna().sum()
                logger.info(f"Filled {n_filled} target_price gaps from Energy Charts")
            else:
                logger.info("No target_price gaps — Energy Charts fallback not needed")
    else:
        logger.warning(f"Column {post_col} not found in new data")

    # Add regime dummies
    new_rows = add_regime_dummies(new_rows)

    # Merge commodities
    new_rows = merge_commodities(new_rows, resolution=resolution)

    # Ensure columns match (add missing columns as NaN)
    for col in existing_df.columns:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA

    # Reorder columns to match existing (use intersection to handle dropped cols)
    common_cols = [c for c in existing_df.columns if c in new_rows.columns]
    new_rows = new_rows[common_cols]

    if source_path is None:
        # Standard update-data mode: trim existing data then append corrections
        existing_trimmed = existing_df[existing_df.index < redundancy_start]
        logger.info(
            f"Trimmed existing data from {len(existing_df)} to {len(existing_trimmed)} rows "
            f"(removed redundancy window)"
        )

        # Normalize to UTC before concat: existing data may be in Europe/Berlin (after normalize_dst)
        # while new_rows are always UTC. Mixed-timezone concat produces an object index which
        # breaks downstream cubicspline interpolation.
        if existing_trimmed.index.tz is not None and str(existing_trimmed.index.tz) != "UTC":
            existing_trimmed = existing_trimmed.copy()
            existing_trimmed.index = existing_trimmed.index.tz_convert("UTC")

        combined = pd.concat([existing_trimmed, new_rows])
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        # add-data mode: write only the delta rows to output_path
        logger.info(f"add-data mode: writing {len(new_rows)} delta rows to {output_path}")
        combined = new_rows.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

    # Handle missing values on full combined dataset
    from src.features.transforms import handle_missing_values

    logger.info("Applying missing value handling")
    combined = handle_missing_values(combined)

    # Normalize DST (hourly only)
    if resolution == "hour":
        from src.features.ts_transforms import normalize_dst

        logger.info("Normalizing DST transitions")
        combined = normalize_dst(combined)

    combined.to_parquet(output_path)
    if source_path is None:
        added_count = len(combined) - len(existing_df)
        logger.info(
            f"Reprocessed redundancy window + added new rows, "
            f"net change: {added_count:+d} rows, total: {len(combined)} rows"
        )
    else:
        logger.info(f"Saved {len(combined)} delta rows to {output_path}")

    return combined


# =============================================================================
# CLI Commands
# =============================================================================


@app.command()
def combine(
    input_dir: Path = typer.Option(
        None,
        "--input-dir",
        "-i",
        help="Input directory with CSV files. If not specified, processes both default directories.",
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        "-o",
        help="Output Parquet file path. Required if --input-dir is specified.",
    ),
    resolution: str = typer.Option(
        "quarterhour",
        "--resolution",
        "-r",
        help="Data resolution: 'quarterhour' or 'hour'. Determines default directories.",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Only process new data since last run (appends to existing parquet).",
    ),
) -> None:
    """Combine CSV files into wide-format Parquet files.

    By default, processes both DE-LU and DE-AT-LU regions.
    Use --input-dir and --output-path to process a single directory.
    Use --resolution to switch between quarterhour and hourly data.
    Use --incremental to only process new data since last run.
    """
    process_fn = combine_data_incremental if incremental else combine_data

    if input_dir is not None:
        if output_path is None:
            raise typer.BadParameter("--output-path is required when --input-dir is specified")
        process_fn(input_dir, output_path)
    else:
        configs = [
            (RAW_DIRS[("DE-LU", resolution)], get_path("combined_de_lu", resolution)),
            (RAW_DIRS[("DE-AT-LU", resolution)], get_path("combined_de_at_lu", resolution)),
        ]

        for dir_path, out_path in configs:
            if dir_path.exists():
                process_fn(dir_path, out_path)
            else:
                logger.warning(f"Directory not found, skipping: {dir_path}")


@app.command()
def merge(
    output_path: Path = typer.Option(
        None,
        "--output-path",
        "-o",
        help="Output Parquet file path. Defaults to data/interim/merged_dataset.parquet.",
    ),
    resolution: str = typer.Option(
        "quarterhour",
        "--resolution",
        "-r",
        help="Data resolution: 'quarterhour' or 'hour'. Determines default paths.",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Only process new data since last run (appends to existing parquet).",
    ),
    source_path: Path = typer.Option(
        None,
        "--source-path",
        help="Source dataset to read base timestamps from (add-data use case). "
        "If set, only new rows are written to output-path.",
    ),
) -> None:
    """Merge DE-AT-LU and DE-LU datasets with regime indicators.

    Creates a unified dataset by:
    1. Taking DE-AT-LU data before the bidding area split (2018-09-30 22:00)
    2. Taking DE-LU data from the split onwards
    3. Creating a unified target_price column
    4. Adding regime dummy variables (regime_de_at_lu, regime_quarter_hourly)

    Use --incremental to only process new data since last run.
    Use --source-path with --incremental for add-data mode (writes delta only).
    """
    if incremental:
        run_merge_pipeline_incremental(output_path, resolution, source_path=source_path)
    else:
        run_merge_pipeline(output_path, resolution)


if __name__ == "__main__":
    app()
