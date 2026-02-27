"""
Data freshness monitoring script.

Checks how old the data is and alerts if it exceeds thresholds.
Run this via cron/task scheduler to monitor data pipeline health.
"""

from pathlib import Path
import sys

from loguru import logger
import pandas as pd

# Freshness thresholds (in hours)
THRESHOLDS = {
    "raw_csvs": 12,  # Raw CSVs should be updated at least every 12 hours
    "combined": 12,  # Combined parquet should match raw freshness
    "merged": 12,  # Merged parquet should match combined freshness
    "commodities": 48,  # Commodity markets closed weekends, allow 48h
}

# Exit codes for monitoring systems
EXIT_OK = 0
EXIT_WARNING = 1
EXIT_CRITICAL = 2


def check_raw_csv_freshness(csv_path: Path, name: str) -> dict:
    """Check freshness of a raw CSV file.

    Returns:
        Dict with status, hours_old, last_valid_timestamp
    """
    if not csv_path.exists():
        return {
            "status": "MISSING",
            "hours_old": None,
            "last_valid": None,
            "message": f"File not found: {csv_path}",
        }

    try:
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
        valid = df[df["value"].notna()]

        if valid.empty:
            return {
                "status": "ERROR",
                "hours_old": None,
                "last_valid": None,
                "message": "No valid data in file",
            }

        now = pd.Timestamp.now(tz="UTC")
        last_valid = valid["time"].max()
        hours_old = (now - last_valid).total_seconds() / 3600

        threshold = THRESHOLDS["raw_csvs"]
        if hours_old > threshold:
            status = "STALE"
            message = f"Data is {hours_old:.1f}h old (threshold: {threshold}h)"
        else:
            status = "OK"
            message = f"Data is {hours_old:.1f}h old"

        return {
            "status": status,
            "hours_old": hours_old,
            "last_valid": last_valid,
            "message": message,
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "hours_old": None,
            "last_valid": None,
            "message": f"Error reading file: {e}",
        }


def check_parquet_freshness(parquet_path: Path, name: str, threshold_key: str) -> dict:
    """Check freshness of a parquet file.

    Returns:
        Dict with status, hours_old, last_valid_timestamp
    """
    if not parquet_path.exists():
        return {
            "status": "MISSING",
            "hours_old": None,
            "last_valid": None,
            "message": f"File not found: {parquet_path}",
        }

    try:
        df = pd.read_parquet(parquet_path)

        # Find latest valid data across all columns
        last_valid_timestamps = [
            df[col].last_valid_index()
            for col in df.columns
            if df[col].last_valid_index() is not None
        ]

        if not last_valid_timestamps:
            return {
                "status": "ERROR",
                "hours_old": None,
                "last_valid": None,
                "message": "No valid data in file",
            }

        now = pd.Timestamp.now(tz="UTC")
        last_valid = max(last_valid_timestamps)
        hours_old = (now - last_valid).total_seconds() / 3600

        threshold = THRESHOLDS[threshold_key]
        if hours_old > threshold:
            status = "STALE"
            message = f"Data is {hours_old:.1f}h old (threshold: {threshold}h)"
        else:
            status = "OK"
            message = f"Data is {hours_old:.1f}h old"

        return {
            "status": status,
            "hours_old": hours_old,
            "last_valid": last_valid,
            "message": message,
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "hours_old": None,
            "last_valid": None,
            "message": f"Error reading file: {e}",
        }


def monitor_data_freshness() -> int:
    """Run data freshness checks and return exit code.

    Returns:
        0 if all OK, 1 if warnings, 2 if critical errors
    """
    logger.info("=" * 80)
    logger.info("DATA FRESHNESS MONITORING")
    logger.info("=" * 80)

    results = {}
    has_errors = False
    has_warnings = False

    # Check key raw CSV files
    logger.info("\n[1] RAW CSV FILES")
    logger.info("-" * 80)

    raw_files = {
        "consumption": "data/raw/smard_api/stromverbrauch_gesamt_(netzlast).csv",
        "wind_onshore": "data/raw/smard_api/stromerzeugung_wind_onshore.csv",
        "photovoltaik": "data/raw/smard_api/stromerzeugung_photovoltaik.csv",
        "price": "data/raw/smard_api/marktpreis_deutschland_luxemburg.csv",
    }

    for name, path in raw_files.items():
        result = check_raw_csv_freshness(Path(path), name)
        results[f"raw_{name}"] = result

        status_symbol = {"OK": "✓", "STALE": "⚠", "ERROR": "✗", "MISSING": "✗"}.get(
            result["status"], "?"
        )

        logger.info(f"  [{status_symbol}] {name:20s}: {result['message']}")

        if result["status"] in ["ERROR", "MISSING"]:
            has_errors = True
        elif result["status"] == "STALE":
            has_warnings = True

    # Check combined parquet
    logger.info("\n[2] COMBINED PARQUET")
    logger.info("-" * 80)

    combined_result = check_parquet_freshness(
        Path("data/interim/combined_de_lu.parquet"), "combined", "combined"
    )
    results["combined"] = combined_result

    status_symbol = {"OK": "✓", "STALE": "⚠", "ERROR": "✗", "MISSING": "✗"}.get(
        combined_result["status"], "?"
    )
    logger.info(f"  [{status_symbol}] combined_de_lu: {combined_result['message']}")

    if combined_result["status"] in ["ERROR", "MISSING"]:
        has_errors = True
    elif combined_result["status"] == "STALE":
        has_warnings = True

    # Check merged parquet
    logger.info("\n[3] MERGED PARQUET")
    logger.info("-" * 80)

    merged_result = check_parquet_freshness(
        Path("data/interim/merged_dataset_hourly.parquet"), "merged", "merged"
    )
    results["merged"] = merged_result

    status_symbol = {"OK": "✓", "STALE": "⚠", "ERROR": "✗", "MISSING": "✗"}.get(
        merged_result["status"], "?"
    )
    logger.info(f"  [{status_symbol}] merged_dataset: {merged_result['message']}")

    if merged_result["status"] in ["ERROR", "MISSING"]:
        has_errors = True
    elif merged_result["status"] == "STALE":
        has_warnings = True

    # Check commodity parquet
    logger.info("\n[4] COMMODITY PRICES")
    logger.info("-" * 80)

    commodity_result = check_parquet_freshness(
        Path("data/interim/commodity_prices_hourly.parquet"),
        "commodities",
        "commodities",
    )
    results["commodities"] = commodity_result

    status_symbol = {"OK": "✓", "STALE": "⚠", "ERROR": "✗", "MISSING": "✗"}.get(
        commodity_result["status"], "?"
    )
    logger.info(f"  [{status_symbol}] commodity_prices: {commodity_result['message']}")

    if commodity_result["status"] in ["ERROR", "MISSING"]:
        has_errors = True
    elif commodity_result["status"] == "STALE":
        has_warnings = True

    # Summary
    logger.info("\n" + "=" * 80)
    if has_errors:
        logger.error("CRITICAL: Data pipeline has errors!")
        return EXIT_CRITICAL
    elif has_warnings:
        logger.warning("WARNING: Some data is stale!")
        return EXIT_WARNING
    else:
        logger.success("OK: All data is fresh!")
        return EXIT_OK


if __name__ == "__main__":
    exit_code = monitor_data_freshness()
    sys.exit(exit_code)
