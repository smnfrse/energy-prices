"""
Reference data for commodity prices pipeline.

Contains ticker mappings, ICAP system IDs, column names, data availability dates,
and other constants used across the commodity data pipeline.
"""

# Yahoo Finance tickers
TICKERS = {
    "ttf": "TTF=F",  # TTF Gas Futures
    "brent": "BZ=F",  # Brent Crude Oil Futures
    "carbon_realtime": "CO2.L",  # SparkChange Physical Carbon EUA ETC (London)
}

# ICAP Carbon Action system IDs
# Different systemIds cover different ETS phases
ICAP_SYSTEMS = {
    "eu_ets_phase3": 33,  # 2014-2018 (Phase 3)
    "eu_ets_phase4": 35,  # 2019-present (Phase 4)
}

# Column mappings (snake_case for consistency with SMARD data)
COLUMN_NAMES = {
    "carbon_price": "carbon_eur_per_ton",
    "carbon_realtime_price": "carbon_realtime_eur_per_ton",
    "ttf_price": "ttf_eur_per_mwh",
    "brent_price": "brent_usd_per_barrel",
}

# Data availability start dates
# These represent the earliest available data from each source
DATA_START = {
    "carbon": "2014-11-03",  # ICAP systemId=33 starts here
    "carbon_realtime": "2021-10-18",  # CO2.L (SparkChange EUA ETC) starts here
    "ttf": "2017-10-23",  # Yahoo Finance TTF limitation
    "brent": "2014-02-03",  # Yahoo Finance Brent (complete coverage)
}

# Known data issues and limitations
DATA_ISSUES = {
    "carbon_publication_lag_days": 60,  # ICAP has ~2 month publication delay
    "carbon_realtime_fills_lag": True,  # CO2.L provides real-time data to fill ICAP lag
    "ttf_missing_period": (
        "2014-12-30",
        "2017-10-22",
    ),  # ~2.8 years missing (filled via FRED reconstruction)
    "ttf_gap_reconstruction": True,  # TTF gap filled using FRED EU monthly + US Henry Hub daily
    "weekend_holidays_missing": True,  # All sources: business days only (forward-filled in processing)
}

# Unix timestamp boundaries (milliseconds) for ICAP API
# Start: Nov 1, 2014 00:00:00 UTC
ICAP_START_TIMESTAMP = 1414800000000
# Phase split: Jan 1, 2019 00:00:00 UTC (approximate boundary between systemId 33 and 35)
ICAP_PHASE_SPLIT_TIMESTAMP = 1546300800000

# Conversion factors (for reference/validation)
CONVERSIONS = {
    "ttf_mwh_to_m3": 0.088,  # Approximate conversion MWh to m3 natural gas
    "brent_barrel_to_liters": 159,  # Standard barrel volume
    "mwh_to_kwh": 1000,  # MWh to kWh
}

# Expected price ranges for validation
# Used to flag potential data quality issues
PRICE_RANGES = {
    "carbon": (3.0, 100.0),  # EUR/ton (historical range ~€3-€98, combined ICAP + CO2.L)
    "ttf": (5.0, 350.0),  # EUR/MWh (highly volatile, peaked €339 in Aug 2022 crisis)
    "brent": (20.0, 150.0),  # USD/barrel (typical range)
}

# ICAP CSV column names as returned by the API
ICAP_COLUMNS = {
    "date": "Date",
    "eur_eur_rate": "Exchange rate EUR/EUR",
    "eur_usd_rate": "Exchange rate EUR/USD",
    "currency": "Market Currency",
    "primary_market": "Primary Market",
    "secondary_market": "Secondary Market",
}

# Yahoo Finance column names
YAHOO_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}

# FRED (Federal Reserve Economic Data) series IDs
FRED_SERIES = {
    "eu_gas_monthly": "PNGASEUUSDM",  # EU Natural Gas Import Price (monthly, USD/MMBtu)
    "us_henryhub_daily": "DHHNGSP",  # Henry Hub Natural Gas Spot Price (daily, USD/MMBtu)
}

# TTF gap reconstruction parameters
TTF_GAP_OVERLAP_START = "2017-10-01"  # Start of overlap period for bias correction calculation
UNIT_CONVERSION_MMBTU_TO_MWH = 0.293  # Energy content: 1 MMBtu ≈ 0.293 MWh
