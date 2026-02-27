"""Column definitions and configuration for data processing pipeline.

This module defines column lists used for missing value handling and data cleaning
in the processing pipeline. Keeping these definitions separate allows them to be
reused across different scripts and keeps the processing logic clean.
"""

import pandas as pd

# =============================================================================
# Key Dates for Regime Changes (UTC)
# =============================================================================

BIDDING_AREA_SPLIT = pd.Timestamp("2018-09-30 22:00:00", tz="UTC")  # DE-AT-LU -> DE-LU transition
QUARTER_HOURLY_START = pd.Timestamp("2025-10-01", tz="UTC")  # Price resolution change
BELGIUM_DATA_START = pd.Timestamp("2017-10-10 22:00:00", tz="UTC")  # Belgium data availability

# =============================================================================
# Missing Value Handling Column Lists
# =============================================================================

# Columns to drop (redundant - values already in target_price or not needed)
DROP_COLUMNS = [
    "marktpreis_deutschland_luxemburg",  # 34% missing, values in target_price
    "marktpres_deutschland_austria_luxembourg",  # 66% missing, values in target_price
    "marktpreis_anrainer_de_lu",  # 44% missing, not needed for modeling
]

# Nuclear: fill 0 after last valid observation (German nuclear decommissioned April 2023)
FILL_ZERO_AFTER_LAST_OBS = [
    "stromerzeugung_kernenergie",
]

# Austria neighbor cross-border flows: fill 0 after bidding area split (2018-09-30 22:00)
# These flows were tracked when Austria was part of DE-AT-LU, but became irrelevant for DE-LU
FILL_ZERO_AFTER_SPLIT = [
    "cross-border_flows_hungary_exports",
    "cross-border_flows_hungary_imports",
    "cross-border_flows_slovenia_exports",
    "cross-border_flows_slovenia_imports",
    "cross-border_flows_northern_italy_exports",
    "cross-border_flows_northern_italy_imports",
]

# Belgium cross-border flows: fill 0 before data availability (2017-10-10 22:00 UTC)
# Data reporting started October 2017, before that cross-border flows weren't recorded
FILL_ZERO_BEFORE_AVAILABILITY = [
    "cross-border_flows_belgium_exports",
    "cross-border_flows_belgium_imports",
]

# Austria cross-border flows and price: fill 0 before bidding area split
# Before split, Austria was part of DE-AT-LU, so no separate flows or price existed
FILL_ZERO_BEFORE_SPLIT = [
    "cross-border_flows_austria_exports",
    "cross-border_flows_austria_imports",
    "marktpreis_österreich",
]

# Norway cross-border flows: fill 0 before first valid observation
# Data reporting started at different times, before that flows weren't recorded
FILL_ZERO_BEFORE_FIRST_VALID = [
    "cross-border_flows_norway_2_exports",
    "cross-border_flows_norway_2_imports",
]

# =============================================================================
# Cross-Border Flow Columns for Net Export Calculation
# =============================================================================

# All export columns (includes both DE-LU and DE-AT-LU era columns)
EXPORT_COLUMNS = [
    "cross-border_flows_denmark_1_exports",
    "cross-border_flows_denmark_2_exports",
    "cross-border_flows_france_exports",
    "cross-border_flows_netherlands_exports",
    "cross-border_flows_poland_exports",
    "cross-border_flows_sweden_4_exports",
    "cross-border_flows_switzerland_exports",
    "cross-border_flows_czech_republic_exports",
    "cross-border_flows_czechia_exports",  # DE-AT-LU era name
    "cross-border_flows_austria_exports",
    "cross-border_flows_belgium_exports",
    "cross-border_flows_norway_2_exports",
    "cross-border_flows_hungary_exports",  # DE-AT-LU era
    "cross-border_flows_slovenia_exports",  # DE-AT-LU era
    "cross-border_flows_northern_italy_exports",  # DE-AT-LU era
]

# All import columns (includes both DE-LU and DE-AT-LU era columns)
IMPORT_COLUMNS = [
    "cross-border_flows_denmark_1_imports",
    "cross-border_flows_denmark_2_imports",
    "cross-border_flows_france_imports",
    "cross-border_flows_netherlands_imports",
    "cross-border_flows_poland_imports",
    "cross-border_flows_sweden_4_imports",
    "cross-border_flows_switzerland_imports",
    "cross-border_flows_czech_republic_imports",
    "cross-border_flows_czechia_imports",  # DE-AT-LU era name
    "cross-border_flows_austria_imports",
    "cross-border_flows_belgium_imports",
    "cross-border_flows_norway_2_imports",
    "cross-border_flows_hungary_imports",  # DE-AT-LU era
    "cross-border_flows_slovenia_imports",  # DE-AT-LU era
    "cross-border_flows_northern_italy_imports",  # DE-AT-LU era
]
