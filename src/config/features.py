"""Column definitions and configuration for feature engineering pipeline.

This module defines column lists and transformation settings used in the
feature engineering pipeline. Keeping these definitions separate allows easy
modification and testing of different feature configurations.
"""

from dataclasses import dataclass

from src.features.ts_transforms import WindowSpec

# =============================================================================
# Transform Configuration Flags
# =============================================================================
# Set to False to disable specific transformations (useful for ablation studies)

TRANSFORM_CONFIG = {
    "price_differencing": True,
    "net_exports": True,
    "generation_percentages": True,
    "prognosticated_generation_percentages": True,
    "drop_columns": True,
}

# =============================================================================
# Columns to Drop
# =============================================================================
# These columns are not useful for modeling:
# - Residuallast is a linear transformation of other columns
# - Pumpspeicher consumption is unlikely to be predictive

DROP_COLUMNS = [
    "stromverbrauch_residuallast",
    "stromverbrauch_pumpspeicher",
    "prognostizierter_verbrauch_residuallast",  # Forecast variant
    "scheduled_commerical_exchanges_net_export",
]

# =============================================================================
# Neighbor Price Columns for Price Spread Calculation
# =============================================================================
# These are prices in neighboring markets. We calculate:
# spread_{country} = target_price - marktpreis_{country}
# A positive spread indicates DE-LU price is higher than neighbor.

NEIGHBOR_PRICES = [
    "marktpreis_belgien",
    "marktpreis_dänemark_1",
    "marktpreis_dänemark_2",
    "marktpreis_frankreich",
    "marktpreis_italien_(nord)",
    "marktpreis_niederlande",
    "marktpreis_norwegen_2",
    "marktpreis_österreich",
    "marktpreis_polen",  # TODO: Fragmented pre-2020, decision pending
    "marktpreis_schweden_4",
    "marktpreis_schweiz",
    "marktpreis_slowenien",
    "marktpreis_tschechien",
    "marktpreis_ungarn",
]

# =============================================================================
# Cross-Border Flow Pairs for Net Export Calculation
# =============================================================================
# Format: (export_col, import_col, country_name)
# Net export = exports - imports
# Positive = net exporter, Negative = net importer

FLOW_PAIRS = [
    (
        "cross-border_flows_austria_exports",
        "cross-border_flows_austria_imports",
        "austria",
    ),
    (
        "cross-border_flows_belgium_exports",
        "cross-border_flows_belgium_imports",
        "belgium",
    ),
    (
        "cross-border_flows_czech_republic_exports",
        "cross-border_flows_czech_republic_imports",
        "czech_republic",
    ),
    (
        "cross-border_flows_denmark_1_exports",
        "cross-border_flows_denmark_1_imports",
        "denmark_1",
    ),
    (
        "cross-border_flows_denmark_2_exports",
        "cross-border_flows_denmark_2_imports",
        "denmark_2",
    ),
    (
        "cross-border_flows_france_exports",
        "cross-border_flows_france_imports",
        "france",
    ),
    (
        "cross-border_flows_hungary_exports",
        "cross-border_flows_hungary_imports",
        "hungary",
    ),
    (
        "cross-border_flows_netherlands_exports",
        "cross-border_flows_netherlands_imports",
        "netherlands",
    ),
    (
        "cross-border_flows_northern_italy_exports",
        "cross-border_flows_northern_italy_imports",
        "northern_italy",
    ),
    (
        "cross-border_flows_norway_2_exports",
        "cross-border_flows_norway_2_imports",
        "norway_2",
    ),
    (
        "cross-border_flows_poland_exports",
        "cross-border_flows_poland_imports",
        "poland",
    ),
    (
        "cross-border_flows_slovenia_exports",
        "cross-border_flows_slovenia_imports",
        "slovenia",
    ),
    (
        "cross-border_flows_sweden_4_exports",
        "cross-border_flows_sweden_4_imports",
        "sweden_4",
    ),
    (
        "cross-border_flows_switzerland_exports",
        "cross-border_flows_switzerland_imports",
        "switzerland",
    ),
]

# =============================================================================
# Generation Columns for Percentage Calculation
# =============================================================================
# These are the actual generation columns (not forecasts or installed capacity).
# We calculate pct_{source} = stromerzeugung_{source} / total_generation

GENERATION_COLUMNS = [
    "stromerzeugung_biomasse",
    "stromerzeugung_braunkohle",
    "stromerzeugung_erdgas",
    "stromerzeugung_kernenergie",
    "stromerzeugung_photovoltaik",
    "stromerzeugung_pumpspeicher",
    "stromerzeugung_sonstige_erneuerbare",
    "stromerzeugung_sonstige_konventionelle",
    "stromerzeugung_steinkohle",
    "stromerzeugung_wasserkraft",
    "stromerzeugung_wind_offshore",
    "stromerzeugung_wind_onshore",
]

# =============================================================================
# Prognosticated Generation Columns for Percentage Calculation
# =============================================================================
# These are forecasted generation columns (not actuals).
# We calculate pct_prog_{source} = prognostizierte_erzeugung_{source} /
#   prognostizierte_erzeugung_gesamt
# Note: Only renewable sources have forecasts (wind, solar, sonstige)

PROGNOSTICATED_GENERATION_COLUMNS = [
    "prognostizierte_erzeugung_sonstige",
    "prognostizierte_erzeugung_wind_und_photovoltaik",
]

# =============================================================================
# Columns to Drop After Feature Engineering
# =============================================================================
# These columns are replaced by calculated feature columns and can be dropped
# to reduce dimensionality:
# - Generation columns -> replaced by pct_{source} percentage columns
#   (stromerzeugung_gesamt is kept as denominator reference)
# - Prognosticated generation columns -> replaced by pct_prog_{source}
#   (prognostizierte_erzeugung_gesamt is kept as denominator reference)
# - Cross-border flow columns -> replaced by net_export_{country} columns
#   (cross-border_flows_net_export is kept as total)
# - Neighbor price columns -> replaced by spread_{country} columns

# Generation columns (replaced by percentage versions)
DROP_GENERATION_COLUMNS = GENERATION_COLUMNS.copy()

# Prognosticated generation columns (replaced by percentage versions)
DROP_PROGNOSTICATED_GENERATION_COLUMNS = PROGNOSTICATED_GENERATION_COLUMNS.copy()

# Cross-border flow columns (replaced by net export calculations)
DROP_FLOW_COLUMNS = []
for export_col, import_col, _ in FLOW_PAIRS:
    DROP_FLOW_COLUMNS.append(export_col)
    DROP_FLOW_COLUMNS.append(import_col)

# Neighbor price columns (replaced by spread calculations)
DROP_PRICE_COLUMNS = NEIGHBOR_PRICES.copy()

# Combined list of all columns to drop after feature engineering
DROP_AFTER_FEATURE_ENGINEERING = (
    DROP_GENERATION_COLUMNS
    + DROP_PROGNOSTICATED_GENERATION_COLUMNS
    + DROP_FLOW_COLUMNS
    + DROP_PRICE_COLUMNS
)

# =============================================================================
# Daily Pipeline Configuration
# =============================================================================

# ---------------------------------------------------------------------------
# Rolling Feature Specs — single source of truth for lagged feature columns
# ---------------------------------------------------------------------------
# Prediction timing: D = day we predict for. Prediction made at D-1 ~11am.
# - Actuals (stromerzeugung_*, etc.): D-2 full day is safe
# - Prices (marktpreis_*, spread_*, ratio_*): full day D-1 known (auction)
# - Commodities: closing price + business day delay -> D-2

# V3 specs (feature refinement - removed cross-border, net_export, spread, ratio)
ROLLING_FEATURE_SPECS_v3 = {
    "actuals_d2": {
        "columns": [
            "stromerzeugung_*",
            "stromverbrauch_gesamt_(netzlast)",
            # REMOVED: cross-border_*, net_export_*
        ],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
    "prices_d1": {
        "columns": [
            "marktpreis_*",
            # REMOVED: spread_*, ratio_*
        ],
        "windows": [WindowSpec(start_day=-1, end_day=-1, agg="mean")],
        "overwrite": True,
    },
    "commodities_d2": {
        "columns": ["ttf_eur_per_mwh", "brent_usd_per_barrel", "carbon_eur_per_ton"],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
}

# V5 specs (hourly pipeline — v5_slim and v5_full)
# Adds total_exports / total_imports (computed by Phase 5b before this runs).
# prices_d1 keeps wildcard — unused neighbour prices are dropped by ColumnDropper.
ROLLING_FEATURE_SPECS_v5 = {
    "actuals_d2": {
        "columns": [
            "stromerzeugung_*",
            "stromverbrauch_gesamt_(netzlast)",
            "total_exports",
            "total_imports",
        ],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
    "prices_d1": {
        "columns": [
            "marktpreis_*",
        ],
        "windows": [WindowSpec(start_day=-1, end_day=-1, agg="mean")],
        "overwrite": True,
    },
    "commodities_d2": {
        "columns": ["ttf_eur_per_mwh", "brent_usd_per_barrel", "carbon_eur_per_ton"],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
}

ROLLING_FEATURE_SPECS = {
    "actuals_d2": {
        "columns": [
            "stromerzeugung_*",
            "stromverbrauch_gesamt_(netzlast)",
            "cross-border_*",
            "net_export_*",
            # "total_generation",
        ],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
    "prices_d1": {
        "columns": [
            "marktpreis_*",
            "spread_*",
            "ratio_*",
        ],
        "windows": [WindowSpec(start_day=-1, end_day=-1, agg="mean")],
        "overwrite": True,
    },
    "commodities_d2": {
        "columns": ["ttf_eur_per_mwh", "brent_usd_per_barrel", "carbon_eur_per_ton"],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
}

# ---------------------------------------------------------------------------
# Aggregation Rules (dynamically generated from rolling specs)
# ---------------------------------------------------------------------------

# Base rules for columns NOT handled by ROLLING_FEATURE_SPECS
BASE_AGGREGATION_RULES = {
    # Forecasts: aggregate for day D (no lag needed)
    "prognostizierte_*": "sum",
    "prognostizierter_*": "sum",
    # Temporal: constant within day
    "installierte_*": "last",
    "regime_*": "first",
    "day_of_week": "first",
    "day_of_week_sin": "first",
    "day_of_week_cos": "first",
    "day_of_month": "first",
    "month": "first",
    "month_sin": "first",
    "month_cos": "first",
    "week_of_year": "first",
    "is_weekend": "first",
    "is_holiday": "first",
}


def build_aggregation_rules():
    """Generate full aggregation rules from base + rolling feature specs.

    Overwritten columns -> "first" (broadcast values, constant within day).
    Non-overwritten rolling output -> "first" (also broadcast).
    Base rules -> as specified.
    """
    rules = dict(BASE_AGGREGATION_RULES)
    for spec in ROLLING_FEATURE_SPECS.values():
        if spec.get("overwrite"):
            for pattern in spec["columns"]:
                rules[pattern] = "first"
    # Rolling stats output columns (non-overwrite, like target_price stats)
    rules["target_price_*"] = "first"
    return rules


def build_aggregation_rules_v3():
    """Generate aggregation rules for v3 pipeline (feature refinement).

    Changes from build_aggregation_rules():
    - Removes cross-border and net_export patterns (no longer created)
    - Removes spread_* and ratio_* patterns (no longer created)
    - Adds EWMA patterns (*_ewma_*) as "first"
    - Adds morning actuals patterns (*_d1_h0-10_mean) as "first"
    - Adds new derived features (pct_renewable, supply_demand_gap, *_range_*)
    """
    rules = dict(BASE_AGGREGATION_RULES)

    # Add overwritten columns from v3 specs
    for spec in ROLLING_FEATURE_SPECS_v3.values():
        if spec.get("overwrite"):
            for pattern in spec["columns"]:
                rules[pattern] = "first"

    # Rolling stats output columns (non-overwrite, like target_price stats)
    rules["target_price_*"] = "first"

    # EWMA features (all spans, all columns)
    rules["*_ewma_*"] = "first"

    # Morning actuals features
    rules["*_d1_h0-10_mean"] = "first"

    # New derived features (created at daily level, already lagged)
    rules["pct_renewable"] = "first"
    rules["supply_demand_gap"] = "first"
    rules["total_generation"] = "first"
    rules["target_price_range_*"] = "first"

    return rules


# ---------------------------------------------------------------------------
# Availability Registry (for leakage validation)
# ---------------------------------------------------------------------------


@dataclass
class AvailabilityRule:
    """Defines when a column's data is available relative to prediction day D.

    Attributes:
        pattern: Column name or wildcard pattern (e.g., "stromerzeugung_*").
        max_offset: Latest day offset allowed. 0=day D ok, -1=D-1 or earlier,
            -2=D-2 or earlier.
        cutoff_hour: Max hour if partial day available (None = full day).
        tier: Category label for documentation/grouping.
    """

    pattern: str
    max_offset: int
    cutoff_hour: int | None
    tier: str


AVAILABILITY_RULES = [
    AvailabilityRule("y_*", 0, None, "target"),
    AvailabilityRule("prognostizierte_*", 0, None, "forecast"),
    AvailabilityRule("prognostizierter_*", 0, None, "forecast"),
    AvailabilityRule("pct_prog_*", 0, None, "forecast"),
    AvailabilityRule("price_lag_*", -1, None, "price"),
    AvailabilityRule("marktpreis_*", -1, None, "price"),
    AvailabilityRule("spread_*", -1, None, "price"),
    AvailabilityRule("ratio_*", -1, None, "price"),
    AvailabilityRule("target_price", -1, 10, "price"),
    AvailabilityRule("stromerzeugung_*", -1, 10, "actual"),
    AvailabilityRule("stromverbrauch_*", -1, 10, "actual"),
    AvailabilityRule("cross-border_*", -1, 10, "actual"),
    AvailabilityRule("net_export_*", -1, 10, "actual"),
    AvailabilityRule("pct_*", -1, 10, "actual"),
    AvailabilityRule("ttf_eur_per_mwh", -2, None, "commodity"),
    AvailabilityRule("brent_usd_per_barrel", -2, None, "commodity"),
    AvailabilityRule("carbon_eur_per_ton", -2, None, "commodity"),
    AvailabilityRule("installierte_*", 0, None, "static"),
    AvailabilityRule("regime_*", 0, None, "static"),
    # V3 pipeline features - derived at daily level (already lagged by construction)
    AvailabilityRule("pct_renewable", 0, None, "derived"),
    AvailabilityRule("supply_demand_gap", 0, None, "derived"),
    AvailabilityRule("total_generation", 0, None, "derived"),
    AvailabilityRule("target_price_range_*", 0, None, "derived"),
    # EWMA features - already correctly lagged by EWMATransformer cutoff
    AvailabilityRule("target_price_ewma_*", 0, None, "derived"),
    AvailabilityRule("marktpreis_frankreich_ewma_*", 0, None, "derived"),
    AvailabilityRule("marktpreis_belgien_ewma_*", 0, None, "derived"),
    AvailabilityRule("marktpreis_niederlande_ewma_*", 0, None, "derived"),
    AvailabilityRule("marktpreis_österreich_ewma_*", 0, None, "derived"),
    AvailabilityRule("stromverbrauch_residuallast_ewma_*", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_wind_onshore_ewma_*", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_photovoltaik_ewma_*", 0, None, "derived"),
    AvailabilityRule("carbon_eur_per_ton_ewma_*", 0, None, "derived"),
    AvailabilityRule("ttf_eur_per_mwh_ewma_*", 0, None, "derived"),
    AvailabilityRule("brent_usd_per_barrel_ewma_*", 0, None, "derived"),
    # Morning actuals features - correctly windowed by RollingStatsTransformer
    AvailabilityRule("stromverbrauch_gesamt_(netzlast)_d1_h0-10_mean", 0, None, "derived"),
    AvailabilityRule("stromverbrauch_residuallast_d1_h0-10_mean", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_wind_onshore_d1_h0-10_mean", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_wind_offshore_d1_h0-10_mean", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_photovoltaik_d1_h0-10_mean", 0, None, "derived"),
    # V4 hourly pipeline features
    AvailabilityRule("target_price_lag_d1", 0, None, "derived"),
    AvailabilityRule("target_price_lag_d7", 0, None, "derived"),
    AvailabilityRule("*_daily_sum", 0, None, "derived"),
    AvailabilityRule("*_daily_mean", 0, None, "derived"),
    AvailabilityRule("*_daily_max", 0, None, "derived"),
    AvailabilityRule("*_share", 0, None, "derived"),
    # V5 hourly pipeline features
    AvailabilityRule("target_price_lag_d2", 0, None, "derived"),
    AvailabilityRule("target_price_lag_d14", 0, None, "derived"),
    AvailabilityRule("marktpreis_frankreich_lag_d1", 0, None, "derived"),
    AvailabilityRule("marktpreis_schweiz_lag_d1", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_wind_onshore_lag_d2", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_wind_offshore_lag_d2", 0, None, "derived"),
    AvailabilityRule("stromerzeugung_photovoltaik_lag_d2", 0, None, "derived"),
    AvailabilityRule("*_ewma_*_h10", 0, None, "derived"),  # h10-cutoff EWMA variants
    AvailabilityRule("total_exports", 0, None, "derived"),  # D-2 mean via Phase 6 overwrite
    AvailabilityRule("total_imports", 0, None, "derived"),  # D-2 mean via Phase 6 overwrite
    AvailabilityRule("day_index", 0, None, "derived"),
    AvailabilityRule("year_index", 0, None, "derived"),
    AvailabilityRule("interact_*", 0, None, "derived"),
    # D-1 daily stats for neighbour prices (full pipeline Phase 5b.5)
    AvailabilityRule("marktpreis_*_d1_*", 0, None, "derived"),
    # Same-hour D-2 lags for prices (full pipeline only)
    AvailabilityRule("marktpreis_niederlande_lag_d1", 0, None, "derived"),
    AvailabilityRule("marktpreis_österreich_lag_d1", 0, None, "derived"),
    AvailabilityRule("marktpreis_*_lag_d2", 0, None, "derived"),
    # Adjacent-hour lags (H-1/H-2 for D-1 prices; H-1/H-2 for same-day forecasts)
    AvailabilityRule("*_lag_25h", 0, None, "derived"),
    AvailabilityRule("*_lag_26h", 0, None, "derived"),
    AvailabilityRule("*_lag_1h", 0, None, "derived"),
    AvailabilityRule("*_lag_2h", 0, None, "derived"),
    # Additional daily aggregate stats for forecast columns (full pipeline Phase 7)
    AvailabilityRule("prognostizierter_verbrauch_residuallast_daily_*", 0, None, "derived"),
    AvailabilityRule("prognostizierte_erzeugung_wind_onshore_daily_*", 0, None, "derived"),
    AvailabilityRule("prognostizierte_erzeugung_photovoltaik_daily_*", 0, None, "derived"),
    AvailabilityRule("prognostizierte_erzeugung_sonstige_daily_*", 0, None, "derived"),
    AvailabilityRule("prognostizierter_verbrauch_residuallast_share", 0, None, "derived"),
    AvailabilityRule("prognostizierte_erzeugung_wind_onshore_share", 0, None, "derived"),
    AvailabilityRule("prognostizierte_erzeugung_photovoltaik_share", 0, None, "derived"),
    AvailabilityRule("prognostizierte_erzeugung_sonstige_share", 0, None, "derived"),
]
