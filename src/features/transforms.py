"""sklearn-compatible transformers for energy price feature engineering.

This module provides:
- handle_missing_values(): Fixed cleaning logic (plain function, not experimentable)
- sklearn BaseEstimator/TransformerMixin classes for composable feature engineering

Usage in experiments:
    from sklearn.pipeline import Pipeline
    from src.features.transforms import (
        handle_missing_values, FeatureScaler, PriceSpreadTransformer,
        NetExportTransformer, GenerationPercentageTransformer, ColumnDropper,
    )

    # Step 1: Clean (plain function, run once)
    df = handle_missing_values(df)

    # Step 2: Feature engineering (sklearn pipeline, experimentable)
    pipeline = Pipeline([
        ("spreads", PriceSpreadTransformer(drop_original=True)),
        ("net_exports", NetExportTransformer(drop_original=True)),
        ("gen_pct", GenerationPercentageTransformer(drop_original=True)),
        ("scale_targets", FeatureScaler(columns=["y_*"], method="log_shift")),
        ("scale_features", FeatureScaler(columns=["stromerzg_*"], method="quantile")),
        ("drop", ColumnDropper()),
    ])
    result = pipeline.fit_transform(df)
"""

import holidays as holidays_lib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from src.config.features import (
    DROP_COLUMNS as FEATURE_DROP_COLUMNS,
)
from src.config.features import (
    FLOW_PAIRS,
    GENERATION_COLUMNS,
    NEIGHBOR_PRICES,
    PROGNOSTICATED_GENERATION_COLUMNS,
)
from src.config.processing import (
    BELGIUM_DATA_START,
    BIDDING_AREA_SPLIT,
    EXPORT_COLUMNS,
    FILL_ZERO_AFTER_LAST_OBS,
    FILL_ZERO_AFTER_SPLIT,
    FILL_ZERO_BEFORE_AVAILABILITY,
    FILL_ZERO_BEFORE_FIRST_VALID,
    FILL_ZERO_BEFORE_SPLIT,
    IMPORT_COLUMNS,
)
from src.config.processing import (
    DROP_COLUMNS as PROCESSING_DROP_COLUMNS,
)
from src.config.temporal import GERMAN_STATE_POPULATIONS

# =============================================================================
# Missing Value Handling (plain functions, fixed logic)
# =============================================================================


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper for handle_missing_values() function.

    No parameters — applies fixed missing value handling logic that includes:
    - Dropping redundant columns
    - Filling zeros for decommissioned/split/unavailable sources
    - Calculating derived columns (prognosticated, net exports, prices)
    - Interpolating small gaps (<=5 consecutive NaNs)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return handle_missing_values(X)


def interpolate_small_gaps(
    df: pd.DataFrame,
    max_gap: int = 2,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate missing values using cubic spline for small gaps.

    For each numeric column, identifies gaps (consecutive NaNs) and applies
    cubic spline interpolation only to gaps of max_gap or fewer. Larger gaps
    are left as-is for downstream handling.

    Args:
        df: DataFrame with datetime index.
        max_gap: Maximum gap size to interpolate (default: 2).
        exclude_columns: Columns to skip (e.g., regime indicators, target).

    Returns:
        DataFrame with small gaps interpolated.
    """
    if exclude_columns is None:
        exclude_columns = ["regime_de_at_lu", "regime_quarter_hourly", "target_price"]

    df = df.copy()
    interpolated_counts = {}

    numeric_cols = df.select_dtypes(include="number").columns
    cols_to_process = [c for c in numeric_cols if c not in exclude_columns]

    for col in cols_to_process:
        if df[col].isna().sum() == 0:
            continue

        is_null = df[col].isna()
        if not is_null.any():
            continue

        null_groups = (~is_null).cumsum()
        null_groups[~is_null] = -1

        gap_sizes = is_null.groupby(null_groups).transform("sum")
        small_gap_mask = is_null & (gap_sizes <= max_gap)

        if small_gap_mask.sum() == 0:
            continue

        interpolated = df[col].interpolate(method="cubicspline", limit_area="inside")
        df.loc[small_gap_mask, col] = interpolated.loc[small_gap_mask]
        interpolated_counts[col] = small_gap_mask.sum()

    if interpolated_counts:
        total = sum(interpolated_counts.values())
        logger.info(
            f"Interpolated {total} values across {len(interpolated_counts)} columns (gaps <= {max_gap})"
        )

    return df


def _fill_zeros(df: pd.DataFrame, columns: list, mask: pd.Series, desc: str) -> list[str]:
    """Fill NaN with 0 for columns where mask is True. Returns log messages."""
    messages = []
    for col in columns:
        if col not in df.columns:
            continue
        col_mask = mask & df[col].isna()
        filled = col_mask.sum()
        if filled > 0:
            df.loc[col_mask, col] = 0.0
            messages.append(f"{col}: {filled} filled {desc} (remaining: {df[col].isna().sum()})")
    return messages


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in merged dataset with documented assumptions.

    Rules applied:
    1. Drop redundant columns (already in target_price or not needed)
    2. Fill 0 after last valid: nuclear (decommissioned April 2023)
    3. Fill 0 after split: Austria neighbor flows (no longer relevant for DE-LU)
    4. Fill 0 before date: Belgium flows (data started Oct 2017)
    5. Fill 0 before split: Austria direct flows/price (Austria was part of DE-AT-LU)
    6. Fill 0 before first valid: Norway flows (late data reporting)
    7. Calculate prognostizierte_erzeugung_sonstige from difference (gesamt - wind/PV)
    8. Fill prognostizierter_verbrauch_gesamt with actual consumption
    9. Fill prognostizierte_erzeugung_gesamt with sum of actual generation
    10. Fill marktpreis_polen with target_price (zero spread assumption)
    11. Calculate net exports from sum(exports) - sum(imports)
    12. Interpolate small gaps (<=2 consecutive NaNs) using cubic spline

    Args:
        df: Merged DataFrame with datetime index.

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    initial_shape = df.shape
    logs = []

    # 1. Drop redundant columns
    to_drop = [c for c in PROCESSING_DROP_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        logs.append(f"Dropped {len(to_drop)} columns: {to_drop}")

    # 2. Fill 0 after last valid observation (nuclear decommissioned)
    for col in FILL_ZERO_AFTER_LAST_OBS:
        if col in df.columns and (last := df[col].last_valid_index()) is not None:
            logs += _fill_zeros(df, [col], df.index > last, f"after {last}")

    # 3. Fill 0 after bidding area split (Austria neighbor flows)
    logs += _fill_zeros(df, FILL_ZERO_AFTER_SPLIT, df.index >= BIDDING_AREA_SPLIT, "after split")

    # 4. Fill 0 before Belgium data availability
    logs += _fill_zeros(
        df, FILL_ZERO_BEFORE_AVAILABILITY, df.index < BELGIUM_DATA_START, "before availability"
    )

    # 5. Fill 0 before bidding area split (Austria direct flows/price)
    logs += _fill_zeros(df, FILL_ZERO_BEFORE_SPLIT, df.index < BIDDING_AREA_SPLIT, "before split")

    # 6. Fill 0 before first valid observation (Norway flows)
    for col in FILL_ZERO_BEFORE_FIRST_VALID:
        if col in df.columns and (first := df[col].first_valid_index()) is not None:
            logs += _fill_zeros(df, [col], df.index < first, f"before {first}")

    # 7. Calculate prognostizierte_erzeugung_sonstige from difference
    sonstige_col = "prognostizierte_erzeugung_sonstige"
    gesamt_col = "prognostizierte_erzeugung_gesamt"
    wind_pv_col = "prognostizierte_erzeugung_wind_und_photovoltaik"

    if all(c in df.columns for c in [sonstige_col, gesamt_col, wind_pv_col]):
        missing = df[sonstige_col].isna()
        calculated = df[gesamt_col] - df[wind_pv_col]
        valid_calc = missing & calculated.notna()
        df.loc[valid_calc, sonstige_col] = calculated[valid_calc]
        logs.append(
            f"{sonstige_col}: {valid_calc.sum()} filled from {gesamt_col} - {wind_pv_col} "
            f"(remaining: {df[sonstige_col].isna().sum()})"
        )

    # 8. Fill prognostizierter_verbrauch_gesamt with actual
    forecast_verb_col = "prognostizierter_verbrauch_gesamt"
    actual_verb_col = "stromverbrauch_gesamt_(netzlast)"

    if forecast_verb_col in df.columns and actual_verb_col in df.columns:
        missing = df[forecast_verb_col].isna()
        df.loc[missing, forecast_verb_col] = df.loc[missing, actual_verb_col]
        logs.append(
            f"{forecast_verb_col}: {missing.sum()} filled with {actual_verb_col} "
            f"(remaining: {df[forecast_verb_col].isna().sum()})"
        )

    # 9. Fill prognostizierte_erzeugung_gesamt
    # Prefer sum of forecast components if available, fallback to sum of actual generation
    forecast_erz_col = "prognostizierte_erzeugung_gesamt"
    sonstige_col = "prognostizierte_erzeugung_sonstige"
    wind_pv_col = "prognostizierte_erzeugung_wind_und_photovoltaik"
    actual_generation_cols = [
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

    if forecast_erz_col in df.columns:
        missing = df[forecast_erz_col].isna()

        # First try to fill from forecast components (ensures consistency)
        if all(c in df.columns for c in [sonstige_col, wind_pv_col]):
            forecast_sum = df[sonstige_col] + df[wind_pv_col]
            valid_forecast = missing & forecast_sum.notna()
            df.loc[valid_forecast, forecast_erz_col] = forecast_sum[valid_forecast]
            filled_from_forecast = valid_forecast.sum()
        else:
            filled_from_forecast = 0

        # Then fill remaining with actual generation sum.
        # For recent rows (last 30 days): fill unconditionally — forecast components may not
        # be published yet (SMARD publication lag) but actuals are available.
        # For historical rows: only fill if at least one forecast component exists, to avoid
        # overwriting the pre-2015 structural gap with actuals.
        available_gen_cols = [c for c in actual_generation_cols if c in df.columns]
        if available_gen_cols and all(c in df.columns for c in [sonstige_col, wind_pv_col]):
            still_missing = df[forecast_erz_col].isna()
            recent_mask = df.index >= (df.index.max() - pd.Timedelta(days=30))
            has_forecast_data = df[sonstige_col].notna() | df[wind_pv_col].notna()
            fill_mask = still_missing & (has_forecast_data | recent_mask)

            # Use min_count=1 to ensure NaN when all components are NaN (not 0)
            actual_total = df[available_gen_cols].sum(axis=1, min_count=1)
            df.loc[fill_mask, forecast_erz_col] = actual_total[fill_mask]
            filled_from_actual = fill_mask.sum()
        else:
            filled_from_actual = 0

        # Log outside the if/else so it always runs when something was filled
        if filled_from_forecast > 0 or filled_from_actual > 0:
            logs.append(
                f"{forecast_erz_col}: {filled_from_forecast} from forecast components, "
                f"{filled_from_actual} from actuals "
                f"(remaining: {df[forecast_erz_col].isna().sum()})"
            )

    # 10. Fill marktpreis_polen with target_price (makes spread zero where missing)
    polen_col = "marktpreis_polen"
    target_col = "target_price"

    if polen_col in df.columns and target_col in df.columns:
        missing = df[polen_col].isna()
        df.loc[missing, polen_col] = df.loc[missing, target_col]
        logs.append(
            f"{polen_col}: {missing.sum()} filled with {target_col} to create zero spread "
            f"(remaining: {df[polen_col].isna().sum()})"
        )

    # 10b. Fill marktpreis_schweiz with target_price (same logic as Poland)
    schweiz_col = "marktpreis_schweiz"
    if schweiz_col in df.columns and target_col in df.columns:
        missing = df[schweiz_col].isna()
        df.loc[missing, schweiz_col] = df.loc[missing, target_col]
        logs.append(
            f"{schweiz_col}: {missing.sum()} filled with {target_col} to create zero spread "
            f"(remaining: {df[schweiz_col].isna().sum()})"
        )

    # 11. Calculate net exports from individual flows
    net_col = "cross-border_flows_net_export"
    if net_col in df.columns:
        export_cols = [c for c in EXPORT_COLUMNS if c in df.columns]
        import_cols = [c for c in IMPORT_COLUMNS if c in df.columns]
        if export_cols and import_cols:
            # Use min_count=1 to ensure NaN when all components are NaN (not 0)
            exports_sum = df[export_cols].sum(axis=1, min_count=1)
            imports_sum = df[import_cols].sum(axis=1, min_count=1)
            calculated = exports_sum - imports_sum
            missing = df[net_col].isna()
            df.loc[missing, net_col] = calculated[missing]
            logs.append(
                f"{net_col}: {missing.sum()} filled from {len(export_cols)} exports - {len(import_cols)} imports"
            )

    # 12. Interpolate small gaps (<=5 consecutive NaNs) using cubic spline
    # Increased from 2 to 5 to handle 2018-09-23 demand gap (5 hours)
    df = interpolate_small_gaps(df, max_gap=5)

    # Log summary
    logger.info(f"Missing value handling: {initial_shape} -> {df.shape}")
    for msg in logs:
        logger.info(f"  {msg}")

    return df


# =============================================================================
# sklearn-Compatible Transformers
# =============================================================================


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop specified columns from the dataset.

    Supports both exact column names and wildcard patterns (e.g., 'marktpreis_*').
    Allows exclude/include semantics where include overrides exclude.

    Parameters:
        exclude: List of column names or patterns to drop. If None, uses FEATURE_DROP_COLUMNS.
        include: List of exact column names to keep even if they match exclude patterns.
    """

    def __init__(self, exclude=None, include=None):
        self.exclude = exclude
        self.include = include

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        exclude = self.exclude if self.exclude is not None else FEATURE_DROP_COLUMNS
        include = self.include if self.include is not None else []
        X = X.copy()

        # Resolve patterns to actual columns
        to_drop = self._resolve_columns(X, exclude, include)

        if to_drop:
            X = X.drop(columns=to_drop)
            logger.info(f"Dropped {len(to_drop)} columns")
        else:
            logger.info("No columns to drop (none found in dataset)")
        return X

    def _resolve_columns(self, X, exclude_patterns, include_columns):
        """Resolve column patterns to actual column names, respecting include list.

        Args:
            X: DataFrame to resolve columns from.
            exclude_patterns: List of column names or wildcard patterns to drop.
            include_columns: List of exact column names to keep (overrides exclude).

        Returns:
            List of unique column names to drop.
        """
        resolved = []
        for pattern in exclude_patterns:
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                resolved.extend([c for c in X.columns if c.startswith(prefix)])
            elif pattern in X.columns:
                resolved.append(pattern)

        # Remove any columns in the include list
        resolved = [c for c in resolved if c not in include_columns]

        return list(set(resolved))  # Remove duplicates


class PriceSpreadTransformer(BaseEstimator, TransformerMixin):
    """Calculate price spreads: target_price - neighbor_price.

    Creates spread_{country} columns.

    Parameters:
        neighbor_prices: List of neighbor price column names. If None, uses NEIGHBOR_PRICES.
        drop_original: If True, drop original neighbor price columns.
    """

    def __init__(self, neighbor_prices=None, drop_original=False):
        self.neighbor_prices = neighbor_prices
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        neighbor_prices = (
            self.neighbor_prices if self.neighbor_prices is not None else NEIGHBOR_PRICES
        )
        X = X.copy()
        created = []

        for neighbor_col in neighbor_prices:
            if neighbor_col not in X.columns:
                continue
            country = neighbor_col.replace("marktpreis_", "")
            spread_col = f"spread_{country}"
            X[spread_col] = X["target_price"] - X[neighbor_col]
            created.append(spread_col)

        logger.info(f"Created {len(created)} price spread columns")

        if self.drop_original and created:
            existing = [c for c in neighbor_prices if c in X.columns]
            X = X.drop(columns=existing)
            logger.info(f"Dropped {len(existing)} original neighbor price columns")

        return X


class PriceRatioTransformer(BaseEstimator, TransformerMixin):
    """Calculate price ratios: target_price / neighbor_price.

    Creates ratio_{country} columns. A ratio > 1 means DE-LU price
    is higher than the neighbor.

    Parameters:
        neighbor_prices: List of neighbor price column names.
            If None, uses NEIGHBOR_PRICES from src/config/features.py.
        drop_original: If True, drop original neighbor price columns.
    """

    def __init__(self, neighbor_prices=None, drop_original=False):
        self.neighbor_prices = neighbor_prices
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        neighbor_prices = (
            self.neighbor_prices if self.neighbor_prices is not None else NEIGHBOR_PRICES
        )
        X = X.copy()
        created = []

        for neighbor_col in neighbor_prices:
            if neighbor_col not in X.columns:
                logger.warning(f"Neighbor price column '{neighbor_col}' not found, skipping ratio")
                continue

            country = neighbor_col.replace("marktpreis_", "")
            ratio_col = f"ratio_{country}"
            X[ratio_col] = X["target_price"] / X[neighbor_col]
            created.append(ratio_col)

        logger.info(f"Created {len(created)} price ratio columns")

        if self.drop_original and created:
            existing = [c for c in neighbor_prices if c in X.columns]
            X = X.drop(columns=existing)
            logger.info(f"Dropped {len(existing)} original neighbor price columns")

        return X


class NetExportTransformer(BaseEstimator, TransformerMixin):
    """Calculate per-country net exports: export - import.

    Creates net_export_{country} columns.

    Parameters:
        flow_pairs: List of (export_col, import_col, country_name) tuples.
            If None, uses FLOW_PAIRS.
        drop_original: If True, drop original export/import columns.
    """

    def __init__(self, flow_pairs=None, drop_original=False):
        self.flow_pairs = flow_pairs
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        flow_pairs = self.flow_pairs if self.flow_pairs is not None else FLOW_PAIRS
        X = X.copy()
        created = []
        dropped = []

        for export_col, import_col, country in flow_pairs:
            if export_col not in X.columns and import_col not in X.columns:
                continue

            net_col = f"net_export_{country}"

            if export_col in X.columns and import_col in X.columns:
                X[net_col] = X[export_col] - X[import_col]
            elif export_col in X.columns:
                X[net_col] = X[export_col]
            else:
                X[net_col] = -X[import_col]

            created.append(net_col)

            if self.drop_original:
                if export_col in X.columns:
                    dropped.append(export_col)
                if import_col in X.columns:
                    dropped.append(import_col)

        logger.info(f"Created {len(created)} net export columns")

        if self.drop_original and dropped:
            X = X.drop(columns=dropped)
            logger.info(f"Dropped {len(dropped)} original flow columns")

        return X


class GenerationPercentageTransformer(BaseEstimator, TransformerMixin):
    """Convert generation columns to percentage of total generation.

    Creates pct_{source} columns and optionally renewable% and supply-demand gap.

    Parameters:
        generation_columns: List of generation column names. If None, uses GENERATION_COLUMNS.
        drop_original: If True, drop original generation columns.
        add_renewable_pct: If True, create pct_renewable column (renewable/total).
        add_supply_demand_gap: If True, create supply_demand_gap column (total_gen - demand).
    """

    def __init__(
        self,
        generation_columns=None,
        drop_original=False,
        add_renewable_pct=False,
        add_supply_demand_gap=False,
    ):
        self.generation_columns = generation_columns
        self.drop_original = drop_original
        self.add_renewable_pct = add_renewable_pct
        self.add_supply_demand_gap = add_supply_demand_gap

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        generation_columns = (
            self.generation_columns if self.generation_columns is not None else GENERATION_COLUMNS
        )
        X = X.copy()
        gen_cols = [c for c in generation_columns if c in X.columns]

        if not gen_cols:
            logger.warning("No generation columns found in dataset")
            return X

        # Use min_count=1 to ensure NaN when all components are NaN (not 0)
        X["total_generation"] = X[gen_cols].sum(axis=1, min_count=1)

        # Warn about problematic denominators
        zero_total = (X["total_generation"] == 0).sum()
        nan_total = X["total_generation"].isna().sum()
        if zero_total > 0:
            logger.warning(
                f"Found {zero_total} rows with zero total generation (percentages will be NaN)"
            )
        if nan_total > 0:
            logger.warning(
                f"Found {nan_total} rows with NaN total generation (percentages will be NaN)"
            )

        created = []
        for col in gen_cols:
            source = col.replace("stromerzeugung_", "")
            pct_col = f"pct_{source}"
            X[pct_col] = X[col] / X["total_generation"]  # Produces NaN when denominator is 0/NaN
            created.append(pct_col)

        logger.info(f"Created {len(created)} generation percentage columns")

        # Add renewable percentage if requested
        if self.add_renewable_pct:
            renewable_cols = [
                "stromerzeugung_wind_onshore",
                "stromerzeugung_wind_offshore",
                "stromerzeugung_photovoltaik",
            ]
            renewable_cols_present = [c for c in renewable_cols if c in X.columns]
            if renewable_cols_present:
                renewable_total = X[renewable_cols_present].sum(axis=1, min_count=1)
                X["pct_renewable"] = renewable_total / X["total_generation"]
                created.append("pct_renewable")
                logger.info("Created pct_renewable column")
            else:
                logger.warning("Renewable columns not found, skipping pct_renewable")

        # Add supply-demand gap if requested
        if self.add_supply_demand_gap:
            demand_col = "stromverbrauch_gesamt_(netzlast)"
            if demand_col in X.columns:
                X["supply_demand_gap"] = X["total_generation"] - X[demand_col]
                created.append("supply_demand_gap")
                logger.info("Created supply_demand_gap column")
            else:
                logger.warning(f"{demand_col} not found, skipping supply_demand_gap")

        if self.drop_original:
            X = X.drop(columns=gen_cols)
            logger.info(f"Dropped {len(gen_cols)} original generation columns")

        return X


class PrognosticatedPercentageTransformer(BaseEstimator, TransformerMixin):
    """Convert prognosticated generation columns to percentage of total forecast.

    Creates pct_prog_{source} columns using prognostizierte_erzeugung_gesamt as denominator.

    Parameters:
        prognosticated_columns: List of prognosticated column names.
            If None, uses PROGNOSTICATED_GENERATION_COLUMNS.
        drop_original: If True, drop original prognosticated columns.
    """

    def __init__(self, prognosticated_columns=None, drop_original=False):
        self.prognosticated_columns = prognosticated_columns
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prognosticated_columns = (
            self.prognosticated_columns
            if self.prognosticated_columns is not None
            else PROGNOSTICATED_GENERATION_COLUMNS
        )
        X = X.copy()

        total_col = "prognostizierte_erzeugung_gesamt"
        if total_col not in X.columns:
            logger.warning(
                f"{total_col} not found in dataset, skipping prognosticated percentages"
            )
            return X

        prog_cols = [c for c in prognosticated_columns if c in X.columns]

        if not prog_cols:
            logger.warning("No prognosticated generation columns found in dataset")
            return X

        # Warn about problematic denominators
        zero_total = (X[total_col] == 0).sum()
        nan_total = X[total_col].isna().sum()
        if zero_total > 0:
            logger.warning(
                f"Found {zero_total} rows with zero total prognosticated generation (percentages will be NaN)"
            )
        if nan_total > 0:
            logger.warning(
                f"Found {nan_total} rows with NaN total prognosticated generation (percentages will be NaN)"
            )

        created = []
        for col in prog_cols:
            source = col.replace("prognostizierte_erzeugung_", "")
            pct_col = f"pct_prog_{source}"
            X[pct_col] = X[col] / X[total_col]  # Produces NaN when denominator is 0/NaN
            created.append(pct_col)

        logger.info(f"Created {len(created)} prognosticated generation percentage columns")

        if self.drop_original:
            X = X.drop(columns=prog_cols)
            logger.info(f"Dropped {len(prog_cols)} original prognosticated generation columns")

        return X


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract calendar features with optional cyclical sin/cos encoding.

    Converts UTC index to local timezone for extraction, keeping the original
    UTC index unchanged.

    Parameters:
        include_cyclical: If True, add sin/cos encoded features for hour, day_of_week, month.
        include_linear: If True, add integer calendar features.
        timezone: Timezone for calendar extraction (default: "Europe/Berlin").
    """

    def __init__(self, include_cyclical=True, include_linear=True, timezone="Europe/Berlin"):
        self.include_cyclical = include_cyclical
        self.include_linear = include_linear
        self.timezone = timezone

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        local_time = X.index.tz_convert(self.timezone)
        created = []

        if self.include_linear:
            X["hour_of_day"] = local_time.hour
            X["day_of_week"] = local_time.dayofweek
            X["day_of_month"] = local_time.day
            X["month"] = local_time.month
            X["week_of_year"] = local_time.isocalendar().week.values.astype(int)
            X["is_weekend"] = (local_time.dayofweek >= 5).astype(int)
            created += [
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "month",
                "week_of_year",
                "is_weekend",
            ]

        if self.include_cyclical:
            X["hour_sin"] = np.sin(2 * np.pi * local_time.hour / 24)
            X["hour_cos"] = np.cos(2 * np.pi * local_time.hour / 24)
            X["day_of_week_sin"] = np.sin(2 * np.pi * local_time.dayofweek / 7)
            X["day_of_week_cos"] = np.cos(2 * np.pi * local_time.dayofweek / 7)
            X["month_sin"] = np.sin(2 * np.pi * (local_time.month - 1) / 12)
            X["month_cos"] = np.cos(2 * np.pi * (local_time.month - 1) / 12)
            created += [
                "hour_sin",
                "hour_cos",
                "day_of_week_sin",
                "day_of_week_cos",
                "month_sin",
                "month_cos",
            ]

        logger.info(f"Created {len(created)} temporal features")
        return X


class GermanHolidayTransformer(BaseEstimator, TransformerMixin):
    """Add population-weighted German holiday indicator.

    For each date, computes the fraction of the German population observing a
    public holiday. National holidays → 1.0, state-specific → population fraction.

    Parameters:
        state_populations: Dict mapping state codes to population (millions).
            If None, uses GERMAN_STATE_POPULATIONS from src/config/temporal.py.
        timezone: Timezone for date extraction (default: "Europe/Berlin").
    """

    def __init__(self, state_populations=None, timezone="Europe/Berlin"):
        self.state_populations = state_populations
        self.timezone = timezone

    def fit(self, X, y=None):
        pops = self.state_populations or GERMAN_STATE_POPULATIONS
        total = sum(pops.values())
        self.state_fractions_ = {state: pop / total for state, pop in pops.items()}
        return self

    def transform(self, X):
        X = X.copy()
        local_time = X.index.tz_convert(self.timezone)
        unique_dates = local_time.date
        unique_date_set = sorted(set(unique_dates))

        # Determine year range
        years = sorted(set(d.year for d in unique_date_set))

        # Build holiday sets per state
        states = list(self.state_fractions_.keys())
        state_holidays = {}
        for state in states:
            state_holidays[state] = holidays_lib.Germany(subdiv=state, years=years)

        # Compute population-weighted holiday score per unique date
        date_scores = {}
        for d in unique_date_set:
            score = 0.0
            for state in states:
                if d in state_holidays[state]:
                    score += self.state_fractions_[state]
            date_scores[d] = score

        # Map back to full index
        X["is_holiday"] = [date_scores[d] for d in unique_dates]

        n_holidays = sum(1 for s in date_scores.values() if s > 0)
        logger.info(
            f"Created is_holiday column: {n_holidays} holiday dates out of "
            f"{len(unique_date_set)} unique dates"
        )
        return X


class CreateCustomColumns(BaseEstimator, TransformerMixin):
    """Bundle multiple column-creating transformers into a single pipeline step.

    Applies a list of sub-transformers sequentially. Each sub-transformer
    remains independently configurable and testable.

    Parameters:
        transformers: List of transformer instances to apply sequentially.
            If None, uses empty list (no-op).

    Examples:
        CreateCustomColumns([
            PriceSpreadTransformer(drop_original=True),
            NetExportTransformer(),
            PriceRatioTransformer(),
        ])
    """

    def __init__(self, transformers=None):
        self.transformers = transformers if transformers is not None else []

    def fit(self, X, y=None):
        X_temp = X.copy()
        for transformer in self.transformers:
            transformer.fit(X_temp, y)
            X_temp = transformer.transform(X_temp)
        return self

    def transform(self, X):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X


class TargetTransformer(BaseEstimator, TransformerMixin):
    """Target scaler for use with TransformedTargetRegressor.

    Works on raw arrays (not DataFrames). Supports log_shift, yeo_johnson,
    quantile, and none.

    Parameters:
        method: Scaling method ("log_shift", "yeo_johnson", "quantile", "none").
        shift: For log_shift only. If None, auto-calculated as abs(min) + 1.
    """

    def __init__(self, method="log_shift", shift=None):
        self.method = method
        self.shift = shift

    def fit(self, y, X=None):
        if self.method == "log_shift":
            self.shift_ = self.shift if self.shift is not None else abs(np.nanmin(y)) + 1
        elif self.method == "yeo_johnson":
            self.scaler_ = PowerTransformer(method="yeo-johnson").fit(y)
        elif self.method == "quantile":
            self.scaler_ = QuantileTransformer(output_distribution="normal").fit(y)
        elif self.method != "none":
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Must be one of: log_shift, yeo_johnson, quantile, none"
            )
        return self

    def transform(self, y):
        if self.method == "log_shift":
            return np.log(y + self.shift_)
        elif self.method in ("yeo_johnson", "quantile"):
            return self.scaler_.transform(y)
        return y

    def inverse_transform(self, y):
        if self.method == "log_shift":
            return np.exp(y) - self.shift_
        elif self.method in ("yeo_johnson", "quantile"):
            return self.scaler_.inverse_transform(y)
        return y


class FeatureScaler(BaseEstimator, TransformerMixin):
    """General-purpose feature scaler with multiple scaling methods.

    Supports log-shift, Yeo-Johnson, quantile, or no-op transformations.
    Can be instantiated multiple times for different column groups.

    Parameters:
        columns: List of column names or wildcard patterns (e.g., ["y_*", "price_lag_*"])
        method: Scaling method ("log_shift", "yeo_johnson", "quantile", "none")
        shift: For log_shift only. If None, auto-calculated from data minimum (abs(min) + 1)
        suffix: Suffix to add to transformed columns (default: "_scaled", "_log" for log_shift)
        keep_original: Whether to keep the original columns (default: True)
    """

    def __init__(
        self,
        columns,
        method="log_shift",
        shift=None,
        suffix=None,
        keep_original=True,
    ):
        self.columns = columns if isinstance(columns, list) else [columns]
        self.method = method
        self.shift = shift
        self.suffix = suffix
        self.keep_original = keep_original

    def fit(self, X, y=None):
        """Fit the scaler to the data."""
        # Resolve column patterns to actual column names
        self.columns_ = self._resolve_columns(X, self.columns)

        if not self.columns_:
            raise ValueError(f"No columns found matching patterns: {self.columns}")

        # Set default suffix based on method
        if self.suffix is None:
            self.suffix_ = "_log" if self.method == "log_shift" else "_scaled"
        else:
            self.suffix_ = self.suffix

        # Fit based on method
        if self.method == "log_shift":
            if self.shift is None:
                all_values = pd.concat(
                    [X[col] for col in self.columns_ if col in X.columns]
                ).values
                data_min = np.nanmin(all_values)
                self.shift_ = abs(data_min) + 1
            else:
                self.shift_ = self.shift
            logger.info(f"FeatureScaler (log_shift): Using shift={self.shift_:.2f}")

        elif self.method == "yeo_johnson":
            self.scaler_ = PowerTransformer(method="yeo-johnson")
            data = X[self.columns_].values
            self.scaler_.fit(data)
            logger.info(f"FeatureScaler (yeo_johnson): Fitted on {len(self.columns_)} columns")

        elif self.method == "quantile":
            self.scaler_ = QuantileTransformer(output_distribution="normal")
            data = X[self.columns_].values
            self.scaler_.fit(data)
            logger.info(f"FeatureScaler (quantile): Fitted on {len(self.columns_)} columns")

        elif self.method == "none":
            logger.info(f"FeatureScaler (none): No-op for {len(self.columns_)} columns")

        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Must be one of: log_shift, yeo_johnson, quantile, none"
            )

        return self

    def transform(self, X):
        """Apply the scaling transformation."""
        X = X.copy()

        if self.method == "log_shift":
            for col in self.columns_:
                if col in X.columns:
                    X[f"{col}{self.suffix_}"] = np.log(X[col] + self.shift_)

        elif self.method in ["yeo_johnson", "quantile"]:
            data = X[self.columns_].values
            transformed = self.scaler_.transform(data)
            for i, col in enumerate(self.columns_):
                X[f"{col}{self.suffix_}"] = transformed[:, i]

        elif self.method == "none":
            for col in self.columns_:
                if col in X.columns:
                    X[f"{col}{self.suffix_}"] = X[col]

        # Drop original columns if requested
        if not self.keep_original:
            X = X.drop(columns=self.columns_, errors="ignore")

        logger.info(f"FeatureScaler ({self.method}): Transformed {len(self.columns_)} columns")
        return X

    def inverse_transform(self, y):
        """Reverse the transformation (needed for predictions)."""
        if not hasattr(self, "columns_"):
            raise ValueError("Must call fit() before inverse_transform()")

        # Handle array or DataFrame input
        is_dataframe = isinstance(y, pd.DataFrame)
        is_series = isinstance(y, pd.Series)

        if self.method == "log_shift":
            if is_dataframe or is_series:
                result = np.exp(y) - self.shift_
            else:
                result = np.exp(y) - self.shift_

        elif self.method in ["yeo_johnson", "quantile"]:
            if is_dataframe:
                result = pd.DataFrame(
                    self.scaler_.inverse_transform(y.values),
                    index=y.index,
                    columns=y.columns,
                )
            elif is_series:
                result = pd.Series(
                    self.scaler_.inverse_transform(y.values.reshape(-1, 1)).flatten(),
                    index=y.index,
                    name=y.name,
                )
            else:
                y_array = np.atleast_2d(y)
                if y_array.shape[0] == 1:
                    y_array = y_array.T
                result = self.scaler_.inverse_transform(y_array)
                if result.shape[1] == 1:
                    result = result.flatten()

        elif self.method == "none":
            result = y

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return result

    def _resolve_columns(self, X, patterns):
        """Resolve column patterns (e.g., 'y_*') to actual column names."""
        resolved = []
        for pattern in patterns:
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                resolved.extend([c for c in X.columns if c.startswith(prefix)])
            elif pattern in X.columns:
                resolved.append(pattern)
        return resolved
