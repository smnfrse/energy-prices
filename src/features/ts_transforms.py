"""Time series transformers for energy price forecasting.

Provides transformers for:
- RollingStatsTransformer: Flexible windowed statistics on hourly data
- EWMATransformer: Exponentially weighted moving averages with cutoff points
- DailyPivotTransformer: Convert hourly to daily rows with configurable pivoting
- SameHourLagTransformer: Same-hour lag features (D-1, D-7, etc.)
- HourlyDailyAggregateTransformer: Daily aggregates broadcast to hourly rows

All transformers follow sklearn's BaseEstimator/TransformerMixin interface.
"""

from dataclasses import dataclass
from datetime import timedelta

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_AGG_FUNCS = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "min": np.nanmin,
    "max": np.nanmax,
    "sum": np.nansum,
}


def _numpy_agg(arr, func_name):
    """Apply named aggregation using numpy (avoids pandas overhead)."""
    return _AGG_FUNCS[func_name](arr)


# ============================================================================
# LAG-BASED TRANSFORMERS (from lag_transforms.py)
# ============================================================================


@dataclass
class WindowSpec:
    """Specification for a rolling window aggregation.

    Defines a relative window of days/hours to aggregate for each prediction day D.

    Parameters:
        start_day: Days before prediction day (e.g., -7 for seven days before).
        end_day: Days before prediction day (e.g., -1 for one day before).
        hours: Filter ALL days to these hours (e.g., range(8, 20) for peak hours).
        end_hour: Cutoff hour on end_day only (e.g., 11 for 11am cutoff).
            Overrides hours filter for the last day.
        agg: Aggregation function: 'mean', 'std', 'min', 'max', 'sum'.
    """

    start_day: int
    end_day: int
    hours: range | list[int] | None = None
    end_hour: int | None = None
    agg: str = "mean"


def _resolve_columns(columns, X):
    """Resolve column patterns (with wildcards) to actual column names.

    Args:
        columns: List of column names or wildcard patterns (e.g., "stromerzeugung_*").
        X: DataFrame to resolve against.

    Returns:
        List of resolved column names.
    """
    resolved = []
    for pattern in columns:
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            matched = [c for c in X.columns if c.startswith(prefix)]
            if not matched:
                logger.warning(f"No columns matched pattern '{pattern}'")
            resolved.extend(matched)
        else:
            resolved.append(pattern)
    return resolved


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Compute windowed statistics on hourly data using flexible WindowSpec.

    Each WindowSpec defines a relative window of days/hours to aggregate.
    The transformer operates on hourly data and outputs hourly data with
    daily-level values (constant within each day). DailyPivotTransformer
    then reduces to daily with "first" aggregation.

    Parameters:
        columns: List of column names or wildcard patterns to compute stats for.
        windows: List of WindowSpec defining the windows to compute.
        overwrite: If True (with exactly 1 window), replace original column
            values with the lagged aggregate instead of creating new columns.
    """

    def __init__(self, columns, windows, overwrite=False):
        self.columns = columns
        self.windows = windows
        self.overwrite = overwrite

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.overwrite and len(self.windows) != 1:
            raise ValueError(f"overwrite=True requires exactly 1 window, got {len(self.windows)}")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        resolved = _resolve_columns(self.columns, X)
        dates = X.index.normalize()
        # Pre-compute date→row indices mapping once (shared across columns/windows)
        date_indices = pd.Series(np.arange(len(dates)), index=dates).groupby(level=0).indices
        hours_arr = X.index.hour.values
        created = []

        for col in resolved:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found, skipping rolling stats")
                continue

            for window in self.windows:
                if self.overwrite:
                    X[col] = self._compute_window(X[col].values, date_indices, hours_arr, window)
                    created.append(col)
                else:
                    new_col = self._make_col_name(col, window)
                    X[new_col] = self._compute_window(
                        X[col].values, date_indices, hours_arr, window
                    )
                    created.append(new_col)

        logger.info(f"Created {len(created)} rolling window features (overwrite={self.overwrite})")
        return X

    def _compute_window(self, values, date_indices, hours_arr, window):
        """Compute window aggregate for each date.

        Uses pre-computed date→indices dict for O(1) lookups per date instead
        of O(n) boolean masks over the full series.
        """
        result = np.full(len(values), np.nan)
        sorted_dates = sorted(date_indices.keys())
        agg_func = window.agg
        hours_set = set(window.hours) if window.hours is not None else None

        for date in sorted_dates:
            # Collect row indices from all days in the window
            window_rows = []
            for day_offset in range(window.start_day, window.end_day + 1):
                offset_date = date + timedelta(days=day_offset)
                if offset_date not in date_indices:
                    continue

                idx = date_indices[offset_date]
                is_end_day = day_offset == window.end_day

                # Apply hour filtering on the small per-day array (~24 rows)
                if hours_set is not None:
                    if window.end_hour is not None and is_end_day:
                        mask = hours_arr[idx] < window.end_hour
                    else:
                        mask = np.isin(hours_arr[idx], list(hours_set))
                    idx = idx[mask]
                elif window.end_hour is not None and is_end_day:
                    mask = hours_arr[idx] < window.end_hour
                    idx = idx[mask]

                window_rows.append(idx)

            # Compute aggregate
            if window_rows:
                all_idx = np.concatenate(window_rows)
                if len(all_idx) > 0:
                    result[date_indices[date]] = _numpy_agg(values[all_idx], agg_func)

        return result

    def _make_col_name(self, col, window):
        """Generate descriptive column name from window specification.

        Format examples:
            target_price_d7_mean        (single day -7)
            target_price_d7_d1_std      (range -7 to -1)
            target_price_d1_to10_mean   (single day -1 with end_hour=10)
            target_price_d7_d1_h8-19_mean  (range with hours filter)
        """
        s = abs(window.start_day)
        e = abs(window.end_day)

        if window.start_day == window.end_day:
            base = f"{col}_d{s}"
        else:
            base = f"{col}_d{s}_d{e}"

        if window.hours is not None:
            hours_list = list(window.hours)
            h_start, h_end = hours_list[0], hours_list[-1]
            base = f"{base}_h{h_start}-{h_end}"
        elif window.end_hour is not None:
            base = f"{base}_to{window.end_hour}"

        return f"{base}_{window.agg}"


class EWMATransformer(BaseEstimator, TransformerMixin):
    """Compute EWMA on hourly data with configurable cutoff.

    Computes exponentially weighted moving average and extracts the value
    at a specified cutoff point for each prediction day. The cutoff value
    is then broadcast to all hours of the prediction day.

    Parameters:
        columns: List of column names or wildcard patterns to compute EWMA for.
        spans: List of EWMA spans in hours.
        cutoff_day: Relative day offset for cutoff (e.g., -1).
        cutoff_hour: Hour cutoff on the cutoff_day (e.g., 11 for 11am).
            If None, uses last hour of cutoff_day.
        adjust: Whether to use adjusted EWMA (default: False).
    """

    def __init__(
        self,
        columns,
        spans,
        cutoff_day=-1,
        cutoff_hour=None,
        adjust=False,
        col_suffix="",
    ):
        self.columns = columns
        self.spans = spans
        self.cutoff_day = cutoff_day
        self.cutoff_hour = cutoff_hour
        self.adjust = adjust
        self.col_suffix = col_suffix

    def __setstate__(self, state):
        state.setdefault("col_suffix", "")
        self.__dict__.update(state)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        resolved = _resolve_columns(self.columns, X)
        dates = X.index.normalize()
        date_indices = pd.Series(np.arange(len(dates)), index=dates).groupby(level=0).indices
        hours_arr = X.index.hour.values
        created = []

        for col in resolved:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found, skipping EWMA")
                continue

            for span in self.spans:
                new_col = f"{col}_ewma_{span}h{self.col_suffix}"
                X[new_col] = self._compute_ewma_with_cutoff(X[col], date_indices, hours_arr, span)
                created.append(new_col)

        logger.info(f"Created {len(created)} EWMA features")
        return X

    def _compute_ewma_with_cutoff(self, series, date_indices, hours_arr, span):
        """Compute EWMA with cutoff for each date."""
        ewma_values = series.ewm(span=span, adjust=self.adjust).mean().values
        result = np.full(len(series), np.nan)

        for date in sorted(date_indices.keys()):
            cutoff_date = date + timedelta(days=self.cutoff_day)
            if cutoff_date not in date_indices:
                continue

            idx = date_indices[cutoff_date]

            if self.cutoff_hour is not None:
                mask = hours_arr[idx] == self.cutoff_hour - 1
                idx = idx[mask]

            if len(idx) > 0:
                result[date_indices[date]] = ewma_values[idx[-1]]

        return pd.Series(result, index=series.index)


# ============================================================================
# DAILY PIVOT TRANSFORMER (from daily_transforms.py)
# ============================================================================


def normalize_dst(df: pd.DataFrame, timezone: str = "Europe/Berlin") -> pd.DataFrame:
    """Normalize DST transitions to ensure exactly 24 hours per day.

    Spring-forward (23h): interpolates missing hour.
    Fall-back (25h): averages duplicate hour.

    Args:
        df: DataFrame with UTC datetime index.
        timezone: Target timezone for conversion (default: Europe/Berlin).

    Returns:
        DataFrame in local timezone with exactly 24 hours per day.
    """
    df = df.copy()

    # Convert UTC to local timezone
    logger.info(f"Converting to {timezone}")
    df.index = df.index.tz_convert(timezone)

    dates = df.index.date
    hours_per_day = pd.Series(dates).value_counts()

    # Identify DST transition days
    spring_days = hours_per_day[hours_per_day == 23].index
    fall_days = hours_per_day[hours_per_day == 25].index

    # Handle spring-forward (missing hour)
    if len(spring_days) > 0:
        logger.info(f"Found {len(spring_days)} spring-forward days (23 hours)")
        for day in spring_days:
            day_mask = df.index.date == day
            day_data = df[day_mask]

            # Find missing hour
            hours = day_data.index.hour
            all_hours = set(range(24))
            present_hours = set(hours)
            missing_hours = all_hours - present_hours

            if not missing_hours:
                continue

            missing_hour = min(missing_hours)
            prev_hour = missing_hour - 1
            next_hour = missing_hour + 1

            prev_data = day_data[day_data.index.hour == prev_hour]
            next_data = day_data[day_data.index.hour == next_hour]

            if len(prev_data) > 0 and len(next_data) > 0:
                # Create timestamp for missing hour
                missing_ts = prev_data.index[0] + pd.Timedelta(hours=1)

                # Interpolate numeric columns, copy others
                new_row = pd.DataFrame(index=[missing_ts])
                for col in df.columns:
                    prev_val = prev_data[col].iloc[0]
                    next_val = next_data[col].iloc[0]
                    if isinstance(prev_val, (int, float, np.integer, np.floating)):
                        new_row[col] = (prev_val + next_val) / 2
                    else:
                        new_row[col] = prev_val

                df = pd.concat([df, new_row]).sort_index()

    # Handle fall-back (duplicate hour)
    if len(fall_days) > 0:
        logger.info(f"Found {len(fall_days)} fall-back days (25 hours)")
        for day in fall_days:
            day_mask = df.index.date == day
            day_data = df[day_mask]

            # Find duplicate hours
            hour_counts = day_data.groupby(day_data.index.hour).size()
            dup_hours = hour_counts[hour_counts > 1].index

            for dup_hour in dup_hours:
                dup_mask = day_mask & (df.index.hour == dup_hour)
                dup_rows = df[dup_mask]

                if len(dup_rows) <= 1:
                    continue

                # Average numeric columns, keep first for others
                avg_row = pd.DataFrame(index=[dup_rows.index[0]])
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        avg_row[col] = dup_rows[col].mean()
                    else:
                        avg_row[col] = dup_rows[col].iloc[0]

                # Remove duplicates and insert averaged row
                df = df[~dup_mask]
                df = pd.concat([df, avg_row]).sort_index()

    return df


@dataclass
class PivotSpec:
    """Specification for pivoting a column into 24 hourly columns.

    Attributes:
        column: Source column to pivot.
        day_offset: 0=today, -1=yesterday, -7=week ago, etc.
        prefix: Output prefix for the 24 columns (default: column name).
    """

    column: str
    day_offset: int = 0
    prefix: str | None = None


class DailyPivotTransformer(BaseEstimator, TransformerMixin):
    """Convert hourly rows to daily rows with configurable column pivoting.

    Pivots specified columns into 24 hourly columns (col_0 to col_23) with
    optional day offsets for lag features. Aggregates remaining columns using
    pattern-based rules.

    This transformer eliminates the need for separate LaggedPriceTransformer
    and LeakagePreventionTransformer by encoding temporal offsets directly
    into the pivot specifications.

    Parameters:
        pivot_specs: List of PivotSpec defining which columns to pivot and how.
        aggregation_rules: Dict mapping column patterns to agg functions.
            If None, uses AGGREGATION_RULES from src/config/features.py.
        timezone: Timezone for daily grouping (default: Europe/Berlin).

    Example:
        >>> transformer = DailyPivotTransformer(
        ...     pivot_specs=[
        ...         PivotSpec(column="target_price", day_offset=0, prefix="y"),
        ...         PivotSpec(column="target_price", day_offset=-1, prefix="price_lag"),
        ...     ]
        ... )
    """

    def __init__(
        self,
        pivot_specs=None,
        aggregation_rules=None,
        timezone="Europe/Berlin",
    ):
        self.pivot_specs = pivot_specs or []
        self.aggregation_rules = aggregation_rules
        self.timezone = timezone

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure index is in the target timezone
        if X.index.tz is None:
            logger.warning(f"Index has no timezone, localizing to {self.timezone}")
            X.index = X.index.tz_localize(self.timezone)
        elif str(X.index.tz) != self.timezone:
            logger.info(f"Converting index from {X.index.tz} to {self.timezone}")
            X.index = X.index.tz_convert(self.timezone)

        # Extract date for grouping
        X["_date"] = X.index.date

        # Group by date
        grouped = X.groupby("_date")

        # Collect all column data to avoid fragmentation
        daily_data = {}

        # Process pivot specifications
        pivoted_columns = set()
        for spec in self.pivot_specs:
            if spec.column not in X.columns:
                logger.warning(f"Pivot column '{spec.column}' not found, skipping")
                continue

            pivoted_columns.add(spec.column)
            prefix = spec.prefix or spec.column

            # Extract 24-hour arrays for each day
            hourly_vals = grouped[spec.column].apply(lambda x: x.values)

            # Create 24 columns
            for hour in range(24):
                col_name = f"{prefix}_{hour}"
                daily_series = hourly_vals.apply(
                    lambda arr, h=hour: (arr[h] if arr is not None and len(arr) > h else np.nan)
                )

                # Apply day offset if specified
                if spec.day_offset != 0:
                    daily_series = daily_series.shift(abs(spec.day_offset))

                daily_data[col_name] = daily_series

            logger.info(
                f"Pivoted '{spec.column}' with offset={spec.day_offset} "
                f"as '{prefix}_0' to '{prefix}_23'"
            )

        # Aggregate remaining columns
        if self.aggregation_rules is not None:
            agg_rules = self.aggregation_rules
        else:
            from src.config.features import build_aggregation_rules

            agg_rules = build_aggregation_rules()
        skip_cols = pivoted_columns | {"_date"}

        for col in X.columns:
            if col in skip_cols or col in daily_data:
                continue

            agg_func = self._get_agg_func(col, agg_rules)
            if agg_func is not None:
                try:
                    daily_data[col] = grouped[col].agg(agg_func)
                except Exception as e:
                    logger.warning(f"Failed to aggregate '{col}' with '{agg_func}': {e}")

        # Build DataFrame from dict
        daily_df = pd.DataFrame(daily_data, index=hourly_vals.index)

        # Set index
        daily_df.index = pd.DatetimeIndex(pd.to_datetime(daily_df.index), name="date").tz_localize(
            self.timezone
        )

        logger.info(
            f"Transformed {len(X)} hourly rows to {len(daily_df)} daily rows, "
            f"{len(daily_df.columns)} columns"
        )
        return daily_df

    def _get_agg_func(self, col_name, rules):
        """Match a column name to its aggregation function via pattern rules."""
        for pattern, func in rules.items():
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if col_name.startswith(prefix):
                    return func
            elif col_name == pattern:
                return func
        # Default: mean for numeric
        return "mean"


# ============================================================================
# SAME-HOUR LAG TRANSFORMER
# ============================================================================


class SameHourLagTransformer(BaseEstimator, TransformerMixin):
    """Create lag features at the same hour of previous days.

    Shifts columns by multiples of 24 hours to get same-hour values
    from D-1, D-7, etc.

    Parameters:
        column: Column name to create lags for.
        day_offsets: List of day offsets (negative = past).
            Default: (-1, -7).
    """

    def __init__(self, column, day_offsets=(-1, -7)):
        self.column = column
        self.day_offsets = day_offsets

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.column not in X.columns:
            logger.warning(f"Column '{self.column}' not found, skipping SameHourLag")
            return X

        created = []
        for offset in self.day_offsets:
            shift = abs(offset) * 24
            col_name = f"{self.column}_lag_d{abs(offset)}"
            X[col_name] = X[self.column].shift(shift)
            created.append(col_name)

        logger.info(f"Created {len(created)} same-hour lag features: {created}")
        return X


# ============================================================================
# HOURLY-DAILY AGGREGATE TRANSFORMER
# ============================================================================


class HourlyDailyAggregateTransformer(BaseEstimator, TransformerMixin):
    """Compute daily aggregates from hourly data and broadcast back.

    For hour-varying columns (forecasts), creates summary stats for the
    full day. Useful for capturing "shape of day" context.

    Parameters:
        columns: List of column names or wildcard patterns.
        aggs: Tuple of aggregation functions.
            Default: ("sum", "mean", "max").
    """

    def __init__(self, columns, aggs=("sum", "mean", "max")):
        self.columns = columns
        self.aggs = aggs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        resolved = _resolve_columns(self.columns, X)
        dates = X.index.normalize()
        created = []

        for col in resolved:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found, skipping daily aggregate")
                continue

            daily_grouped = X[col].groupby(dates)

            for agg in self.aggs:
                new_col = f"{col}_daily_{agg}"
                daily_vals = daily_grouped.transform(agg)
                X[new_col] = daily_vals
                created.append(new_col)

            # Add share feature: value / daily_sum
            daily_sum = daily_grouped.transform("sum")
            share_col = f"{col}_share"
            X[share_col] = X[col] / daily_sum.replace(0, np.nan)
            created.append(share_col)

        logger.info(f"Created {len(created)} daily aggregate features")
        return X
