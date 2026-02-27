"""Metrics for time series forecasting evaluation.

Single function that calculates all metrics and returns a flat dict
suitable for mlflow.log_metrics().
"""

from typing import Literal

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray | Literal["default"] | None = None,
) -> dict:
    """Calculate all evaluation metrics.

    Args:
        y_true: True values — either (N,) / (N,1) for scalar or
            (N, 24) for daily
        y_pred: Predicted values — must match y_true shape
        y_baseline: Baseline predictions for skill score calculation.
            - None: skill metrics not calculated
            - 'default': load from data/baseline_predictions.npy
            - np.ndarray: user-provided baseline (must match y_true shape)

    Returns:
        Flat dict of metric_name -> value, ready for mlflow.log_metrics().

    Raises:
        ValueError: If y_baseline shape doesn't match y_true
        FileNotFoundError: If y_baseline='default' but baseline file not found
    """
    # Squeeze (N,1) to (N,)
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
    if y_baseline is not None and isinstance(y_baseline, np.ndarray):
        if y_baseline.ndim == 2 and y_baseline.shape[1] == 1:
            y_baseline = y_baseline.ravel()

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        msg = f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        raise ValueError(msg)

    is_scalar = y_true.ndim == 1

    if is_scalar:
        # Scalar target: flat metrics only
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        me = float(np.mean(y_pred - y_true))
        r2 = r2_score(y_true, y_pred)
        metrics = {"rmse": rmse, "mae": mae, "me": me, "r2": r2}
    else:
        # 2D target (N, 24): existing behavior unchanged
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        me = float(np.mean(y_pred_flat - y_true_flat))
        r2 = r2_score(y_true_flat, y_pred_flat)

        hourly_rmse = np.array(
            [
                np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
                for h in range(y_true.shape[1])
            ]
        )
        daily_max_rmse = hourly_rmse.max()
        rmse_std = hourly_rmse.std()

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "me": me,
            "r2": r2,
            "daily_max_rmse": daily_max_rmse,
            "rmse_std": rmse_std,
        }

    # Skill scores if baseline provided (same logic for 1D and 2D)
    if y_baseline is not None:
        # Load default baseline if requested
        if isinstance(y_baseline, str) and y_baseline == "default":
            from src.modeling.baselines import (
                get_default_baseline_predictions,
            )

            y_baseline = get_default_baseline_predictions()

        # Handle baseline shape mismatches
        if y_baseline.shape != y_true.shape:
            # If baseline has more rows, slice from end
            # (time series alignment)
            if y_baseline.shape[0] > y_true.shape[0]:
                from loguru import logger

                logger.warning(
                    f"Baseline has {y_baseline.shape[0]} rows but "
                    f"y_true has {y_true.shape[0]}. Auto-slicing "
                    f"baseline from end to match."
                )
                y_baseline = y_baseline[-len(y_true) :]

                # Verify shape after slicing
                if y_baseline.shape != y_true.shape:
                    logger.warning(
                        f"After slicing, baseline shape "
                        f"{y_baseline.shape} still doesn't match "
                        f"y_true {y_true.shape}. Skipping skill "
                        f"scores."
                    )
                    return metrics
            else:
                # Baseline has fewer rows - cannot align reliably
                from loguru import logger

                logger.warning(
                    f"Baseline has {y_baseline.shape[0]} rows but "
                    f"y_true has {y_true.shape[0]}. Cannot align - "
                    f"skipping skill scores."
                )
                return metrics

        y_baseline_flat = y_baseline.flatten()
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        rmse_baseline = np.sqrt(mean_squared_error(y_true_flat, y_baseline_flat))
        mae_baseline = mean_absolute_error(y_true_flat, y_baseline_flat)

        metrics["rmse_skill"] = 1 - (metrics["rmse"] / rmse_baseline)
        metrics["mae_skill"] = 1 - (metrics["mae"] / mae_baseline)

    return metrics


def reshape_hourly_to_daily(
    y_hourly: np.ndarray,
    n_hours: int = 24,
) -> np.ndarray:
    """Reshape (N*24,) hourly predictions to (N, 24) for daily-level.

    Args:
        y_hourly: Hourly predictions with shape (N*24,) or (N*24, 1)
        n_hours: Hours per day (default 24)

    Returns:
        Reshaped array with shape (N, 24)

    Raises:
        ValueError: If length is not divisible by n_hours
    """
    if y_hourly.ndim == 2 and y_hourly.shape[1] == 1:
        y_hourly = y_hourly.ravel()
    if len(y_hourly) % n_hours != 0:
        msg = f"Length {len(y_hourly)} is not divisible by {n_hours}"
        raise ValueError(msg)
    return y_hourly.reshape(-1, n_hours)


def calculate_hourly_model_metrics(
    y_true_hourly: np.ndarray,
    y_pred_hourly: np.ndarray,
    y_baseline: np.ndarray | None = None,
) -> dict:
    """Calculate both per-hour and aggregate metrics.

    Computes scalar metrics on hourly data, then reshapes to (N, 24)
    for daily-level analysis. Returns both prefixed metrics.

    Args:
        y_true_hourly: Hourly true values with shape (N*24,)
        y_pred_hourly: Hourly predictions with shape (N*24,)
        y_baseline: Optional baseline predictions (N*24,) or (N, 24)

    Returns:
        Dict with both 'hourly_' and 'daily_' prefixed metrics
    """
    # Scalar metrics on flattened hourly data
    scalar = calculate_metrics(
        y_true_hourly,
        y_pred_hourly,
        y_baseline=None,
    )

    # Reshape to (N, 24) for daily-level analysis
    y_true_daily = reshape_hourly_to_daily(y_true_hourly)
    y_pred_daily = reshape_hourly_to_daily(y_pred_hourly)
    daily = calculate_metrics(
        y_true_daily,
        y_pred_daily,
        y_baseline=y_baseline,
    )

    # Merge with prefixes to distinguish
    result = {f"hourly_{k}": v for k, v in scalar.items()}
    result.update({f"daily_{k}": v for k, v in daily.items()})
    return result


def group_feature_importance(importance_df):
    """
    Group features by regex patterns.

    Args:
        importance_df: DataFrame with columns [feature, importance]

    Returns:
        DataFrame with columns [group, total_importance, n_features, avg_importance]
    """
    # Define feature groups with regex patterns
    feature_groups = [
        ("Actual Generation", r"^stromerzeugung_"),
        ("Generation Percentages", r"^pct_(?!prog)"),
        ("Forecasted Generation", r"^prognostizierte_erzeugung_"),
        ("Forecasted Percentages", r"^pct_prog_"),
        ("Neighbor Prices", r"^marktpreis_"),
        ("Price Spreads", r"^spread_"),
        ("Price Ratios", r"^ratio_"),
        ("Cross-Border Flows", r"^cross-border_flows_"),
        ("Net Exports", r"^net_export_"),
        ("Target Price Features", r"^target_price"),
        ("Hour Temporal", r"^hour_"),
        ("Day Temporal", r"^day_"),
        ("Month Temporal", r"^month_"),
        ("Week Temporal", r"^week_"),
        ("Binary Temporal", r"^is_"),
        ("Target Values", r"^y_\d+$"),
        ("Lagged Prices", r"^price_lag_\d+$"),
        ("Forecasted Consumption", r"^prog_verbrauch_\d+$"),
        ("Forecasted Generation (Hourly)", r"^prog_erzeugung_\d+$"),
        ("Forecasted Wind+PV", r"^prog_wind_pv_\d+$"),
        ("Carbon Price", r"^carbon_"),
        ("TTF Gas Price", r"^ttf_"),
        ("Brent Oil Price", r"^brent_"),
    ]

    # Assign group to each feature
    importance_df = importance_df.copy()
    importance_df["group"] = "Other"

    for group_name, pattern in feature_groups:
        mask = importance_df["feature"].str.match(pattern)
        importance_df.loc[mask, "group"] = group_name

    # Aggregate by group
    grouped = (
        importance_df.groupby("group")
        .agg(
            total_importance=("importance", "sum"),
            n_features=("feature", "count"),
            avg_importance=("importance", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values("total_importance", ascending=False).reset_index(drop=True)

    # Check for unmatched features
    unmatched = importance_df[importance_df["group"] == "Other"]
    if len(unmatched) > 0:
        print(f"⚠️  {len(unmatched)} features did not match any pattern:")
        print(unmatched["feature"].head(10).tolist())

    return grouped
