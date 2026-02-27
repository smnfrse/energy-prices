"""Baseline models for day-ahead energy price forecasting.

Simple baseline forecasters that work on univariate hourly time series:
- Naive baselines (daily/weekly persistence)
- ARIMA (classical statistical forecasting)
- Exponential Smoothing (Holt-Winters with seasonality)
- Prophet (Facebook's forecasting tool)

These baselines fit once on the **full dataset** and report in-sample performance metrics.
In-sample metrics are optimistic (no holdout), but this is acceptable for a quick benchmark
comparison against ML models. The baseline_predictions.npy output covers the full dataset.
"""

import logging

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR
from src.config.modeling import EXPERIMENTS
from src.features.transforms import TargetTransformer
from src.modeling.metrics import calculate_metrics

# =============================================================================
# Simple Baseline Functions
# =============================================================================


def naive_baseline(y: pd.Series, lag: int = 24) -> np.ndarray:
    """Naive in-sample forecast: use values from 'lag' hours ago.

    Returns in-sample predictions of length len(y). The first `lag` values will
    be NaN (no lagged value available); these are dropped when computing metrics.

    Args:
        y: Full time series
        lag: Hours to look back (24 for daily persistence, 168 for weekly seasonality)

    Returns:
        Prediction array of length len(y); first `lag` entries are NaN.
    """
    predictions = np.full(len(y), np.nan)
    predictions[lag:] = y.values[:-lag]
    return predictions


def arima_baseline(y: pd.Series, order: tuple = (2, 1, 2)) -> np.ndarray:
    """ARIMA baseline — fit once on full series, return in-sample fitted values.

    Args:
        y: Full time series
        order: ARIMA(p,d,q) order

    Returns:
        In-sample prediction array of length len(y).
    """
    model = ARIMA(y, order=order)
    fitted = model.fit()
    return fitted.fittedvalues.values


def exponential_smoothing_baseline(y: pd.Series) -> np.ndarray:
    """Holt-Winters exponential smoothing — fit once, return in-sample fitted values.

    Args:
        y: Full time series

    Returns:
        In-sample prediction array of length len(y).
    """
    model = ExponentialSmoothing(
        y,
        seasonal_periods=24,
        trend="add",
        seasonal="add",
        initialization_method="estimated",
    )
    fitted = model.fit()
    return fitted.fittedvalues.values


def prophet_baseline(y: pd.Series) -> np.ndarray:
    """Facebook Prophet — fit once on full series, return in-sample predictions.

    Args:
        y: Full time series with datetime index

    Returns:
        In-sample prediction array of length len(y).
    """
    from prophet import Prophet

    index = y.index
    if hasattr(index, "tz") and index.tz is not None:
        index = index.tz_localize(None)

    df = pd.DataFrame({"ds": index, "y": y.values})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
    )

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model.fit(df)
    forecast = model.predict(df)
    return forecast["yhat"].values


# =============================================================================
# Transformation Helpers
# =============================================================================


def apply_transformation(y: pd.Series, method: str) -> tuple[pd.Series, object]:
    """Apply transformation, return transformed series and transformer.

    Args:
        y: Original time series
        method: One of ["none", "log_shift", "yeo_johnson", "quantile"]

    Returns:
        (y_transformed, transformer) tuple
    """
    if method == "none":
        return y, None

    transformer = TargetTransformer(method=method)
    y_transformed = transformer.fit_transform(y.values.reshape(-1, 1)).ravel()
    return pd.Series(y_transformed, index=y.index), transformer


def inverse_transform(y_pred: np.ndarray, transformer) -> np.ndarray:
    """Inverse transform predictions.

    Args:
        y_pred: Predictions in transformed space
        transformer: Fitted transformer (or None if no transformation)

    Returns:
        Predictions in original space
    """
    if transformer is None:
        return y_pred
    return transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()


# =============================================================================
# Main Baseline Calculation Function
# =============================================================================


def calculate_baselines(
    data_path: str | None = None,
    transformations: list[str] | None = None,
    experiment: str | None = None,
    baselines: list[str] | None = None,
) -> dict[str, str]:
    """Train and log baseline models to MLflow using full-dataset in-sample evaluation.

    Each model is fit ONCE on the entire dataset and evaluated on in-sample predictions.

    IMPORTANT — horizon mismatch: ARIMA and ETS fittedvalues are ONE-STEP-AHEAD (h=1)
    predictions, not 24-step-ahead (day-ahead). Since adjacent hours are highly
    autocorrelated, h=1 gives artificially low MAE (~6-7) vs the actual day-ahead task.
    ARIMA and ETS are therefore excluded from this function's defaults; use
    calculate_rolling_baselines() for proper 24-step-ahead evaluation.

    Naive baselines (lag=24h, lag=168h) correctly measure day-ahead-equivalent performance
    because each prediction uses a value from the same hour the previous day/week.

    Args:
        data_path: Path to hourly parquet file. Defaults to
            PROCESSED_DATA_DIR/features_hourly.parquet
        transformations: List of transformations to try. Defaults to ["none", "log_shift"]
        experiment: MLflow experiment name. Defaults to EXPERIMENTS["baselines"]
        baselines: Baseline names to run. Options: ["naive_daily", "naive_weekly",
            "prophet"]. "arima" and "ets" are excluded by default (see horizon note above).

    Returns:
        Dict mapping "model_transformation" to MLflow run_id
    """
    if data_path is None:
        data_path = PROCESSED_DATA_DIR / "features_hourly.parquet"
    if transformations is None:
        transformations = ["none", "log_shift"]
    if experiment is None:
        experiment = EXPERIMENTS["baselines"]
    if baselines is None:
        baselines = ["naive_daily", "naive_weekly", "prophet"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    y = df["target_price"]
    logger.info(f"Loaded {len(y)} hourly observations")

    baseline_funcs = {
        "naive_daily": (naive_baseline, {"lag": 24}),
        "naive_weekly": (naive_baseline, {"lag": 168}),
        "arima": (arima_baseline, {"order": (2, 1, 2)}),
        "ets": (exponential_smoothing_baseline, {}),
        "prophet": (prophet_baseline, {}),
    }

    run_ids = {}

    for baseline in baselines:
        if baseline not in baseline_funcs:
            logger.warning(f"Unknown baseline '{baseline}', skipping")
            continue

        baseline_func, baseline_kwargs = baseline_funcs[baseline]

        if baseline == "prophet":
            try:
                import prophet  # noqa: F401
            except ImportError:
                logger.warning(
                    "Prophet not installed, skipping. Install with: pip install prophet"
                )
                continue

        for transformation in transformations:
            run_name = f"{baseline}_{transformation}"
            logger.info(f"Running {run_name}...")

            try:
                # Statistical models require NaN-free input; drop leading structural NaN
                if baseline in ("arima", "ets", "prophet"):
                    y_fit = y.dropna()
                else:
                    y_fit = y

                y_transformed, transformer = apply_transformation(y_fit, transformation)
                predictions_transformed = baseline_func(y_transformed, **baseline_kwargs)
                predictions = inverse_transform(predictions_transformed, transformer)

                # Align predictions back to full y length so metrics compare on same scale
                if len(predictions) < len(y):
                    full_predictions = np.full(len(y), np.nan)
                    valid_positions = ~y.isna().values
                    full_predictions[valid_positions] = predictions
                    predictions = full_predictions

                # Drop NaN entries (first `lag` values for naive baselines)
                valid_mask = ~np.isnan(predictions)
                y_eval = y.values[valid_mask]
                predictions_eval = predictions[valid_mask]

                metrics = calculate_metrics(y_eval, predictions_eval)

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_metrics(metrics)
                    mlflow.log_params(
                        {
                            "model": baseline,
                            "transformation": transformation,
                            "n_total_hours": len(y),
                            "n_valid_hours": int(valid_mask.sum()),
                            "data_path": str(data_path),
                        }
                    )
                    if baseline == "naive_daily":
                        mlflow.log_param("lag_hours", 24)
                    elif baseline == "naive_weekly":
                        mlflow.log_param("lag_hours", 168)
                    elif baseline == "arima":
                        mlflow.log_param("order", str(baseline_kwargs["order"]))
                    elif baseline == "ets":
                        mlflow.log_param("seasonal_periods", 24)

                    mlflow.set_tag("model_type", "baseline")
                    mlflow.set_tag("forecast_type", "in_sample")

                    run_id = mlflow.active_run().info.run_id
                    run_ids[run_name] = run_id

                logger.info(
                    f"  {run_name}: RMSE={metrics['rmse']:.2f}, "
                    f"MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to run {run_name}: {e}")
                continue

    logger.info(f"Completed {len(run_ids)} baseline runs")
    return run_ids


# =============================================================================
# Baseline Prediction Utilities
# =============================================================================


def get_default_baseline_predictions() -> np.ndarray:
    """Load default baseline predictions for skill score calculations.

    Loads cached baseline predictions from data/baseline_predictions.npy.
    Shape is (n_total_hours,) — a flat 1D array covering the full dataset.

    Note: These are in-sample predictions (no holdout). Skill scores computed
    against them are therefore also in-sample and optimistic. This is acceptable
    for a quick benchmark comparison against ML models.

    Returns:
        Baseline predictions array of shape (n_total_hours,).

    Raises:
        FileNotFoundError: If baseline predictions file not found
    """
    from src.config import DATA_DIR

    baseline_path = DATA_DIR / "baseline_predictions.npy"

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline predictions not found at {baseline_path}. "
            "Generate baseline predictions first using calculate_baselines() "
            "and save with save_baseline_from_mlflow(run_id)."
        )

    return np.load(baseline_path)


def save_baseline_from_mlflow(run_id: str, output_path: str | None = None) -> None:
    """Regenerate and save baseline predictions from an MLflow baseline run.

    Args:
        run_id: MLflow run_id of the baseline model
        output_path: Path to save predictions. Defaults to DATA_DIR/baseline_predictions.npy

    Raises:
        ValueError: If run_id is missing required params
        FileNotFoundError: If data file not found
    """
    from pathlib import Path

    from src.config import DATA_DIR, MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run = mlflow.get_run(run_id)

    params = run.data.params
    required_params = ["model", "transformation", "data_path"]
    missing = [p for p in required_params if p not in params]
    if missing:
        raise ValueError(
            f"Run {run_id} is missing required parameters: {missing}. "
            "Is this a baseline model run?"
        )

    baseline_name = params["model"]
    transformation = params["transformation"]
    data_path = params["data_path"]

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    y = df["target_price"]

    y_transformed, transformer = apply_transformation(y, transformation)

    baseline_funcs = {
        "naive_daily": (naive_baseline, {"lag": 24}),
        "naive_weekly": (naive_baseline, {"lag": 168}),
        "arima": (arima_baseline, {"order": (2, 1, 2)}),
        "ets": (exponential_smoothing_baseline, {}),
        "prophet": (prophet_baseline, {}),
    }

    if baseline_name not in baseline_funcs:
        raise ValueError(
            f"Unknown baseline model: {baseline_name}. "
            f"Expected one of {list(baseline_funcs.keys())}"
        )

    baseline_func, baseline_kwargs = baseline_funcs[baseline_name]
    predictions_transformed = baseline_func(y_transformed, **baseline_kwargs)
    predictions = inverse_transform(predictions_transformed, transformer)

    if output_path is None:
        output_path = DATA_DIR / "baseline_predictions.npy"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)

    logger.info(f"Saved baseline predictions from run {run_id}")
    logger.info(f"  Model: {baseline_name} ({transformation})")
    logger.info(f"  Shape: {predictions.shape}")
    logger.info(f"  Path: {output_path}")
    logger.info(f"  RMSE: {run.data.metrics.get('rmse', 'N/A'):.2f}")
    logger.info(f"  MAE: {run.data.metrics.get('mae', 'N/A'):.2f}")


# =============================================================================
# Rolling Holdout Evaluation
# =============================================================================


def _ets_rolling_preds(fit, y_holdout: pd.Series, holdout_days: int) -> np.ndarray:
    """Manual Holt-Winters (additive trend + additive seasonal) rolling forecast.

    statsmodels HoltWintersResults has no .append() method, so we implement the
    state update recursion directly using the fitted parameters:

        l_t = α*(y_t - s_{t-m}) + (1-α)*(l_{t-1} + b_{t-1})
        b_t = β*(l_t - l_{t-1}) + (1-β)*b_{t-1}
        s_t = γ*(y_t - l_{t-1} - b_{t-1}) + (1-γ)*s_{t-m}
        ŷ_{t+h} = l_t + h*b_t + s_{t+h-m}

    Args:
        fit: Fitted HoltWintersResultsWrapper from ExponentialSmoothing.fit().
        y_holdout: Holdout observations (ground truth to feed back each day).
        holdout_days: Number of days to roll through.

    Returns:
        Array of shape (holdout_days * 24,) with rolling 24h-ahead predictions.
    """
    m = 24
    alpha = float(fit.params["smoothing_level"])
    beta = float(fit.params["smoothing_trend"])
    gamma = float(fit.params["smoothing_seasonal"])

    # State at end of training: seasons[-m:] = [s_{t-m+1}, ..., s_t]
    # seasons[0] is the oldest, used for h=1 forecast (s_{t+1-m}).
    level = float(fit.level.iloc[-1])
    trend = float(fit.trend.iloc[-1])
    seasons = fit.season.iloc[-m:].values.copy().astype(float)

    # pos tracks which slot to write/read next.
    # pos=0 means seasons[0] is s_{t-m+1} → used for the first h=1 forecast.
    pos = 0

    all_preds = []
    for day in range(holdout_days):
        # Forecast next 24h: ŷ_{t+h} = level + h*trend + s_{t+h-m}
        day_preds = [level + (h + 1) * trend + seasons[(pos + h) % m] for h in range(m)]
        all_preds.extend(day_preds)

        # Update state with the 24 actual observations
        actual = y_holdout.iloc[day * m : (day + 1) * m].values
        for y_obs in actual:
            level_prev, trend_prev = level, trend
            s_old = seasons[pos % m]
            level = alpha * (y_obs - s_old) + (1 - alpha) * (level_prev + trend_prev)
            trend = beta * (level - level_prev) + (1 - beta) * trend_prev
            seasons[pos % m] = gamma * (y_obs - level_prev - trend_prev) + (1 - gamma) * s_old
            pos += 1

    return np.array(all_preds)


def rolling_day_ahead_preds(
    y: pd.Series,
    holdout_days: int = 90,
    model_type: str = "sarima",
    sarima_order: tuple = (1, 1, 1),
    sarima_seasonal_order: tuple = (1, 1, 1, 24),
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling 24h-ahead forecasts over the holdout period.

    Fits once on the training portion (all data before the last holdout_days),
    then for each holdout day:
      1. Forecast the next 24 hours
      2. Update model state with the 24 actual observations (no parameter refit)

    For SARIMA, uses statsmodels .append(refit=False) — a cheap Kalman filter step.
    For ETS, uses a manual Holt-Winters state update (HoltWintersResults has no
    .append() in statsmodels 0.14.x).

    Args:
        y: Full hourly target series (no NaNs).
        holdout_days: Number of days for holdout evaluation.
        model_type: "sarima" or "ets".
        sarima_order: ARIMA (p,d,q) order (sarima only).
        sarima_seasonal_order: Seasonal (P,D,Q,s) order (sarima only).

    Returns:
        (holdout_preds, y_holdout_values) — both shape (holdout_days * 24,).
        Returns y_holdout so callers can compute metrics without re-slicing.
    """
    split_idx = len(y) - holdout_days * 24
    y_train = y.iloc[:split_idx]
    y_holdout = y.iloc[split_idx:]

    if model_type == "sarima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        logger.info(
            f"Fitting SARIMA{sarima_order}x{sarima_seasonal_order} on "
            f"{len(y_train):,} training obs..."
        )
        results = SARIMAX(
            y_train,
            order=sarima_order,
            seasonal_order=sarima_seasonal_order,
        ).fit(disp=False)

        logger.info(f"Rolling forecast: {holdout_days} days × 24h...")
        all_preds = []
        for day in range(holdout_days):
            day_forecast = results.forecast(steps=24)
            all_preds.extend(day_forecast.values)
            actual = y_holdout.iloc[day * 24 : (day + 1) * 24]
            results = results.append(actual, refit=False)
        return np.array(all_preds), y_holdout.values[: holdout_days * 24]

    elif model_type == "ets":
        logger.info(f"Fitting ETS on {len(y_train):,} training obs...")
        fit = ExponentialSmoothing(
            y_train,
            seasonal_periods=24,
            trend="add",
            seasonal="add",
            initialization_method="estimated",
        ).fit()
        logger.info(f"Rolling forecast: {holdout_days} days × 24h...")
        preds = _ets_rolling_preds(fit, y_holdout, holdout_days)
        return preds, y_holdout.values[: holdout_days * 24]

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'sarima' or 'ets'.")


def calculate_rolling_baselines(
    data_path=None,
    holdout_days: int = 90,
    models: list[str] | None = None,
    experiment: str | None = None,
) -> dict[str, dict]:
    """Compute rolling holdout forecasts for SARIMA and/or ETS and log to MLflow.

    Fits once on the training portion, then rolls through the holdout period using
    warm-start .append(refit=False) — cheap Kalman/EWM state updates, no refitting.
    Also prints a comparison table against blend_config.json holdout MAEs if available.

    Args:
        data_path: Path to features_hourly.parquet. Defaults to PROCESSED_DATA_DIR.
        holdout_days: Number of holdout days. Should match BLEND_HOLDOUT_DAYS for
            apples-to-apples comparison against ML models.
        models: List of model types to evaluate. Defaults to ["sarima", "ets"].
        experiment: MLflow experiment name. Defaults to EXPERIMENTS["baselines"].

    Returns:
        Dict of {model_name: {"mae": float, "rmse": float, "run_id": str}}.
    """
    import json

    from src.config import MODELS_DIR

    if data_path is None:
        data_path = PROCESSED_DATA_DIR / "features_hourly.parquet"
    if models is None:
        models = ["sarima", "ets"]
    if experiment is None:
        experiment = EXPERIMENTS["baselines"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    y = df["target_price"].dropna()
    logger.info(f"Loaded {len(y):,} hourly observations (after dropna)")

    results_out: dict[str, dict] = {}

    for model_type in models:
        logger.info(f"=== {model_type.upper()} rolling holdout ({holdout_days}d) ===")
        try:
            preds, y_true = rolling_day_ahead_preds(
                y, holdout_days=holdout_days, model_type=model_type
            )
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(y_true, preds)
            rmse = float(np.sqrt(mean_squared_error(y_true, preds)))

            run_name = f"rolling_{model_type}_{holdout_days}d"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({"mae": mae, "rmse": rmse})
                mlflow.log_params(
                    {
                        "model": model_type,
                        "forecast_type": "rolling_holdout",
                        "holdout_days": holdout_days,
                        "n_train_hours": len(y) - holdout_days * 24,
                        "data_path": str(data_path),
                    }
                )
                mlflow.set_tag("model_type", "baseline")
                mlflow.set_tag("forecast_type", "rolling_holdout")
                run_id = mlflow.active_run().info.run_id

            results_out[model_type] = {"mae": mae, "rmse": rmse, "run_id": run_id}
            logger.info(f"  {model_type}: MAE={mae:.2f}, RMSE={rmse:.2f}")

        except Exception as e:
            logger.error(f"Failed to run {model_type}: {e}")
            continue

    # Print comparison table against blend_config.json if available
    blend_path = MODELS_DIR / "production" / "blend_config.json"
    if blend_path.exists() and results_out:
        try:
            config = json.loads(blend_path.read_text())
            best_ml_mae = min(m["holdout_mae"] for m in config["models"])
            blend_mae = config["blend_mae"]

            logger.info("")
            logger.info("=" * 55)
            logger.info("Holdout MAE Comparison (lower is better)")
            logger.info("=" * 55)
            logger.info(f"  {'Model':<20} {'MAE':>8}  {'vs blend':>10}")
            logger.info(f"  {'-' * 20}  {'-' * 7}  {'-' * 9}")
            logger.info(f"  {'Blend ensemble':<20} {blend_mae:>8.2f}  {'(reference)':>10}")
            logger.info(
                f"  {'Best ML model':<20} {best_ml_mae:>8.2f}  {best_ml_mae / blend_mae - 1:>+9.1%}"
            )
            for name, res in results_out.items():
                logger.info(
                    f"  {name:<20} {res['mae']:>8.2f}  {res['mae'] / blend_mae - 1:>+9.1%}"
                )
            logger.info("=" * 55)
        except Exception as e:
            logger.warning(f"Could not load blend_config.json for comparison: {e}")

    return results_out


if __name__ == "__main__":
    run_ids = calculate_baselines()
    logger.info(f"Baseline run IDs: {run_ids}")
