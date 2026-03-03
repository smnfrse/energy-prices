"""One-off backfill: generate retroactive forecast_history entries.

Truncates the merged dataset to simulate what would have been available at each
forecast date, runs the blend ensemble, and writes entries to forecast_history.json.

Designed to run in CI after normal inference (which handles data update + actuals).

Usage:
    python scripts/backfill_history.py --days 8
"""

from datetime import datetime, timedelta, timezone
import json

from loguru import logger
import numpy as np
import pandas as pd

from src.config import get_path
from src.deploy.inference import (
    DEPLOY_DATA_DIR,
    HISTORY_FILE,
    HISTORY_MAX_DAYS,
    _group_models_by_dataset,
    _impute_inference_gaps,
    _load_dataset_pipeline,
)
from src.modeling.blend import _flatten_daily_to_hourly, load_blend


def backfill(days: int = 8) -> None:
    """Generate forecast_history entries for the last N days."""
    merged_path = get_path("merged", "hour")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged dataset not found at {merged_path}")

    logger.info(f"Loading data from {merged_path}")
    df_full = pd.read_parquet(merged_path)
    logger.info(
        f"Loaded {len(df_full)} rows, index range {df_full.index.min()} to {df_full.index.max()}"
    )

    # Determine the last CET date in the data
    last_cet = df_full.index.max().tz_convert("Europe/Berlin").date()
    logger.info(f"Last CET date in data: {last_cet}")

    # Load blend ensemble once
    models, weights, config = load_blend()
    groups = _group_models_by_dataset(config["models"])

    # Load existing history
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupt forecast_history.json, starting fresh")

    existing_dates = {h["date"] for h in history}
    logger.info(f"Existing history: {len(history)} entries ({sorted(existing_dates)})")

    # Generate forecasts for each target date.
    # Forecast date D predicts prices for day D using data through D-1 (CET).
    # Cap at tomorrow (same as production inference) — we can't forecast further out.
    import zoneinfo

    generated_at = datetime.now(timezone.utc).isoformat()
    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    tomorrow_cet = datetime.now(tz).date() + timedelta(days=1)
    latest_forecast = min(last_cet + timedelta(days=1), tomorrow_cet)
    target_dates = [latest_forecast - timedelta(days=i) for i in range(days - 1, -1, -1)]
    logger.info(f"Backfill target dates: {[str(d) for d in target_dates]}")

    for forecast_date in target_dates:
        # Data cutoff: end of D-1 in CET.
        # CET=UTC+1 (winter), so 23:00 CET = 22:00 UTC
        data_last_date = forecast_date - timedelta(days=1)
        cutoff = pd.Timestamp(data_last_date, tz="UTC") + pd.Timedelta(hours=22)
        if cutoff > df_full.index.max():
            logger.info(f"  Skipping {forecast_date}: cutoff {cutoff} beyond data range")
            continue

        forecast_date_str = str(forecast_date)
        logger.info(
            f"  Generating forecast for {forecast_date_str} "
            f"(data through {data_last_date}, cutoff={cutoff})"
        )

        # Truncate data
        df_trunc = df_full.loc[:cutoff]
        if len(df_trunc) < 8760:  # need at least ~1 year of history
            logger.warning(
                f"  Skipping {forecast_date_str}: insufficient data ({len(df_trunc)} rows)"
            )
            continue

        # Run prediction pipeline (same logic as inference.py _predict_all_models)
        try:
            forecast_hourly = _predict_for_date(df_trunc, models, weights, config, groups)
        except Exception as e:
            logger.error(f"  Failed for {forecast_date_str}: {e}")
            continue

        # Update history (overwrite existing entry for this date)
        history = [h for h in history if h["date"] != forecast_date_str]
        history.append(
            {
                "date": forecast_date_str,
                "generated_at": generated_at,
                "prices": [round(float(p), 2) for p in forecast_hourly],
            }
        )
        logger.success(f"  Generated forecast for {forecast_date_str}")

    # Sort and trim
    history = sorted(history, key=lambda h: h["date"])[-HISTORY_MAX_DAYS:]

    # Write
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    logger.success(f"Wrote {len(history)} entries to {HISTORY_FILE}")

    for entry in history:
        logger.info(
            f"  {entry['date']}: [{entry['prices'][0]:.1f}, ..., {entry['prices'][-1]:.1f}]"
        )


def _predict_for_date(
    df: pd.DataFrame,
    models: list,
    weights: np.ndarray,
    config: dict,
    groups: dict[str, list[int]],
) -> np.ndarray:
    """Run blend prediction on truncated data, return 24h forecast."""
    all_preds: list[np.ndarray | None] = [None] * len(models)

    for dataset_run_id, model_indices in groups.items():
        pipeline, feature_names_out = _load_dataset_pipeline(dataset_run_id)

        X_transformed = pipeline.fit_transform(df)
        y_cols = X_transformed.filter(regex=r"^y(_\d+)?$").columns.tolist()
        X_all = X_transformed.drop(columns=y_cols)

        if feature_names_out:
            available = [c for c in feature_names_out if c in X_all.columns]
            X_features = X_all[available]
        else:
            X_features = X_all

        X_features = _impute_inference_gaps(X_features)

        for idx in model_indices:
            info = config["models"][idx]
            model = models[idx]
            gs = info["group_size"]

            X_pred = X_features.tail(1) if gs == 1 else X_features.tail(24)
            y_pred = model.predict(X_pred)
            all_preds[idx] = _flatten_daily_to_hourly(y_pred)

    # Weighted blend
    forecast_hourly = np.zeros(24)
    for i, pred in enumerate(all_preds):
        forecast_hourly += weights[i] * pred[:24]

    return forecast_hourly


if __name__ == "__main__":
    import typer

    def main(
        days: int = typer.Option(8, help="Number of days to backfill"),
    ):
        """Generate retroactive forecast history entries."""
        backfill(days=days)

    typer.run(main)
