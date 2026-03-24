"""Daily inference pipeline: generate 24h price forecast from blend ensemble.

Loads data from the merged dataset, applies preprocessing pipelines per dataset
group, generates predictions from all blend models, and outputs JSON files for
the dashboard.  Feature snapshots are saved as parquet for audit/analysis.

Usage:
    python -m src.deploy.inference              # Generate forecast with data update
    python -m src.deploy.inference --skip-update # Skip data download, use existing parquet
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, MODELS_DIR, PROJ_ROOT, get_path
from src.modeling.blend import _flatten_daily_to_hourly, load_blend

DEPLOY_DATA_DIR = PROJ_ROOT / "deploy" / "data"
SNAPSHOT_DIR = DEPLOY_DATA_DIR / "snapshots"
HISTORY_FILE = DEPLOY_DATA_DIR / "forecast_history.json"
HISTORY_MAX_DAYS = 30
SOURCE_SNAPSHOT_DAYS = 7  # Keep rolling window of source data snapshots


def run_inference(skip_update: bool = False, skip_ema: bool = False) -> dict:
    """Run the full daily inference pipeline.

    Args:
        skip_update: If True, skip data download (use existing inference parquet).
        skip_ema: If True, skip EMA overlay (use SMARD forecasts as-is).

    Returns:
        Dict with forecast data written to deploy/data/.
    """
    if not skip_update:
        _update_data()

    if not skip_ema:
        _apply_ema_overlay()

    merged_path = get_path("merged", "hour")
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {merged_path}. Run 'make update-data' first."
        )

    logger.info(f"Loading data from {merged_path}")
    df_full = pd.read_parquet(merged_path)
    logger.info(f"Loaded {len(df_full)} rows, {len(df_full.columns)} columns")
    df = _extend_to_forecast_date(df_full)

    # Load blend ensemble
    models, weights, config = load_blend()

    # Group models by dataset_run_id to avoid redundant pipeline applications
    groups = _group_models_by_dataset(config["models"])

    # Apply pipelines per group and collect predictions
    all_preds, snapshot_data = _predict_all_models(df, models, config["models"], groups)

    # Weighted blend (full ensemble, all models)
    forecast_hourly = np.zeros(24)
    for i, pred in enumerate(all_preds):
        forecast_hourly += weights[i] * pred[:24]

    # Stamp full blend onto each group's snapshot
    for ds_key, snap in snapshot_data.items():
        if "hour" in snap.columns and (snap["hour"] == -1).all():
            # Daily snapshot: store as blend_forecast_h0..h23
            for h in range(24):
                snap[f"blend_forecast_h{h}"] = forecast_hourly[h]
        else:
            # Hourly snapshot: one value per row
            snap["blend_forecast"] = forecast_hourly

    # Build forecast timestamps (next day, hours 0-23 CET)
    forecast_date = _get_forecast_date()
    forecast_timestamps = _build_forecast_timestamps(forecast_date)

    # Load recent actuals
    actuals = _get_recent_actuals(df_full)

    # Write output
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    result = _write_output(
        forecast_hourly,
        forecast_timestamps,
        forecast_date,
        actuals,
        config,
        generated_at,
        all_preds,
    )

    # Compute per-model errors from forecast history + full actuals
    _compute_model_errors(df_full)

    # Copy retrain_history.json to deploy/data/ if it exists
    _copy_retrain_history()

    # Write snapshots
    _write_feature_snapshots(snapshot_data, forecast_date, generated_at)
    _write_source_snapshot(df_full, forecast_date)

    logger.success(f"Forecast written for {forecast_date}")
    return result


RELEASE_ASSET_URL = (
    "https://github.com/smnfrse/energy-prices/releases/download/"
    "data-latest/merged_dataset_hourly.parquet"
)


def _download_release_asset(dest_path: Path) -> bool:
    """Download merged dataset from GitHub Release (public repo, no auth)."""
    import urllib.request

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading baseline data from release...")
        urllib.request.urlretrieve(RELEASE_ASSET_URL, dest_path)
        size_mb = dest_path.stat().st_size / 1024 / 1024
        logger.success(f"Downloaded {size_mb:.1f} MB to {dest_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download release asset: {e}")
        return False


def _update_data() -> None:
    """Download / update data via Makefile.

    Three paths depending on what data exists locally:
    1. Raw CSVs exist → ``make update`` (normal incremental, ~2 min)
    2. No raw CSVs but merged parquet exists → ``make update`` (bootstraps CSVs)
    3. Nothing exists → download merged parquet from GitHub Release, then ``make update``

    Falls back to ``make data`` (full download, ~50 min) only if the release
    download fails.
    """
    import subprocess

    from src.config import DATA_DIR

    raw_smard = DATA_DIR / "raw" / "smard_hourly"
    has_existing_data = raw_smard.exists() and any(raw_smard.glob("*.csv"))
    merged_path = get_path("merged", "hour")

    if has_existing_data:
        target = "update"
    elif merged_path.exists():
        target = "update"
    else:
        if _download_release_asset(merged_path):
            target = "update"
        else:
            logger.warning("Release download failed, falling back to full data download")
            target = "data"

    logger.info(
        f"Running make {target} ({'incremental' if target == 'update' else 'full bootstrap'})..."
    )
    result = subprocess.run(
        ["make", target],
        cwd=str(PROJ_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"make {target} failed:\n{result.stderr}")
        raise RuntimeError("Data update failed. Check logs above.")
    logger.success("Data update complete")


def _apply_ema_overlay() -> None:
    """Apply EMA overlay to merged dataset (historical + snapshots + live).

    Fetches today's live EMA forecast, combines with historical data and
    snapshots, then replaces SMARD forecast columns where EMA data exists.
    If no EMA data is available, logs a warning and continues.
    """
    from src.data.ema import build_ema_training_data, get_combined_ema_data

    merged_path = get_path("merged", "hour")
    if not merged_path.exists():
        logger.warning(f"Merged dataset not found at {merged_path}, skipping EMA overlay")
        return

    merged_df = pd.read_parquet(merged_path)

    ema_data = get_combined_ema_data(include_live=True)
    if ema_data.empty:
        logger.warning("No EMA data available, skipping overlay")
        return

    result = build_ema_training_data(merged_df, ema_data)
    result.to_parquet(merged_path)
    logger.info(f"EMA overlay applied ({len(ema_data)} EMA hours)")


def _group_models_by_dataset(model_infos: list[dict]) -> dict[str, list[int]]:
    """Group model indices by dataset_run_id."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, info in enumerate(model_infos):
        groups[info["dataset_run_id"]].append(i)
    return dict(groups)


def _load_dataset_pipeline(dataset_run_id: str) -> tuple:
    """Load preprocessing pipeline and expected output columns for a dataset.

    Checks models/production/pipeline_{run_id[:8]}/ first (pre-exported, no MLflow
    needed — works in CI). Falls back to MLflow for select_columns() datasets whose
    source run_id resolves to the parent build_pipeline() run.

    Args:
        dataset_run_id: MLflow run_id of the dataset.

    Returns:
        (pipeline, feature_names_out) tuple.

    Raises:
        ValueError: If no pipeline can be found for this dataset or its source.
    """
    PRODUCTION_DIR = PROJ_ROOT / "models" / "production"

    # Load metadata: check pre-exported file first (works in CI without mlruns/).
    # Falls back to MLflow artifact download for local dev / new datasets.
    local_meta = PRODUCTION_DIR / f"metadata_{dataset_run_id[:8]}.json"
    if local_meta.exists():
        metadata = json.loads(local_meta.read_text())
        logger.info(f"Loaded metadata from local file (dataset={dataset_run_id[:8]}...)")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = mlflow.artifacts.download_artifacts(
                run_id=dataset_run_id,
                artifact_path="metadata",
                dst_path=tmpdir,
            )
            metadata = json.loads((Path(local_dir) / "metadata.json").read_text())

    feature_names_out: list[str] = metadata.get("feature_names_out", [])
    source_run_id: str | None = metadata.get("source_run_id")

    # Try loading pipeline from pre-exported files in models/production/ first.
    # These are committed to git so CI doesn't need the full mlruns/ directory.
    for run_id in filter(None, [dataset_run_id, source_run_id]):
        local_path = PRODUCTION_DIR / f"pipeline_{run_id[:8]}"
        if local_path.exists():
            pipeline = mlflow.sklearn.load_model(str(local_path))
            logger.info(
                f"Loaded pipeline from local file (run_id={run_id[:8]}..., "
                f"dataset={dataset_run_id[:8]}...)"
            )
            return pipeline, feature_names_out

    # Fallback: load directly from MLflow artifact store (local dev only)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for run_id in filter(None, [dataset_run_id, source_run_id]):
        try:
            pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline")
            logger.info(
                f"Loaded pipeline from MLflow run_id={run_id[:8]}... "
                f"(dataset={dataset_run_id[:8]}...)"
            )
            return pipeline, feature_names_out
        except Exception:
            continue

    raise ValueError(
        f"No pipeline artifact found for dataset {dataset_run_id[:8]}... "
        f"and its source_run_id={source_run_id}. "
        "Export pipelines via 'python -m src.deploy.export_pipelines' or "
        "ensure mlruns/ contains the artifact."
    )


def _impute_inference_gaps(X_features: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaN values in the last 48 rows for known slow-moving features.

    Applied after pipeline transforms but before prediction. Only touches columns
    matching known patterns; everything else is left as NaN and logged.

    Args:
        X_features: Feature matrix after pipeline fit_transform.

    Returns:
        Imputed DataFrame.
    """
    FFILL_PATTERNS = (
        "pct_prog_",
        "pct_erdgas",
        "pct_",
        "_d1_h0-6_mean",
        "supply_demand_gap",
        "marktpreis_",
        "carbon_",
        "ttf_",
        "brent_",
    )

    df = X_features.copy()
    tail_idx = df.index[-48:] if len(df) >= 48 else df.index
    n_filled_total = 0

    nan_in_tail = df.loc[tail_idx].isna().any()
    cols_with_nan = nan_in_tail[nan_in_tail].index.tolist()

    for col in cols_with_nan:
        if any(col.startswith(p) or p in col for p in FFILL_PATTERNS):
            before = df[col].isna().sum()
            df[col] = df[col].ffill()
            filled = before - df[col].isna().sum()
            n_filled_total += filled

    if n_filled_total > 0:
        logger.info(f"Imputed {n_filled_total} NaN values via forward-fill")

    return df


def _predict_all_models(
    df: pd.DataFrame,
    models: list,
    model_infos: list[dict],
    groups: dict[str, list[int]],
) -> tuple[list[np.ndarray], dict[str, pd.DataFrame]]:
    """Apply pipeline per dataset group, predict with each model.

    For each dataset group:
    1. Load the unfitted pipeline (tracing to source if needed)
    2. fit_transform the full data
    3. Impute slow-moving gaps
    4. Select only the columns expected by models in this group
    5. Run each model's predict() and flatten to hourly

    Returns:
        (all_preds, snapshot_data) where snapshot_data maps dataset_run_id[:8]
        to a DataFrame of features + per-model predictions ready for parquet append.
    """
    all_preds: list[np.ndarray | None] = [None] * len(models)
    snapshot_data: dict[str, pd.DataFrame] = {}

    for dataset_run_id, model_indices in groups.items():
        logger.info(
            f"Processing dataset group {dataset_run_id[:8]}... ({len(model_indices)} models)"
        )

        pipeline, feature_names_out = _load_dataset_pipeline(dataset_run_id)

        # Apply unfitted pipeline to inference window
        X_transformed = pipeline.fit_transform(df)

        # Drop target columns (y / y_0 .. y_23)
        y_cols = X_transformed.filter(regex=r"^y(_\d+)?$").columns.tolist()
        X_all = X_transformed.drop(columns=y_cols)

        # Select only the columns this dataset's models expect
        if feature_names_out:
            missing = [c for c in feature_names_out if c not in X_all.columns]
            if missing:
                logger.warning(
                    f"  {len(missing)} expected columns not found in transformed data "
                    f"(e.g. {missing[:3]}). Using intersection."
                )
                feature_names_out = [c for c in feature_names_out if c in X_all.columns]
            X_features = X_all[feature_names_out]
        else:
            X_features = X_all

        # Impute slow-moving features before prediction
        X_features = _impute_inference_gaps(X_features)

        # Determine group_size (all models in a group share the same group_size)
        group_size = model_infos[model_indices[0]]["group_size"]

        # Build snapshot DataFrame from the prediction rows
        if group_size == 1:
            # Daily model: last 1 row, predictions are 24-element vectors
            snap = X_features.tail(1).copy()
            snap["hour"] = -1  # sentinel for daily
        else:
            # Hourly model: last 24 rows
            snap = X_features.tail(24).copy()
            snap["hour"] = list(range(24))

        # Predict with each model in this group
        for idx in model_indices:
            info = model_infos[idx]
            model = models[idx]
            model_name = info["name"]
            gs = info["group_size"]

            if gs == 1:
                X_pred = X_features.tail(1)
            else:
                X_pred = X_features.tail(24)

            logger.info(
                f"  Predicting {model_name} (group_size={gs}, X_pred shape={X_pred.shape})"
            )

            y_pred = model.predict(X_pred)
            hourly_pred = _flatten_daily_to_hourly(y_pred)
            all_preds[idx] = hourly_pred

            # Add predictions to snapshot
            if gs == 1:
                # Daily: 24 prediction values stored as separate columns
                for h in range(min(24, len(hourly_pred))):
                    snap[f"pred_{model_name}_h{h}"] = hourly_pred[h]
            else:
                # Hourly: one prediction per row
                snap[f"pred_{model_name}"] = hourly_pred[:24]

        snap = snap.reset_index(drop=True)
        snapshot_data[dataset_run_id[:8]] = snap

    return all_preds, snapshot_data


def _write_feature_snapshots(
    snapshot_data: dict[str, pd.DataFrame], forecast_date: str, generated_at: str
) -> None:
    """Append feature + prediction snapshots to per-group parquet files."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for ds_key, snap in snapshot_data.items():
        snap = snap.copy()
        snap.insert(0, "forecast_date", forecast_date)
        snap.insert(1, "generated_at", generated_at)

        path = SNAPSHOT_DIR / f"features_{ds_key}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, snap], ignore_index=True)
        else:
            combined = snap

        combined.to_parquet(path, index=False)
        logger.info(f"  Snapshot: {path.name} ({len(snap)} new rows, {len(combined)} total)")


def _write_source_snapshot(df: pd.DataFrame, forecast_date: str) -> None:
    """Save last 336 rows (14 days) of merged data; delete snapshots older than 7 days."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete old source snapshots
    for old in SNAPSHOT_DIR.glob("source_*.parquet"):
        try:
            file_date = old.stem.replace("source_", "")
            age = (
                datetime.strptime(forecast_date, "%Y-%m-%d").date()
                - datetime.strptime(file_date, "%Y-%m-%d").date()
            )
            if age.days > SOURCE_SNAPSHOT_DAYS:
                old.unlink()
                logger.info(f"  Deleted old source snapshot: {old.name}")
        except ValueError:
            continue

    # Save current source snapshot (last 336 rows = 14 days * 24 hours)
    tail = df.tail(336).copy()
    path = SNAPSHOT_DIR / f"source_{forecast_date}.parquet"
    tail.to_parquet(path)
    logger.info(f"  Source snapshot: {path.name} ({len(tail)} rows)")


def _extend_to_forecast_date(df: pd.DataFrame) -> pd.DataFrame:
    """Extend the merged dataset to include tomorrow's rows using EMA data.

    At 08:00 UTC on day D, SMARD data only extends to D (today). The EMA live
    forecast covers 168h ahead and already has D+1's generation/load data.
    This function appends 24 in-memory rows for D+1, populated with EMA
    forecast columns, so that tail(24) selects the correct prediction day.

    Raises RuntimeError if EMA data doesn't cover all 24 hours of D+1.
    """
    import zoneinfo

    from src.config import EMA_DATA_DIR

    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    forecast_date = (datetime.now(tz) + timedelta(days=1)).date()

    # Check if data already covers the forecast date (e.g. evening debug runs)
    last_cet = df.index.max()
    if hasattr(last_cet, "tz_convert"):
        last_cet = last_cet.tz_convert("Europe/Berlin")
    last_date = last_cet.date()
    if last_date >= forecast_date:
        logger.info(
            f"Data already extends to {last_date}, "
            f"no extension needed for forecast date {forecast_date}"
        )
        return df

    # Load latest EMA snapshot (saved by _apply_ema_overlay → save_ema_snapshot)
    snapshots = sorted(EMA_DATA_DIR.glob("*.parquet"))
    if not snapshots:
        raise RuntimeError(
            f"No EMA snapshots in {EMA_DATA_DIR}. Cannot extend data to {forecast_date}. "
            "Run with EMA enabled or ensure EMA data is available."
        )

    ema = pd.read_parquet(snapshots[-1])
    logger.info(f"Loaded EMA snapshot {snapshots[-1].name} ({len(ema)} rows)")

    # Build 24 hourly CET timestamps for the forecast date
    target_hours = pd.date_range(
        start=pd.Timestamp(forecast_date, tz=tz),
        periods=24,
        freq="h",
    )

    # Match EMA (UTC) to our target hours (CET)
    ema_cet = ema.copy()
    ema_cet.index = ema_cet.index.tz_convert(tz)
    ema_for_target = ema_cet.reindex(target_hours)

    n_covered = ema_for_target.notna().all(axis=1).sum()
    if n_covered < 24:
        raise RuntimeError(
            f"EMA data only covers {n_covered}/24 hours for {forecast_date}. "
            f"EMA snapshot range: {ema.index.min()} to {ema.index.max()} (UTC). "
            f"Cannot produce forecast for {forecast_date}."
        )

    # Build new rows preserving dtypes from the original DataFrame
    new_rows = pd.DataFrame(np.nan, index=target_hours, columns=df.columns, dtype="float64")

    # Populate the 8 EMA forecast columns
    for col in ema_for_target.columns:
        if col in new_rows.columns:
            new_rows[col] = ema_for_target[col].values

    # Set regime/metadata columns from the last existing row
    last_row = df.iloc[-1]
    for col in ("regime_de_at_lu", "regime_quarter_hourly"):
        if col in df.columns:
            new_rows[col] = last_row[col]
    new_rows["forecast_source"] = 1.0
    if "is_true_forecast" in df.columns:
        new_rows["is_true_forecast"] = 1.0

    result = pd.concat([df, new_rows])
    # pd.concat with all-NaN rows downgrades DatetimeIndex to plain Index;
    # restore it so downstream pipeline transformers work correctly.
    result.index = pd.to_datetime(result.index, utc=True).tz_convert("Europe/Berlin")
    logger.info(
        f"Extended data to {forecast_date} with {len(new_rows)} EMA rows "
        f"(total: {len(result)} rows)"
    )
    return result


def _get_forecast_date() -> str:
    """Return tomorrow's date (CET) as the forecast target.

    The pipeline extends the merged dataset to D+1 using EMA data before
    prediction. If extension fails, the pipeline aborts — so this always
    matches what tail(24) selects.
    """
    import zoneinfo

    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    return str(datetime.now(tz).date() + timedelta(days=1))


def _build_forecast_timestamps(forecast_date: str) -> list[str]:
    """Build ISO timestamps for each hour of the forecast date (CET/CEST)."""
    import zoneinfo

    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    base = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=tz)
    return [(base + pd.Timedelta(hours=h)).isoformat() for h in range(24)]


def _get_recent_actuals(df: pd.DataFrame) -> list[dict]:
    """Extract recent actual prices for dashboard overlay (last 7 complete days)."""
    if "target_price" not in df.columns:
        logger.warning("No target_price column in data, skipping actuals")
        return []

    df_berlin = df.copy()
    if hasattr(df_berlin.index, "tz_convert"):
        df_berlin.index = df_berlin.index.tz_convert("Europe/Berlin")

    actuals = []
    df_berlin["date"] = df_berlin.index.date
    for date, group in df_berlin.groupby("date"):
        prices = group["target_price"].values
        if len(prices) == 24 and not np.any(np.isnan(prices.astype(float))):
            actuals.append(
                {
                    "date": str(date),
                    "prices": [round(float(p), 2) for p in prices],
                }
            )

    return actuals[-7:]


def _write_output(
    forecast: np.ndarray,
    timestamps: list[str],
    forecast_date: str,
    actuals: list[dict],
    config: dict,
    generated_at: str,
    all_preds: list[np.ndarray] | None = None,
) -> dict:
    """Write JSON output files to deploy/data/."""
    # Build per-model predictions dict
    model_predictions = {}
    if all_preds:
        for pred, info in zip(all_preds, config["models"]):
            model_predictions[info["name"]] = [round(float(p), 2) for p in pred[:24]]

    # forecast_latest.json
    forecast_data = {
        "date": forecast_date,
        "generated_at": generated_at,
        "unit": "EUR/MWh",
        "timestamps": timestamps,
        "prices": [round(float(p), 2) for p in forecast],
    }
    _write_json(DEPLOY_DATA_DIR / "forecast_latest.json", forecast_data)

    # actuals_latest.json
    actuals_data = {
        "generated_at": generated_at,
        "unit": "EUR/MWh",
        "days": actuals,
    }
    _write_json(DEPLOY_DATA_DIR / "actuals_latest.json", actuals_data)

    # metadata.json
    metadata = {
        "last_updated": generated_at,
        "forecast_date": forecast_date,
        "blend_mae": config.get("blend_mae"),
        "blend_rmse": config.get("blend_rmse"),
        "blend_me": config.get("blend_me"),
        "blend_r2": config.get("blend_r2"),
        "n_models": len(config["models"]),
        "needs_reselection": config.get("needs_reselection", False),
        "model_categories": list({m["category"] for m in config["models"]}),
        "last_retrain": config.get("updated"),
        "model_details": [
            {
                "name": m["name"],
                "category": m["category"],
                "weight": m["weight"],
                "holdout_mae": m.get("holdout_mae"),
                "holdout_rmse": m.get("holdout_rmse"),
            }
            for m in config["models"]
        ],
    }
    _write_json(DEPLOY_DATA_DIR / "metadata.json", metadata)

    # Append to forecast_history.json (last 30 days)
    _append_history(forecast_date, forecast, generated_at, model_predictions)

    logger.info("Written forecast_latest.json, actuals_latest.json, metadata.json")
    return forecast_data


def _append_history(
    forecast_date: str,
    forecast: np.ndarray,
    generated_at: str,
    model_predictions: dict | None = None,
) -> None:
    """Append forecast to history file, keeping last HISTORY_MAX_DAYS entries."""
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupt forecast_history.json, starting fresh")

    # Remove existing entry for same date then append updated one
    history = [h for h in history if h["date"] != forecast_date]
    entry = {
        "date": forecast_date,
        "generated_at": generated_at,
        "prices": [round(float(p), 2) for p in forecast],
    }
    if model_predictions:
        entry["model_predictions"] = model_predictions
    history.append(entry)

    history = sorted(history, key=lambda h: h["date"])[-HISTORY_MAX_DAYS:]
    _write_json(HISTORY_FILE, history)


def _write_json(path: Path, data) -> None:
    """Write JSON file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _compute_model_errors(df_full: pd.DataFrame) -> None:
    """Compute per-model daily RMSE/MAE from forecast history and actuals in df_full."""
    if not HISTORY_FILE.exists():
        return
    try:
        history = json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return

    if "target_price" not in df_full.columns:
        return

    # Build actuals lookup from df_full (up to 30 days of complete data)
    df = df_full.copy()
    if hasattr(df.index, "tz_convert"):
        df.index = df.index.tz_convert("Europe/Berlin")
    df["date"] = df.index.date

    actuals_by_date: dict[str, np.ndarray] = {}
    for date, group in df.groupby("date"):
        prices = group["target_price"].values
        if len(prices) == 24 and not np.any(np.isnan(prices.astype(float))):
            actuals_by_date[str(date)] = prices

    # Collect all model names across history
    all_model_names: set[str] = set()
    for entry in history:
        if "model_predictions" in entry:
            all_model_names.update(entry["model_predictions"].keys())

    dates: list[str] = []
    rmse: dict[str, list] = defaultdict(list)
    mae: dict[str, list] = defaultdict(list)

    for entry in sorted(history, key=lambda h: h["date"]):
        date = entry["date"]
        if date not in actuals_by_date:
            continue

        actual = actuals_by_date[date]
        dates.append(date)

        # Blend errors (always available)
        forecast = np.array(entry["prices"])
        diff = forecast - actual
        rmse["blend"].append(round(float(np.sqrt(np.mean(diff**2))), 2))
        mae["blend"].append(round(float(np.mean(np.abs(diff))), 2))

        # Per-model errors (only for entries with model_predictions)
        model_preds = entry.get("model_predictions", {})
        for model_name in all_model_names:
            if model_name in model_preds:
                pred_arr = np.array(model_preds[model_name])
                diff = pred_arr - actual
                rmse[model_name].append(round(float(np.sqrt(np.mean(diff**2))), 2))
                mae[model_name].append(round(float(np.mean(np.abs(diff))), 2))
            else:
                rmse[model_name].append(None)
                mae[model_name].append(None)

    if not dates:
        return

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dates": dates,
        "rmse": dict(rmse),
        "mae": dict(mae),
    }
    _write_json(DEPLOY_DATA_DIR / "model_errors.json", result)
    logger.info(f"Written model_errors.json ({len(dates)} dates)")


def _copy_retrain_history() -> None:
    """Copy retrain_history.json from models/production/ to deploy/data/."""
    import shutil

    src = MODELS_DIR / "production" / "retrain_history.json"
    if src.exists():
        shutil.copy2(src, DEPLOY_DATA_DIR / "retrain_history.json")
        logger.info("Copied retrain_history.json to deploy/data/")


# --- CLI ---

if __name__ == "__main__":
    import typer

    def main(
        skip_update: bool = typer.Option(
            False, "--skip-update", help="Skip data download, use existing parquet."
        ),
        skip_ema: bool = typer.Option(
            False, "--skip-ema", help="Skip EMA overlay, use SMARD forecasts as-is."
        ),
    ):
        """Generate 24h price forecast from blend ensemble."""
        run_inference(skip_update=skip_update, skip_ema=skip_ema)

    typer.run(main)
