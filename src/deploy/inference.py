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

from src.config import MLFLOW_TRACKING_URI, PROJ_ROOT, get_path
from src.modeling.blend import _flatten_daily_to_hourly, load_blend

DEPLOY_DATA_DIR = PROJ_ROOT / "deploy" / "data"
SNAPSHOT_DIR = DEPLOY_DATA_DIR / "snapshots"
HISTORY_FILE = DEPLOY_DATA_DIR / "forecast_history.json"
HISTORY_MAX_DAYS = 30
SOURCE_SNAPSHOT_DAYS = 7  # Keep rolling window of source data snapshots


def run_inference(skip_update: bool = False) -> dict:
    """Run the full daily inference pipeline.

    Args:
        skip_update: If True, skip data download (use existing inference parquet).

    Returns:
        Dict with forecast data written to deploy/data/.
    """
    if not skip_update:
        _update_data()

    merged_path = get_path("merged", "hour")
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {merged_path}. Run 'make update-data' first."
        )

    logger.info(f"Loading data from {merged_path}")
    df_full = pd.read_parquet(merged_path)
    logger.info(f"Loaded {len(df_full)} rows, {len(df_full.columns)} columns")
    df = df_full

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
    forecast_date = _get_forecast_date(df)
    forecast_timestamps = _build_forecast_timestamps(forecast_date)

    # Load recent actuals
    actuals = _get_recent_actuals(df_full)

    # Write output
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    result = _write_output(
        forecast_hourly, forecast_timestamps, forecast_date, actuals, config, generated_at
    )

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
        "_d1_h0-10_mean",
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


def _get_forecast_date(df: pd.DataFrame) -> str:
    """Determine the forecast target date (day after latest data).

    Capped at tomorrow (CET) so that afternoon debugging runs target the same
    day as production runs (which always execute before SMARD publishes
    day-ahead prices at ~12:00 CET). In production the cap has no effect.
    """
    import zoneinfo

    last_ts = df.index.max()
    if hasattr(last_ts, "tz_convert"):
        last_date = last_ts.tz_convert("Europe/Berlin").date()
    else:
        last_date = pd.Timestamp(last_ts).date()
    data_driven = last_date + timedelta(days=1)

    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    tomorrow_cet = datetime.now(tz).date() + timedelta(days=1)

    if data_driven > tomorrow_cet:
        logger.info(
            f"Forecast date capped at {tomorrow_cet} "
            f"(data extends to {last_date}, afternoon run detected)"
        )
        return str(tomorrow_cet)
    return str(data_driven)


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
) -> dict:
    """Write JSON output files to deploy/data/."""
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
        "blend_r2": config.get("blend_r2"),
        "n_models": len(config["models"]),
        "needs_reselection": config.get("needs_reselection", False),
        "model_categories": list({m["category"] for m in config["models"]}),
    }
    _write_json(DEPLOY_DATA_DIR / "metadata.json", metadata)

    # Append to forecast_history.json (last 30 days)
    _append_history(forecast_date, forecast, generated_at)

    logger.info("Written forecast_latest.json, actuals_latest.json, metadata.json")
    return forecast_data


def _append_history(forecast_date: str, forecast: np.ndarray, generated_at: str) -> None:
    """Append forecast to history file, keeping last HISTORY_MAX_DAYS entries."""
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupt forecast_history.json, starting fresh")

    # Remove existing entry for same date then append updated one
    history = [h for h in history if h["date"] != forecast_date]
    history.append(
        {
            "date": forecast_date,
            "generated_at": generated_at,
            "prices": [round(float(p), 2) for p in forecast],
        }
    )

    history = sorted(history, key=lambda h: h["date"])[-HISTORY_MAX_DAYS:]
    _write_json(HISTORY_FILE, history)


def _write_json(path: Path, data) -> None:
    """Write JSON file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# --- CLI ---

if __name__ == "__main__":
    import typer

    def main(
        skip_update: bool = typer.Option(
            False, "--skip-update", help="Skip data download, use existing parquet."
        ),
    ):
        """Generate 24h price forecast from blend ensemble."""
        run_inference(skip_update=skip_update)

    typer.run(main)
