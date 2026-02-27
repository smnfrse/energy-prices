"""Preprocessing pipeline registration and dataset caching via MLflow.

Provides a unified workflow for feature engineering experiments:
1. Build pipeline → validates, fits on full data, saves dataset + unfitted pipeline
2. Load dataset → retrieve cached preprocessed features for fast model training

Usage:
    from src.features.preprocessors import build_pipeline, load_dataset
    from sklearn.pipeline import Pipeline

    # Define preprocessing pipeline
    pipe = Pipeline([("step1", SomeTransformer()), ...])

    # Build and cache dataset (validates + fits + saves everything)
    dataset_id = build_pipeline(pipe, name="my_features_v1")

    # Load for model training (fast - no re-preprocessing)
    X, y, metadata = load_dataset("my_features_v1")
"""

import json
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.config import MLFLOW_TRACKING_URI, get_path
from src.features.transforms import ColumnDropper, FeatureScaler


def build_pipeline(
    pipeline: Pipeline,
    name: str,
    description: str = "",
    tags: dict | None = None,
    resolution: str = "hour",
) -> str:
    """Validate, fit, and register a preprocessing pipeline with full dataset.

    This is the main entry point for feature engineering experiments. It:
    1. Validates pipeline structure and checks for data leakage
    2. Loads full merged data from processed/
    3. Fits and transforms the pipeline
    4. Splits features (X) and targets (y matching r"^y_\\d+$")
    5. Saves X, y, unfitted pipeline, and metadata to MLflow under experiment "datasets"

    The unfitted pipeline is saved for reproducibility - you can reload it to
    apply the same transformations to new data.

    Args:
        pipeline: Unfitted sklearn Pipeline to register.
        name: Dataset name for versioning (same name = new version).
        description: Optional description.
        tags: Optional dict of MLflow tags.
        resolution: Data resolution for loading merged data ("hour" or "quarterhour").

    Returns:
        MLflow run_id for the cached dataset.

    Raises:
        ValueError: If pipeline is invalid or wrong number of target columns found.
    """
    # Validate pipeline
    _validate_pipeline_structure(pipeline)
    _validate_leakage_if_applicable(pipeline)

    # Load full merged data
    logger.info(f"Loading merged data (resolution={resolution})...")
    X_raw = _load_merged_data(resolution)

    # Fit and transform
    logger.info("Fitting and transforming pipeline...")
    fitted_pipeline = clone(pipeline)
    X_transformed = fitted_pipeline.fit_transform(X_raw)

    # Split X and y
    y_cols = X_transformed.filter(regex=r"^y(_\d+)?$").columns.tolist()
    if len(y_cols) not in (1, 24):
        raise ValueError(
            f"Expected 1 (hourly) or 24 (daily) target columns, found {len(y_cols)}: {y_cols}"
        )

    y = X_transformed[y_cols]
    X = X_transformed.drop(columns=y_cols)

    # Drop rows where any target column is NaN
    rows_with_nan_targets = y.isna().any(axis=1)
    n_rows_to_drop = rows_with_nan_targets.sum()

    if n_rows_to_drop > 0:
        first_dropped = y[rows_with_nan_targets].index.min()
        last_dropped = y[rows_with_nan_targets].index.max()
        logger.warning(
            f"Dropping {n_rows_to_drop} rows with missing targets "
            f"(first: {first_dropped}, last: {last_dropped})"
        )
        valid_mask = ~rows_with_nan_targets
        X = X[valid_mask]
        y = y[valid_mask]

    logger.info(f"Dataset created: X={X.shape}, y={y.shape}")

    # Extract metadata
    metadata = _build_metadata(
        pipeline=pipeline,
        fitted_pipeline=fitted_pipeline,
        X_raw=X_raw,
        X=X,
        y=y,
        name=name,
    )

    all_tags = {
        "dataset_name": name,
        "resolution": resolution,
    }
    if tags:
        all_tags.update(tags)

    return _save_dataset_to_mlflow(X, y, metadata, name, all_tags, description, pipeline=pipeline)


def load_dataset(
    name: str | None = None,
    version: int | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a preprocessed dataset from MLflow by name or run_id.

    Args:
        name: Dataset name (requires exactly one of name or run_id).
        version: Optional version number (only with name, defaults to latest, 1-indexed).
        run_id: MLflow run_id (alternative to name).

    Returns:
        (X, y, metadata) where:
        - X: Feature DataFrame with DatetimeIndex
        - y: Target DataFrame with DatetimeIndex and columns y_0..y_23
        - metadata: dict with dataset info, preprocessing lineage, feature names

    Raises:
        ValueError: If both name and run_id provided, or neither provided.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Validate arguments
    if (name is None) == (run_id is None):
        raise ValueError("Must provide exactly one of name or run_id")

    if run_id is not None:
        # Load by run_id directly
        if version is not None:
            raise ValueError("version parameter not allowed when using run_id")

        logger.info(f"Loading dataset from run_id={run_id}")
    else:
        # Load by name (original behavior)
        # Get experiment
        experiment = mlflow.get_experiment_by_name("datasets")
        if experiment is None:
            raise ValueError("No datasets experiment found")

        # Query for datasets with matching name
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.dataset_name = '{name}' AND attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
        )

        if runs.empty:
            raise ValueError(f"No datasets found with name '{name}'")

        # Select version
        if version is not None:
            if version < 1 or version > len(runs):
                raise ValueError(f"Version {version} not found (available: 1-{len(runs)})")
            run_row = runs.iloc[version - 1]
        else:
            run_row = runs.iloc[0]  # Latest

        run_id = run_row["run_id"]
        logger.info(
            f"Loading dataset '{name}' (version {version or len(runs)}) from run_id={run_id}"
        )

    # Download artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="metadata",
            dst_path=tmpdir,
        )
        local_path = Path(local_dir)

        # Load dataframes
        X = pd.read_parquet(local_path / "X.parquet")
        y = pd.read_parquet(local_path / "y.parquet")

        # Load metadata
        metadata = json.loads((local_path / "metadata.json").read_text())

    metadata["run_id"] = run_id
    logger.info(f"Loaded dataset: X={X.shape}, y={y.shape}")
    return X, y, metadata


def load_pipeline(
    name: str | None = None,
    version: int | None = None,
    run_id: str | None = None,
) -> Pipeline:
    """Load the unfitted preprocessing pipeline from a dataset.

    Useful for applying the same transformations to new data or inspecting
    the pipeline definition.

    Args:
        name: Dataset name (requires exactly one of name or run_id).
        version: Optional version number (only with name, defaults to latest, 1-indexed).
        run_id: MLflow run_id (alternative to name).

    Returns:
        Unfitted sklearn Pipeline.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if run_id is not None:
        if name is not None or version is not None:
            raise ValueError("Cannot combine run_id with name or version")
        model_uri = f"runs:/{run_id}/pipeline"
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded pipeline from run_id={run_id}")
        return pipeline

    if name is None:
        raise ValueError("Must provide either name or run_id")

    # Get experiment
    experiment = mlflow.get_experiment_by_name("datasets")
    if experiment is None:
        raise ValueError("No datasets experiment found")

    # Query for datasets with matching name
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.dataset_name = '{name}' AND attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        raise ValueError(f"No datasets found with name '{name}'")

    # Select version
    if version is not None:
        if version < 1 or version > len(runs):
            raise ValueError(f"Version {version} not found (available: 1-{len(runs)})")
        run_row = runs.iloc[version - 1]
    else:
        run_row = runs.iloc[0]  # Latest

    run_id = run_row["run_id"]
    model_uri = f"runs:/{run_id}/pipeline"
    pipeline = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Loaded pipeline from dataset '{name}' (run_id={run_id})")
    return pipeline


def select_columns(
    source_run_id: str,
    name: str,
    columns: list[str],
    description: str = "",
    tags: dict | None = None,
) -> str:
    """Create a new dataset by selecting a subset of X columns from an existing one.

    Loads a dataset by run_id, subsets X to the requested columns, and saves the
    result as a new MLflow run under the 'datasets' experiment.

    Args:
        source_run_id: MLflow run_id of the source dataset.
        name: Dataset name for the new run.
        columns: List of X feature columns to keep.
        description: Optional description.
        tags: Optional dict of MLflow tags.

    Returns:
        MLflow run_id for the new dataset.

    Raises:
        ValueError: If any requested columns are not present in the source dataset.
    """
    X, y, source_meta = load_dataset(run_id=source_run_id)

    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"Columns not found in source dataset: {missing}")

    X_selected = X[columns]
    dropped = sorted(c for c in X.columns if c not in columns)

    metadata = {
        "dataset_name": name,
        "n_samples": len(X_selected),
        "n_features": len(X_selected.columns),
        "n_targets": len(y.columns),
        "date_start": str(X_selected.index.min().date()),
        "date_end": str(X_selected.index.max().date()),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "format": source_meta.get("format", "hourly"),
        "n_features_in": len(X.columns),
        "n_features_out": len(X_selected.columns),
        "feature_names_in": list(X.columns),
        "feature_names_out": list(X_selected.columns),
        "target_names": list(y.columns),
        "dropped_columns": dropped,
        "scaler_info": {},
        "source_run_id": source_run_id,
    }

    all_tags: dict = {"dataset_name": name, "source_run_id": source_run_id}
    if tags:
        all_tags.update(tags)

    return _save_dataset_to_mlflow(
        X_selected, y, metadata, name, all_tags, description, pipeline=None
    )


def list_datasets(tags: dict | None = None) -> pd.DataFrame:
    """List registered datasets from MLflow.

    Args:
        tags: Optional filter tags (e.g., {"resolution": "hour"}).

    Returns:
        DataFrame with columns: run_id, dataset_name, version, n_samples,
        n_features, n_targets, n_steps, created_at.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Get experiment
    experiment = mlflow.get_experiment_by_name("datasets")
    if experiment is None:
        return pd.DataFrame()

    # Build filter
    filter_parts = ["attributes.status = 'FINISHED'"]
    if tags:
        for key, value in tags.items():
            filter_parts.append(f"tags.{key} = '{value}'")
    filter_string = " AND ".join(filter_parts)

    # Query runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
    )

    if runs.empty:
        return runs

    # Extract relevant columns
    result = pd.DataFrame()
    result["run_id"] = runs["run_id"]
    result["dataset_name"] = runs["tags.dataset_name"]
    result["n_samples"] = runs["params.n_samples"].astype(int)
    result["n_features"] = runs["params.n_features"].astype(int)
    result["n_targets"] = runs["params.n_targets"].astype(int)
    result["n_steps"] = runs["params.n_steps"].astype(int)
    result["created_at"] = pd.to_datetime(runs["start_time"])

    # Add version numbers (grouped by dataset_name)
    result["version"] = result.groupby("dataset_name").cumcount(ascending=False) + 1

    # Reorder columns
    result = result[
        [
            "run_id",
            "dataset_name",
            "version",
            "n_samples",
            "n_features",
            "n_targets",
            "n_steps",
            "created_at",
        ]
    ]

    return result.reset_index(drop=True)


# =============================================================================
# Data loading
# =============================================================================


def _load_merged_data(resolution: str = "hour") -> pd.DataFrame:
    """Load the merged dataset from processed directory.

    Args:
        resolution: Data resolution - "hour" or "quarterhour".

    Returns:
        DataFrame with datetime index.
    """
    path = get_path("merged", resolution)
    logger.info(f"Loading merged dataset from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_pipeline_structure(pipeline: Pipeline) -> None:
    """Validate basic pipeline structure.

    Raises:
        ValueError: If pipeline is empty or has duplicate step names.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(pipeline).__name__}")

    if len(pipeline.steps) == 0:
        raise ValueError("Pipeline has no steps")

    names = [name for name, _ in pipeline.steps]
    duplicates = [n for n in names if names.count(n) > 1]
    if duplicates:
        raise ValueError(f"Duplicate step names: {set(duplicates)}")


def _validate_leakage_if_applicable(pipeline: Pipeline) -> None:
    """Run leakage validation if pipeline contains time series transformers."""
    from src.features.ts_transforms import (
        DailyPivotTransformer,
        EWMATransformer,
        HourlyDailyAggregateTransformer,
        RollingStatsTransformer,
        SameHourLagTransformer,
    )

    has_ts_transformers = any(
        isinstance(
            t,
            (
                RollingStatsTransformer,
                DailyPivotTransformer,
                EWMATransformer,
                SameHourLagTransformer,
                HourlyDailyAggregateTransformer,
            ),
        )
        for _, t in pipeline.steps
    )

    if has_ts_transformers:
        from src.features.validation import validate_pipeline_leakage

        validate_pipeline_leakage(pipeline)


# =============================================================================
# Metadata extraction
# =============================================================================


def _build_metadata(
    pipeline: Pipeline,
    fitted_pipeline: Pipeline,
    X_raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.DataFrame,
    name: str,
) -> dict:
    """Build comprehensive metadata dict for the dataset.

    Returns dict with:
        - Dataset info: n_samples, n_features, n_targets, date_start, date_end
        - Preprocessing info: n_features_in/out, feature_names_in/out, dropped_columns, scaler_info
        - Lineage: dataset_name, created_at
    """
    metadata = {
        # Dataset info
        "dataset_name": name,
        "n_samples": len(X),
        "n_features": len(X.columns),
        "n_targets": len(y.columns),
        "date_start": str(X.index.min().date()),
        "date_end": str(X.index.max().date()),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "format": "hourly" if len(y.columns) == 1 else "daily",
        # Preprocessing info
        "n_features_in": len(X_raw.columns),
        "n_features_out": len(X.columns),
        "feature_names_in": list(X_raw.columns),
        "feature_names_out": list(X.columns),
        "target_names": list(y.columns),
        "dropped_columns": [],
        "scaler_info": {},
    }

    # Find dropped columns
    for step_name, transformer in fitted_pipeline.steps:
        if isinstance(transformer, ColumnDropper):
            dropped = set(X_raw.columns) - set(X.columns) - set(y.columns)
            metadata["dropped_columns"] = sorted(dropped)

    # Find scaler info
    for step_name, transformer in fitted_pipeline.steps:
        if isinstance(transformer, FeatureScaler):
            info = {
                "step_name": step_name,
                "method": transformer.method,
                "columns": list(transformer.columns_) if hasattr(transformer, "columns_") else [],
            }
            if hasattr(transformer, "shift_"):
                info["shift"] = float(transformer.shift_)
            metadata["scaler_info"] = info

    return metadata


# =============================================================================
# Artifact helpers
# =============================================================================


def _save_list_file(items: list, path: Path) -> None:
    """Save a list as a line-delimited text file."""
    path.write_text("\n".join(str(item) for item in items))


def _save_dataset_to_mlflow(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: dict,
    name: str,
    tags: dict,
    description: str,
    pipeline: Pipeline | None = None,
) -> str:
    """Save a preprocessed dataset to MLflow under the 'datasets' experiment.

    Args:
        X: Feature DataFrame.
        y: Target DataFrame.
        metadata: Metadata dict with keys: n_samples, n_features, n_targets,
            n_features_in, n_features_out, feature_names_in, feature_names_out,
            dropped_columns, scaler_info.
        name: MLflow run name.
        tags: MLflow tags to set on the run.
        description: Optional run description.
        pipeline: Unfitted sklearn Pipeline to log as an artifact; skipped if None.

    Returns:
        MLflow run_id.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("datasets")

    with mlflow.start_run(run_name=name) as run:
        # Log pipeline structure (or indicate no pipeline)
        if pipeline is not None:
            for i, (step_name, transformer) in enumerate(pipeline.steps):
                mlflow.log_param(f"step_{i}_name", step_name)
                mlflow.log_param(f"step_{i}_class", type(transformer).__name__)
            mlflow.log_param("n_steps", len(pipeline.steps))
        else:
            mlflow.log_param("n_steps", 0)

        # Log dataset info as params
        mlflow.log_param("n_samples", metadata["n_samples"])
        mlflow.log_param("n_features", metadata["n_features"])
        mlflow.log_param("n_targets", metadata["n_targets"])

        # Log description
        if description:
            mlflow.set_tag("mlflow.note.content", description)

        # Log scalar metadata as metrics
        mlflow.log_metric("n_features_in", metadata["n_features_in"])
        mlflow.log_metric("n_features_out", metadata["n_features_out"])

        # Save artifacts to temp dir and log
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save dataframes
            X.to_parquet(tmpdir_path / "X.parquet")
            y.to_parquet(tmpdir_path / "y.parquet")
            pd.Series(X.index, name="index").to_frame().to_parquet(tmpdir_path / "index.parquet")

            # Save metadata as JSON
            (tmpdir_path / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

            # Save list artifacts
            _save_list_file(metadata["feature_names_in"], tmpdir_path / "feature_names_in.txt")
            _save_list_file(metadata["feature_names_out"], tmpdir_path / "feature_names_out.txt")

            if metadata["dropped_columns"]:
                _save_list_file(metadata["dropped_columns"], tmpdir_path / "dropped_columns.txt")

            if metadata["scaler_info"]:
                (tmpdir_path / "scaler_info.json").write_text(
                    json.dumps(metadata["scaler_info"], indent=2, default=str)
                )

            # Log unfitted pipeline as artifact
            if pipeline is not None:
                mlflow.sklearn.log_model(pipeline, "pipeline")

            # Log all artifacts
            mlflow.log_artifacts(tmpdir, artifact_path="metadata")

        mlflow.set_tags({**tags, "run_type": "dataset"})

        run_id = run.info.run_id
        logger.info(f"Registered dataset '{name}' with run_id={run_id}")
        return run_id
