"""Training, evaluation, and cross-validation utilities for MLflow-tracked experiments.

Provides:
- train_and_log(): Load dataset, compose ML pipeline + model, fit, evaluate, log to MLflow
- evaluate(): Calculate metrics on predictions
- TimeSeriesSplitter: Expanding/sliding window cross-validation
- evaluate_pipeline(): Per-fold CV evaluation with precomputed datasets

Usage (in notebooks):
    from src.modeling.training import train_and_log
    from sklearn.linear_model import Ridge

    run_id = train_and_log(
        dataset_run_id="abc123",
        model=Ridge(),
        model_name="ridge_baseline",
        experiment="linear",
    )
"""

from collections.abc import Iterator
from typing import Literal

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.config import MLFLOW_TRACKING_URI
from src.features.transforms import TargetTransformer
from src.modeling.metrics import calculate_metrics

# =============================================================================
# Training with MLflow logging
# =============================================================================


def train_and_log(
    dataset_run_id: str,
    model,
    model_name: str,
    test_size: float = 0.2,
    val_size: float | None = None,
    ml_pipeline: Pipeline | None = None,
    target_transform: str = "log_shift",
    experiment: str = "experiments",
    run_name: str | None = None,
    description: str = "",
    tags: dict | None = None,
    log_model: bool = True,
    y_baseline: np.ndarray | Literal["default"] | None = "default",
    group_size: int = 1,
    weight_half_life: float | None = None,
) -> str:
    """Train using a precomputed dataset from MLflow.

    Steps:
    1. Load dataset (X, y, metadata) using load_dataset()
    2. Split into train/val/test (time series split)
    3. Build ML pipeline: ml_pipeline (scaling) + model
    4. Fit on train, evaluate on val (if provided) and test
    5. Log to MLflow with dataset_run_id linkage
    6. Register model under model_name for version tracking

    Args:
        dataset_run_id: MLflow run_id of preprocessed dataset
        model: sklearn-compatible estimator
        model_name: Name for MLflow model registry (groups versions)
        test_size: Fraction for test split
        val_size: Optional fraction for validation split. If provided, splits into
            [--- train ---|--- val ---|--- test ---]. If None (default), only train/test.
        ml_pipeline: Optional ML pipeline. If None, uses reference_ml_pipeline()
        target_transform: Target scaling method for TransformedTargetRegressor.
            One of "log_shift", "yeo_johnson", "quantile", "none".
        experiment: MLflow experiment name
        run_name: Optional run name
        description: Optional description
        tags: Optional tags
        log_model: Whether to log fitted model
        y_baseline: Baseline predictions for skill score calculation.
            - 'default': load from data/baseline_predictions.npy (default)
            - None: skip skill metrics
            - np.ndarray: user-provided baseline
        group_size: Group size for time series splitting (e.g., 24 for daily groups
            in hourly data). When > 1, ensures all rows in a group stay together.
        weight_half_life: If provided, compute exponential decay sample weights from
            X_train['day_index'] with this half-life in days (e.g., 730 = 2 years).
            If None (default), no sample weights are used.

    Returns:
        MLflow run_id
    """
    from src.features.preprocessors import load_dataset

    # Load dataset by run_id
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    dataset_run = mlflow.get_run(dataset_run_id)
    dataset_name = dataset_run.data.tags.get("dataset_name", "unknown")

    X, y, _ = load_dataset(run_id=dataset_run_id)
    y = y.values  # Convert to numpy array

    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    # Round split to group boundary
    split_idx = (split_idx // group_size) * group_size

    if val_size is not None:
        # Three-way split: train / val / test
        val_idx = int(split_idx * (1 - val_size))
        X_train, X_val, X_test = X.iloc[:val_idx], X.iloc[val_idx:split_idx], X.iloc[split_idx:]
        y_train, y_val, y_test = y[:val_idx], y[val_idx:split_idx], y[split_idx:]
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    else:
        # Two-way split: train / test
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_val, y_val = None, None
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Wrap model in TransformedTargetRegressor for target scaling
    if target_transform != "none":
        wrapped_model = TransformedTargetRegressor(
            regressor=model,
            transformer=TargetTransformer(method=target_transform),
        )
    else:
        wrapped_model = model

    # Compose ml_pipeline + model
    if ml_pipeline is None:
        full_pipeline = Pipeline([("model", wrapped_model)])
    else:
        full_pipeline = clone(ml_pipeline)
        full_pipeline.steps.append(("model", wrapped_model))

    # Compute sample weights from training data (after split to avoid leakage)
    weights = None
    if weight_half_life is not None:
        weights = compute_sample_weights(X_train, half_life_days=weight_half_life)

    # Fit
    logger.info("Fitting pipeline...")
    if weights is not None:
        full_pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        full_pipeline.fit(X_train, y_train)

    # Predict and evaluate on validation set if provided
    val_metrics = {}
    if X_val is not None:
        y_val_pred = full_pipeline.predict(X_val)
        val_metrics_raw = calculate_metrics(y_val, y_val_pred, y_baseline=None)
        # Add val_ prefix
        val_metrics = {f"val_{k}": v for k, v in val_metrics_raw.items()}

    # Predict and evaluate on test set (TTR automatically inverse-transforms predictions)
    y_pred = full_pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_baseline=y_baseline)

    # Combine all metrics
    all_metrics = {**metrics, **val_metrics}

    # Log to MLflow
    mlflow.set_experiment(experiment)

    if run_name is None:
        run_name = type(model).__name__

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics(all_metrics)

        # Log description
        if description:
            mlflow.set_tag("mlflow.note.content", description)

        params = {
            "model_class": type(model).__name__,
            "dataset_run_id": dataset_run_id,
            "dataset_name": dataset_name,
            "test_size": test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "group_size": group_size,
        }

        if val_size is not None:
            params["val_size"] = val_size
            params["n_val"] = len(X_val)

        if weight_half_life is not None:
            params["weight_half_life"] = weight_half_life

        mlflow.log_params(params)

        # Log model hyperparameters (sklearn get_params)
        try:
            model_params = model.get_params()
            # Filter to simple types only
            for k, v in model_params.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    mlflow.log_param(f"model_{k}", v)
        except Exception:
            pass

        if log_model:
            # Log with model registry for version tracking
            mlflow.sklearn.log_model(
                full_pipeline,
                "model",
                registered_model_name=model_name,
            )

        # Add strategy tag
        all_tags = tags.copy() if tags else {}
        if group_size > 1:
            all_tags.setdefault("strategy", "hourly_global")

        if all_tags:
            mlflow.set_tags(all_tags)

        run_id = run.info.run_id
        logger.info(f"Logged run '{run_name}' to experiment '{experiment}' (run_id={run_id})")
        logger.info(f"Registered as model '{model_name}' in MLflow Model Registry")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        return run_id


def tune_and_log(
    dataset_run_id: str,
    search_cv,
    model_name: str,
    test_size: float = 0.2,
    experiment: str = "experiments",
    run_name: str | None = None,
    description: str = "",
    tags: dict | None = None,
    log_model: bool = True,
    y_baseline: np.ndarray | Literal["default"] | None = "default",
    group_size: int = 1,
) -> tuple[str, dict]:
    """Hyperparameter tuning with CV, then final test evaluation + MLflow logging.

    Accepts any sklearn-compatible search CV object (GridSearchCV, RandomizedSearchCV,
    OptunaSearchCV from optuna-integration, etc.). The search_cv should be configured
    with the estimator, param grid/distributions, cv splitter, and scoring.

    Steps:
    1. Load dataset by run_id
    2. Hold out test set (last test_size fraction)
    3. Fit search_cv on train portion (runs CV internally)
    4. Extract best estimator, evaluate on test set
    5. Log to MLflow: CV scores, best params, test metrics, model

    Usage examples:
        # GridSearchCV
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplitter(),
                              scoring="neg_root_mean_squared_error")
        run_id, best = tune_and_log(dataset_run_id, search, "ridge_tuned")

        # OptunaSearchCV
        from optuna.integration.sklearn import OptunaSearchCV
        search = OptunaSearchCV(pipeline, param_distributions,
                                cv=TimeSeriesSplitter(), n_trials=50)
        run_id, best = tune_and_log(dataset_run_id, search, "ridge_optuna")

    Args:
        dataset_run_id: MLflow run_id of preprocessed dataset
        search_cv: Configured sklearn search CV object (fitted or unfitted)
        model_name: Name for MLflow model registry (groups versions)
        test_size: Fraction for test split
        experiment: MLflow experiment name
        run_name: Optional run name
        description: Optional description
        tags: Optional tags
        log_model: Whether to log fitted model
        y_baseline: Baseline predictions for skill score calculation.
            - 'default': load from data/baseline_predictions.npy (default)
            - None: skip skill metrics
            - np.ndarray: user-provided baseline
        group_size: Group size for time series splitting (e.g., 24 for daily groups
            in hourly data). When > 1, ensures the split lands on a group boundary.

    Returns:
        (run_id, best_params) tuple
    """
    from src.features.preprocessors import load_dataset

    # Load dataset
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    dataset_run = mlflow.get_run(dataset_run_id)
    dataset_name = dataset_run.data.tags.get("dataset_name", "unknown")

    X, y, _ = load_dataset(run_id=dataset_run_id)
    y = y.values  # Convert to numpy array

    # Hold out test set, rounded to group boundary
    split_idx = int(len(X) * (1 - test_size))
    split_idx = (split_idx // group_size) * group_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Running hyperparameter search with {type(search_cv).__name__}...")

    # Fit search on train portion (CV happens internally)
    search_cv.fit(X_train, y_train)

    # Extract best estimator
    best_estimator = search_cv.best_estimator_
    best_params = search_cv.best_params_

    logger.info(f"Best params: {best_params}")

    # Evaluate on test set
    y_pred = best_estimator.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_baseline=y_baseline)

    # Log to MLflow
    mlflow.set_experiment(experiment)

    if run_name is None:
        run_name = f"{type(search_cv.estimator).__name__}_tuned"

    with mlflow.start_run(run_name=run_name) as run:
        # Log test metrics
        mlflow.log_metrics(metrics)

        # Log CV results
        mlflow.log_metric("cv_best_score", search_cv.best_score_)
        if hasattr(search_cv, "cv_results_"):
            # Log mean and std of CV scores (ignoring NaN from failed trials)
            cv_scores = search_cv.cv_results_["mean_test_score"]
            cv_mean = float(np.nanmean(cv_scores))
            cv_std = float(np.nanstd(cv_scores))
            # Only log if valid (prevents duplicate NaN entries)
            if not np.isnan(cv_mean):
                mlflow.log_metric("cv_mean_score", cv_mean)
            if not np.isnan(cv_std):
                mlflow.log_metric("cv_std_score", cv_std)

        # Log best params
        for param_name, param_value in best_params.items():
            if isinstance(param_value, (int, float, str, bool, type(None))):
                mlflow.log_param(f"best_{param_name}", param_value)

        # Log metadata
        if description:
            mlflow.set_tag("mlflow.note.content", description)

        mlflow.log_params(
            {
                "search_class": type(search_cv).__name__,
                "dataset_run_id": dataset_run_id,
                "dataset_name": dataset_name,
                "test_size": test_size,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "group_size": group_size,
            }
        )

        # Log model
        if log_model:
            mlflow.sklearn.log_model(
                best_estimator,
                "model",
                registered_model_name=model_name,
            )

        if tags:
            mlflow.set_tags(tags)

        run_id = run.info.run_id
        logger.info(
            f"Logged tuning run '{run_name}' to experiment '{experiment}' (run_id={run_id})"
        )
        logger.info(f"Registered as model '{model_name}' in MLflow Model Registry")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        return run_id, best_params


def compute_sample_weights(
    X: pd.DataFrame,
    half_life_days: float = 730.0,
) -> np.ndarray:
    """Compute exponential decay sample weights from the day_index column.

    More recent observations receive higher weight. The most recent observation
    always gets weight 1.0; older observations decay exponentially.

    Pass X_train (not the full dataset) to avoid computing weights for the test set.

    Args:
        X: Feature DataFrame containing a 'day_index' column (produced by the v5 pipeline).
        half_life_days: Number of days after which weight halves. Default 730 (2 years).
            - 730 days: observation 2 years old → weight 0.50
            - 730 days: observation 5 years old → weight 0.16
            Reasonable range: 365–1460 days.

    Returns:
        1-D float array of sample weights, same length as X.

    Example:
        >>> weights = compute_sample_weights(X_train, half_life_days=730)
        >>> model.fit(X_train, y_train, sample_weight=weights)
    """
    if "day_index" not in X.columns:
        raise ValueError(
            "'day_index' column not found. Run the v5 pipeline before computing weights."
        )
    t = X["day_index"].values.astype(float)
    decay_rate = np.log(2) / half_life_days
    weights = np.exp(decay_rate * (t - t.max()))
    return weights


# =============================================================================
# Time Series Cross-Validation
# =============================================================================


class TimeSeriesSplitter:
    """Time series cross-validation splitter with expanding or sliding windows.

    Implements sklearn's splitter interface (get_n_splits, split).

    Args:
        n_splits: Number of CV folds.
        test_size: Size of each test fold. If float, fraction of total samples.
            If int, absolute number of samples.
        mode: "expanding" (growing train) or "sliding" (fixed-size train window).
        gap: Number of observations to skip between train and test sets
            (prevents leakage from correlated time steps).
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float | int = 0.2,
        mode: Literal["expanding", "sliding"] = "expanding",
        gap: int = 0,
        group_size: int = 1,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.mode = mode
        self.gap = gap
        self.group_size = group_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) tuples.

        For expanding mode: train grows from a minimum size, test is fixed.
        For sliding mode: train is fixed size, slides forward with test.
        When group_size > 1, ensures all rows in a group stay together.
        """
        n_samples = len(X)

        # Work at group level when group_size > 1
        g = self.group_size
        n_groups = n_samples // g

        # Resolve test_size to absolute number of groups
        if isinstance(self.test_size, float):
            test_len = max(1, int(n_groups * self.test_size / self.n_splits))
        else:
            test_len = self.test_size // g if g > 1 else self.test_size

        gap_groups = self.gap // g if g > 1 else self.gap

        for i in range(self.n_splits):
            test_end = n_groups - i * test_len
            test_start = test_end - test_len

            if test_start <= 0:
                break

            train_end = test_start - gap_groups

            if train_end <= 0:
                break

            if self.mode == "expanding":
                train_start = 0
            else:
                earliest_test_start = n_groups - self.n_splits * test_len
                train_size = earliest_test_start - gap_groups
                train_start = max(0, train_end - train_size)

            # Expand group indices to row indices
            train_idx = np.arange(train_start * g, train_end * g)
            test_idx = np.arange(test_start * g, test_end * g)

            yield train_idx, test_idx


def evaluate_pipeline(
    dataset_run_id: str,
    model,
    ml_pipeline: Pipeline | None = None,
    target_transform: str = "log_shift",
    splitter: TimeSeriesSplitter | None = None,
    y_baseline: np.ndarray | Literal["default"] | None = "default",
    group_size: int = 1,
) -> pd.DataFrame:
    """Cross-validation using a precomputed dataset.

    Args:
        dataset_run_id: MLflow run_id of preprocessed dataset
        model: sklearn-compatible estimator
        ml_pipeline: Optional ML pipeline for scaling
        target_transform: Target scaling method for TransformedTargetRegressor.
            One of "log_shift", "yeo_johnson", "quantile", "none".
        splitter: TimeSeriesSplitter instance
        y_baseline: Baseline predictions for skill score calculation.
            - 'default': load from data/baseline_predictions.npy (default)
            - None: skip skill metrics
            - np.ndarray: user-provided baseline
        group_size: Group size for time series splitting (e.g., 24 for daily groups
            in hourly data). When > 1, ensures all rows in a group stay together.

    Returns:
        DataFrame with per-fold metrics
    """
    from src.features.preprocessors import load_dataset

    # Load dataset
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    X, y, _ = load_dataset(run_id=dataset_run_id)
    y = y.values  # Convert to numpy array

    logger.info(f"Loaded dataset: X={X.shape}, y={y.shape}")

    # Default splitter
    if splitter is None:
        splitter = TimeSeriesSplitter(n_splits=5, test_size=0.2, group_size=group_size)

    # Default ML pipeline
    if ml_pipeline is None:
        from src.config.pipelines import reference_ml_pipeline

        ml_pipeline = reference_ml_pipeline()

    # Wrap model in TransformedTargetRegressor for target scaling
    if target_transform != "none":
        wrapped_model = TransformedTargetRegressor(
            regressor=model,
            transformer=TargetTransformer(method=target_transform),
        )
    else:
        wrapped_model = model

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone ml_pipeline and model for this fold
        fold_pipeline = clone(ml_pipeline)
        fold_pipeline.steps.append(("model", clone(wrapped_model)))

        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)

        # TTR automatically inverse-transforms predictions
        # Extract corresponding baseline slice if provided
        y_baseline_fold = None
        if y_baseline is not None:
            if isinstance(y_baseline, str) and y_baseline == "default":
                # Load full baseline and slice for this fold
                from src.modeling.baselines import get_default_baseline_predictions

                y_baseline_full = get_default_baseline_predictions()
                y_baseline_fold = y_baseline_full[test_idx]
            else:
                # User provided baseline - slice for this fold
                y_baseline_fold = y_baseline[test_idx]

        metrics = calculate_metrics(y_test, y_pred, y_baseline=y_baseline_fold)
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        fold_metrics.append(metrics)

        logger.info(
            f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}, "
            f"RMSE={metrics.get('rmse', float('nan')):.4f}"
        )

    return pd.DataFrame(fold_metrics)
