"""Production ensemble pipeline: select, validate, train, and blend models.

Bridges MLflow experimentation and production inference by selecting the best
models, validating with sliding-window CV, and computing inverse-MAE blend weights.

Each model carries its own dataset_run_id (stored in MLflow params). All dataset
loading is per-model — no shared dataset_run_id is required at the blend level.

Usage (via CLI):
    python -m src.cli blend select   # Full pipeline: select → CV → train → blend
    python -m src.cli blend update   # Daily: incremental tree fit + weight refresh
    python -m src.cli blend retrain  # Biweekly: full retrain from scratch
    python -m src.cli blend info     # Print blend_config.json summary
"""

from datetime import datetime, timezone
import json

import joblib
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import MLFLOW_TRACKING_URI, MODELS_DIR
from src.config.modeling import (
    BLEND_CANDIDATES_RANDOM,
    BLEND_CANDIDATES_RANDOM_POOL,
    BLEND_CANDIDATES_TOP,
    BLEND_CATEGORY_MATCHERS,
    BLEND_DEGRADATION_THRESHOLD,
    BLEND_FORCE_RUN_IDS,
    BLEND_HOLDOUT_DAYS,
)

PRODUCTION_DIR = MODELS_DIR / "production"
BLEND_CONFIG_PATH = PRODUCTION_DIR / "blend_config.json"


# =============================================================================
# Helpers
# =============================================================================


def _classify_model(model_class: str) -> str | None:
    """Map a model_class string to a blend category."""
    for category, names in BLEND_CATEGORY_MATCHERS.items():
        if model_class in names:
            return category
    return None


def _flatten_daily_to_hourly(preds: np.ndarray) -> np.ndarray:
    """Flatten (N_days, 24) daily predictions to (N_hours,) hourly array."""
    if preds.ndim == 2 and preds.shape[1] == 24:
        return preds.flatten()
    if preds.ndim == 1:
        return preds
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds.ravel()
    raise ValueError(f"Unexpected prediction shape: {preds.shape}")


def _flatten_y_true(y: np.ndarray, group_size: int) -> np.ndarray:
    """Flatten y_true to 1-D hourly.

    group_size=1 (daily model): y is (N_days, 24) → flatten to (N_hours,)
    group_size=24 (hourly model): y is already (N_hours,) or (N_hours, 1)
    """
    if group_size == 1 and y.ndim == 2:
        return y.flatten()
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    return y.ravel() if y.ndim > 1 else y


def _compute_inverse_mae_weights(maes: np.ndarray) -> np.ndarray:
    """Compute inverse-MAE weights, normalized to sum to 1."""
    inv_mae = 1.0 / (maes + 1e-6)
    return inv_mae / inv_mae.sum()


def _load_datasets(model_infos: list[dict]) -> dict[str, tuple]:
    """Load and cache datasets by dataset_run_id.

    Args:
        model_infos: List of dicts, each containing 'dataset_run_id'.

    Returns:
        Dict mapping dataset_run_id → (X, y_values).
    """
    from src.features.preprocessors import load_dataset

    cache: dict[str, tuple] = {}
    for info in model_infos:
        run_id = info["dataset_run_id"]
        if run_id not in cache:
            logger.info(f"Loading dataset {run_id[:8]}...")
            X, y_df, _ = load_dataset(run_id=run_id)
            cache[run_id] = (X, y_df.values)
    return cache


# =============================================================================
# 1. Select candidates from MLflow
# =============================================================================


def select_candidates() -> pd.DataFrame:
    """Query MLflow, classify runs by model_class, and select diverse candidates.

    Per category: top BLEND_CANDIDATES_TOP by MAE + BLEND_CANDIDATES_RANDOM random
    from the next BLEND_CANDIDATES_RANDOM_POOL ranks. Force-includes any run IDs
    listed in BLEND_FORCE_RUN_IDS.

    Returns:
        DataFrame with columns: run_id, model_class, category, group_size,
        mae, rmse, dataset_run_id
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Search all experiments for model runs with valid MAE.
    # Exclude dataset runs (tagged run_type="dataset" by build_pipeline).
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="metrics.mae > 0 AND tags.run_type != 'dataset'",
        order_by=["metrics.mae ASC"],
    )

    if runs.empty:
        raise ValueError("No MLflow runs found with valid metrics.mae")

    df = pd.DataFrame(
        {
            "run_id": runs["run_id"],
            "model_class": runs["params.model_class"],
            "group_size": runs["params.group_size"].fillna("1").astype(int),
            "mae": runs["metrics.mae"],
            "rmse": runs["metrics.rmse"],
            "dataset_run_id": runs["params.dataset_run_id"],
        }
    ).dropna(subset=["model_class", "mae", "dataset_run_id"])

    df["category"] = df["model_class"].map(_classify_model)
    df = df.dropna(subset=["category"])

    if df.empty:
        raise ValueError("No runs matched any blend category")

    selected = []
    for category, group in df.groupby("category"):
        group = group.sort_values("mae")
        selected.append(group.head(BLEND_CANDIDATES_TOP))

        remaining = group.iloc[
            BLEND_CANDIDATES_TOP : BLEND_CANDIDATES_TOP + BLEND_CANDIDATES_RANDOM_POOL
        ]
        if len(remaining) > 0:
            n_random = min(BLEND_CANDIDATES_RANDOM, len(remaining))
            selected.append(remaining.sample(n=n_random, random_state=42))

    result = pd.concat(selected, ignore_index=True)

    if BLEND_FORCE_RUN_IDS:
        forced = df[df["run_id"].isin(BLEND_FORCE_RUN_IDS)]
        forced = forced[~forced["run_id"].isin(result["run_id"])]
        if len(forced) > 0:
            result = pd.concat([result, forced], ignore_index=True)

    logger.info(
        f"Selected {len(result)} candidates across {result['category'].nunique()} categories"
    )
    return result


# =============================================================================
# 2. Validate candidates with sliding-window CV
# =============================================================================


def validate_candidates(
    candidates_df: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Retrain each candidate with sliding CV to assess robustness.

    Each candidate is validated against its own dataset (loaded via its
    dataset_run_id stored in candidates_df). Datasets are cached to avoid
    reloading the same parquet when multiple candidates share a dataset.

    Args:
        candidates_df: Output of select_candidates().
        n_splits: Number of CV folds.

    Returns:
        Enriched DataFrame with cv_mae_mean, cv_mae_std, cv_rmse_mean, cv_rmse_std.
    """
    from src.modeling.training import TimeSeriesSplitter

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Cache datasets to avoid redundant loads
    dataset_cache = _load_datasets(candidates_df.to_dict("records"))

    cv_results = []

    for _, row in candidates_df.iterrows():
        run_id = row["run_id"]
        group_size = int(row["group_size"])
        X, y = dataset_cache[row["dataset_run_id"]]

        logger.info(
            f"Validating {row['model_class']} (run={run_id[:8]}..., group_size={group_size})"
        )

        try:
            fitted_pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            pipeline_template = clone(fitted_pipeline)
        except Exception as e:
            logger.warning(f"Could not load model for run {run_id}: {e}")
            cv_results.append(
                {
                    "cv_mae_mean": np.nan,
                    "cv_mae_std": np.nan,
                    "cv_rmse_mean": np.nan,
                    "cv_rmse_std": np.nan,
                }
            )
            continue

        splitter = TimeSeriesSplitter(
            n_splits=n_splits,
            mode="sliding",
            group_size=group_size,
        )

        fold_maes, fold_rmses = [], []
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_pipe = clone(pipeline_template)
            fold_pipe.fit(X_train, y_train)
            y_pred = fold_pipe.predict(X_test)

            y_pred_flat = _flatten_daily_to_hourly(y_pred)
            y_test_flat = _flatten_y_true(y_test, group_size)

            min_len = min(len(y_pred_flat), len(y_test_flat))
            fold_maes.append(mean_absolute_error(y_test_flat[:min_len], y_pred_flat[:min_len]))
            fold_rmses.append(
                float(np.sqrt(np.mean((y_test_flat[:min_len] - y_pred_flat[:min_len]) ** 2)))
            )

        cv_results.append(
            {
                "cv_mae_mean": np.mean(fold_maes),
                "cv_mae_std": np.std(fold_maes),
                "cv_rmse_mean": np.mean(fold_rmses),
                "cv_rmse_std": np.std(fold_rmses),
            }
        )

    cv_df = pd.DataFrame(cv_results)
    return pd.concat([candidates_df.reset_index(drop=True), cv_df], axis=1)


# =============================================================================
# 3. Select final models (best MAE + best RMSE per category)
# =============================================================================


def select_final_models(validated_df: pd.DataFrame) -> pd.DataFrame:
    """Per category, select the best-MAE and best-RMSE model.

    If the same model wins both metrics, take the 2nd-best on the other metric.
    """
    validated_df = validated_df.dropna(subset=["cv_mae_mean"])
    selected = []

    for category, group in validated_df.groupby("category"):
        picks = set()

        best_mae_idx = group["cv_mae_mean"].idxmin()
        picks.add(best_mae_idx)

        best_rmse_idx = group["cv_rmse_mean"].idxmin()
        if best_rmse_idx == best_mae_idx:
            remaining = group.drop(index=best_rmse_idx)
            if not remaining.empty:
                picks.add(remaining["cv_rmse_mean"].idxmin())
        else:
            picks.add(best_rmse_idx)

        selected.append(group.loc[list(picks)])

    result = pd.concat(selected, ignore_index=True)
    logger.info(
        f"Selected {len(result)} final models across {result['category'].nunique()} categories"
    )
    return result


# =============================================================================
# 4. Train final models and compute blend weights
# =============================================================================


def train_and_blend(
    selected_df: pd.DataFrame,
    holdout_days: int = BLEND_HOLDOUT_DAYS,
) -> dict:
    """Train selected models on (all - holdout), compute inverse-MAE blend weights.

    Each model uses its own dataset (loaded via its dataset_run_id). The holdout
    split is per-model: last holdout_days * group_size rows. All predictions are
    flattened to hourly for weight computation.

    Args:
        selected_df: Output of select_final_models().
        holdout_days: Days held out for weight computation.

    Returns:
        Blend config dict (also saved to blend_config.json).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    dataset_cache = _load_datasets(selected_df.to_dict("records"))

    models_info = []
    holdout_preds = []
    holdout_maes = []
    ref_y_holdout_flat: np.ndarray | None = None  # hourly y_true reference

    for i, (_, row) in enumerate(selected_df.iterrows()):
        run_id = row["run_id"]
        category = row["category"]
        group_size = int(row["group_size"])
        dataset_run_id = row["dataset_run_id"]
        name = f"{category}_{i}"

        X, y = dataset_cache[dataset_run_id]
        holdout_rows = holdout_days * group_size
        if len(X) <= holdout_rows:
            raise ValueError(
                f"Not enough rows in dataset {dataset_run_id[:8]} for "
                f"{holdout_days}-day holdout (have {len(X)}, need >{holdout_rows})"
            )

        split_idx = len(X) - holdout_rows
        X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_holdout = y[:split_idx], y[split_idx:]

        logger.info(f"Training {name} ({row['model_class']}, run={run_id[:8]}...)")

        fitted_pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        pipeline = clone(fitted_pipeline)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_holdout)
        y_pred_flat = _flatten_daily_to_hourly(y_pred)
        y_holdout_flat = _flatten_y_true(y_holdout, group_size)

        min_len = min(len(y_pred_flat), len(y_holdout_flat))
        y_pred_flat = y_pred_flat[:min_len]
        y_holdout_flat = y_holdout_flat[:min_len]

        holdout_mae = mean_absolute_error(y_holdout_flat, y_pred_flat)
        holdout_rmse = float(np.sqrt(np.mean((y_holdout_flat - y_pred_flat) ** 2)))

        holdout_preds.append(y_pred_flat)
        holdout_maes.append(holdout_mae)

        # Use the first model's y_holdout as the blend evaluation reference
        if ref_y_holdout_flat is None:
            ref_y_holdout_flat = y_holdout_flat

        model_file = f"{name}.joblib"
        joblib.dump(pipeline, PRODUCTION_DIR / model_file)

        models_info.append(
            {
                "name": name,
                "category": category,
                "file": model_file,
                "source_run_id": run_id,
                "dataset_run_id": dataset_run_id,
                "group_size": group_size,
                "weight": 0.0,
                "holdout_mae": round(holdout_mae, 4),
                "holdout_rmse": round(holdout_rmse, 4),
                "cv_mae_mean": round(float(row.get("cv_mae_mean", np.nan)), 4),
                "cv_mae_std": round(float(row.get("cv_mae_std", np.nan)), 4),
            }
        )

    maes = np.array(holdout_maes)
    weights = _compute_inverse_mae_weights(maes)
    for info, w in zip(models_info, weights):
        info["weight"] = round(float(w), 6)

    # Blend evaluation on holdout
    min_len = min(len(p) for p in holdout_preds)
    preds_matrix = np.column_stack([p[:min_len] for p in holdout_preds])
    y_blend = preds_matrix @ weights
    y_ref = ref_y_holdout_flat[:min_len]
    blend_mae = float(mean_absolute_error(y_ref, y_blend))
    blend_rmse = float(np.sqrt(np.mean((y_ref - y_blend) ** 2)))

    now = datetime.now(timezone.utc).isoformat()
    config = {
        "version": 1,
        "created": now,
        "updated": now,
        "holdout_days": holdout_days,
        "models": models_info,
        "blend_mae": round(blend_mae, 4),
        "blend_rmse": round(blend_rmse, 4),
        "blend_me": round(float(np.mean(y_blend - y_ref)), 4),
        "blend_r2": round(float(r2_score(y_ref, y_blend)), 4),
        "needs_reselection": False,
    }

    _log_blend_to_mlflow(config, y_ref, y_blend)
    logger.success(f"Blend config saved to {BLEND_CONFIG_PATH}")
    _print_summary(config)
    return config


# =============================================================================
# 5. Daily incremental update
# =============================================================================


def update_blend_daily() -> dict:
    """Lightweight daily update: incremental tree fit + weight recomputation.

    Tree models (LightGBM, XGBoost, CatBoost) are warm-started on recent data.
    Linear models are skipped (wait for biweekly full retrain).
    Each model uses its own dataset_run_id stored in blend_config.json.

    Returns:
        Updated blend config dict.
    """
    config = _load_config()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    holdout_days = config["holdout_days"]
    dataset_cache = _load_datasets(config["models"])

    holdout_preds = []
    holdout_maes = []
    ref_y_holdout_flat: np.ndarray | None = None

    for model_info in config["models"]:
        model_path = PRODUCTION_DIR / model_info["file"]
        pipeline = joblib.load(model_path)
        category = model_info["category"]
        group_size = model_info["group_size"]

        X, y = dataset_cache[model_info["dataset_run_id"]]
        holdout_rows = holdout_days * group_size
        X_holdout = X.iloc[-holdout_rows:]
        y_holdout = y[-holdout_rows:]

        if category in ("lgbm", "xgboost", "catboost"):
            _try_incremental_fit(pipeline, X, y, category, holdout_rows)
            joblib.dump(pipeline, model_path)

        y_pred = pipeline.predict(X_holdout)
        y_pred_flat = _flatten_daily_to_hourly(y_pred)
        y_holdout_flat = _flatten_y_true(y_holdout, group_size)

        min_len = min(len(y_pred_flat), len(y_holdout_flat))
        y_pred_flat = y_pred_flat[:min_len]
        y_holdout_flat = y_holdout_flat[:min_len]

        holdout_mae = mean_absolute_error(y_holdout_flat, y_pred_flat)
        holdout_preds.append(y_pred_flat)
        holdout_maes.append(holdout_mae)
        model_info["holdout_mae"] = round(holdout_mae, 4)

        if ref_y_holdout_flat is None:
            ref_y_holdout_flat = y_holdout_flat

    maes = np.array(holdout_maes)
    weights = _compute_inverse_mae_weights(maes)
    for info, w in zip(config["models"], weights):
        info["weight"] = round(float(w), 6)

    min_len = min(len(p) for p in holdout_preds)
    preds_matrix = np.column_stack([p[:min_len] for p in holdout_preds])
    y_blend = preds_matrix @ weights
    y_ref = ref_y_holdout_flat[:min_len]
    config["blend_mae"] = round(float(mean_absolute_error(y_ref, y_blend)), 4)
    config["blend_rmse"] = round(float(np.sqrt(np.mean((y_ref - y_blend) ** 2))), 4)
    config["updated"] = datetime.now(timezone.utc).isoformat()

    BLEND_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    logger.success("Daily blend update complete")
    _print_summary(config)
    return config


def _try_incremental_fit(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    category: str,
    holdout_rows: int,
) -> None:
    """Warm-start incremental fit for tree models.

    All production models use target_transform="none", so the pipeline is
    Pipeline([...preprocessing..., ("model", raw_estimator)]).
    We transform X_new through preprocessing steps, then warm-start the
    tree estimator using its library-specific API.
    """
    X_new = X.iloc[-holdout_rows:]
    y_new = y[-holdout_rows:]

    try:
        X_transformed = pipeline[:-1].transform(X_new) if len(pipeline) > 1 else X_new
        model = pipeline[-1]

        if category == "lgbm":
            model.fit(X_transformed, y_new, init_model=model)
        elif category == "xgboost":
            model.fit(X_transformed, y_new, xgb_model=model.get_booster())
        elif category == "catboost":
            model.fit(X_transformed, y_new, init_model=model)

        logger.info(f"Incremental fit for {category} on {len(X_new)} rows")
    except Exception as e:
        logger.warning(f"Incremental fit failed for {category}: {e}")


# =============================================================================
# 6. Biweekly full retrain
# =============================================================================


def retrain_blend(holdout_days: int | None = None) -> dict:
    """Full retrain: refit all models from scratch + recompute weights.

    Checks for degradation: if new blend_mae > old × (1 + threshold),
    sets needs_reselection flag. Each model is retrained on its own dataset.

    Args:
        holdout_days: Override holdout days (default: use config value).

    Returns:
        Updated blend config dict.
    """
    config = _load_config()
    old_blend_mae = config["blend_mae"]
    if holdout_days is None:
        holdout_days = config["holdout_days"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    dataset_cache = _load_datasets(config["models"])

    holdout_preds = []
    holdout_maes = []
    ref_y_holdout_flat: np.ndarray | None = None

    for model_info in config["models"]:
        model_path = PRODUCTION_DIR / model_info["file"]
        pipeline = joblib.load(model_path)
        group_size = model_info["group_size"]

        X, y = dataset_cache[model_info["dataset_run_id"]]
        holdout_rows = holdout_days * group_size
        split_idx = len(X) - holdout_rows
        X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_holdout = y[:split_idx], y[split_idx:]

        fresh_pipeline = clone(pipeline)
        fresh_pipeline.fit(X_train, y_train)

        y_pred = fresh_pipeline.predict(X_holdout)
        y_pred_flat = _flatten_daily_to_hourly(y_pred)
        y_holdout_flat = _flatten_y_true(y_holdout, group_size)

        min_len = min(len(y_pred_flat), len(y_holdout_flat))
        y_pred_flat = y_pred_flat[:min_len]
        y_holdout_flat = y_holdout_flat[:min_len]

        holdout_mae = mean_absolute_error(y_holdout_flat, y_pred_flat)
        holdout_rmse = float(np.sqrt(np.mean((y_holdout_flat - y_pred_flat) ** 2)))

        holdout_preds.append(y_pred_flat)
        holdout_maes.append(holdout_mae)
        model_info["holdout_mae"] = round(holdout_mae, 4)
        model_info["holdout_rmse"] = round(holdout_rmse, 4)

        if ref_y_holdout_flat is None:
            ref_y_holdout_flat = y_holdout_flat

        joblib.dump(fresh_pipeline, model_path)

    maes = np.array(holdout_maes)
    weights = _compute_inverse_mae_weights(maes)
    for info, w in zip(config["models"], weights):
        info["weight"] = round(float(w), 6)

    min_len = min(len(p) for p in holdout_preds)
    preds_matrix = np.column_stack([p[:min_len] for p in holdout_preds])
    y_blend = preds_matrix @ weights
    y_ref = ref_y_holdout_flat[:min_len]
    new_blend_mae = float(mean_absolute_error(y_ref, y_blend))
    new_blend_rmse = float(np.sqrt(np.mean((y_ref - y_blend) ** 2)))

    needs_reselection = False
    if old_blend_mae > 0:
        degradation = (new_blend_mae - old_blend_mae) / old_blend_mae
        if degradation > BLEND_DEGRADATION_THRESHOLD:
            logger.warning(
                f"Blend MAE degraded by {degradation:.1%} "
                f"({old_blend_mae:.2f} → {new_blend_mae:.2f}). "
                f"Flagging for reselection."
            )
            needs_reselection = True

    config["blend_mae"] = round(new_blend_mae, 4)
    config["blend_rmse"] = round(new_blend_rmse, 4)
    config["blend_me"] = round(float(np.mean(y_blend - y_ref)), 4)
    config["blend_r2"] = round(float(r2_score(y_ref, y_blend)), 4)
    config["holdout_days"] = holdout_days
    config["needs_reselection"] = needs_reselection
    config["updated"] = datetime.now(timezone.utc).isoformat()

    _log_blend_to_mlflow(config, y_ref, y_blend)
    logger.success("Full retrain complete")
    _print_summary(config)
    return config


# =============================================================================
# 7. Load production ensemble
# =============================================================================


def load_blend() -> tuple[list, np.ndarray, dict]:
    """Load production ensemble for inference.

    Returns:
        (models, weights, config) where:
        - models: list of fitted sklearn pipelines
        - weights: 1-D array of blend weights
        - config: full blend config dict (includes group_size and dataset_run_id per model)
    """
    config = _load_config()
    models = [joblib.load(PRODUCTION_DIR / m["file"]) for m in config["models"]]
    weights = np.array([m["weight"] for m in config["models"]])
    return models, weights, config


# =============================================================================
# 8. Blended prediction
# =============================================================================


def predict_blend(
    X: pd.DataFrame,
    models: list | None = None,
    weights: np.ndarray | None = None,
    config: dict | None = None,
) -> np.ndarray:
    """Generate blended prediction from ensemble.

    Args:
        X: Feature DataFrame. Each model uses the same X — callers are
           responsible for ensuring X has the features required by all
           models in the blend (i.e., use a feature-union dataset or
           a per-model preprocessing step before calling this function).
        models: List of fitted pipelines. If None, loaded from disk.
        weights: Blend weights array. If None, loaded from disk.
        config: Blend config dict. If None, loaded from disk.

    Returns:
        Hourly predictions array of shape (N_hours,).
    """
    if models is None or weights is None or config is None:
        models, weights, config = load_blend()

    preds = []
    for model in models:
        y_pred = model.predict(X)
        preds.append(_flatten_daily_to_hourly(y_pred))

    min_len = min(len(p) for p in preds)
    preds_matrix = np.column_stack([p[:min_len] for p in preds])
    return preds_matrix @ weights


# =============================================================================
# Internal helpers
# =============================================================================


def _log_blend_to_mlflow(config: dict, y_ref: np.ndarray, y_blend: np.ndarray) -> None:
    """Log blend config and metrics to MLflow 'final_models' experiment.

    Constituent models are not re-serialized — they are already tracked under
    their own source_run_ids. This run records weights, metrics, and the
    blend_config.json artifact for traceability.

    Writes blend_run_id back into config and saves blend_config.json to disk.
    """
    mlflow.set_experiment("final_models")
    with mlflow.start_run(run_name="blend") as blend_run:
        mlflow.log_param("n_models", len(config["models"]))
        mlflow.log_param("holdout_days", config["holdout_days"])
        for i, info in enumerate(config["models"]):
            mlflow.log_param(f"model_{i}_run_id", info["source_run_id"])
            mlflow.log_param(f"model_{i}_category", info["category"])
            mlflow.log_param(f"model_{i}_group_size", info["group_size"])
            mlflow.log_param(f"model_{i}_weight", info["weight"])
        mlflow.log_metric("blend_mae", config["blend_mae"])
        mlflow.log_metric("blend_rmse", config["blend_rmse"])
        mlflow.log_metric("blend_me", config["blend_me"])
        mlflow.log_metric("blend_r2", config["blend_r2"])
        for info in config["models"]:
            mlflow.log_metric(f"{info['name']}_holdout_mae", info["holdout_mae"])
            mlflow.log_metric(f"{info['name']}_holdout_rmse", info["holdout_rmse"])
        config["blend_run_id"] = blend_run.info.run_id
        BLEND_CONFIG_PATH.write_text(json.dumps(config, indent=2))
        mlflow.log_artifact(str(BLEND_CONFIG_PATH), artifact_path="config")


def _load_config() -> dict:
    """Load blend_config.json."""
    if not BLEND_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Blend config not found at {BLEND_CONFIG_PATH}. Run 'blend select' first."
        )
    return json.loads(BLEND_CONFIG_PATH.read_text())


def _print_summary(config: dict) -> None:
    """Print a human-readable summary of the blend config."""
    logger.info("=" * 60)
    logger.info("Blend Ensemble Summary")
    logger.info("=" * 60)
    logger.info(f"  Models: {len(config['models'])}")
    logger.info(f"  Blend MAE:  {config['blend_mae']:.4f}")
    logger.info(f"  Blend RMSE: {config['blend_rmse']:.4f}")
    logger.info(f"  Blend ME:   {config.get('blend_me', float('nan')):.4f}")
    logger.info(f"  Blend R²:   {config.get('blend_r2', float('nan')):.4f}")
    logger.info(f"  Holdout:    {config['holdout_days']} days")
    logger.info(f"  Updated:    {config['updated']}")
    if config.get("needs_reselection"):
        logger.warning("  ⚠ needs_reselection = true")
    logger.info("-" * 60)
    for m in config["models"]:
        logger.info(
            f"  {m['name']:20s}  w={m['weight']:.4f}  "
            f"MAE={m['holdout_mae']:.2f}  ({m['category']}, "
            f"ds={m['dataset_run_id'][:8]}...)"
        )
    logger.info("=" * 60)
