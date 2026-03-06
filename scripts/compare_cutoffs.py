"""Three-way cutoff comparison: 10am CET vs 7am CET vs midnight.

Builds v5_slim datasets for 7am and midnight variants, trains LightGBM
with lgbm_2's hyperparameters on each, and compares against the existing
10am baseline (dataset 247ccd98).

Usage:
    # Build datasets only
    python scripts/compare_cutoffs.py --build

    # Train models on existing datasets
    python scripts/compare_cutoffs.py --train

    # Build + train + compare
    python scripts/compare_cutoffs.py --all
"""

import argparse
import json
from pathlib import Path
import sys

from loguru import logger

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Existing baseline dataset
BASELINE_DATASET_RUN_ID = "247ccd98f2614cc5aa0bf834c1f1835e"

# lgbm_2 hyperparameters from blend_hyperparams.json
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "n_estimators": 1160,
    "learning_rate": 0.008460389319539744,
    "max_depth": 10,
    "num_leaves": 38,
    "colsample_bytree": 0.5732422713108168,
    "subsample": 0.747640945202763,
    "min_child_samples": 108,
    "min_child_weight": 22,
    "reg_alpha": 6.943300649628005,
    "reg_lambda": 6.957404718670572,
    "min_split_gain": 0.290969352827906,
    "objective": "mae",
    "metric": "mae",
    "verbosity": -1,
}

TRAIN_KWARGS = {
    "test_size": 0.1,
    "group_size": 24,
    "weight_half_life": 730,
    "target_transform": "log_shift",
    "experiment": "cutoff_comparison",
    "log_model": False,
    "y_baseline": None,
}


def build_datasets():
    """Build 7am and midnight variant datasets via MLflow."""
    from src.config.pipelines import preprocessor_v5_slim_hourly
    from src.features.preprocessors import build_pipeline

    results = {}

    for cutoff, label in [(7, "7am"), (None, "midnight")]:
        logger.info(f"Building dataset for {label} cutoff (morning_cutoff_cet={cutoff})")
        pipe = preprocessor_v5_slim_hourly(morning_cutoff_cet=cutoff)
        name = f"dataset_v5_slim_hourly_cutoff_{label}"
        run_id = build_pipeline(
            pipe,
            name=name,
            description=f"v5_slim with morning_cutoff_cet={cutoff} + nuclear removal",
            tags={"morning_cutoff_cet": str(cutoff)},
        )
        results[label] = run_id
        logger.info(f"  → {label}: run_id={run_id}")

    return results


def train_models(dataset_ids: dict[str, str]):
    """Train LightGBM on each dataset variant."""
    from lightgbm import LGBMRegressor

    from src.modeling.training import train_and_log

    run_ids = {}

    for label, dataset_run_id in dataset_ids.items():
        logger.info(f"Training LightGBM on {label} dataset ({dataset_run_id[:8]})")
        model = LGBMRegressor(**LGBM_PARAMS)
        run_id = train_and_log(
            dataset_run_id=dataset_run_id,
            model=model,
            model_name=f"lgbm_cutoff_{label}",
            run_name=f"lgbm_{label}",
            description=f"LightGBM with lgbm_2 hyperparams on {label} cutoff dataset",
            **TRAIN_KWARGS,
        )
        run_ids[label] = run_id
        logger.info(f"  → {label}: run_id={run_id}")

    return run_ids


def compare_results(model_run_ids: dict[str, str]):
    """Print side-by-side comparison table of metrics."""
    import mlflow

    from src.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    headers = ["Metric", *model_run_ids.keys()]
    rows = []

    # Collect metrics from each run
    all_metrics = {}
    for label, run_id in model_run_ids.items():
        run = mlflow.get_run(run_id)
        all_metrics[label] = run.data.metrics

    # Find common metric names
    metric_names = sorted(
        set().union(*(m.keys() for m in all_metrics.values()))
    )

    # Key metrics first
    priority = ["mae", "rmse", "mape", "peak_mae", "r2"]
    ordered = [m for m in priority if m in metric_names]
    ordered += [m for m in metric_names if m not in priority]

    for metric in ordered:
        row = [metric]
        for label in model_run_ids:
            val = all_metrics[label].get(metric, float("nan"))
            row.append(f"{val:.4f}")
        rows.append(row)

    # Print table
    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print()
    print("=" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    print("CUTOFF COMPARISON RESULTS")
    print("=" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def main():
    parser = argparse.ArgumentParser(description="Three-way cutoff comparison")
    parser.add_argument("--build", action="store_true", help="Build 7am and midnight datasets")
    parser.add_argument("--train", action="store_true", help="Train LightGBM on all variants")
    parser.add_argument("--all", action="store_true", help="Build + train + compare")
    parser.add_argument(
        "--dataset-7am", type=str, default=None,
        help="MLflow run_id for 7am dataset (skip building)",
    )
    parser.add_argument(
        "--dataset-midnight", type=str, default=None,
        help="MLflow run_id for midnight dataset (skip building)",
    )
    args = parser.parse_args()

    if not any([args.build, args.train, args.all]):
        parser.print_help()
        return

    # Resolve dataset IDs
    dataset_ids = {"10am_baseline": BASELINE_DATASET_RUN_ID}

    if args.build or args.all:
        new_datasets = build_datasets()
        dataset_ids["7am"] = new_datasets["7am"]
        dataset_ids["midnight"] = new_datasets["midnight"]
    else:
        if args.dataset_7am:
            dataset_ids["7am"] = args.dataset_7am
        if args.dataset_midnight:
            dataset_ids["midnight"] = args.dataset_midnight

    if args.train or args.all:
        model_run_ids = train_models(dataset_ids)
        compare_results(model_run_ids)

        # Save run IDs for reference
        output_path = Path(__file__).parent.parent / "models" / "cutoff_comparison.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(
            {"dataset_ids": dataset_ids, "model_run_ids": model_run_ids},
            indent=2,
        ))
        logger.info(f"Saved run IDs to {output_path}")


if __name__ == "__main__":
    main()
