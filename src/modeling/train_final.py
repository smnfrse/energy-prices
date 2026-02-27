"""Retrain the production blend from scratch using committed hyperparameters.

    python -m src.modeling.train_final [--dry-run]

Reads models/production/blend_hyperparams.json, trains each model with its
stored hyperparameters, then calls train_and_blend() to compute fresh weights
and write a new blend_config.json. MLflow runs are tagged retrain=true.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import MODELS_DIR
from src.modeling.blend import train_and_blend
from src.modeling.training import train_and_log

HYPERPARAMS_PATH = MODELS_DIR / "production" / "blend_hyperparams.json"


def _coerce(v: str):
    """Convert an MLflow-stored string parameter back to a Python scalar."""
    if v == "None":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _build_model(category: str, raw_params: dict):
    """Instantiate the model for a given blend category from stored params."""
    mp = {
        k[6:]: _coerce(v)
        for k, v in raw_params.items()
        if k.startswith("model_") and k != "model_class"
    }

    if category == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**{k: v for k, v in mp.items() if v is not None})

    if category == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**{k: v for k, v in mp.items() if v is not None})

    if category == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**{k: v for k, v in mp.items() if v is not None})

    if category == "linear":
        from sklearn.multioutput import MultiOutputRegressor

        estimator_params = {k[11:]: v for k, v in mp.items() if k.startswith("estimator__")}
        kw = {k: v for k, v in estimator_params.items() if v is not None}
        # Detect Lasso vs Ridge: 'selection' is Lasso-specific
        if "selection" in estimator_params:
            from sklearn.linear_model import Lasso

            estimator = Lasso(**kw)
        else:
            from sklearn.linear_model import Ridge

            estimator = Ridge(**kw)
        return MultiOutputRegressor(estimator, n_jobs=mp.get("n_jobs"))

    raise ValueError(f"Unknown category: {category!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain production blend from committed hyperparameters."
    )
    parser.add_argument(
        "--hyperparams",
        default=str(HYPERPARAMS_PATH),
        help="Path to blend_hyperparams.json (default: models/production/blend_hyperparams.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print training plan without running",
    )
    args = parser.parse_args()

    with open(args.hyperparams) as f:
        spec = json.load(f)

    if args.dry_run:
        print(f"Would train {len(spec['models'])} models:")
        for m in spec["models"]:
            p = m["params"]
            print(
                f"  {m['name']:20s}  {m['model_class']:25s}  dataset={p['dataset_run_id'][:8]}..."
            )
        return

    rows = []
    for entry in spec["models"]:
        params = entry["params"]
        category = entry["category"]
        dataset_run_id = params["dataset_run_id"]
        test_size = float(params.get("test_size", 0.1))
        group_size = int(params.get("group_size", 1))
        whl = params.get("weight_half_life")
        weight_half_life = float(whl) if whl and whl != "None" else None

        model = _build_model(category, params)
        run_id = train_and_log(
            dataset_run_id=dataset_run_id,
            model=model,
            model_name=entry["name"],
            test_size=test_size,
            group_size=group_size,
            weight_half_life=weight_half_life,
            experiment=f"{category}_retrain",
            tags={"retrain": "true"},
        )
        rows.append(
            {
                "run_id": run_id,
                "model_class": type(model).__name__,
                "category": category,
                "group_size": group_size,
                "mae": 0.0,
                "rmse": 0.0,
                "dataset_run_id": dataset_run_id,
            }
        )

    train_and_blend(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
