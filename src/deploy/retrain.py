"""Biweekly retraining script for the blend ensemble.

Refits all production models from scratch, recomputes blend weights, and checks
for model degradation. Applies EMA overlay before retraining so models train on
EMA forecasts instead of SMARD's leaked/unavailable prognostizierte_* columns.

Usage:
    python -m src.deploy.retrain
    python -m src.deploy.retrain --holdout-days 60
"""

from loguru import logger
import pandas as pd

from src.config import get_path
from src.modeling.blend import retrain_blend


def _apply_ema_overlay() -> None:
    """Apply EMA overlay to merged dataset (historical + snapshots, no live).

    For retraining we use historical + snapshot data only (no live fetch needed
    since we're training on past data, not predicting tomorrow).
    """
    from src.data.ema import build_ema_training_data, get_combined_ema_data

    merged_path = get_path("merged", "hour")
    if not merged_path.exists():
        logger.warning(f"Merged dataset not found at {merged_path}, skipping EMA overlay")
        return

    merged_df = pd.read_parquet(merged_path)

    ema_data = get_combined_ema_data(include_live=False)
    if ema_data.empty:
        logger.warning("No EMA data available, skipping overlay")
        return

    result = build_ema_training_data(merged_df, ema_data)
    result.to_parquet(merged_path)
    logger.info(f"EMA overlay applied for retrain ({len(ema_data)} EMA hours)")


def run_retrain(holdout_days: int | None = None) -> dict:
    """Run full biweekly retrain of the blend ensemble.

    Applies EMA overlay to the merged dataset, then calls retrain_blend() which
    refits all models from scratch, recomputes inverse-MAE blend weights, and
    checks for degradation against the previous blend MAE. Logs a warning if
    the needs_reselection flag is set.

    Args:
        holdout_days: Override holdout days used for weight computation.
                      If None, the value stored in blend_config.json is used.

    Returns:
        Updated blend config dict.
    """
    logger.info("Starting biweekly blend retrain...")
    _apply_ema_overlay()
    config = retrain_blend(holdout_days=holdout_days)

    if config.get("needs_reselection"):
        logger.warning(
            "needs_reselection=True: blend MAE degraded beyond threshold. "
            "Run 'python -m src.cli blend select' to reselect candidates."
        )

    return config


# --- CLI ---

if __name__ == "__main__":
    import typer

    def main(
        holdout_days: int = typer.Option(
            None, "--holdout-days", help="Override holdout days (default: use config value)."
        ),
    ):
        """Biweekly full retrain: refit all blend models from scratch."""
        run_retrain(holdout_days=holdout_days)

    typer.run(main)
