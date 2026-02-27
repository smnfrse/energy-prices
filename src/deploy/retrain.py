"""Biweekly retraining script for the blend ensemble.

Refits all production models from scratch, recomputes blend weights, and checks
for model degradation. Data updates are handled separately by the GitHub Actions
workflow before this script is invoked.

Usage:
    python -m src.deploy.retrain
    python -m src.deploy.retrain --holdout-days 60
"""

from loguru import logger

from src.modeling.blend import retrain_blend


def run_retrain(holdout_days: int | None = None) -> dict:
    """Run full biweekly retrain of the blend ensemble.

    Calls retrain_blend(), which refits all models from scratch, recomputes
    inverse-MAE blend weights, and checks for degradation against the previous
    blend MAE. Logs a warning if the needs_reselection flag is set.

    Args:
        holdout_days: Override holdout days used for weight computation.
                      If None, the value stored in blend_config.json is used.

    Returns:
        Updated blend config dict.
    """
    logger.info("Starting biweekly blend retrain...")
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
