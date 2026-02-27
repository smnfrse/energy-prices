"""Unified CLI entry point for the energy prices pipeline.

Provides a single command-line interface with subcommands for all pipeline stages:
download, update, combine, merge, and pipeline (full end-to-end).
"""

from pathlib import Path

from loguru import logger
import typer

from src.config import RAW_DATA_DIR, RAW_DIRS, get_path

app = typer.Typer(help="Energy prices data pipeline CLI.")


@app.command()
def download(
    source: str = typer.Argument("smard", help="Data source: 'smard' or 'commodities'."),
    keys: str = typer.Option(None, help="Comma-separated SMARD filter keys (smard only)."),
    region: str = typer.Option("DE-LU", help="Region: 'DE-LU' or 'DE-AT-LU'."),
    resolution: str = typer.Option("hour", help="Resolution: 'quarterhour' or 'hour'."),
    include_capacity: bool = typer.Option(
        False, "--include-capacity", help="Include capacity keys (smard only)."
    ),
    start_date: str = typer.Option("2014-01-01", help="Start date for commodities (YYYY-MM-DD)."),
):
    """Download data from scratch (SMARD or commodities)."""
    if source == "smard":
        from src.data.smard import _parse_keys
        from src.data.sources import SmardSource

        dir_key = (region, resolution)
        if dir_key not in RAW_DIRS:
            raise typer.BadParameter(f"No raw dir configured for {dir_key}")
        output_dir = RAW_DIRS[dir_key]
        output_dir.mkdir(parents=True, exist_ok=True)

        keys_list = _parse_keys(keys, include_capacity, region)
        logger.info(f"Downloading SMARD data: {region}, {resolution}, {len(keys_list)} keys")

        src = SmardSource(region=region, resolution=resolution)
        src.download(output_dir, keys=keys_list)
        logger.success("SMARD download complete.")

    elif source == "commodities":
        from src.data.commodities import download_all_commodities

        output_dir = RAW_DATA_DIR / "prices"
        download_all_commodities(output_dir, start_date)

    else:
        raise typer.BadParameter(f"Unknown source '{source}'. Use 'smard' or 'commodities'.")


@app.command()
def update(
    source: str = typer.Argument("smard", help="Data source: 'smard' or 'commodities'."),
    keys: str = typer.Option(None, help="Comma-separated SMARD filter keys (smard only)."),
    region: str = typer.Option("DE-LU", help="Region: 'DE-LU' or 'DE-AT-LU'."),
    resolution: str = typer.Option("hour", help="Resolution: 'quarterhour' or 'hour'."),
    redundancy_days: int = typer.Option(14, help="Days to re-fetch for corrections."),
    include_capacity: bool = typer.Option(
        False, "--include-capacity", help="Include capacity keys (smard only)."
    ),
):
    """Update existing data incrementally."""
    if source == "smard":
        from src.data.smard import _parse_keys
        from src.data.sources import SmardSource

        dir_key = (region, resolution)
        if dir_key not in RAW_DIRS:
            raise typer.BadParameter(f"No raw dir configured for {dir_key}")
        output_dir = RAW_DIRS[dir_key]

        keys_list = _parse_keys(keys, include_capacity, region)
        logger.info(f"Updating SMARD data: {region}, {resolution}, {len(keys_list)} keys")

        src = SmardSource(region=region, resolution=resolution)
        src.update(output_dir, keys=keys_list, redundancy_days=redundancy_days)
        logger.success("SMARD update complete.")

    elif source == "commodities":
        from src.data.commodities import update_all_commodities

        data_dir = RAW_DATA_DIR / "prices"
        update_all_commodities(data_dir, redundancy_days)

    else:
        raise typer.BadParameter(f"Unknown source '{source}'. Use 'smard' or 'commodities'.")


@app.command()
def combine(
    resolution: str = typer.Option("quarterhour", "--resolution", "-r", help="Data resolution."),
    incremental: bool = typer.Option(False, "--incremental", help="Only process new data."),
):
    """Combine raw CSVs into wide-format Parquet files."""
    from src.data.processing import combine_data, combine_data_incremental

    process_fn = combine_data_incremental if incremental else combine_data

    configs = [
        (RAW_DIRS[("DE-LU", resolution)], get_path("combined_de_lu", resolution)),
        (RAW_DIRS[("DE-AT-LU", resolution)], get_path("combined_de_at_lu", resolution)),
    ]

    for dir_path, out_path in configs:
        if dir_path.exists():
            process_fn(dir_path, out_path)
        else:
            logger.warning(f"Directory not found, skipping: {dir_path}")


@app.command()
def merge(
    resolution: str = typer.Option("quarterhour", "--resolution", "-r", help="Data resolution."),
    incremental: bool = typer.Option(False, "--incremental", help="Only process new data."),
    output_path: Path = typer.Option(None, "--output-path", "-o", help="Output Parquet path."),
):
    """Merge DE-AT-LU and DE-LU datasets with regime indicators."""
    from src.data.processing import run_merge_pipeline, run_merge_pipeline_incremental

    if incremental:
        run_merge_pipeline_incremental(output_path, resolution)
    else:
        run_merge_pipeline(output_path, resolution)


@app.command()
def pipeline(
    resolution: str = typer.Option("hour", "--resolution", "-r", help="Data resolution."),
    region: str = typer.Option("DE-LU", help="Region for SMARD download."),
):
    """Run full pipeline: download -> combine -> merge -> commodities."""
    from src.commodity_processing import process_commodities
    from src.config.smard import get_filter_dict_for_region
    from src.data.commodities import download_all_commodities
    from src.data.processing import combine_data, run_merge_pipeline
    from src.data.sources import SmardSource

    # 1. Download SMARD
    logger.info("=" * 60)
    logger.info("[1/4] Downloading SMARD data...")
    logger.info("=" * 60)

    dir_key = (region, resolution)
    output_dir = RAW_DIRS[dir_key]
    output_dir.mkdir(parents=True, exist_ok=True)

    keys_list = list(get_filter_dict_for_region(region).keys())
    src = SmardSource(region=region, resolution=resolution)
    src.download(output_dir, keys=keys_list)

    # Also download DE-AT-LU if region is DE-LU
    if region == "DE-LU":
        de_at_lu_dir = RAW_DIRS[("DE-AT-LU", resolution)]
        de_at_lu_dir.mkdir(parents=True, exist_ok=True)
        de_at_lu_keys = list(get_filter_dict_for_region("DE-AT-LU").keys())
        SmardSource(region="DE-AT-LU", resolution=resolution).download(
            de_at_lu_dir, keys=de_at_lu_keys
        )

    # 2. Combine
    logger.info("=" * 60)
    logger.info("[2/4] Combining raw CSVs...")
    logger.info("=" * 60)

    combine_data(RAW_DIRS[("DE-LU", resolution)], get_path("combined_de_lu", resolution))
    de_at_lu_raw = RAW_DIRS[("DE-AT-LU", resolution)]
    if de_at_lu_raw.exists():
        combine_data(de_at_lu_raw, get_path("combined_de_at_lu", resolution))

    # 3. Commodities
    logger.info("=" * 60)
    logger.info("[3/4] Downloading and processing commodities...")
    logger.info("=" * 60)
    commodity_dir = RAW_DATA_DIR / "prices"
    download_all_commodities(commodity_dir)
    process_commodities(
        raw_dir=commodity_dir,
        output_dir=get_path("commodity_hourly", resolution).parent,
        smard_path=get_path("merged", resolution),
    )

    # 4. Merge (includes missing value handling + DST normalization)
    logger.info("=" * 60)
    logger.info("[4/4] Merging datasets...")
    logger.info("=" * 60)
    run_merge_pipeline(resolution=resolution)

    logger.success("Full pipeline complete.")


# =============================================================================
# Blend subcommands
# =============================================================================

blend_app = typer.Typer(help="Production ensemble blend commands.")
app.add_typer(blend_app, name="blend")


@blend_app.command("select")
def blend_select(
    holdout_days: int = typer.Option(90, help="Days held out for weight computation."),
    n_splits: int = typer.Option(5, help="Number of CV folds for validation."),
):
    """Full blend pipeline: select candidates, CV validate, train, and blend.

    Each model uses its own dataset (resolved from its MLflow dataset_run_id).
    No shared dataset needs to be specified.
    """
    from src.modeling.blend import (
        select_candidates,
        select_final_models,
        train_and_blend,
        validate_candidates,
    )

    logger.info("[1/4] Selecting candidates from MLflow...")
    candidates = select_candidates()

    logger.info("[2/4] Validating candidates with sliding-window CV...")
    validated = validate_candidates(candidates, n_splits=n_splits)

    logger.info("[3/4] Selecting final models...")
    final = select_final_models(validated)

    logger.info("[4/4] Training and computing blend weights...")
    config = train_and_blend(final, holdout_days=holdout_days)

    logger.success(f"Blend complete. MAE={config['blend_mae']:.4f}")


@blend_app.command("update")
def blend_update():
    """Daily update: incremental tree fit + weight refresh."""
    from src.modeling.blend import update_blend_daily

    update_blend_daily()


@blend_app.command("retrain")
def blend_retrain(
    holdout_days: int = typer.Option(None, help="Override holdout days."),
):
    """Biweekly full retrain: refit all models from scratch."""
    from src.modeling.blend import retrain_blend

    retrain_blend(holdout_days=holdout_days)


@blend_app.command("info")
def blend_info():
    """Print blend_config.json summary."""
    from src.modeling.blend import _load_config, _print_summary

    config = _load_config()
    _print_summary(config)


# =============================================================================
# Baselines subcommands
# =============================================================================

baselines_app = typer.Typer(help="Baseline model commands.")
app.add_typer(baselines_app, name="baselines")


@baselines_app.command("insample")
def baselines_insample(
    transformations: str = typer.Option(
        "none,log_shift", help="Comma-separated transformations to try."
    ),
    models: str = typer.Option(
        "naive_daily,naive_weekly,prophet", help="Comma-separated baseline names."
    ),
):
    """Fit baselines on full dataset and log in-sample metrics to MLflow.

    NOTE: arima and ets are excluded because their fittedvalues are 1-step-ahead (h=1),
    not 24-step-ahead. Use 'baselines rolling' for proper ARIMA/ETS evaluation.
    """
    from src.modeling.baselines import calculate_baselines

    transformation_list = [t.strip() for t in transformations.split(",")]
    model_list = [m.strip() for m in models.split(",")]
    run_ids = calculate_baselines(transformations=transformation_list, baselines=model_list)
    logger.info(f"Completed {len(run_ids)} baseline runs.")


@baselines_app.command("rolling")
def baselines_rolling(
    holdout_days: int = typer.Option(90, help="Holdout days to evaluate."),
    models: str = typer.Option("sarima,ets", help="Comma-separated: sarima, ets"),
):
    """Compute rolling day-ahead holdout forecasts for SARIMA and/or ETS.

    Fits once on the training portion, then rolls through the holdout window
    using warm-start .append(refit=False). Results are logged to MLflow and
    compared against blend_config.json holdout MAEs.
    """
    from src.modeling.baselines import calculate_rolling_baselines

    model_list = [m.strip() for m in models.split(",")]
    results = calculate_rolling_baselines(holdout_days=holdout_days, models=model_list)
    for name, metrics in results.items():
        logger.info(f"{name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")


# =============================================================================
# Inference command
# =============================================================================


@app.command()
def forecast(
    skip_update: bool = typer.Option(
        False, "--skip-update", help="Skip data download, use existing parquet."
    ),
):
    """Generate 24h price forecast from blend ensemble."""
    from src.deploy.inference import run_inference

    run_inference(skip_update=skip_update)


if __name__ == "__main__":
    app()
