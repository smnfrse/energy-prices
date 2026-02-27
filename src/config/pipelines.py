"""Pipeline builder functions for feature engineering.

Contains reference pipelines (v1 daily/hourly) and versioned preprocessor
pipelines (v2–v5) used across notebooks and training scripts.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline  # noqa: F401 (used in return type annotations)

from src.config.features import (
    ROLLING_FEATURE_SPECS,
    ROLLING_FEATURE_SPECS_v3,
    ROLLING_FEATURE_SPECS_v5,
    build_aggregation_rules,
    build_aggregation_rules_v3,
)
from src.features.ts_transforms import WindowSpec

# =============================================================================
# Rolling Feature Specs v2
# =============================================================================

ROLLING_FEATURE_SPECS_v2 = {
    "actuals_d2": {
        "columns": [
            "stromerzeugung_*",
            "stromverbrauch_gesamt_(netzlast)",
            "cross-border_*",
            "net_export_*",
            # "total_generation",
        ],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
    "prices_d1": {
        "columns": [
            "marktpreis_*",
            "spread_*",
            "ratio_*",
        ],
        "windows": [WindowSpec(start_day=-1, end_day=-1, agg="mean")],
        "overwrite": True,
    },
    "commodities_d2": {
        "columns": ["ttf_eur_per_mwh", "brent_usd_per_barrel", "carbon_eur_per_ton"],
        "windows": [WindowSpec(start_day=-2, end_day=-2, agg="mean")],
        "overwrite": True,
    },
}


# =============================================================================
# Reference Pipelines (v1)
# =============================================================================


def reference_hourly_pipeline() -> "Pipeline":
    """Reference hourly feature pipeline (formerly build_default_pipeline).

    Returns an sklearn Pipeline that:
    1. Creates custom columns (temporal, holiday, spreads, net exports, ratios,
       generation percentages, prognosticated percentages)
    2. Drops redundant columns

    Returns:
        Unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline

    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GenerationPercentageTransformer,
        GermanHolidayTransformer,
        NetExportTransformer,
        PriceRatioTransformer,
        PriceSpreadTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )

    return Pipeline(
        [
            (
                "custom_cols",
                CreateCustomColumns(
                    [
                        TemporalFeatureTransformer(),
                        GermanHolidayTransformer(),
                        PriceSpreadTransformer(),
                        NetExportTransformer(),
                        PriceRatioTransformer(),
                        GenerationPercentageTransformer(),
                        PrognosticatedPercentageTransformer(),
                    ]
                ),
            ),
            ("drop_cols", ColumnDropper()),
        ]
    )


def reference_daily_pipeline() -> "Pipeline":
    """Reference daily feature pipeline (formerly build_daily_pipeline).

    Transforms hourly data into daily format with 24-element target arrays,
    properly lagged features, and feature scaling.

    .. deprecated::
        Use reference_preprocessor() + reference_ml_pipeline() for dataset caching.
        This function will be removed in a future version.

    Returns:
        Unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline

    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        FeatureScaler,
        GenerationPercentageTransformer,
        GermanHolidayTransformer,
        NetExportTransformer,
        PriceRatioTransformer,
        PriceSpreadTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        DailyPivotTransformer,
        EWMATransformer,
        PivotSpec,
        RollingStatsTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    # Build rolling steps dynamically from ROLLING_FEATURE_SPECS
    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            # Phase 1: Hourly custom features (spreads, net exports, ratios)
            (
                "hourly_features",
                CreateCustomColumns(
                    [
                        PriceSpreadTransformer(),
                        NetExportTransformer(),
                        PriceRatioTransformer(),
                    ]
                ),
            ),
            # Phase 2: Rolling stats on target_price (overwrite=False, new columns)
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                    ],
                ),
            ),
            # Phase 3: EWMA on target_price (cutoff_hour=10)
            (
                "ewma",
                EWMATransformer(
                    columns=["target_price"],
                    spans=[6, 24, 168, 720, 2160],
                    cutoff_day=-1,
                ),
            ),
            # Phase 4: Lagged features — overwrite originals from ROLLING_FEATURE_SPECS
            *rolling_steps,
            # Phase 5: Hourly -> daily (DailyPivotTransformer)
            (
                "daily_pivot",
                DailyPivotTransformer(
                    pivot_specs=[
                        PivotSpec(column="target_price", day_offset=0, prefix="y"),
                        PivotSpec(
                            column="target_price",
                            day_offset=-1,
                            prefix="price_lag",
                        ),
                        PivotSpec(
                            column="prognostizierter_verbrauch_gesamt",
                            day_offset=0,
                            prefix="prog_verbrauch",
                        ),
                        PivotSpec(
                            column="prognostizierte_erzeugung_gesamt",
                            day_offset=0,
                            prefix="prog_erzeugung",
                        ),
                        PivotSpec(
                            column="prognostizierte_erzeugung_wind_und_photovoltaik",
                            day_offset=0,
                            prefix="prog_wind_pv",
                        ),
                    ],
                    aggregation_rules=build_aggregation_rules(),
                ),
            ),
            # Phase 6: Daily features (after aggregation)
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        GenerationPercentageTransformer(),
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                        GermanHolidayTransformer(),
                    ]
                ),
            ),
            # Phase 7: Drop unnecessary columns
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        "hour_of_day",
                        "hour_sin",
                        "hour_cos",
                    ]
                ),
            ),
            # Phase 8: Scale targets (log-shift)
            (
                "scale_targets",
                FeatureScaler(
                    columns=["y_*", "price_lag_*"],
                    method="log_shift",
                ),
            ),
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe


def reference_preprocessor() -> "Pipeline":
    """Data preparation pipeline for dataset creation (no scaling).

    This is the expensive preprocessing that should be cached. Does NOT
    include FeatureScaler to avoid data leakage during cross-validation.

    Returns the same pipeline as reference_daily_pipeline() but WITHOUT
    the "scale_targets" step.

    Returns:
        Unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline

    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GenerationPercentageTransformer,
        GermanHolidayTransformer,
        NetExportTransformer,
        PriceSpreadTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        DailyPivotTransformer,
        EWMATransformer,
        PivotSpec,
        RollingStatsTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    # Build rolling steps dynamically from ROLLING_FEATURE_SPECS
    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            # Phase 1: Hourly custom features (spreads, net exports, ratios)
            (
                "hourly_features",
                CreateCustomColumns(
                    [
                        PriceSpreadTransformer(),
                        NetExportTransformer(),
                        # PriceRatioTransformer(),
                    ]
                ),
            ),
            # Phase 2: Rolling stats on target_price (overwrite=False, new columns)
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                    ],
                ),
            ),
            # Phase 3: EWMA on target_price (cutoff_hour=10)
            (
                "ewma",
                EWMATransformer(
                    columns=["target_price"],
                    spans=[6, 24, 168, 720, 2160],
                    cutoff_day=-1,
                ),
            ),
            # Phase 4: Lagged features — overwrite originals from ROLLING_FEATURE_SPECS
            *rolling_steps,
            # Phase 5: Hourly -> daily (DailyPivotTransformer)
            (
                "daily_pivot",
                DailyPivotTransformer(
                    pivot_specs=[
                        PivotSpec(column="target_price", day_offset=0, prefix="y"),
                        PivotSpec(
                            column="target_price",
                            day_offset=-1,
                            prefix="price_lag",
                        ),
                        PivotSpec(
                            column="prognostizierter_verbrauch_gesamt",
                            day_offset=0,
                            prefix="prog_verbrauch",
                        ),
                        PivotSpec(
                            column="prognostizierte_erzeugung_gesamt",
                            day_offset=0,
                            prefix="prog_erzeugung",
                        ),
                        PivotSpec(
                            column="prognostizierte_erzeugung_wind_und_photovoltaik",
                            day_offset=0,
                            prefix="prog_wind_pv",
                        ),
                    ],
                    aggregation_rules=build_aggregation_rules(),
                ),
            ),
            # Phase 6: Daily features (after aggregation)
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        GenerationPercentageTransformer(),
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                        GermanHolidayTransformer(),
                    ]
                ),
            ),
            # Phase 7: Drop unnecessary columns
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        "hour_of_day",
                        "hour_sin",
                        "hour_cos",
                    ]
                ),
            ),
            # NOTE: NO scale_targets step (moved to reference_ml_pipeline)
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe


def reference_ml_pipeline() -> "Pipeline":
    """ML pipeline for training (no model).

    Target scaling is now handled by TransformedTargetRegressor in
    train_and_log() / evaluate_pipeline(), so this pipeline is a passthrough.

    Returns:
        Pipeline with passthrough step (no-op).
    """
    from sklearn.pipeline import Pipeline

    return Pipeline([("passthrough", "passthrough")])


# =============================================================================
# Version 2 Pipeline
# =============================================================================


def preprocessor_v2() -> "Pipeline":
    """Data preparation pipeline for dataset creation (no scaling).

    This is the expensive preprocessing that should be cached. Does NOT
    include FeatureScaler to avoid data leakage during cross-validation.

    Version 2 of the pipeline removing low value features

    Returns:
        Unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline

    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GermanHolidayTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        DailyPivotTransformer,
        EWMATransformer,
        PivotSpec,
        RollingStatsTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    # Build rolling steps dynamically from ROLLING_FEATURE_SPECS
    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS_v2.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            # Phase 2: Rolling stats on target_price (overwrite=False, new columns)
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                    ],
                ),
            ),
            # Phase 3: EWMA on target_price (cutoff_hour=10)
            (
                "ewma",
                EWMATransformer(
                    columns=["target_price"],
                    spans=[6, 24, 168, 720, 2160],
                    cutoff_day=-1,
                ),
            ),
            # Phase 4: Lagged features — overwrite originals from ROLLING_FEATURE_SPECS
            *rolling_steps,
            # Phase 5: Hourly -> daily (DailyPivotTransformer)
            (
                "daily_pivot",
                DailyPivotTransformer(
                    pivot_specs=[
                        PivotSpec(column="target_price", day_offset=0, prefix="y"),
                        PivotSpec(
                            column="target_price",
                            day_offset=-1,
                            prefix="price_lag",
                        ),
                        PivotSpec(
                            column="prognostizierte_erzeugung_wind_und_photovoltaik",
                            day_offset=0,
                            prefix="prog_wind_pv",
                        ),
                    ],
                    aggregation_rules=build_aggregation_rules(),
                ),
            ),
            # Phase 6: Daily features (after aggregation)
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                        GermanHolidayTransformer(),
                    ]
                ),
            ),
            # Phase 7: Drop unnecessary columns
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        "hour_of_day",
                        "hour_sin",
                        "hour_cos",
                    ]
                ),
            ),
            # NOTE: NO scale_targets step (moved to reference_ml_pipeline)
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe


# =============================================================================
# Version 3: Feature Refinement Pipeline
# =============================================================================


def preprocessor_v3() -> "Pipeline":
    """Feature refinement pipeline v3 - removes noise, adds high-value features.

    Changes from v2:
    - Removed: cross-border flows, net exports, price spreads, hourly prog
      consumption/generation
    - Removed: month, holiday, regime temporal features
    - Added: Generation percentages (re-enabled), renewable %, supply-demand gap
    - Added: Morning-of-D-1 actuals (hours 0-10)
    - Added: Expanded EWMA features (prices, actuals, commodities with
      appropriate cutoffs)
    - Added: Price volatility range features (7d, 30d)

    Returns:
        Unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GenerationPercentageTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        DailyPivotTransformer,
        EWMATransformer,
        PivotSpec,
        RollingStatsTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    # Helper for price range features
    def add_price_ranges(X):
        X = X.copy()
        if "target_price_d7_d1_max" in X.columns and "target_price_d7_d1_min" in X.columns:
            X["target_price_range_7d"] = X["target_price_d7_d1_max"] - X["target_price_d7_d1_min"]
        if "target_price_d30_d1_max" in X.columns and "target_price_d30_d1_min" in X.columns:
            X["target_price_range_30d"] = (
                X["target_price_d30_d1_max"] - X["target_price_d30_d1_min"]
            )
        return X

    # Build rolling steps dynamically from ROLLING_FEATURE_SPECS_v3
    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS_v3.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            # Phase 1: Rolling stats on target_price (keep existing 10 windows)
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                    ],
                ),
            ),
            # Phase 2: EWMA - prices (cutoff_day=-1, no hour cutoff - full D-1)
            (
                "ewma_prices",
                EWMATransformer(
                    columns=[
                        "target_price",
                        "marktpreis_frankreich",
                        "marktpreis_belgien",
                        "marktpreis_niederlande",
                        "marktpreis_österreich",
                    ],
                    spans=[6, 24, 2160],
                    cutoff_day=-1,
                ),
            ),
            # Phase 3: EWMA - actuals (cutoff_day=-1, cutoff_hour=10 - morning D-1)
            (
                "ewma_actuals",
                EWMATransformer(
                    columns=[
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    spans=[24, 168, 2160],
                    cutoff_day=-1,
                    cutoff_hour=10,
                ),
            ),
            # Phase 4: EWMA - commodities (cutoff_day=-2, no hour cutoff - full D-2)
            (
                "ewma_commodities",
                EWMATransformer(
                    columns=[
                        "carbon_eur_per_ton",
                        "ttf_eur_per_mwh",
                        "brent_usd_per_barrel",
                    ],
                    spans=[24, 720, 2160],
                    cutoff_day=-2,
                ),
            ),
            # Phase 5: Morning-of-D-1 actuals (hours 0-10)
            (
                "morning_actuals",
                RollingStatsTransformer(
                    columns=[
                        "stromverbrauch_gesamt_(netzlast)",
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_wind_offshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    windows=[
                        WindowSpec(start_day=-1, end_day=-1, hours=list(range(0, 11)), agg="mean")
                    ],
                    overwrite=False,
                ),
            ),
            # Phase 6: Lag overwrite (ROLLING_FEATURE_SPECS_v3)
            *rolling_steps,
            # Phase 7: Daily pivot (target today + yesterday + prog_wind_pv only)
            (
                "daily_pivot",
                DailyPivotTransformer(
                    pivot_specs=[
                        PivotSpec(column="target_price", day_offset=0, prefix="y"),
                        PivotSpec(column="target_price", day_offset=-1, prefix="price_lag"),
                        PivotSpec(
                            column="prognostizierte_erzeugung_wind_und_photovoltaik",
                            day_offset=0,
                            prefix="prog_wind_pv",
                        ),
                    ],
                    aggregation_rules=build_aggregation_rules_v3(),
                ),
            ),
            # Phase 8: Daily features (generation %, prognosticated %, temporal)
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        GenerationPercentageTransformer(
                            add_renewable_pct=True, add_supply_demand_gap=True
                        ),
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                    ]
                ),
            ),
            # Phase 8b: Price range features
            ("price_ranges", FunctionTransformer(add_price_ranges, validate=False)),
            # Phase 9: Drop unnecessary columns
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        "cross-border_*",
                        "net_export_*",
                        "month",
                        "month_sin",
                        "month_cos",
                        "regime_*",
                    ],
                    include=[
                        "day_of_month",  # Keep this even though it has "month" in name
                    ],
                ),
            ),
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe


# =============================================================================
# Version 4: Hourly Global Model Pipeline
# =============================================================================


def preprocessor_v4_hourly() -> "Pipeline":
    """Hourly global model pipeline — no daily pivot.

    Reuses v3 phases 1-6 (rolling stats, EWMA, morning actuals, lag
    overwrite), then adds hourly-specific transformers instead of
    DailyPivotTransformer.

    Output: one row per hour, single scalar target column 'y'.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GenerationPercentageTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        EWMATransformer,
        HourlyDailyAggregateTransformer,
        RollingStatsTransformer,
        SameHourLagTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    # Helper for price range features (same as v3)
    def add_price_ranges(X):
        X = X.copy()
        if "target_price_d7_d1_max" in X.columns and "target_price_d7_d1_min" in X.columns:
            X["target_price_range_7d"] = X["target_price_d7_d1_max"] - X["target_price_d7_d1_min"]
        if "target_price_d30_d1_max" in X.columns and "target_price_d30_d1_min" in X.columns:
            X["target_price_range_30d"] = (
                X["target_price_d30_d1_max"] - X["target_price_d30_d1_min"]
            )
        return X

    # Rename target_price to y (hourly single target)
    def rename_target(X):
        X = X.copy()
        if "target_price" in X.columns:
            X = X.rename(columns={"target_price": "y"})
        return X

    # Build rolling lag overwrite steps (same as v3)
    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS_v3.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            # Phase 1: Rolling stats on target_price (same as v3)
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                    ],
                ),
            ),
            # Phase 2: EWMA prices (same as v3)
            (
                "ewma_prices",
                EWMATransformer(
                    columns=[
                        "target_price",
                        "marktpreis_frankreich",
                        "marktpreis_belgien",
                        "marktpreis_niederlande",
                        "marktpreis_österreich",
                    ],
                    spans=[6, 24, 2160],
                    cutoff_day=-1,
                ),
            ),
            # Phase 3: EWMA actuals (same as v3)
            (
                "ewma_actuals",
                EWMATransformer(
                    columns=[
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    spans=[24, 168, 2160],
                    cutoff_day=-1,
                    cutoff_hour=10,
                ),
            ),
            # Phase 4: EWMA commodities (same as v3)
            (
                "ewma_commodities",
                EWMATransformer(
                    columns=[
                        "carbon_eur_per_ton",
                        "ttf_eur_per_mwh",
                        "brent_usd_per_barrel",
                    ],
                    spans=[24, 720, 2160],
                    cutoff_day=-2,
                ),
            ),
            # Phase 5: Morning-of-D-1 actuals (same as v3)
            (
                "morning_actuals",
                RollingStatsTransformer(
                    columns=[
                        "stromverbrauch_gesamt_(netzlast)",
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_wind_offshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    windows=[
                        WindowSpec(
                            start_day=-1,
                            end_day=-1,
                            hours=list(range(0, 11)),
                            agg="mean",
                        ),
                    ],
                    overwrite=False,
                ),
            ),
            # Phase 6: Lag overwrite (same as v3)
            *rolling_steps,
            # --- HOURLY-SPECIFIC PHASES (diverge from v3) ---
            # Phase 7: Same-hour price lags (D-1, D-7)
            (
                "same_hour_lags",
                SameHourLagTransformer(
                    column="target_price",
                    day_offsets=(-1, -7),
                ),
            ),
            # Phase 8: Daily aggregates of forecasts (broadcast back)
            (
                "daily_aggregates",
                HourlyDailyAggregateTransformer(
                    columns=[
                        "prognostizierte_erzeugung_wind_und_photovoltaik",
                        "prognostizierter_verbrauch_gesamt",
                    ],
                    aggs=("sum", "mean", "max"),
                ),
            ),
            # Phase 9: Daily features (generation %, prog %, temporal)
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        GenerationPercentageTransformer(
                            add_renewable_pct=True,
                            add_supply_demand_gap=True,
                        ),
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                    ]
                ),
            ),
            # Phase 9b: Price range features (same as v3)
            (
                "price_ranges",
                FunctionTransformer(
                    add_price_ranges,
                    validate=False,
                ),
            ),
            # Phase 10: Rename target_price -> y
            (
                "rename_target",
                FunctionTransformer(
                    rename_target,
                    validate=False,
                ),
            ),
            # Phase 11: Drop unnecessary columns (hourly-adapted)
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        "cross-border_*",
                        "net_export_*",
                        "month",
                        "month_sin",
                        "month_cos",
                        "regime_*",
                    ],
                    include=[
                        "day_of_month",
                    ],
                ),
            ),
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe


# =============================================================================
# Version 5: Slim Hourly Pipeline (85 features, curated)
# =============================================================================


_DAY_INDEX_EPOCH = pd.Timestamp("2015-01-04 23:00:00", tz="UTC")  # 2015-01-05 00:00 CET
_YEAR_INDEX_BASE = 2015


def preprocessor_v5_slim_hourly() -> "Pipeline":
    """Curated 85-feature hourly pipeline with leakage fixes and new time features.

    Key changes from v4:
    - Leakage fixes: drops stromverbrauch_residuallast and
      stromverbrauch_pumpspeicher
    - Aggressive feature pruning: 126 -> 85 features
    - New: same-hour lags for target (D-2, D-14), marktpreis
      (frankreich/schweiz D-1), generation actuals (wind_onshore/offshore,
      photovoltaik D-2)
    - New: target EWMA variants with D-1 10am cutoff (_h10 suffix)
    - New: total_imports / total_exports (sum of cross-border flows, D-2 mean)
    - New: day_index + year_index (time trend features)
    - New: 5 interaction terms (top features x day_index)

    Returns:
        Unfitted sklearn Pipeline producing 85 features + target column 'y'.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from src.config.processing import EXPORT_COLUMNS, IMPORT_COLUMNS
    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GenerationPercentageTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        EWMATransformer,
        HourlyDailyAggregateTransformer,
        RollingStatsTransformer,
        SameHourLagTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    def add_price_ranges(X):
        X = X.copy()
        if "target_price_d7_d1_max" in X.columns and "target_price_d7_d1_min" in X.columns:
            X["target_price_range_7d"] = X["target_price_d7_d1_max"] - X["target_price_d7_d1_min"]
        if "target_price_d30_d1_max" in X.columns and "target_price_d30_d1_min" in X.columns:
            X["target_price_range_30d"] = (
                X["target_price_d30_d1_max"] - X["target_price_d30_d1_min"]
            )
        return X

    def rename_target(X):
        X = X.copy()
        if "target_price" in X.columns:
            X = X.rename(columns={"target_price": "y"})
        return X

    def add_total_flows(X):
        X = X.copy()
        export_cols = [c for c in EXPORT_COLUMNS if c in X.columns]
        import_cols = [c for c in IMPORT_COLUMNS if c in X.columns]
        X["total_exports"] = X[export_cols].sum(axis=1, min_count=1)
        X["total_imports"] = X[import_cols].sum(axis=1, min_count=1)
        return X

    def add_day_index_and_interactions(X):
        X = X.copy()
        # day_index: absolute days since 2015-01-05 00:00 CET (epoch)
        normalized = X.index.normalize()
        X["day_index"] = ((normalized - _DAY_INDEX_EPOCH).total_seconds() // 86400).astype("int64")
        # year_index: 0 = 2015
        X["year_index"] = (X.index.year - _YEAR_INDEX_BASE).astype("int64")
        # Interaction terms: top-5 features x day_index
        pairs = [
            ("prognostizierter_verbrauch_residuallast", "interact_residuallast_forecast"),
            ("target_price_ewma_6h", "interact_target_ewma_6h"),
            ("ttf_eur_per_mwh_ewma_720h", "interact_ttf_ewma_720h"),
            ("ttf_eur_per_mwh_ewma_24h", "interact_ttf_ewma_24h"),
            ("prognostizierte_erzeugung_sonstige", "interact_sonstige_forecast"),
        ]
        for src, dst in pairs:
            if src in X.columns:
                X[dst] = X[src] * X["day_index"]
        return X

    # Build rolling overwrite steps from ROLLING_FEATURE_SPECS_v5
    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS_v5.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            # Phase 1: Rolling stats on target_price
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                        WindowSpec(start_day=-2, end_day=-1, agg="std"),  # 48h volatility
                        WindowSpec(start_day=-3, end_day=-1, agg="std"),  # 72h volatility
                    ],
                ),
            ),
            # Phase 2: EWMA prices — end-of-D-1 cutoff (target + frankreich only)
            (
                "ewma_prices",
                EWMATransformer(
                    columns=["target_price", "marktpreis_frankreich"],
                    spans=[6, 24, 2160],
                    cutoff_day=-1,
                ),
            ),
            # Phase 2b: EWMA target price — D-1 10am cutoff, _h10 suffix
            (
                "ewma_prices_h10",
                EWMATransformer(
                    columns=["target_price"],
                    spans=[6, 24, 2160],
                    cutoff_day=-1,
                    cutoff_hour=10,
                    col_suffix="_h10",
                ),
            ),
            # Phase 3: EWMA actuals — D-1 10am cutoff
            (
                "ewma_actuals",
                EWMATransformer(
                    columns=[
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    spans=[24, 168, 2160],
                    cutoff_day=-1,
                    cutoff_hour=10,
                ),
            ),
            # Phase 4: EWMA commodities — full D-2 cutoff
            (
                "ewma_commodities",
                EWMATransformer(
                    columns=[
                        "carbon_eur_per_ton",
                        "ttf_eur_per_mwh",
                        "brent_usd_per_barrel",
                    ],
                    spans=[24, 720, 2160],
                    cutoff_day=-2,
                ),
            ),
            # Phase 5: Morning-of-D-1 actuals (hours 0-10, 3 columns vs v4's 5)
            (
                "morning_actuals",
                RollingStatsTransformer(
                    columns=[
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_wind_offshore",
                    ],
                    windows=[
                        WindowSpec(start_day=-1, end_day=-1, hours=list(range(0, 11)), agg="mean"),
                    ],
                    overwrite=False,
                ),
            ),
            # Phase 5b: Total cross-border flows — BEFORE Phase 6 overwrite
            (
                "total_flows",
                FunctionTransformer(add_total_flows, validate=False),
            ),
            # Phase 5c: Same-hour price lags — BEFORE Phase 6 overwrites marktpreis_*
            (
                "same_hour_lag_frankreich",
                SameHourLagTransformer(column="marktpreis_frankreich", day_offsets=(-1,)),
            ),
            (
                "same_hour_lag_schweiz",
                SameHourLagTransformer(column="marktpreis_schweiz", day_offsets=(-1,)),
            ),
            # Phase 5d: Same-hour gen lags — BEFORE Phase 6 overwrites stromerzeugung_*
            (
                "same_hour_lag_wind_onshore",
                SameHourLagTransformer(column="stromerzeugung_wind_onshore", day_offsets=(-2,)),
            ),
            (
                "same_hour_lag_wind_offshore",
                SameHourLagTransformer(column="stromerzeugung_wind_offshore", day_offsets=(-2,)),
            ),
            (
                "same_hour_lag_photovoltaik",
                SameHourLagTransformer(column="stromerzeugung_photovoltaik", day_offsets=(-2,)),
            ),
            # Phase 5e: Same-hour target price lags (D-1, D-2, D-7, D-14)
            (
                "same_hour_lags_target",
                SameHourLagTransformer(column="target_price", day_offsets=(-1, -2, -7, -14)),
            ),
            # Phase 6: Lag overwrites (daily means, no forecast columns overwritten)
            *rolling_steps,
            # Phase 7: Daily aggregates of forecast columns (broadcast to all hours)
            (
                "daily_aggregates",
                HourlyDailyAggregateTransformer(
                    columns=[
                        "prognostizierte_erzeugung_wind_und_photovoltaik",
                        "prognostizierter_verbrauch_gesamt",
                    ],
                    aggs=("sum", "mean", "max"),
                ),
            ),
            # Phase 8: Derived columns (percentages, temporal)
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        GenerationPercentageTransformer(
                            add_renewable_pct=False, add_supply_demand_gap=True
                        ),
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                    ]
                ),
            ),
            # Phase 9: Price range features
            ("price_ranges", FunctionTransformer(add_price_ranges, validate=False)),
            # Phase 10: Rename target_price -> y
            ("rename_target", FunctionTransformer(rename_target, validate=False)),
            # Phase 11: Drop unused columns (leakage fixes + low-importance pruning)
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        # Structural
                        "cross-border_*",
                        "net_export_*",
                        "month",
                        "month_sin",
                        "month_cos",
                        "regime_*",
                        # Leakage fixes
                        "stromverbrauch_residuallast",
                        "stromverbrauch_pumpspeicher",
                        # Low-importance temporal
                        "is_weekend",
                        "day_of_month",
                        "week_of_year",
                        # Degenerate / low-importance daily aggregates
                        "prognostizierte_erzeugung_wind_und_photovoltaik_daily_mean",
                        "prognostizierter_verbrauch_gesamt_daily_mean",
                        "prognostizierte_erzeugung_wind_und_photovoltaik_daily_sum",
                        "prognostizierter_verbrauch_gesamt_daily_sum",
                        "prognostizierter_verbrauch_gesamt_daily_max",
                        # Generation mix pct (low importance)
                        "pct_renewable",
                        "pct_biomasse",
                        "pct_braunkohle",
                        "pct_photovoltaik",
                        "pct_sonstige_erneuerbare",
                        "pct_steinkohle",
                        "pct_wind_onshore",
                        "total_generation",
                        # Neighbour prices (keep frankreich/schweiz/niederlande/osterreich)
                        "marktpreis_belgien",
                        "marktpreis_dänemark_1",
                        "marktpreis_dänemark_2",
                        "marktpreis_italien_(nord)",
                        "marktpreis_norwegen_2",
                        "marktpreis_polen",
                        "marktpreis_schweden_4",
                        "marktpreis_slowenien",
                        "marktpreis_tschechien",
                        "marktpreis_ungarn",
                        # Neighbour EWMAs (keep frankreich only)
                        "marktpreis_belgien_ewma_*",
                        "marktpreis_niederlande_ewma_*",
                        "marktpreis_österreich_ewma_*",
                        # Generation actuals (keep wind_onshore/offshore, pv, erdgas)
                        "stromerzeugung_biomasse",
                        "stromerzeugung_braunkohle",
                        "stromerzeugung_kernenergie",
                        "stromerzeugung_pumpspeicher",
                        "stromerzeugung_sonstige_erneuerbare",
                        "stromerzeugung_sonstige_konventionelle",
                        "stromerzeugung_steinkohle",
                        "stromerzeugung_wasserkraft",
                        # Generation EWMA spans (keep one per source)
                        "stromerzeugung_wind_onshore_ewma_24h",
                        "stromerzeugung_wind_onshore_ewma_2160h",
                        "stromerzeugung_photovoltaik_ewma_24h",
                        "stromerzeugung_photovoltaik_ewma_168h",
                        "stromverbrauch_residuallast_ewma_168h",
                        "stromverbrauch_residuallast_ewma_2160h",
                        # Commodity EWMAs (keep carbon_24h, ttf_24h, ttf_720h)
                        "carbon_eur_per_ton_ewma_720h",
                        "carbon_eur_per_ton_ewma_2160h",
                        "ttf_eur_per_mwh_ewma_2160h",
                        "brent_usd_per_barrel_ewma_24h",
                        "brent_usd_per_barrel_ewma_720h",
                        "brent_usd_per_barrel_ewma_2160h",
                        # Morning actuals (drop gesamt and photovoltaik)
                        "stromverbrauch_gesamt_(netzlast)_d1_h0-10_mean",
                        "stromerzeugung_photovoltaik_d1_h0-10_mean",
                        # Low-importance rolling target stats
                        "target_price_d30_d1_max",
                        "target_price_d30_d1_min",
                        "target_price_d7_d1_h8-19_mean",
                    ],
                ),
            ),
            # Phase 12: Time index features + interaction terms (after drop_cols)
            (
                "day_index_interactions",
                FunctionTransformer(add_day_index_and_interactions, validate=False),
            ),
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe


# =============================================================================
# Version 5: Full Hourly Pipeline (~138 features, for feature selection)
# =============================================================================


def preprocessor_v5_full_hourly() -> "Pipeline":
    """Maximal feature set hourly pipeline for feature selection experiments.

    Same phase structure as preprocessor_v5_slim_hourly() but with a minimal
    ColumnDropper: only leakage bugs and structural columns are removed.

    Returns:
        Unfitted sklearn Pipeline producing ~138 features + target column 'y'.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from src.config.processing import EXPORT_COLUMNS, IMPORT_COLUMNS
    from src.features.transforms import (
        ColumnDropper,
        CreateCustomColumns,
        GenerationPercentageTransformer,
        PrognosticatedPercentageTransformer,
        TemporalFeatureTransformer,
    )
    from src.features.ts_transforms import (
        EWMATransformer,
        HourlyDailyAggregateTransformer,
        RollingStatsTransformer,
        SameHourLagTransformer,
    )
    from src.features.validation import validate_pipeline_leakage

    def add_price_ranges(X):
        X = X.copy()
        if "target_price_d7_d1_max" in X.columns and "target_price_d7_d1_min" in X.columns:
            X["target_price_range_7d"] = X["target_price_d7_d1_max"] - X["target_price_d7_d1_min"]
        if "target_price_d30_d1_max" in X.columns and "target_price_d30_d1_min" in X.columns:
            X["target_price_range_30d"] = (
                X["target_price_d30_d1_max"] - X["target_price_d30_d1_min"]
            )
        return X

    def rename_target(X):
        X = X.copy()
        if "target_price" in X.columns:
            X = X.rename(columns={"target_price": "y"})
        return X

    def add_total_flows(X):
        X = X.copy()
        export_cols = [c for c in EXPORT_COLUMNS if c in X.columns]
        import_cols = [c for c in IMPORT_COLUMNS if c in X.columns]
        X["total_exports"] = X[export_cols].sum(axis=1, min_count=1)
        X["total_imports"] = X[import_cols].sum(axis=1, min_count=1)
        return X

    def add_adjacent_hour_lags(X):
        """H-1/H-2 lags: 25/26 row shifts for D-1 prices; 1/2 for same-day forecasts."""
        X = X.copy()
        for col in ["target_price", "marktpreis_frankreich"]:
            if col in X.columns:
                X[f"{col}_lag_25h"] = X[col].shift(25)
                X[f"{col}_lag_26h"] = X[col].shift(26)
        for col in [
            "prognostizierter_verbrauch_residuallast",
            "prognostizierte_erzeugung_wind_und_photovoltaik",
        ]:
            if col in X.columns:
                X[f"{col}_lag_1h"] = X[col].shift(1)
                X[f"{col}_lag_2h"] = X[col].shift(2)
        return X

    def add_day_index_and_interactions(X):
        X = X.copy()
        # day_index: absolute days since 2015-01-05 00:00 CET (epoch)
        normalized = X.index.normalize()
        X["day_index"] = ((normalized - _DAY_INDEX_EPOCH).total_seconds() // 86400).astype("int64")
        # year_index: 0 = 2015
        X["year_index"] = (X.index.year - _YEAR_INDEX_BASE).astype("int64")
        pairs = [
            ("prognostizierter_verbrauch_residuallast", "interact_residuallast_forecast"),
            ("target_price_ewma_6h", "interact_target_ewma_6h"),
            ("ttf_eur_per_mwh_ewma_720h", "interact_ttf_ewma_720h"),
            ("ttf_eur_per_mwh_ewma_24h", "interact_ttf_ewma_24h"),
            ("prognostizierte_erzeugung_sonstige", "interact_sonstige_forecast"),
            # Neighbour prices x day_index
            ("marktpreis_frankreich", "interact_marktpreis_frankreich"),
            ("marktpreis_schweiz", "interact_marktpreis_schweiz"),
            # More forecast columns x day_index
            (
                "prognostizierte_erzeugung_wind_und_photovoltaik",
                "interact_wind_pv_forecast",
            ),
            (
                "prognostizierte_erzeugung_wind_und_photovoltaik_daily_max",
                "interact_wind_pv_daily_max",
            ),
            # Commodity prices x day_index (raw D-2 daily means after Phase 6)
            ("ttf_eur_per_mwh", "interact_ttf_raw"),
            ("carbon_eur_per_ton", "interact_carbon_raw"),
            # Generation actuals x day_index (same-hour D-2 lags)
            ("stromerzeugung_wind_onshore_lag_d2", "interact_wind_onshore_lag"),
            ("stromerzeugung_photovoltaik_lag_d2", "interact_photovoltaik_lag"),
        ]
        for src, dst in pairs:
            if src in X.columns:
                X[dst] = X[src] * X["day_index"]
        return X

    rolling_steps = []
    for name, spec in ROLLING_FEATURE_SPECS_v5.items():
        rolling_steps.append(
            (
                f"rolling_{name}",
                RollingStatsTransformer(
                    columns=spec["columns"],
                    windows=spec["windows"],
                    overwrite=spec.get("overwrite", False),
                ),
            )
        )

    pipe = Pipeline(
        [
            (
                "rolling_target",
                RollingStatsTransformer(
                    columns=["target_price"],
                    windows=[
                        WindowSpec(start_day=-7, end_day=-7, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-7, end_day=-1, agg="std"),
                        WindowSpec(start_day=-7, end_day=-1, agg="max"),
                        WindowSpec(start_day=-7, end_day=-1, agg="min"),
                        WindowSpec(start_day=-7, end_day=-1, hours=list(range(8, 20)), agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="mean"),
                        WindowSpec(start_day=-30, end_day=-1, agg="std"),
                        WindowSpec(start_day=-30, end_day=-1, agg="min"),
                        WindowSpec(start_day=-30, end_day=-1, agg="max"),
                        WindowSpec(start_day=-2, end_day=-1, agg="std"),
                        WindowSpec(start_day=-3, end_day=-1, agg="std"),
                        # D-1 intra-day stats (single previous day)
                        WindowSpec(start_day=-1, end_day=-1, agg="max"),
                        WindowSpec(start_day=-1, end_day=-1, agg="min"),
                        WindowSpec(start_day=-1, end_day=-1, agg="std"),
                    ],
                ),
            ),
            (
                "ewma_prices",
                EWMATransformer(
                    columns=["target_price", "marktpreis_frankreich"],
                    spans=[6, 24, 2160],
                    cutoff_day=-1,
                ),
            ),
            (
                "ewma_prices_h10",
                EWMATransformer(
                    columns=["target_price"],
                    spans=[6, 24, 2160],
                    cutoff_day=-1,
                    cutoff_hour=10,
                    col_suffix="_h10",
                ),
            ),
            (
                "ewma_actuals",
                EWMATransformer(
                    columns=[
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    spans=[24, 168, 2160],
                    cutoff_day=-1,
                    cutoff_hour=10,
                ),
            ),
            (
                "ewma_commodities",
                EWMATransformer(
                    columns=[
                        "carbon_eur_per_ton",
                        "ttf_eur_per_mwh",
                        "brent_usd_per_barrel",
                    ],
                    spans=[24, 720, 2160],
                    cutoff_day=-2,
                ),
            ),
            # Full pipeline keeps all 5 morning actual columns
            (
                "morning_actuals",
                RollingStatsTransformer(
                    columns=[
                        "stromverbrauch_gesamt_(netzlast)",
                        "stromverbrauch_residuallast",
                        "stromerzeugung_wind_onshore",
                        "stromerzeugung_wind_offshore",
                        "stromerzeugung_photovoltaik",
                    ],
                    windows=[
                        WindowSpec(start_day=-1, end_day=-1, hours=list(range(0, 11)), agg="mean"),
                    ],
                    overwrite=False,
                ),
            ),
            (
                "total_flows",
                FunctionTransformer(add_total_flows, validate=False),
            ),
            # Phase 5b.5: D-1 daily max/min/std for neighbour prices
            (
                "neighbour_price_stats",
                RollingStatsTransformer(
                    columns=[
                        "marktpreis_frankreich",
                        "marktpreis_schweiz",
                        "marktpreis_niederlande",
                        "marktpreis_österreich",
                    ],
                    windows=[
                        WindowSpec(start_day=-1, end_day=-1, agg="max"),
                        WindowSpec(start_day=-1, end_day=-1, agg="min"),
                        WindowSpec(start_day=-1, end_day=-1, agg="std"),
                    ],
                    overwrite=False,
                ),
            ),
            # Phase 5c: Same-hour lags — extended to D-2 for prices
            (
                "same_hour_lag_frankreich",
                SameHourLagTransformer(column="marktpreis_frankreich", day_offsets=(-1, -2)),
            ),
            (
                "same_hour_lag_schweiz",
                SameHourLagTransformer(column="marktpreis_schweiz", day_offsets=(-1, -2)),
            ),
            (
                "same_hour_lag_niederlande",
                SameHourLagTransformer(column="marktpreis_niederlande", day_offsets=(-1,)),
            ),
            (
                "same_hour_lag_österreich",
                SameHourLagTransformer(column="marktpreis_österreich", day_offsets=(-1,)),
            ),
            (
                "same_hour_lag_wind_onshore",
                SameHourLagTransformer(column="stromerzeugung_wind_onshore", day_offsets=(-2,)),
            ),
            (
                "same_hour_lag_wind_offshore",
                SameHourLagTransformer(column="stromerzeugung_wind_offshore", day_offsets=(-2,)),
            ),
            (
                "same_hour_lag_photovoltaik",
                SameHourLagTransformer(column="stromerzeugung_photovoltaik", day_offsets=(-2,)),
            ),
            (
                "same_hour_lags_target",
                SameHourLagTransformer(column="target_price", day_offsets=(-1, -2, -7, -14)),
            ),
            # Phase 5f: Adjacent-hour lags
            (
                "hour_adjacent_lags",
                FunctionTransformer(add_adjacent_hour_lags, validate=False),
            ),
            *rolling_steps,
            (
                "daily_aggregates",
                HourlyDailyAggregateTransformer(
                    columns=[
                        "prognostizierte_erzeugung_wind_und_photovoltaik",
                        "prognostizierter_verbrauch_gesamt",
                        # Additional forecast columns (full pipeline only)
                        "prognostizierter_verbrauch_residuallast",
                        "prognostizierte_erzeugung_photovoltaik",
                        "prognostizierte_erzeugung_sonstige",
                    ],
                    aggs=("sum", "mean", "max", "min", "std"),
                ),
            ),
            # Full pipeline: add_renewable_pct=True
            (
                "daily_features",
                CreateCustomColumns(
                    [
                        GenerationPercentageTransformer(
                            add_renewable_pct=True, add_supply_demand_gap=True
                        ),
                        PrognosticatedPercentageTransformer(),
                        TemporalFeatureTransformer(),
                    ]
                ),
            ),
            ("price_ranges", FunctionTransformer(add_price_ranges, validate=False)),
            ("rename_target", FunctionTransformer(rename_target, validate=False)),
            # Minimal drop: leakage + structural only
            (
                "drop_cols",
                ColumnDropper(
                    exclude=[
                        "cross-border_*",
                        "net_export_*",
                        "month",
                        "month_sin",
                        "month_cos",
                        "regime_*",
                        "stromverbrauch_residuallast",
                        "stromverbrauch_pumpspeicher",
                    ],
                ),
            ),
            (
                "day_index_interactions",
                FunctionTransformer(add_day_index_and_interactions, validate=False),
            ),
        ]
    )

    validate_pipeline_leakage(pipe)
    return pipe
