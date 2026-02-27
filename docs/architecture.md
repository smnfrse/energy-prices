# Architecture Overview

## Module Structure

```
src/
├── config/             # Configuration layer (constants + pipeline builders)
├── data/               # Data acquisition and processing
├── features/           # Feature engineering (sklearn transformers)
├── modeling/           # Training, evaluation, baselines
└── cli.py              # Unified CLI entry point
```

## Data Flow

```
[1] Download  → DataSource classes (SmardSource, IcapSource, YahooSource, FredSource)
               Output: raw CSVs in data/raw/

[2] Combine   → combine_raw_data() groups CSVs → wide-format parquet
               Output: data/interim/combined_de_lu_hourly.parquet

[3] Merge     → merge_datasets() joins DE-LU + DE-AT-LU + regime indicators
               Output: data/processed/merged_dataset_hourly.parquet

[4] Clean     → handle_missing_values() applies rule-based imputation
               Rules defined in src/config/processing.py

[5] Engineer  → sklearn Pipeline (configurable, versioned v1–v5)
               Pipelines defined in src/config/pipelines.py

[6] Model     → MLflow-tracked training with cross-validation
               Training loop in src/modeling/training.py
```

## Configuration (`src/config/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Path constants (`DATA_DIR`, `MODELS_DIR`, etc.), MLflow URI |
| `smard.py` | SMARD API key→name mappings, cross-border flow codes |
| `commodities.py` | Yahoo/ICAP ticker mappings, data availability dates |
| `processing.py` | Column lists for missing value handling, regime dates |
| `temporal.py` | German state populations (holiday weighting), cyclical periods |
| `features.py` | Feature column lists, rolling specs, aggregation rules, availability rules |
| `modeling.py` | Experiment names, target columns, test split config |
| `pipelines.py` | Pipeline builder functions (reference v1, v2–v5 preprocessors) |

## Feature Engineering

All transformers inherit from `sklearn.base.BaseEstimator, TransformerMixin` and implement `fit(X, y=None)` / `transform(X)`.

**Key transformers** (`src/features/transforms.py`):
- `TemporalFeatureTransformer` — hour, day-of-week, month (cyclical encoding)
- `GermanHolidayTransformer` — population-weighted holiday indicator
- `PriceSpreadTransformer` — DE-LU vs neighbor price spreads
- `NetExportTransformer` — per-country net cross-border flows
- `GenerationPercentageTransformer` — generation mix as percentages
- `ColumnDropper` — wildcard-based column removal with include/exclude

**Time series transformers** (`src/features/ts_transforms.py`):
- `RollingStatsTransformer` — windowed aggregates with day-offset semantics
- `EWMATransformer` — exponentially weighted means with cutoff-hour support
- `SameHourLagTransformer` — same-hour values from D-N
- `DailyPivotTransformer` — hourly→daily with 24-element target arrays
- `HourlyDailyAggregateTransformer` — daily stats broadcast to hourly rows

**Leakage validation** (`src/features/validation.py`):
- `validate_pipeline_leakage()` inspects pipeline steps against availability rules
- Rules defined in `src/config/features.py` specify max day-offset per column pattern

## Pipeline Versions

| Version | Architecture | Features | Use Case |
|---------|-------------|----------|----------|
| v1 (reference) | Daily (24-target) | Full feature set | Legacy baseline |
| v2 | Daily (24-target) | Pruned low-value features | Improved daily |
| v3 | Daily (24-target) | + EWMA, morning actuals, price ranges | Feature refinement |
| v4 | Hourly (scalar target) | v3 + same-hour lags, daily aggregates | First hourly model |
| v5 slim | Hourly (scalar target) | 85 curated features, leakage fixes | Production candidate |
| v5 full | Hourly (scalar target) | ~138 features | Feature selection experiments |

## MLflow Integration

- Tracking URI: `sqlite:///models/mlflow.db`
- Artifacts: `models/mlruns/`
- Experiments: `0-baselines`, `1-linear`, etc.
- Training function `train_and_log()` handles logging, cross-validation, and metric computation

## Deployment / Inference

The daily forecast pipeline lives in `src/deploy/inference.py`. Key design decisions:

**Single dataset** — Inference reads `data/processed/merged_dataset_hourly.parquet` directly (the same file used for training). There is no separate `inference_dataset`. This ensures EWMA and rolling features (e.g. EWMA-2160h, 30-day rolling windows) always have sufficient history to converge correctly.

**Absolute `day_index`** — `preprocessor_v5_slim_hourly` and `preprocessor_v5_full_hourly` compute `day_index` as days elapsed since a fixed epoch (`2015-01-05 00:00 CET`) and `year_index` as `year - 2015`. This matches the values seen during training regardless of the size of the inference window, keeping interaction terms (feature × day_index) on the correct scale.

**Data update** — `make update-data` is the single command for incremental data refresh. It updates raw SMARD data, recombines, updates commodities, and re-runs the merge pipeline — all writing into `merged_dataset_hourly.parquet`. There is no separate `add-data` target or staging parquet.

**Output** — Forecast JSON files are written to `deploy/data/` and consumed by the dashboard:
- `forecast_latest.json` — 24-hour hourly price forecast for the next day
- `actuals_latest.json` — Last 7 days of actual prices
- `forecast_history.json` — Rolling 30-day forecast archive
- `metadata.json` — Blend ensemble metadata (MAE, RMSE, model count)
