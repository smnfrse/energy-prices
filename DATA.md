# Data Documentation

Documentation of data sources, assumptions, structural breaks, and missing value handling decisions.

## Data Sources and Collection

### Primary Data Source: SMARD API

The **SMARD API** (Strommarktdaten) is the official German energy market data platform operated by the Bundesnetzagentur (Federal Network Agency), accessed via `smard.api.proxy.bund.dev`.

**Data categories**: Market prices (DE-LU and neighbors), power generation (by source), consumption, cross-border flows, forecasts, installed capacity

**Resolution**: Quarter-hourly (96/day, default) or hourly (24/day). As of October 2025, day-ahead prices transitioned from hourly to quarter-hourly resolution.

**Timestamp convention**: All timestamps stored in UTC (`datetime64[ns, UTC]`). Convert to German local time: `df.index.tz_convert('Europe/Berlin')`

### External Data Sources: Commodity Prices

To enhance predictive power, the project integrates external commodity price data that influences energy markets.

#### 1. EU Carbon Allowances (Dual-Source Strategy)

**Dual-source approach** provides both historical accuracy and real-time prices:

| Source | Coverage | Frequency | Purpose |
|--------|----------|-----------|---------|
| **ICAP Carbon Action** | Nov 2014 - Dec 2025 | Business days | Historical (official source, ~2 month lag) |
| **CO2.L (Yahoo Finance)** | Oct 2021 - Present | Business days | Real-time (fills ICAP publication lag) |

**Unified carbon column methodology**:
1. Calculate bias correction from overlap period (Oct 2021 - Dec 2025): ICAP - CO2.L = +1.08 EUR/ton
2. Merge series: Use ICAP where available, fill gaps with bias-corrected CO2.L
3. Forward-fill weekends/holidays within valid data range

**Result**: Complete carbon price series (`carbon_eur_per_ton`) with 0% missing values.

**Validation** (914 days overlap): Correlation 0.9857, mean difference 1.08 EUR/ton, 99% within ±5 EUR tracking accuracy. Excellent proxy quality across all price regimes and volatility periods.

#### 2. TTF Gas Futures (Dual-Source Strategy)

| Source | Coverage | Purpose |
|--------|----------|---------|
| **Yahoo Finance (TTF=F)** | Oct 2017 - Present | Direct TTF futures prices in EUR/MWh |
| **FRED reconstruction** | Dec 2014 - Oct 2017 | Gap filling via EU monthly + US Henry Hub daily |

**Gap reconstruction** uses FRED PNGASEUUSDM (EU gas monthly) + DHHNGSP (US Henry Hub daily) with bias correction calculated from overlap period (Oct 2017 - Dec 2025). Validation: correlation 0.9656, fills 1,391 days, ~0% missing after reconstruction.

**Note**: Reconstructed values are estimates for broad trends, not precise market prices. Highly volatile during 2021-2022 energy crisis (peak: €339/MWh Aug 2022).

#### 3. Brent Crude Oil Futures

**Source**: Yahoo Finance (`BZ=F`)
- **Coverage**: Jan 2021 - Present
- **Quality**: Standard commodity futures (range: $51-$128)
- **Missing**: ~54% (structural gap before 2021)

#### Commodity Missing Value Strategy

All commodity data undergoes **forward-fill** for weekend/holiday gaps (markets closed):
- Forward-fill applied **only** within valid data range (first to last observation)
- Structural gaps preserved as NaN
- **Final missingness**: Carbon 0%, TTF ~0% (with reconstruction), Brent ~54%

## Data Processing Flow

The data undergoes four transformation stages:

### Stage 1: Raw Data Collection
Raw SMARD data fetched from API and saved to CSV files. Two separate directories maintain pre-split (DE-AT-LU) and post-split (DE-LU) data due to the 2018 bidding area split.

### Stage 2: Data Combination
Individual measure CSVs combined into wide-format Parquet files:
- Each time series becomes a column
- Timestamps form the index
- Produces separate files for DE-AT-LU historical and DE-LU current periods

**Outputs**: `combined_de_lu.parquet` (420,864 rows × 52 cols), `combined_de_at_lu.parquet` (389,088 rows × 50 cols)

### Stage 3: Dataset Merging
Merges historical and current datasets with structural break handling:
- Concatenates pre-split and post-split data at cutoff (2018-09-30 22:00 UTC)
- Creates unified `target_price` from both price series
- Adds regime indicators: `regime_de_at_lu`, `regime_quarter_hourly`

**Outputs**: `merged_dataset.parquet` (420,864 rows × 54 cols quarter-hourly), `merged_dataset_hourly.parquet` (97,344 rows × 69 cols)

**Note**: Missing value handling is no longer part of the merge pipeline. It is applied separately during feature engineering via `handle_missing_values()` in `src/features/transforms.py`.

### Stage 4: Commodity Data Processing
Processes external commodity prices:
- Downloads carbon (ICAP systemIds 33 + 35), CO2.L, TTF, Brent, FRED gas data
- Reconstructs TTF gap (Dec 2014 - Oct 2017) using FRED EU monthly + US Henry Hub daily
- Calculates bias correction for CO2.L from overlap period (+1.08 EUR/ton)
- Creates unified carbon column: ICAP where available, bias-corrected CO2.L otherwise
- Forward-fills weekends/holidays within valid data ranges
- Replicates daily prices to hourly frequency (each day's price applies to all 24 hours)

**Outputs**: `commodity_prices_daily.parquet` (3 cols), `commodity_prices_hourly.parquet` (97,344 rows × 3 cols)

**Columns**: `carbon_eur_per_ton` (0% missing), `ttf_eur_per_mwh` (~0% missing), `brent_usd_per_barrel` (~54% missing)

### Stage 5: Feature Engineering

Two pipeline variants depending on forecasting horizon:

#### 5a. Hourly Pipeline (for hourly forecasting)
Applies missing value handling and sklearn-compatible feature transformers:
1. **Missing value handling** (`handle_missing_values()` in `src/features/transforms.py`): Applies documented rules (drop redundant, structural fills, calculations, cubic spline interpolation for small gaps)
2. **Commodity merge** (`CommodityMerger`): Left-joins commodity prices onto main dataset
3. **Column dropping** (`ColumnDropper`): Removes columns not useful for modeling
4. **Price spreads** (`PriceSpreadTransformer`): Calculates `target_price - neighbor_price` per country
5. **Net exports** (`NetExportTransformer`): Calculates per-country `export - import`
6. **Generation percentages** (`GenerationPercentageTransformer`): Converts generation to % of total
7. **Prognosticated percentages** (`PrognosticatedPercentageTransformer`): Converts forecast generation to % of total forecast

Steps 2-7 are sklearn `BaseEstimator`/`TransformerMixin` classes composable via `sklearn.pipeline.Pipeline` for experimentation with `GridSearchCV`.

**Output**: `data/processed/features_hourly.parquet`

#### 5b. Daily Pipeline (for day-ahead 24-hour forecasting)
End-to-end transformation from hourly to daily features with proper lag structure:

**Phase 1 - Hourly feature engineering**:
1. Commodity merge, temporal features, holidays
2. Column dropping, price spreads, net exports
3. Rolling window statistics (24h, 168h, 720h windows with mean/std/min/max)
4. EWMA features (6h, 24h, 168h, 720h, 2160h spans)

**Phase 2 - Daily aggregation** (`DailyAggregator`):
1. Convert UTC → CET/CEST timezone
2. Handle DST transitions (interpolate missing hours, average duplicate hours)
3. Group by date, create 24-element target arrays
4. Expand forecast columns into 24 separate hourly columns (e.g., `prognostizierte_erzeugung_gesamt_hour_0` through `_hour_23`)
5. Aggregate other features using type-appropriate functions (sum for generation/flows, mean for prices, last for EWMA)
6. Recalculate generation percentages from daily sums (not weighted averages of hourly percentages)

**Phase 3 - Lag features** (`LaggedPriceTransformer`):
1. Extract D-1 individual hourly prices (24 columns: `price_lag_1` through `price_lag_24`)
2. Calculate D-2 summary statistics (`price_d2_mean`, `price_d2_std`)
3. Calculate D-7 summary statistics (`price_d7_mean`, `price_d7_std`)

**Phase 4 - Leakage prevention** (`LeakagePreventionTransformer`):
Apply time shifts based on real-world data availability:
- **No shift**: Forecast columns (available 11am D-1 for day D), temporal features, regime indicators
- **D-1 shift**: Price spreads, rolling stats, EWMA (based on day-ahead auction prices)
- **D-2 shift**: Actuals (generation, consumption, net exports), commodities (reporting delays)

**Phase 5 - Log transformation** (`PriceLogTransformer`):
1. Calculate auto-shift for log transform (abs(min) + 1) from target columns
2. Apply log(x + shift) to all price-related columns:
   - Target columns (y_0 to y_23) → y_0_log to y_23_log
   - Lagged prices (price_lag_0 to price_lag_23) → price_lag_0_log to price_lag_23_log
   - Summary lags (price_d2_mean, price_d7_mean, etc.) → *_log versions
   - Neighbor prices (marktpreis_*) → marktpreis_*_log
   - Spreads (spread_*) → spread_*_log
   - Rolling stats (*_rolling_*) → *_rolling_*_log
   - EWMA features (*_ewma_*) → *_ewma_*_log
3. All price columns use same shift for consistency and comparability

**Phase 6 - Outlier detection** (`OutlierHandler`):
1. Flag outliers in log-transformed targets using specified method (std/iqr) and threshold
2. Creates binary `is_outlier` flag for days with any hour outside bounds

**Phase 7 - Cleanup** (`ColumnDropper`):
1. Drop all non-logged price columns (y_0 to y_23, price_lag_*, marktpreis_*, spread_*, rolling, EWMA)
2. Keep only log-transformed versions for modeling

**Output**: `data/processed/daily_forecast_features_hourly.parquet`

**Key features**:
- Target: 24 separate log-price columns (y_0_log to y_23_log, one per hour of day D)
- Lagged prices: Individual D-1 hours + D-2/D-7 summaries (all logged)
- Forecasts: 24 hourly columns per forecast type (consumption, total generation, wind+PV)
- All prices in log space: Consistent transformation across all price features
- No data leakage: All features properly shifted to match real-world availability

## Structural Breaks

### 1. Bidding Area Split (2018-09-30 22:00 UTC)

**Background**: DE-AT-LU split into separate DE-LU and AT bidding areas due to border congestion.

**Data handling**:
- `target_price`: Uses DE-AT-LU price pre-split, DE-LU price post-split
- `regime_de_at_lu`: 1 before split, 0 after
- Austria flows/price: Filled with 0 before split (didn't exist as separate entities)
- Austria neighbor flows (Hungary, Slovenia, Italy): Filled with 0 after split (no longer relevant)

**Assumption**: Pre-split DE-AT-LU price equivalent to post-split DE-LU (Germany dominated combined area).

### 2. Price Resolution Change (2025-10-01)

**Background**: Day-ahead prices transitioned from hourly (24/day) to quarter-hourly (96/day).

**Data handling**:
- `regime_quarter_hourly`: 1 from 2025-10-01 onwards, 0 before
- Quarter-hourly data preserved by default; can be aggregated to hourly if needed

## Missing Value Handling

Handled systematically by `handle_missing_values()` in `src/features/transforms.py`. All rules configured in `src/config/processing.py`.

### Summary of Rules

| Column(s) | Rule | Rationale |
|-----------|------|-----------|
| `marktpreis_deutschland_luxemburg`, `marktpres_deutschland_austria_luxembourg`, `marktpreis_anrainer_de_lu` | DROP | Redundant (merged into `target_price`) or not needed |
| `stromerzeugung_kernenergie` | Fill 0 after last obs | Nuclear decommissioned April 2023 |
| Austria neighbor flows (HU, SI, IT) | Fill 0 after 2018-09-30 22:00 | No longer relevant for DE-LU after split |
| Belgium flows | Fill 0 before 2017-10-10 22:00 | Data reporting started Oct 2017 |
| Austria direct flows and price | Fill 0 before 2018-09-30 22:00 | Didn't exist as separate entities before split |
| Norway flows | Fill 0 before first valid obs | Data reporting started at specific point |
| `cross-border_flows_net_export` | Calculate from individual flows | Sum(exports) - Sum(imports) |
| `prognostizierte_erzeugung_sonstige` | Calculate from difference | gesamt - wind/PV |
| `prognostizierter_verbrauch_gesamt` | Fill with actual consumption | Correlation 0.9744 |
| `prognostizierte_erzeugung_gesamt` | Fill with sum of actual generation | Verified measurements |
| `marktpreis_polen` | Fill with `target_price` | Zero spread assumption (fragmented pre-2020) |
| Small gaps (≤2 consecutive NaNs) | Interpolate with cubic spline | Final cleanup |

**Processing order**: Drop → structural fills → calculations → interpolation

## Final Dataset Schema

### SMARD Energy Data

**File**: `data/interim/merged_dataset_hourly.parquet`

| Column Type | Count | Examples |
|-------------|-------|----------|
| Target | 1 | `target_price` |
| Regime indicators | 2 | `regime_de_at_lu`, `regime_quarter_hourly` |
| Market prices | ~13 | `marktpreis_frankreich`, `marktpreis_nederland`, `marktpreis_polen` |
| Generation | ~12 | `stromerzeugung_photovoltaik`, `stromerzeugung_wind_onshore`, `stromerzeugung_erdgas` |
| Consumption | 3 | `stromverbrauch_gesamt_(netzlast)`, `stromverbrauch_residuallast` |
| Cross-border flows | ~30 | `cross-border_flows_france_exports`, `cross-border_flows_net_export` |
| Forecasts | ~7 | `prognostizierte_erzeugung_photovoltaik`, `prognostizierte_erzeugung_wind_offshore` |
| Installed capacity | ~12 | `installierte_erzeugungsleistung_photovoltaik` |

**Statistics**: 97,344 rows × 69 columns, hourly resolution, 2014-12-30 23:00 to 2026-02-06 22:00 (UTC)

### Commodity Price Data

**File**: `data/interim/commodity_prices_hourly.parquet`

| Column | Source | Coverage | Missing |
|--------|--------|----------|---------|
| `carbon_eur_per_ton` | ICAP + CO2.L (unified) | Nov 2014 - Present | 0% |
| `ttf_eur_per_mwh` | Yahoo Finance + FRED reconstruction | Dec 2014 - Present | ~0% |
| `brent_usd_per_barrel` | Yahoo Finance | Jan 2021 - Present | ~54% |

**Statistics**: 97,344 rows × 3 columns, hourly (forward-filled from daily), perfectly aligned with SMARD timestamps

### Dataset Separation Rationale

Datasets are kept separate until feature engineering for flexibility:
- Update commodity data independently without reprocessing SMARD
- Experiment with different commodity feature engineering approaches
- Cleaner pipeline separation and easier debugging

Indexes are perfectly aligned (both 97,344 rows, identical timestamps) to enable simple joins.

## Data Quality Notes

### Known Issues

**SMARD**: Poland price fragmented pre-2020; resolution change Oct 2025 may require regime-specific modeling; historic Austria neighbor flows only relevant pre-split

**Commodities**: TTF structural gap Dec 2014 - Oct 2017 (resolved via FRED reconstruction); Brent structural gap before Jan 2021 (~54% missing); weekend/holiday gaps (resolved via forward-fill); ICAP ~2 month lag (resolved via CO2.L real-time proxy)

### Validation Checks

**SMARD**:
- Timestamps in UTC
- `target_price` complete (except known gaps)
- Regime indicators align with cutoff dates
- Net exports match calculated values

**Commodities**:
- Timestamps match SMARD: `commodities.index.equals(smard.index)`
- Row counts match: 97,344 (hourly)
- Missing values: Carbon 0%, TTF ~0%, Brent ~54%
- Daily prices replicated across 24 hours
- Bias correction applied to CO2.L
- Smooth ICAP-to-CO2.L transition (no jumps > 3 EUR)

## Further Reading

### SMARD Data
- API Documentation: https://www.smard.de/en/downloadcenter/download-market-data
- Bidding Area Split: https://www.bundesnetzagentur.de/EN/Areas/Energy/Companies/SecurityOfSupply/CrossBorderElectricityTrading/start.html
- Column mappings: `src/config/smard.py`
- Processing config: `src/config/processing.py`

### Commodity Data
- ICAP Carbon Action: https://icapcarbonaction.com/en/ets-prices
- CO2.L ETC: Yahoo Finance `CO2.L` (correlation 0.9857, bias +1.08 EUR/ton)
- EU ETS Background: https://climate.ec.europa.eu/eu-action/eu-emissions-trading-system-eu-ets_en
- TTF/Brent: Yahoo Finance `TTF=F`, `BZ=F`
- Configuration: `src/config/commodities.py`

### Deployment & CI/CD
- GitHub Actions workflows: `.github/workflows/`

## Deployment & Production

### Blend Ensemble

The production model is a blended ensemble of 8 individual models (2 per category: linear, LightGBM, XGBoost, CatBoost). Models are selected from MLflow experiments using a 90-day holdout and weighted by inverse-MAE:

```
weight_i = (1 / MAE_i) / sum(1 / MAE_j for all j)
```

Current blend (as of 2026-02-20): MAE 9.86 EUR/MWh, RMSE 18.52 EUR/MWh on 90-day holdout.

### Inference Outputs

All inference artifacts are written to `deploy/data/`:

| File | Description |
|------|-------------|
| `forecast_latest.json` | 24-hour day-ahead forecast for tomorrow |
| `actuals_latest.json` | Most recent 7 days of actual prices |
| `metadata.json` | Run timestamp, model version, data coverage |
| `forecast_history.json` | Rolling 30-day archive of past forecasts |

### GitHub Actions Automation

| Workflow | Schedule | Steps |
|----------|----------|-------|
| `daily_forecast.yml` | 08:00 UTC daily | `make add-data` → inference → update `deploy/data/` |
| `retrain.yml` | 06:00 UTC on 1st and 15th | Full model retrain → blend weight refresh |
