# Day-Ahead Energy Price Forecasting (DE-LU)

**[Live Dashboard](https://smnfrse.github.io/energy-prices/)** — daily 24-hour price forecasts

<img width="922" height="829" alt="grafik" src="https://github.com/user-attachments/assets/7cebb60d-4fca-48f3-a0dc-b0165c09eea3" />

Predicting hourly day-ahead electricity prices in the Germany/Luxembourg (DE-LU) bidding area using gradient-boosted tree models with engineered features from energy market fundamentals, cross-border flows, and commodity prices.

## Motivation

Day-ahead electricity prices are determined by auction at noon for all 24 hours of the following day. Accurate forecasts enable better trading decisions, grid management, and renewable integration planning. This project builds a complete ML pipeline from raw market data to production-ready predictions.

## Acknowledgements

I have drawn significantly on two repos to produce this project and want to give my thanks to their authors: energy-market-analysis https://github.com/vsevolodnedora/energy_market_analysis from Vsevolod Nedora and Modern Times Series Forecasting from Manu Joseph https://github.com/PacktPublishing/Modern-Time-Series-Forecasting-with-Python. I also want to thank all the instructors at the WBS coding school.


## Limitations

**Forecast timing**: The current pipeline uses SMARD generation and load forecasts as key features. These are published at ~18:00 CET — six hours after the 12:00 CET day-ahead auction that sets electricity prices. This means the live dashboard currently operates as a next-day backtesting system using oracle forecasts rather than a true pre-auction forecasting tool. A companion project ([energy_market_analysis](https://github.com/smnfrse/energy_market_analysis)) produces weather-based generation and load forecasts that are available before the auction, and work to integrate these as a replacement forecast source is underway.

**Data sources**: This repo uses exclusively free data sources. This means that it is missing full data on commodity and electricity markets — only an imperfect reconstruction of daily closing commodity prices rather than any information about trading volumes, intraday prices, or futures. The forecasts would likely improve with access to this information.

## Methodology

**Data Sources:**
- **SMARD API** (Bundesnetzagentur): Power generation by source, consumption, cross-border physical flows, market prices, and TSO forecasts. Hourly resolution, Dec 2014 – present.
- **Commodity prices**: EU carbon allowances (ICAP), TTF natural gas futures (Yahoo Finance & FRED), Brent crude oil futures (Yahoo Finance).

**Feature Engineering:**
- Temporal features (cyclical hour/day encoding, holidays)
- Lagged price statistics (rolling mean/std/min/max over 7d and 30d windows)
- Exponentially weighted averages with different spans
- Same-hour lags (D-1, D-2, D-7, D-14)
- Generation mix percentages, supply-demand gap
- Cross-border flow aggregates (total imports/exports)


All transformations are implemented as sklearn-compatible transformers with built-in leakage validation.

**Models:**
- Baselines: naive persistence, ARIMA, ETS, Prophet
- Ridge, Lasso, Elasticnet, LightGBM, XGboost, CatBoost with hourly global model architecture
- Versioned preprocessing pipelines (v2–v5) for systematic ablation

## Production Deployment

The production model is a blended ensemble of 8 individual models (2 per category: linear, LightGBM, XGBoost, CatBoost). Blend weights are computed via inverse-MAE on a 90-day holdout:

```
weight_i = (1 / MAE_i) / sum(1 / MAE_j for all j)
```

**GitHub Actions automation:**
- `daily_forecast.yml` — runs at 19:00 UTC daily: `make add-data` → inference → updates `deploy/data/`
- `retrain.yml` — runs at 06:00 UTC on the 1st and 15th of each month: full model retrain + blend weight refresh

**Dashboard**: Available at GitHub Pages — 24-hour day-ahead forecast with 7-day actuals overlay, EN/DE toggle.

## Project Structure

```
├── data/                   # Data directory (gitignored)
│   ├── raw/                # Raw CSV downloads from SMARD API
│   ├── interim/            # Combined parquet files by region
│   ├── processed/          # Merged and feature-engineered datasets
│   └── external/           # Third-party data
│
├── models/                 # MLflow database, artifacts, production models
│
├── src/                    # Source code
│   ├── config/             # Configuration and constants
│   │   ├── __init__.py     # Path constants, MLflow config
│   │   ├── smard.py        # SMARD API key mappings and filter dicts
│   │   ├── commodities.py  # Commodity ticker mappings and constants
│   │   ├── energy_charts.py# Energy-Charts API config
│   │   ├── processing.py   # Column lists for missing value handling
│   │   ├── temporal.py     # Holiday/calendar encoding constants
│   │   ├── features.py     # Feature engineering column definitions
│   │   ├── modeling.py     # Experiment names, test config
│   │   └── pipelines.py    # Pipeline builder functions (v1–v5)
│   ├── data/               # Data acquisition and processing
│   │   ├── smard.py        # SMARD API client
│   │   ├── energy_charts.py# Energy-Charts API client
│   │   ├── sources.py      # DataSource classes (Smard, ICAP, Yahoo, FRED)
│   │   ├── processing.py   # Combine, merge, clean pipeline
│   │   └── commodities.py  # Commodity price pipeline
│   ├── features/           # Feature engineering
│   │   ├── transforms.py   # sklearn transformers (spreads, ratios, holidays)
│   │   ├── ts_transforms.py# Time series transformers (rolling, EWMA, pivot)
│   │   ├── validation.py   # Leakage validation framework
│   │   └── preprocessors.py# Additional preprocessing utilities
│   ├── modeling/           # Model training and evaluation
│   │   ├── baselines.py    # Baseline models and prediction utilities
│   │   ├── training.py     # MLflow-integrated training loop
│   │   ├── train_final.py  # Reproducible blend retrain from committed hyperparams
│   │   ├── blend.py        # Blend ensemble weight computation
│   │   └── metrics.py      # Metric calculations (RMSE, MAE, skill scores)
│   ├── deploy/             # Deployment pipeline
│   │   ├── inference.py    # Daily forecast generation
│   │   └── retrain.py      # Biweekly retrain wrapper
│   └── cli.py              # Unified CLI entry point
│
├── docs/                   # Documentation
├── .github/workflows/      # GitHub Actions CI/CD
├── Makefile                # Development commands
├── pyproject.toml          # Package metadata and tool config
└── DATA.md                 # Comprehensive data documentation
```

## Setup

### Prerequisites
- Python 3.13.5
- Conda (for environment management)

### Installation

```bash
# Create and activate environment
make create_environment
conda activate energy_prices

# Install dependencies
make requirements
```

### Data Pipeline

```bash
# Download and process all data (SMARD + commodities)
make data

# Incremental update (only download new data)
make update-data
```

### Training

```bash
# Run baseline models
make train-baselines

# Retrain blend ensemble from committed hyperparameters
make train-final

# View results in MLflow UI
make mlflow
```

### Full pipeline from scratch

```bash
make full-project
```

### Development

```bash
make lint     # Run ruff linter
make format   # Auto-format code
```

## Data Sources

### SMARD API (Primary)
Official German energy market data from Bundesnetzagentur. Covers generation by source, consumption, cross-border physical flows, and day-ahead market prices. Hourly resolution from December 2014 to present.

### Commodity Prices
- **EU Carbon Allowances**: ICAP Carbon Action (Nov 2014 – present, ~2 month publication lag filled with CO2.L equity proxy, bias +1.08 EUR/ton)
- **TTF Natural Gas Futures**: Yahoo Finance (Oct 2017 – present), gap-filled Dec 2014 – Oct 2017 via FRED EU monthly + US Henry Hub daily reconstruction
- **Brent Crude Oil**: Yahoo Finance (Jan 2021 – present)

See [DATA.md](DATA.md) for detailed documentation on data sources, structural breaks, missing value handling, and commodity price reconstruction methodology.

## Future Plans

- Longer forecast horizon
- Deep learning models (LSTM, Temporal Fusion Transformer)
- Probabilistic forecasts (prediction intervals)
- Multi-region models (per-neighbour bidding area)
- Intra-day price forecasting

## License

MIT
