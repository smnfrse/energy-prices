import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
ENERGY_CHARTS_DIR = RAW_DATA_DIR / "energy_charts"

MODELS_DIR = PROJ_ROOT / "models"

# MLflow configuration
MLFLOW_DB_PATH = MODELS_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_ROOT = MODELS_DIR / "mlruns"

# Set default artifact location to ensure all experiments store artifacts in the same place
# This prevents MLflow from creating mlruns/ in the current working directory
os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = str(MLFLOW_ARTIFACT_ROOT.absolute())

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_path(dataset: str, resolution: str = "hour") -> Path:
    """Get the canonical path for a dataset file.

    Centralizes resolution-dependent path logic to eliminate scattered conditionals.

    Args:
        dataset: One of "combined_de_lu", "combined_de_at_lu", "merged",
                 "commodity_hourly", "commodity_daily".
        resolution: "hour" or "quarterhour".

    Returns:
        Path to the dataset file.
    """
    suffix = "_hourly" if resolution == "hour" else ""
    paths = {
        "combined_de_lu": INTERIM_DATA_DIR / f"combined_de_lu{suffix}.parquet",
        "combined_de_at_lu": INTERIM_DATA_DIR / f"combined_de_at_lu{suffix}.parquet",
        "merged": PROCESSED_DATA_DIR / f"merged_dataset{suffix}.parquet",
        "commodity_hourly": INTERIM_DATA_DIR / "commodity_prices_hourly.parquet",
        "commodity_daily": INTERIM_DATA_DIR / "commodity_prices_daily.parquet",
    }
    if dataset not in paths:
        raise KeyError(f"Unknown dataset '{dataset}'. Valid: {list(paths.keys())}")
    return paths[dataset]


RAW_DIRS = {
    ("DE-LU", "hour"): RAW_DATA_DIR / "smard_hourly",
    ("DE-LU", "quarterhour"): RAW_DATA_DIR / "smard_api",
    ("DE-AT-LU", "hour"): RAW_DATA_DIR / "smard_hourly_de_at_lu",
    ("DE-AT-LU", "quarterhour"): RAW_DATA_DIR / "DE_AT_LU",
}
