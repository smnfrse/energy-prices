"""Energy Charts API download and update functions.

Fetches day-ahead price data from energy-charts.info as a fallback
when SMARD API data is unavailable or delayed.

CLI commands:
- update: Incrementally update the local energy charts CSV
"""

from loguru import logger
import pandas as pd
import requests
import typer

from src.config import ENERGY_CHARTS_DIR
from src.config.energy_charts import ENERGY_CHARTS_BASE_URL, SERIES

app = typer.Typer()


def fetch_price(bzn: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch day-ahead price from energy-charts.info.

    Args:
        bzn: Bidding zone, e.g. "DE-LU".
        start: Start timestamp (inclusive).
        end: End timestamp (inclusive).

    Returns:
        DataFrame with UTC DatetimeIndex named "time" and a "value" column (EUR/MWh).
        Rows with null prices are dropped.
    """
    url = ENERGY_CHARTS_BASE_URL + "/price"
    params = {
        "bzn": bzn,
        "start": start.strftime("%Y-%m-%dT%H:%M+00:00"),
        "end": end.strftime("%Y-%m-%dT%H:%M+00:00"),
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    unix_seconds = data.get("unix_seconds", [])
    prices = data.get("price", [])

    df = pd.DataFrame({"time": unix_seconds, "value": prices})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.dropna(subset=["value"])

    return df


@app.command()
def update(
    series: str = typer.Option("da_price_de_lu", help="Series name from config."),
    redundancy_days: int = typer.Option(14, help="Days to re-fetch for corrections."),
):
    """Incrementally update Energy Charts data."""
    from src.data.sources import EnergyChartsSource

    if series not in SERIES:
        raise typer.BadParameter(f"Unknown series '{series}'. Valid: {list(SERIES.keys())}")

    output_path = ENERGY_CHARTS_DIR / SERIES[series]["filename"]
    ENERGY_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    source = EnergyChartsSource(series)

    if not output_path.exists():
        logger.info(f"No existing file found, downloading full history for {series}")
        source.download(output_path)
    else:
        source.update(output_path, redundancy_days=redundancy_days)

    logger.success(f"Energy Charts update complete: {output_path}")


if __name__ == "__main__":
    app()
