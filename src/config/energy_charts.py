"""Energy Charts API configuration."""

ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info"

SERIES = {
    "da_price_de_lu": {
        "endpoint": "/price",
        "params": {"bzn": "DE-LU"},
        "filename": "da_price_de_lu.csv",
        "description": "Day-ahead auction price (DE-LU bidding zone)",
    }
}
