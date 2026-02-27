"""Constants for temporal feature engineering.

German state populations (2023 estimates) for population-weighted holiday features,
and cyclical encoding periods for calendar features.
"""

# =============================================================================
# German State Populations (2023 estimates, millions)
# =============================================================================
# Used to weight holiday indicators: national holidays → 1.0,
# state-specific holidays → fraction of population observing.

GERMAN_STATE_POPULATIONS = {
    "NW": 17.9,  # Nordrhein-Westfalen
    "BY": 13.2,  # Bayern
    "BW": 11.1,  # Baden-Württemberg
    "NI": 8.0,  # Niedersachsen
    "HE": 6.3,  # Hessen
    "SN": 4.1,  # Sachsen
    "RP": 4.1,  # Rheinland-Pfalz
    "BE": 3.7,  # Berlin
    "SH": 2.9,  # Schleswig-Holstein
    "BB": 2.5,  # Brandenburg
    "ST": 2.2,  # Sachsen-Anhalt
    "TH": 2.1,  # Thüringen
    "HH": 1.9,  # Hamburg
    "MV": 1.6,  # Mecklenburg-Vorpommern
    "SL": 1.0,  # Saarland
    "HB": 0.7,  # Bremen
}

TOTAL_POPULATION = sum(GERMAN_STATE_POPULATIONS.values())

STATE_CODES = list(GERMAN_STATE_POPULATIONS.keys())

# =============================================================================
# Cyclical Encoding Periods
# =============================================================================
# Used for sin/cos encoding of calendar features.

CYCLICAL_PERIODS = {
    "hour": 24,
    "day_of_week": 7,
    "month": 12,
}
