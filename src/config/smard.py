import re

# Map resolution names to periods per day
resolution_periods = {
    "quarterhour": 96,  # 24 hours * 4 quarters
    "hour": 24,  # 24 hours
    "day": 1,  # 1 day
    "week": 1 / 7,  # ~0.143 days per week
    "month": 1 / 30,  # ~0.033 days per month (approximation)
    "year": 1 / 365,  # ~0.0027 days per year
}

# Installed capacity keys - rarely useful for short-term price prediction
INSTALLED_CAPACITY_KEYS = [186, 188, 189, 194, 198, 207, 3792, 4072, 4073, 4074, 4075, 4076]
SCHEDULED_COMMERCIAL_KEYS = [
    "4546",
    "4404",
    "4548",
    "4406",
    "4712",
    "4998",
    "4553",
    "4412",
    "4552",
    "4410",
    "4550",
    "4408",
    "4724",
    "4722",
    "4545",
    "4403",
    "4551",
    "4409",
    "4547",
    "4405",
    "4549",
    "4407",
    "4629",
]
EXCLUDED_KEYS = INSTALLED_CAPACITY_KEYS + SCHEDULED_COMMERCIAL_KEYS

# Cross-border physical flow keys for DE-LU bidding area (post-2018-09-30)
CROSS_BORDER_FLOW_KEYS_DE_LU = [
    4963,  # net export
    # Imports
    4840,
    4841,
    4842,
    4843,
    4844,
    4845,
    4846,
    4847,
    4848,
    4978,
    4982,
    # Exports
    4821,
    4822,
    4823,
    4824,
    4825,
    4826,
    4827,
    4828,
    4829,
    4976,
    4980,
]

# Cross-border physical flow keys for DE-AT-LU bidding area (pre-2018-09-30)
CROSS_BORDER_FLOW_KEYS_DE_AT_LU = [
    4963,  # net export
    # Denmark 1 & 2
    4872,
    4726,
    4869,
    4727,
    # Netherlands
    4870,
    4730,
    # Northern Italy (Austria's neighbor)
    4873,
    4729,
    # Switzerland
    4732,
    4876,
    # Czech Republic
    4734,
    4878,
    # France
    4728,
    4871,
    # Sweden 4
    4857,
    4875,
    # Hungary (Austria's neighbor)
    4735,
    4879,
    # Slovenia (Austria's neighbor)
    4733,
    4877,
    # Poland
    4731,
    4874,
    # Belgium
    4984,
    4986,
]

filter_dict = {
    # Generation
    1223: "Stromerzeugung: Braunkohle",
    1224: "Stromerzeugung: Kernenergie",
    1225: "Stromerzeugung: Wind Offshore",
    1226: "Stromerzeugung: Wasserkraft",
    1227: "Stromerzeugung: Sonstige Konventionelle",
    1228: "Stromerzeugung: Sonstige Erneuerbare",
    4066: "Stromerzeugung: Biomasse",
    4067: "Stromerzeugung: Wind Onshore",
    4068: "Stromerzeugung: Photovoltaik",
    4069: "Stromerzeugung: Steinkohle",
    4070: "Stromerzeugung: Pumpspeicher",
    4071: "Stromerzeugung: Erdgas",
    # Forecasted generation
    3791: "Prognostizierte Erzeugung: Offshore",
    123: "Prognostizierte Erzeugung: Onshore",
    125: "Prognostizierte Erzeugung: Photovoltaik",
    715: "Prognostizierte Erzeugung: Sonstige",
    5097: "Prognostizierte Erzeugung: Wind und Photovoltaik",
    122: "Prognostizierte Erzeugung: Gesamt",
    # Demand
    410: "Stromverbrauch: Gesamt (Netzlast)",
    4359: "Stromverbrauch: Residuallast",
    4387: "Stromverbrauch: Pumpspeicher",
    # Forecasted demand
    411: "Prognostizierter Verbrauch: Gesamt",
    4362: "Prognostizierter Verbrauch: Residuallast",
    # Installed capacity
    186: "Installierte Erzeugungsleistung: Kernenergie",
    188: "Installierte Erzeugungsleistung: Braunkohle",
    189: "Installierte Erzeugungsleistung: Steinkohle",
    194: "Installierte Erzeugungsleistung: Erdgas",
    198: "Installierte Erzeugungsleistung: Pumpspeicher",
    207: "Installierte Erzeugungsleistung: Sonstige Konventionelle",
    3792: "Installierte Erzeugungsleistung: Wind Offshore",
    4072: "Installierte Erzeugungsleistung: Biomasse",
    4073: "Installierte Erzeugungsleistung: Wasserkraft",
    4074: "Installierte Erzeugungsleistung: Wind Onshore",
    4075: "Installierte Erzeugungsleistung: Photovoltaik",
    4076: "Installierte Erzeugungsleistung: Sonstige Erneuerbare",
    # Prices
    4169: "Marktpreis: Deutschland/Luxemburg",
    5078: "Marktpreis: Anrainer DE/LU",
    4996: "Marktpreis: Belgien",
    4997: "Marktpreis: Norwegen 2",
    4170: "Marktpreis: Österreich",
    251: "Marktpres: Deutschland/Austria/Luxembourg",
    252: "Marktpreis: Dänemark 1",
    253: "Marktpreis: Dänemark 2",
    254: "Marktpreis: Frankreich",
    255: "Marktpreis: Italien (Nord)",
    256: "Marktpreis: Niederlande",
    257: "Marktpreis: Polen",
    258: "Marktpreis: Schweden 4",
    259: "Marktpreis: Schweiz",
    260: "Marktpreis: Slowenien",
    261: "Marktpreis: Tschechien",
    262: "Marktpreis: Ungarn",
}


cross_border_de_lu_flows = {  # Cross-border physical flows (DE-LU bidding area)
    4963: "Cross-Border Flows: net export",
    # Imports
    4840: "Cross-Border Flows: Denmark 1 imports",
    4841: "Cross-Border Flows: Denmark 2 imports",
    4842: "Cross-Border Flows: France imports",
    4843: "Cross-Border Flows: Netherlands imports",
    4844: "Cross-Border Flows: Poland imports",
    4845: "Cross-Border Flows: Sweden 4 imports",
    4846: "Cross-Border Flows: Switzerland imports",
    4847: "Cross-Border Flows: Czech Republic imports",
    4848: "Cross-Border Flows: Austria imports",
    4978: "Cross-Border Flows: Norway 2 imports",
    4982: "Cross-Border Flows: Belgium imports",
    # Exports
    4821: "Cross-Border Flows: Denmark 1 exports",
    4822: "Cross-Border Flows: Denmark 2 exports",
    4823: "Cross-Border Flows: France exports",
    4824: "Cross-Border Flows: Netherlands exports",
    4825: "Cross-Border Flows: Poland exports",
    4826: "Cross-Border Flows: Sweden 4 exports",
    4827: "Cross-Border Flows: Switzerland exports",
    4828: "Cross-Border Flows: Czech Republic exports",
    4829: "Cross-Border Flows: Austria exports",
    4976: "Cross-Border Flows: Norway 2 exports",
    4980: "Cross-Border Flows: Belgium exports",
}
# Cross-border physical flow codes for DE-AT-LU bidding area (pre-2018-09-30)
cross_border_de_at_lu_flows = {
    4963: "Cross-Border Flows: net export",
    # Denmark 1
    4872: "Cross-Border Flows: Denmark 1 exports",
    4726: "Cross-Border Flows: Denmark 1 imports",
    # Denmark 2
    4869: "Cross-Border Flows: Denmark 2 exports",
    4727: "Cross-Border Flows: Denmark 2 imports",
    # Netherlands
    4870: "Cross-Border Flows: Netherlands exports",
    4730: "Cross-Border Flows: Netherlands imports",
    # Northern Italy (Austria's neighbor - only in DE-AT-LU)
    4873: "Cross-Border Flows: Northern Italy exports",
    4729: "Cross-Border Flows: Northern Italy imports",
    # Switzerland
    4732: "Cross-Border Flows: Switzerland exports",
    4876: "Cross-Border Flows: Switzerland imports",
    # Czech Republic
    4734: "Cross-Border Flows: Czech Republic exports",
    4878: "Cross-Border Flows: Czech Republic imports",
    # France
    4728: "Cross-Border Flows: France exports",
    4871: "Cross-Border Flows: France imports",
    # Sweden 4
    4857: "Cross-Border Flows: Sweden 4 exports",
    4875: "Cross-Border Flows: Sweden 4 imports",
    # Hungary (Austria's neighbor - only in DE-AT-LU)
    4735: "Cross-Border Flows: Hungary exports",
    4879: "Cross-Border Flows: Hungary imports",
    # Slovenia (Austria's neighbor - only in DE-AT-LU)
    4733: "Cross-Border Flows: Slovenia exports",
    4877: "Cross-Border Flows: Slovenia imports",
    # Poland
    4731: "Cross-Border Flows: Poland exports",
    4874: "Cross-Border Flows: Poland imports",
    # Belgium (no data before ~2020)
    4984: "Cross-Border Flows: Belgium exports",
    4986: "Cross-Border Flows: Belgium imports",
}


# Legacy cross-border flow codes for DE region (not DE-LU bidding area)
# Kept for reference - these were used before the bidding area correction
filter_dict_de_legacy = {
    4963: "Cross-Border Flows: net export",
    # Imports
    4880: "Cross-Border Flows: Denmark imports",
    4881: "Cross-Border Flows: France imports",
    4882: "Cross-Border Flows: Luxembourg imports",
    4883: "Cross-Border Flows: Netherlands imports",
    4884: "Cross-Border Flows: Austria imports",
    4885: "Cross-Border Flows: Poland imports",
    4886: "Cross-Border Flows: Sweden imports",
    4887: "Cross-Border Flows: Switzerland imports",
    4888: "Cross-Border Flows: czech_republic imports",
    4990: "Cross-Border Flows: Norway imports",
    4994: "Cross-Border Flows: Belgium imports",
    # Exports
    4736: "Cross-Border Flows: Denmark exports",
    4737: "Cross-Border Flows: France exports",
    4738: "Cross-Border Flows: Luxembourg exports",
    4739: "Cross-Border Flows: Netherlands exports",
    4740: "Cross-Border Flows: Austria exports",
    4741: "Cross-Border Flows: Poland exports",
    4742: "Cross-Border Flows: Sweden exports",
    4743: "Cross-Border Flows: Switzerland exports",
    4744: "Cross-Border Flows: czech_republic exports",
    4988: "Cross-Border Flows: Norway exports",
    4992: "Cross-Border Flows: Belgium exports",
}

scheduled_commerical_exchanges = {
    # Scheduled comercial exchanges
    4546: "Scheduled commercial exchanges: France imports",
    4404: "Scheduled commercial exchanges: France exports",
    4548: "Scheduled commercial exchanges: Netherlands imports",
    4406: "Scheduled commercial exchanges: Netherlands exports",
    4712: "Scheduled commercial exchanges: Belgium imports",
    4998: "Scheduled commercial exchanges: Belgium exports",
    4553: "Scheduled commercial exchanges: czech_republic imports",
    4412: "Scheduled commercial exchanges: czech_republic exports",
    4552: "Scheduled commercial exchanges: Switzerland imports",
    4410: "Scheduled commercial exchanges: Switzerland exports",
    4550: "Scheduled commercial exchanges: Poland imports",
    4408: "Scheduled commercial exchanges: Poland exports",
    4724: "Scheduled commercial exchanges: Norway imports",
    4722: "Scheduled commercial exchanges: Norway exports",
    4545: "Scheduled commercial exchanges: Denmark imports",
    4403: "Scheduled commercial exchanges: Denmark exports",
    4551: "Scheduled commercial exchanges: Sweden imports",
    4409: "Scheduled commercial exchanges: Sweden exports",
    4547: "Scheduled commercial exchanges: Luxembourg imports",
    4405: "Scheduled commercial exchanges: Luxembourg exports",
    4549: "Scheduled commercial exchanges: Austria imports",
    4407: "Scheduled commercial exchanges: Austria exports",
    4629: "Scheduled commercial exchanges: net exports",
}


def clean_filename(text):
    """
    Remove or replace problematic characters for filenames.
    """
    text = text.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(":", "_")
    # Replace other problematic characters
    text = re.sub(r'[<>:"|?*]', "", text)
    # Remove multiple consecutive underscores
    text = re.sub(r"_+", "_", text)
    # Remove leading/trailing underscores
    text = text.strip("_")
    return text.lower()


# Create a version of filter_dict that can be used for saving files
camel_dict = {key: clean_filename(value) for key, value in filter_dict.items()}

# Create camel dicts for each cross-border flow set
camel_dict_cross_border_de_lu = {
    key: clean_filename(value) for key, value in cross_border_de_lu_flows.items()
}
camel_dict_cross_border_de_at_lu = {
    key: clean_filename(value) for key, value in cross_border_de_at_lu_flows.items()
}

# Combined camel dict for all regions - used in processing for column naming
# Both DE-LU and DE-AT-LU cross-border flows have the same descriptions,
# so they'll produce the same column names and merge properly
camel_dict_all_regions = camel_dict.copy()
camel_dict_all_regions.update(camel_dict_cross_border_de_lu)
camel_dict_all_regions.update(camel_dict_cross_border_de_at_lu)

# Filtered dict excluding installed capacity (for faster downloads)
filter_dict_no_capacity = {k: v for k, v in filter_dict.items() if k not in EXCLUDED_KEYS}


def get_filter_dict_for_region(region="DE-LU", include_capacity=True):
    """Get the complete filter dict including region-specific cross-border flows.

    Args:
        region: 'DE-LU' (default) or 'DE-AT-LU'
        include_capacity: If False, exclude installed capacity keys

    Returns:
        Combined filter dict with appropriate cross-border flows for the region
    """
    base = filter_dict.copy() if include_capacity else filter_dict_no_capacity.copy()
    if region == "DE-AT-LU":
        base.update(cross_border_de_at_lu_flows)
    else:  # DE-LU is default
        base.update(cross_border_de_lu_flows)
    return base


def get_camel_dict_for_region(region="DE-LU"):
    """Get the complete camel dict including region-specific cross-border flows.

    Args:
        region: 'DE-LU' (default) or 'DE-AT-LU'

    Returns:
        Combined camel dict with appropriate cross-border flow filenames for the region
    """
    base = camel_dict.copy()
    if region == "DE-AT-LU":
        base.update(camel_dict_cross_border_de_at_lu)
    else:  # DE-LU is default
        base.update(camel_dict_cross_border_de_lu)
    return base
