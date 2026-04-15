"""
sector_map.py — Necessity mapping for Odin's Blindspot Index.
Maps GICS sub-industries to necessity scores (0-100) across 5 levels.
"""

# Level 100 — Civilization-critical: energy, nuclear, agriculture, water
# Level 80  — High necessity: defense, steel, fertilizers, shipping, copper
# Level 60  — Important: semiconductors, infrastructure, telecom, pharma
# Level 30  — Discretionary: consumer, finance, media
# Level 0   — Non-essential: luxury, crypto, meme, social media

NECESSITY_MAP = {
    # === Level 100: Civilization-critical ===
    "oil": 100, "gas": 100, "petroleum": 100, "integrated oil": 100,
    "oil & gas exploration": 100, "oil & gas production": 100,
    "oil & gas refining": 100, "oil & gas equipment": 100,
    "oil & gas drilling": 100, "oil & gas storage": 100,
    "nuclear": 100, "uranium": 100, "nuclear fuel": 100,
    "electric utilities": 100, "power": 100, "utilities": 100,
    "independent power": 100, "renewable electricity": 100,
    "water utilities": 100, "water": 100,
    "agricultural products": 100, "agriculture": 100,
    "crop": 100, "grain": 100, "food products": 100,
    "packaged foods": 100, "fertilizers": 100,
    "agricultural chemicals": 100,

    # === Level 80: High necessity ===
    "defense": 80, "aerospace & defense": 80,
    "steel": 80, "iron": 80, "metals & mining": 80,
    "copper": 80, "aluminum": 80, "industrial metals": 80,
    "diversified metals": 80, "precious metals": 80,
    "gold": 80, "silver": 80,
    "marine": 80, "marine shipping": 80, "shipping": 80,
    "marine transport": 80, "dry bulk": 80,
    "construction materials": 80, "building products": 80,
    "industrial machinery": 80, "heavy machinery": 80,
    "farm & heavy machinery": 80,
    "railroads": 80, "trucking": 80, "freight": 80,

    # === Level 60: Important ===
    "semiconductors": 60, "semiconductor equipment": 60,
    "semiconductor materials": 60,
    "telecom": 60, "telecommunications": 60,
    "integrated telecommunication": 60, "wireless": 60,
    "pharmaceuticals": 60, "biotech": 60, "biotechnology": 60,
    "health care equipment": 60, "medical devices": 60,
    "construction & engineering": 60, "infrastructure": 60,
    "industrial conglomerates": 60,
    "electrical components": 60, "electrical equipment": 60,
    "chemicals": 60, "specialty chemicals": 60,
    "diversified chemicals": 60, "commodity chemicals": 60,

    # === Level 30: Discretionary ===
    "consumer": 30, "consumer discretionary": 30,
    "retail": 30, "apparel": 30, "restaurants": 30,
    "hotels": 30, "leisure": 30, "gaming": 30,
    "banks": 30, "banking": 30, "diversified banks": 30,
    "regional banks": 30, "financial services": 30,
    "insurance": 30, "asset management": 30,
    "real estate": 30, "reits": 30,
    "media": 30, "broadcasting": 30, "advertising": 30,
    "auto": 30, "automobile": 30, "auto manufacturers": 30,

    # === Level 0: Non-essential ===
    "luxury": 0, "luxury goods": 0,
    "crypto": 0, "cryptocurrency": 0, "digital assets": 0,
    "social media": 0, "interactive media": 0,
    "entertainment": 0, "movies": 0, "music": 0,
    "casinos": 0, "gambling": 0,
    "application software": 0, "systems software": 0,
    "internet services": 0, "it services": 0,
}

# Ticker-level overrides for specific companies
TICKER_OVERRIDES = {
    # Uranium
    "CCJ": ("Uranium", 100),
    "UEC": ("Uranium", 100),
    "DNN": ("Uranium", 100),
    "NXE": ("Uranium", 100),
    "UUUU": ("Uranium", 100),
    "LEU": ("Uranium", 100),
    # Oil & Gas
    "OXY": ("Oil & Gas", 100),
    "DVN": ("Oil & Gas", 100),
    "EQNR.OL": ("Oil & Gas", 100),
    "CVE": ("Oil & Gas", 100),
    "SU": ("Oil & Gas", 100),
    "PBR": ("Oil & Gas", 100),
    # Copper
    "FCX": ("Copper", 80),
    "SCCO": ("Copper", 80),
    "TECK": ("Metals & Mining", 80),
    "BOL.ST": ("Metals & Mining", 80),
    # Steel
    "CLF": ("Steel", 80),
    "X": ("Steel", 80),
    "NUE": ("Steel", 80),
    "SSAB-A.ST": ("Steel", 80),
    # Fertilizers
    "MOS": ("Fertilizers", 100),
    "NTR": ("Fertilizers", 100),
    "CF": ("Fertilizers", 100),
    "YARA.OL": ("Fertilizers", 100),
    # Shipping
    "GOGL": ("Shipping", 80),
    "SBLK": ("Shipping", 80),
    "DAC": ("Shipping", 80),
    # Power
    "CEG": ("Nuclear Power", 100),
    "VST": ("Power", 100),
    "NRG": ("Power", 100),
    # Defense
    "LMT": ("Defense", 80),
    "RTX": ("Defense", 80),
    "NOC": ("Defense", 80),
    "SAAB-B.ST": ("Defense", 80),
    # Semiconductors
    "INTC": ("Semiconductors", 60),
    "MU": ("Semiconductors", 60),
    "TXN": ("Semiconductors", 60),
    "ASML": ("Semiconductor Equipment", 60),
    # Infrastructure
    "VMC": ("Construction Materials", 80),
    "MLM": ("Construction Materials", 80),
    "URI": ("Equipment Rental", 80),
    # Agriculture
    "ADM": ("Agriculture", 100),
    "BG": ("Agriculture", 100),
    # Nordic industrials
    "VOLV-B.ST": ("Heavy Machinery", 80),
    "NHY.OL": ("Aluminum", 80),
    "AKER.OL": ("Industrial Conglomerate", 60),
}
